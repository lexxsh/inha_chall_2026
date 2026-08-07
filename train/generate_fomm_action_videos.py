"""Generate challenge videos from an action-to-keypoint FOMM checkpoint.

Unlike the renderer oracle audit, this path never reads future/target video.
Its only per-sample inputs are the challenge source image and 16 SO-100 actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"
KIT_SRC = REPO / "open" / "baseline" / "challenge_kit" / "src"
for dependency in (str(TRAIN), str(KIT_SRC)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

from ldwma.datasets.lerobot_so100 import preprocess_video  # noqa: E402
from source_anchored_fomm import ActionKeypointPredictor, SourceAnchoredFOMM  # noqa: E402
from train_fomm_action_kp import features, load_scales  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--renderer-checkpoint", type=Path)
    parser.add_argument("--challenge-root", type=Path, default=REPO / "open/data/eval")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json", type=Path)
    return parser.parse_args()


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(path.stem for path in (root / "images").glob(f"{prefix}*.png"))


def load_source(path: Path, height: int, width: int, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"))[None]
    # Same pad=True preprocessing used by the oracle renderer's training clip.
    source = preprocess_video(image, height, width, pad=True)[:, 0]
    return source.add(1.0).mul(0.5).unsqueeze(0).to(device)


def save_video(path: Path, frames: torch.Tensor, fps: float) -> None:
    array = frames.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.width, stream.height, stream.pix_fmt = array.shape[2], array.shape[1], "yuv420p"
        stream.options = {"crf": "12", "preset": "slow"}
        for image in array:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    args = parse_args()
    predictor_path = args.checkpoint.resolve()
    if not predictor_path.is_file():
        raise FileNotFoundError(predictor_path)
    config = json.loads((predictor_path.parent / "config.json").read_text())
    renderer_path = (args.renderer_checkpoint or Path(config["renderer_checkpoint"])).resolve()
    if not renderer_path.is_file():
        raise FileNotFoundError(renderer_path)

    root = args.challenge_root.resolve()
    output = args.prediction_root.resolve()
    ids = sample_ids(root)
    selected = ids[: args.limit] if args.limit else ids
    if not selected:
        raise ValueError(f"No samples found under {root}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assigned = selected[rank::world_size]
    if not assigned:
        print(f"[FOMM action generate] rank {rank}/{world_size}: no assigned samples")
        return
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    renderer = SourceAnchoredFOMM(config["num_kp"], config["model_size"])
    renderer.load_state_dict(load_file(str(renderer_path), device="cpu"))
    renderer.to(device).eval().requires_grad_(False)
    predictor = ActionKeypointPredictor(
        config["num_kp"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    predictor.load_state_dict(load_file(str(predictor_path), device="cpu"))
    predictor.to(device).eval().requires_grad_(False)
    scales = load_scales(device)

    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(local_rank)
    started = time.perf_counter()
    generated = 0
    with torch.inference_mode():
        for local_index, sample_id in enumerate(assigned, start=1):
            target = output / f"{sample_id}.mp4"
            if target.exists() and not args.overwrite:
                continue
            source = load_source(
                root / "images" / f"{sample_id}.png",
                config["height"],
                config["width"],
                device,
            )
            action = torch.from_numpy(
                np.load(root / "actions" / f"{sample_id}.npy").astype(np.float32)
            ).unsqueeze(0).to(device)
            if action.shape != (1, 16, 6):
                raise ValueError(f"{sample_id}: expected actions (16,6), got {tuple(action.shape[1:])}")
            source_kp = renderer.kp_detector(source)
            predicted_kp = predictor(features(action, scales), source_kp)
            driving_kp = {
                "value": predicted_kp["value"][0],
                "jacobian": predicted_kp["jacobian"][0],
            }
            source_frames = source.expand(16, -1, -1, -1)
            video = renderer(source_frames, driving_kp=driving_kp)["prediction"]
            video = torch.cat([source, video[1:]], dim=0)
            save_video(target, video, args.fps)
            generated += 1
            print(f"[FOMM action] {local_index}/{len(assigned)} -> {target}", flush=True)

    report = {
        "checkpoint": str(predictor_path),
        "renderer_checkpoint": str(renderer_path),
        "challenge_root": str(root),
        "prediction_root": str(output),
        "rank": rank,
        "world_size": world_size,
        "assigned": len(assigned),
        "generated": generated,
        "seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        "input_contract": "source RGB + 16x6 actions only; no future video",
    }
    report_path = args.benchmark_json or output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(f"{report_path.stem}.rank{rank}.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
