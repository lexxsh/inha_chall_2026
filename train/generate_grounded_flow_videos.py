"""Generate challenge videos with a frozen grounding probe and flow renderer."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
SUBMISSION_KIT = REPO / "open/submission_kit"
for path in (REPO, REPO / "train", SUBMISSION_KIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_csv_utils import load_dino_model, resolve_dino_image_size  # noqa: E402
from flow_world_model import ConservativeFlowWorldModel  # noqa: E402
from tools.cache_so100_grounding_features import extract_ragged_dino_features  # noqa: E402
from train_grounded_flow_renderer import load_grounder, motion_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-checkpoint", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, default=REPO / "valset_holdout")
    parser.add_argument(
        "--prediction-root", type=Path, default=REPO / "diagnostics/grounded_flow_renderer"
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--action-ablation", choices=("none", "zero", "batch-roll"), default="none"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json", type=Path)
    return parser.parse_args()


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(path.stem for path in (root / "images").glob(f"{prefix}*.png"))


def padded_source(image: np.ndarray, width: int, height: int) -> torch.Tensor:
    pil = ImageOps.pad(
        Image.fromarray(image),
        (width, height),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )
    return torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float().div(255)


def save_mp4(video: torch.Tensor, path: Path, fps: int = 6) -> None:
    frames = video.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Generation requires an explicitly selected CUDA GPU.")
    device = torch.device("cuda")
    ids_all = sample_ids(args.challenge_root)
    ids = ids_all[: args.limit] if args.limit else ids_all
    args.prediction_root.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    dino_dim = int(config["grounding_dim"])
    renderer = ConservativeFlowWorldModel(
        grounding_dim=dino_dim, pretrained_encoder=False
    )
    renderer.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    renderer = renderer.to(device).eval()
    grounder = load_grounder(args.grounding_checkpoint, dino_dim, device)
    dino_name = "vit_small_patch14_dinov2.lvd142m"
    dino = load_dino_model(device, dino_name, pretrained=True)
    dino_size = resolve_dino_image_size(dino, 0)
    stats = json.loads((REPO / "open/data/train/so100_action_statistics.json").read_text())
    action_mean = torch.tensor(stats["mean"], device=device)
    action_std = torch.tensor(stats["std"], device=device).clamp_min(1e-6)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    done = 0
    for begin in range(0, len(ids), args.batch_size):
        batch_ids = ids[begin : begin + args.batch_size]
        raw_images = [
            np.asarray(Image.open(args.challenge_root / "images" / f"{sid}.png").convert("RGB"))
            for sid in batch_ids
        ]
        source_dino = extract_ragged_dino_features(
            [image[None] for image in raw_images], dino, device, dino_size
        )[:, 0].to(device)
        sources = torch.stack(
            [padded_source(image, args.width, args.height) for image in raw_images]
        ).to(device)

        action_ids = batch_ids
        if args.action_ablation == "batch-roll":
            action_ids = [ids_all[(ids_all.index(sid) + 1) % len(ids_all)] for sid in batch_ids]
        actions = torch.stack(
            [
                torch.from_numpy(
                    np.load(args.challenge_root / "actions" / f"{sid}.npy").astype(np.float32)
                )
                for sid in action_ids
            ]
        ).to(device)
        if args.action_ablation == "zero":
            actions = actions[:, :1].expand_as(actions)
        normalized_actions = (actions - action_mean) / action_std

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = motion_tokens(grounder, source_dino, normalized_actions)
            prediction = renderer(sources, motion_tokens=tokens).float()
        for sid, video in zip(batch_ids, prediction, strict=True):
            destination = args.prediction_root / f"{sid}.mp4"
            if args.overwrite or not destination.exists():
                save_mp4(video, destination)
            done += 1
            print(f"[Grounded flow] {done}/{len(ids)}")

    elapsed = time.perf_counter() - started
    report = {
        "samples": done,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(done, 1),
        "projected_total_samples": len(ids_all),
        "projected_seconds": elapsed / max(done, 1) * len(ids_all),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "frames_saved": 16,
        "action_ablation": args.action_ablation,
    }
    print(json.dumps(report, indent=2))
    if args.benchmark_json:
        args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
