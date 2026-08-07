"""Generate competition-format videos with the conservative flow model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
from data_module import load_delta_scale, transform_actions  # noqa: E402
from flow_world_model import ConservativeFlowWorldModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    p.add_argument("--prediction-root", default=str(REPO / "diagnostics/flow_world"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--action-ablation", choices=("none", "zero", "reverse-time", "batch-roll"), default="none")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--benchmark-json")
    return p.parse_args()


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(p.stem for p in (root / "images").glob(f"{prefix}*.png"))


def prepare_image(path: Path, width: int, height: int) -> torch.Tensor:
    image = ImageOps.pad(
        Image.open(path).convert("RGB"),
        (width, height),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)


def prepare_action(raw: np.ndarray) -> torch.Tensor:
    stats = json.loads((REPO / "open/data/train/so100_action_statistics.json").read_text())
    action = torch.from_numpy(raw.astype(np.float32))
    action = (action - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])
    scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), "hybrid")
    return transform_actions(action, scale, "hybrid", 0)


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
    root = Path(args.challenge_root)
    output = Path(args.prediction_root)
    output.mkdir(parents=True, exist_ok=True)
    ids = sample_ids(root)[: args.limit]
    all_ids = sample_ids(root)

    model = ConservativeFlowWorldModel(pretrained_encoder=False)
    state = load_file(args.checkpoint, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model = model.cuda().eval()

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    done = 0
    for start in range(0, len(ids), args.batch_size):
        batch_ids = ids[start : start + args.batch_size]
        images = []
        actions = []
        for sid in batch_ids:
            images.append(prepare_image(root / "images" / f"{sid}.png", args.width, args.height))
            action_sid = sid
            if args.action_ablation == "batch-roll":
                action_sid = all_ids[(all_ids.index(sid) + 1) % len(all_ids)]
            action = prepare_action(np.load(root / "actions" / f"{action_sid}.npy"))
            if args.action_ablation == "zero":
                action.zero_()
            elif args.action_ablation == "reverse-time":
                action = action.flip(0)
            actions.append(action)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(torch.stack(images).cuda(), torch.stack(actions).cuda()).float()
        for sid, video in zip(batch_ids, prediction):
            path = output / f"{sid}.mp4"
            if args.overwrite or not path.exists():
                save_mp4(video, path)
            done += 1
            elapsed = time.perf_counter() - started
            print(f"[Flow generate] {done}/{len(ids)}, {elapsed / done:.2f}s/sample")

    elapsed = time.perf_counter() - started
    report = {
        "samples": done,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(done, 1),
        "projected_total_samples": len(all_ids),
        "projected_seconds": elapsed / max(done, 1) * len(all_ids),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frames_saved": 16,
    }
    print(json.dumps(report, indent=2))
    if args.benchmark_json:
        target = Path(args.benchmark_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
