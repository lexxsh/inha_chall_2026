"""Generate challenge videos with the native SO100 BWM/Wan2.2 model."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
for dependency in (
    REPO / "train",
    REPO / "third_party/boundless-world-model",
    REPO / "third_party/DiffSynth-Studio",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from diffsynth.core import ModelConfig  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402
from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline  # noqa: E402
from bwm_native_so100 import (  # noqa: E402
    ACTION_DIM,
    build_action_tokens,
    local_wan22_model_paths,
    normalize_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B"))
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--stats-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--limit", type=int, default=8, help="0 means all")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--video-quality", type=int, default=9)
    parser.add_argument(
        "--action-ablation",
        choices=("none", "zero-motion", "batch-roll", "all"),
        default="none",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json")
    return parser.parse_args()


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(path.stem for path in (root / "images").glob(f"{prefix}*.png"))


def letterbox(image: Image.Image, size: tuple[int, int]):
    image = image.convert("RGB")
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (target_w - width) // 2, (target_h - height) // 2
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


def input_video(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return (torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0)[None, :, None]


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("height and width must be divisible by 32")
    root = Path(args.challenge_root).resolve()
    output = Path(args.prediction_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids_all = sample_ids(root)
    selected = ids_all[: args.limit] if args.limit else ids_all
    if not selected:
        raise ValueError(f"No challenge samples found under {root}")
    if args.action_ablation in ("batch-roll", "all") and len(selected) < 2:
        raise ValueError("batch-roll needs at least two samples")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assigned = selected[rank::world_size]
    if not assigned:
        print(f"[BWM native] rank {rank}/{world_size} has no samples")
        return
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    shards, vae = local_wan22_model_paths(args.model_root)
    pipe = build_wan_video_action_pipeline(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=shards), ModelConfig(path=vae)],
        tokenizer_config=None,
        redirect_common_files=False,
        ckpt_path=args.checkpoint,
        action_dim=ACTION_DIM,
        action_mode="adaln",
    )
    pipe.use_gradient_checkpointing = False
    pipe.use_gradient_checkpointing_offload = False
    pipe.eval()

    normalized = {}
    for sid in selected:
        raw = torch.from_numpy(np.load(root / "actions" / f"{sid}.npy").astype(np.float32))
        if raw.shape != (16, ACTION_DIM):
            raise ValueError(f"{sid}: expected [16,6], got {tuple(raw.shape)}")
        normalized[sid] = normalize_actions(raw, args.stats_root)

    modes = (
        ("none", "zero-motion", "batch-roll")
        if args.action_ablation == "all"
        else (args.action_ablation,)
    )
    torch.cuda.reset_peak_memory_stats(local_rank)
    started = time.perf_counter()
    total = 0
    reports = {}
    for mode in modes:
        mode_output = output / mode if args.action_ablation == "all" else output
        mode_output.mkdir(parents=True, exist_ok=True)
        mode_started, done = time.perf_counter(), 0
        for sid in assigned:
            target = mode_output / f"{sid}.mp4"
            if target.exists() and not args.overwrite:
                done += 1
                total += 1
                continue
            index = selected.index(sid)
            action = normalized[sid]
            if mode == "zero-motion":
                action = action[:1].expand_as(action).clone()
            elif mode == "batch-roll":
                action = normalized[selected[(index + 1) % len(selected)]]
            tokens = build_action_tokens(action)
            original = Image.open(root / "images" / f"{sid}.png").convert("RGB")
            conditioned, box = letterbox(original, (args.width, args.height))
            frames = pipe(
                input_video=input_video(conditioned),
                action=tokens[None],
                seed=args.seed + index,
                rand_device="cpu",
                height=args.height,
                width=args.width,
                num_frames=17,
                num_history_frames=1,
                cfg_scale=1.0,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                tiled=False,
                output_type="quantized",
            )
            if len(frames) != 17:
                raise RuntimeError(f"Expected 17 decoded frames for {sid}, got {len(frames)}")
            future = [
                frame.convert("RGB").crop(box).resize(original.size, Image.Resampling.LANCZOS)
                for frame in frames[1:16]
            ]
            # Exact source replacement is part of the competition contract.
            save_video([original] + future, str(target), fps=args.fps, quality=args.video_quality)
            done += 1
            total += 1
            print(f"[BWM native {mode}] {done}/{len(assigned)}, "
                  f"{(time.perf_counter() - mode_started) / done:.1f}s/sample")
        reports[mode] = {
            "samples": done,
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(mode_output),
        }

    elapsed = time.perf_counter() - started
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "initialization": "vanilla Wan2.2-TI2V-5B; no public BWM checkpoint",
        "rank": rank,
        "world_size": world_size,
        "samples": total,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(total, 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        "action_contract": "17x6=[a0 history proxy, a0..a15 future], clipped 3sigma",
        "source_frame_zero_replaced": True,
        "modes": reports,
    }
    print(json.dumps(report, indent=2))
    report_path = Path(args.benchmark_json) if args.benchmark_json else output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(f"{report_path.stem}.rank{rank}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
