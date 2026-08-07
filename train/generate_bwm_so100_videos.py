"""Generate competition-format SO100 videos from the BWM-5B LoRA model."""
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
DIFFSYNTH = REPO / "third_party/DiffSynth-Studio"
BWM_REPO = REPO / "third_party/boundless-world-model"
for dependency in (str(BWM_REPO), str(DIFFSYNTH)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from diffsynth.utils.data import save_video  # noqa: E402

from bwm_so100 import (  # noqa: E402
    build_bwm_action_tokens,
    load_delta_scale,
    normalize_so100_actions,
)
from train_bwm_so100_lora import BWMSO100TrainingModule  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B")
    )
    parser.add_argument(
        "--bwm-checkpoint",
        default=str(REPO / "checkpoints/Boundless-World-Model/step-12000.safetensors"),
    )
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--stats-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--limit", type=int, default=8, help="0 means all samples")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=32)
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


def letterbox(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Fit without distortion and return the content box for inverse mapping."""
    image = image.convert("RGB")
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized_w = max(1, round(image.width * scale))
    resized_h = max(1, round(image.height * scale))
    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized_w, top + resized_h)


def to_bwm_video(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
    return tensor[None, :, None]


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("BWM height and width must be divisible by 32")
    root = Path(args.challenge_root).resolve()
    output = Path(args.prediction_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids_all = sample_ids(root)
    selected = ids_all[: args.limit] if args.limit else ids_all
    if not selected:
        raise ValueError(f"No challenge samples found under {root}")
    if args.action_ablation in ("batch-roll", "all") and len(selected) < 2:
        raise ValueError("batch-roll needs at least two selected samples")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assigned = selected[rank::world_size]
    if not assigned:
        print(f"[BWM generate] rank {rank}/{world_size} has no samples")
        return
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = BWMSO100TrainingModule(
        model_root=args.model_root,
        bwm_checkpoint=args.bwm_checkpoint,
        lora_rank=args.lora_rank,
        resume_checkpoint=args.checkpoint,
        use_gradient_checkpointing=False,
        device=device,
    )
    model.eval()
    pipe = model.pipe

    delta_scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), "hybrid")
    raw_by_id: dict[str, torch.Tensor] = {}
    normalized_by_id: dict[str, torch.Tensor] = {}
    for sid in selected:
        raw = torch.from_numpy(np.load(root / "actions" / f"{sid}.npy").astype(np.float32))
        if raw.shape != (16, 6):
            raise ValueError(f"Expected {sid} actions [16,6], got {tuple(raw.shape)}")
        raw_by_id[sid] = raw
        normalized_by_id[sid] = normalize_so100_actions(raw, args.stats_root)

    modes = (
        ("none", "zero-motion", "batch-roll")
        if args.action_ablation == "all"
        else (args.action_ablation,)
    )
    torch.cuda.reset_peak_memory_stats(local_rank)
    started = time.perf_counter()
    generated = 0
    reports: dict[str, dict] = {}
    for mode in modes:
        mode_output = output / mode if args.action_ablation == "all" else output
        mode_output.mkdir(parents=True, exist_ok=True)
        mode_started = time.perf_counter()
        mode_done = 0
        for sid in assigned:
            target = mode_output / f"{sid}.mp4"
            if target.exists() and not args.overwrite:
                mode_done += 1
                generated += 1
                continue
            global_index = selected.index(sid)
            source_actions = normalized_by_id[sid]
            future_actions = source_actions
            if mode == "zero-motion":
                future_actions = source_actions[:1].expand_as(source_actions).clone()
            elif mode == "batch-roll":
                future_sid = selected[(global_index + 1) % len(selected)]
                future_actions = normalized_by_id[future_sid]
            tokens = build_bwm_action_tokens(
                future_actions,
                delta_scale,
                source_action=source_actions[0],
            )

            original = Image.open(root / "images" / f"{sid}.png").convert("RGB")
            conditioned, content_box = letterbox(original, (args.width, args.height))
            frames = pipe(
                input_video=to_bwm_video(conditioned),
                action=tokens[None],
                seed=args.seed + global_index,
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
                raise RuntimeError(f"BWM returned {len(frames)} frames for {sid}, expected 17")
            # Undo model letterboxing for challenge scoring. Frame zero is the
            # exact observed RGB image; only generated future frames are mapped.
            future = [
                frame.convert("RGB").crop(content_box).resize(
                    original.size, Image.Resampling.LANCZOS
                )
                for frame in frames[1:16]
            ]
            save_video(
                [original] + future,
                str(target),
                fps=args.fps,
                quality=args.video_quality,
            )
            mode_done += 1
            generated += 1
            print(
                f"[BWM {mode}] {mode_done}/{len(assigned)}, "
                f"{(time.perf_counter() - mode_started) / mode_done:.1f}s/sample"
            )
        reports[mode] = {
            "samples": mode_done,
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(mode_output),
        }

    elapsed = time.perf_counter() - started
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "bwm_checkpoint": str(Path(args.bwm_checkpoint).resolve()),
        "rank": rank,
        "world_size": world_size,
        "assigned_ids": assigned,
        "samples": generated,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(generated, 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        "num_inference_steps": args.num_inference_steps,
        "action_ablation": args.action_ablation,
        "action_contract": "17x18=[absolute6,source_delta6,step_delta6], clipped 3sigma to [-1,1]",
        "source_frame_zero_replaced": True,
        "inverse_letterbox": True,
        "modes": reports,
    }
    print(json.dumps(report, indent=2))
    report_path = Path(args.benchmark_json) if args.benchmark_json else output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(
            f"{report_path.stem}.rank{rank}{report_path.suffix or '.json'}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
