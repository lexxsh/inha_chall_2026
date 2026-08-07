"""Generate challenge videos with source-anchored SO100 EEF14 controls."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bwm_so100_eef import DEFAULT_STATS, SO100EEFConverter
from generate_bwm_so100_videos import letterbox, sample_ids, to_bwm_video
from train_bwm_so100_eef_lora import ACTION_CONTRACT
from train_bwm_so100_lora import BWMSO100TrainingModule, REPO
from diffsynth.utils.data import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Omit for the untouched public-BWM zero-shot audit")
    parser.add_argument("--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B"))
    parser.add_argument(
        "--bwm-checkpoint",
        default=str(REPO / "checkpoints/Boundless-World-Model/step-12000.safetensors"),
    )
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS))
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--video-quality", type=int, default=9)
    parser.add_argument(
        "--action-ablation", choices=("none", "zero-motion", "batch-roll", "all"), default="none"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json")
    return parser.parse_args()


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
        raise ValueError("batch-roll needs at least two samples")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assigned = selected[rank::world_size]
    if not assigned:
        print(f"[BWM EEF14] rank {rank}/{world_size} has no samples")
        return
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = BWMSO100TrainingModule(
        model_root=args.model_root,
        bwm_checkpoint=args.bwm_checkpoint,
        lora_rank=args.lora_rank,
        resume_checkpoint=args.checkpoint,
        use_gradient_checkpointing=False,
        action_dim=14,
        expand_action_inputs=False,
        action_contract=ACTION_CONTRACT,
        device=device,
    )
    model.eval()
    pipe = model.pipe
    converter = SO100EEFConverter(args.stats_path)
    raw_by_id = {
        sid: np.load(root / "actions" / f"{sid}.npy").astype(np.float32) for sid in selected
    }
    for sid, raw in raw_by_id.items():
        if raw.shape != (16, 6):
            raise ValueError(f"Expected {sid} action [16,6], got {raw.shape}")

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
            future = raw_by_id[sid]
            if mode == "zero-motion":
                future = np.repeat(future[:1], len(future), axis=0)
            elif mode == "batch-roll":
                future = raw_by_id[selected[(global_index + 1) % len(selected)]]
            # Every trajectory is anchored at its own first command.  Thus the
            # batch-roll control swaps motion, not unrelated servo zero offsets.
            tokens = converter.build_tokens(future, source_action=future[0])

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
                raise RuntimeError(f"BWM returned {len(frames)} frames for {sid}")
            future_frames = [
                frame.convert("RGB").crop(content_box).resize(original.size, Image.Resampling.LANCZOS)
                for frame in frames[1:16]
            ]
            save_video([original] + future_frames, str(target), fps=args.fps, quality=args.video_quality)
            mode_done += 1
            generated += 1
            print(
                f"[BWM EEF14 {mode}] {mode_done}/{len(assigned)}, "
                f"{(time.perf_counter() - mode_started) / mode_done:.1f}s/sample"
            )
        reports[mode] = {
            "samples": mode_done,
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(mode_output),
        }
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "public_bwm_zero_shot": args.checkpoint is None,
        "bwm_checkpoint": str(Path(args.bwm_checkpoint).resolve()),
        "rank": rank,
        "world_size": world_size,
        "assigned_ids": assigned,
        "samples": generated,
        "seconds": time.perf_counter() - started,
        "seconds_per_sample": (time.perf_counter() - started) / max(generated, 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        "num_inference_steps": args.num_inference_steps,
        "action_ablation": args.action_ablation,
        "action_contract": ACTION_CONTRACT,
        "source_frame_zero_replaced": True,
        "inverse_letterbox": True,
        "modes": reports,
    }
    print(json.dumps(report, indent=2))
    report_path = Path(args.benchmark_json) if args.benchmark_json else output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(f"{report_path.stem}.rank{rank}{report_path.suffix or '.json'}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
