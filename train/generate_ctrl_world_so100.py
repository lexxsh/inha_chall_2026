"""Autoregressive competition rollout with the released Ctrl-World pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
CTRL_WORLD = REPO / "third_party/Ctrl-World"
for dependency in (str(REPO), str(CTRL_WORLD)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

from models.ctrl_world import CrtlWorld  # noqa: E402
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402
# See train_ctrl_world_so100.py: direct path execution puts this directory at
# sys.path[0], where train.py would otherwise shadow the ``train`` namespace.
from ctrl_world_so100 import (  # noqa: E402
    build_rollout_pose_tokens,
    ctrl_world_args,
    load_ctrl_world_checkpoint,
    load_pose_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--svd-model-path",
        type=Path,
        default=REPO / "checkpoints/stable-video-diffusion-img2vid",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=REPO / "results/ctrl_world_so100_pose_stats.json",
    )
    parser.add_argument("--challenge-root", type=Path, default=REPO / "valset_holdout")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8, help="0 means all")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--action-ablation",
        choices=("none", "zero-motion", "batch-roll", "all"),
        default="none",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json", type=Path)
    return parser.parse_args()


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(path.stem for path in (root / "images").glob(f"{prefix}*.png"))


def letterbox(image: Image.Image, width: int, height: int):
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized_width, top + resized_height)


def pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        .div(127.5)
        .sub(1.0)
    )


@torch.inference_mode()
def encode_source(model: CrtlWorld, image: Image.Image, device, dtype) -> torch.Tensor:
    tensor = pil_to_tensor(image, device, dtype)
    latent = model.vae.encode(tensor).latent_dist.sample()
    return latent.mul(model.vae.config.scaling_factor)


@torch.inference_mode()
def decode_chunk(model: CrtlWorld, latent: torch.Tensor) -> list[Image.Image]:
    batch, frames = latent.shape[:2]
    if batch != 1:
        raise ValueError(f"Generation decoder expects batch one, got {batch}")
    value = latent.flatten(0, 1) / model.vae.config.scaling_factor
    decoded = model.vae.decode(value, num_frames=frames).sample
    decoded = decoded.reshape(batch, frames, *decoded.shape[1:])[0]
    decoded = decoded.float().add(1.0).div(2.0).clamp(0, 1)
    arrays = decoded.permute(0, 2, 3, 1).mul(255).round().byte().cpu().numpy()
    return [Image.fromarray(array, mode="RGB") for array in arrays]


@torch.inference_mode()
def generate_one(
    model: CrtlWorld,
    source: Image.Image,
    pose_chunks: list[torch.Tensor],
    *,
    height: int,
    width: int,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[list[Image.Image], dict]:
    device = model.unet.device
    dtype = model.unet.dtype
    conditioned, content_box = letterbox(source, width, height)
    generator = torch.Generator(device=device).manual_seed(seed)
    source_latent = encode_source(model, conditioned, device, dtype)
    latent_history = [source_latent.clone() for _ in range(24)]
    history_offsets = (0, 0, -8, -6, -4, -2)
    current_latent = source_latent
    generated: list[Image.Image] = []
    chunk_reports = []
    for chunk_index, poses in enumerate(pose_chunks):
        history = torch.cat([latent_history[offset] for offset in history_offsets], dim=0)
        history = history.unsqueeze(0)
        poses = poses.unsqueeze(0).to(device=device, dtype=dtype)
        hidden = model.action_encoder(poses, frame_level_cond=True)
        started = time.perf_counter()
        _, predicted = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline,
            image=current_latent,
            text=hidden,
            width=width,
            height=height,
            num_frames=5,
            history=history,
            num_inference_steps=inference_steps,
            decode_chunk_size=5,
            max_guidance_scale=guidance_scale,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            generator=generator,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
            his_cond_zero=False,
        )
        decoded = decode_chunk(model, predicted)
        # Predicted frame zero reconstructs the current frame.  Only the four
        # following outcomes correspond to new action-conditioned observations.
        for frame in decoded[1:]:
            frame = frame.crop(content_box).resize(source.size, Image.Resampling.LANCZOS)
            generated.append(frame)
        current_latent = predicted[:, -1]
        latent_history.append(current_latent.clone())
        chunk_reports.append(
            {
                "chunk": chunk_index,
                "seconds": time.perf_counter() - started,
                "latent_shape": list(predicted.shape),
            }
        )
    # The challenge scores frame0 + 15 outcomes. action[15] targets frame16,
    # which is generated for native chunk alignment and deliberately dropped.
    return [source] + generated[:15], {"chunks": chunk_reports, "generated_then_kept": [16, 15]}


def main() -> None:
    args = parse_args()
    if not args.svd_model_path.is_dir():
        raise FileNotFoundError(f"Missing official SVD base: {args.svd_model_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by the SVD VAE factor 8")
    root = args.challenge_root.resolve()
    output = args.prediction_root.resolve()
    ids = sample_ids(root)
    selected = ids[: args.limit] if args.limit else ids
    if not selected:
        raise ValueError(f"No challenge samples under {root}")
    if args.action_ablation in {"batch-roll", "all"} and len(selected) < 2:
        raise ValueError("batch-roll requires at least two selected samples")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assigned = selected[rank::world_size]
    if not assigned:
        print(f"[Ctrl-World generate] rank {rank}/{world_size}: no assigned samples")
        return
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    model = CrtlWorld(
        ctrl_world_args(
            svd_model_path=args.svd_model_path,
            width=args.width,
            height=args.height,
        )
    )
    load_report = load_ctrl_world_checkpoint(model, args.checkpoint)
    model.to(device=device, dtype=dtype).eval()
    low, high = load_pose_statistics(args.stats_path)
    raw_actions = {
        sample_id: torch.from_numpy(
            np.load(root / "actions" / f"{sample_id}.npy").astype(np.float32)
        )
        for sample_id in selected
    }
    modes = (
        ("none", "zero-motion", "batch-roll")
        if args.action_ablation == "all"
        else (args.action_ablation,)
    )
    torch.cuda.reset_peak_memory_stats(local_rank)
    started_all = time.perf_counter()
    reports = {}
    for mode in modes:
        mode_output = output / mode if args.action_ablation == "all" else output
        mode_output.mkdir(parents=True, exist_ok=True)
        mode_started = time.perf_counter()
        samples = []
        for sample_index, sample_id in enumerate(assigned, start=1):
            target = mode_output / f"{sample_id}.mp4"
            if target.exists() and not args.overwrite:
                continue
            global_index = selected.index(sample_id)
            source_actions = raw_actions[sample_id]
            future_actions = source_actions
            if mode == "zero-motion":
                future_actions = source_actions[:1].expand_as(source_actions).clone()
            elif mode == "batch-roll":
                rolled_id = selected[(global_index + 1) % len(selected)]
                future_actions = raw_actions[rolled_id]
            pose_chunks, _ = build_rollout_pose_tokens(
                future_actions,
                low,
                high,
                source_pose=source_actions[0],
            )
            source = Image.open(root / "images" / f"{sample_id}.png").convert("RGB")
            frames, sample_report = generate_one(
                model,
                source,
                pose_chunks,
                height=args.height,
                width=args.width,
                inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + global_index,
            )
            if len(frames) != 16:
                raise RuntimeError(f"Expected 16 submission frames, got {len(frames)}")
            iio.imwrite(target, np.stack([np.asarray(frame) for frame in frames]), fps=args.fps)
            samples.append({"sample_id": sample_id, **sample_report})
            print(f"[Ctrl-World {mode}] {sample_index}/{len(assigned)} -> {target}")
        reports[mode] = {
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(mode_output),
            "samples": samples,
        }

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "svd_model_path": str(args.svd_model_path.resolve()),
        "checkpoint_load": load_report,
        "rank": rank,
        "world_size": world_size,
        "assigned_ids": assigned,
        "seconds": time.perf_counter() - started_all,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        "num_inference_steps": args.num_inference_steps,
        "action_ablation": args.action_ablation,
        "contract": "released Ctrl-World 6-history + current reconstruction + 4 futures, 4 AR chunks",
        "source_pose_proxy": "sample's action[0] in native SO100 joint coordinates",
        "reports": reports,
    }
    report_path = args.benchmark_json or output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(
            f"{report_path.stem}.rank{rank}{report_path.suffix or '.json'}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {report_path.resolve()}")


if __name__ == "__main__":
    main()
