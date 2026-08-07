"""Generate competition-format videos from an SO-100-adapted IRASim model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IRASIM_SRC = REPO / "third_party/IRASim"
if str(IRASIM_SRC) not in sys.path:
    sys.path.insert(0, str(IRASIM_SRC))

from diffusers.models import AutoencoderKL  # noqa: E402
from diffusers.schedulers import PNDMScheduler  # noqa: E402
from irasim_so100 import (  # noqa: E402
    build_irasim,
    load_irasim_checkpoint,
    preprocess_irasim_frames,
    prepare_eval_actions,
)
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--weight-branch",
        choices=("ema", "model"),
        default="ema",
        help="Checkpoint branch to evaluate. Short transfer runs must compare both; "
        "EMA decay 0.9999 intentionally lags the main weights.",
    )
    p.add_argument(
        "--public-zero-shot",
        action="store_true",
        help="Allow the exact public RT-1 checkpoint with the zero-initialized SO-100 adapter. "
        "This diagnoses the native video prior only; it cannot follow SO-100 actions.",
    )
    p.add_argument("--vae", default=str(REPO / "models/IRASim/sdxl-base"))
    p.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    p.add_argument("--prediction-root", default=str(REPO / "diagnostics/irasim_action"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1, help="Reserved; official pipeline runs one sample at a time.")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--action-mode", choices=("absolute", "delta", "delta_step"), default="absolute")
    p.add_argument(
        "--source-state-root",
        type=Path,
        help="Train-only oracle/predicted source-state .npy directory. Required for relative "
        "modes because challenge evaluation does not provide proprioception.",
    )
    p.add_argument(
        "--action-ablation",
        choices=("none", "zero", "reverse-time", "batch-roll"),
        default="none",
    )
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--benchmark-json")
    return p.parse_args()


def ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(path.stem for path in (root / "images").glob(f"{prefix}*.png"))


def source_latent(image_path: Path, vae, device, generator: torch.Generator) -> torch.Tensor:
    array = np.asarray(Image.open(image_path).convert("RGB"))[None].copy()
    # [C,T,H,W] -> [T,C,H,W]; identical helper and interpolation settings to
    # training, rather than a separate PIL resize implementation.
    tensor = preprocess_irasim_frames(array).permute(1, 0, 2, 3)
    tensor = tensor.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample(generator=generator)
        latent = latent * vae.config.scaling_factor
    return latent.unsqueeze(1)


def save_competition_video(video: torch.Tensor, source_path: Path, path: Path) -> None:
    """Save source frame 0 plus generated frames 1..15.

    Official IRASim evaluation also replaces the VAE-decoded masked frame with
    the true source RGB.  The challenge GT confirms that its input PNG equals
    target frame 0 exactly.
    """
    if video.shape[1] != 16:
        raise ValueError(f"Expected 16 IRASim frames, got {video.shape[1]}")
    frames = ((video[0].float().clamp(-1, 1) + 1) * 127.5).byte()
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    restored = [
        np.asarray(Image.fromarray(frame).resize((640, 480), Image.Resampling.BILINEAR))
        for frame in frames
    ]
    restored[0] = np.asarray(
        Image.open(source_path).convert("RGB").resize((640, 480), Image.Resampling.BILINEAR)
    )
    writer = imageio.get_writer(path, fps=6, codec="libx264", quality=5)
    for frame in restored:
        writer.append_data(frame)
    writer.close()


def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 4:
        raise ValueError(
            "IRASim's official PNDM scheduler requires at least 4 inference steps "
            "for its PRK warm-up. Use --num-inference-steps 4 or greater."
        )
    root = Path(args.challenge_root)
    output = Path(args.prediction_root)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    model = build_irasim(action_dim=6, num_frames=16)
    report = load_irasim_checkpoint(
        model, args.checkpoint, branch=args.weight_branch
    )
    if report["format"] == "public_rt1" and not args.public_zero_shot:
        raise RuntimeError(
            "The public RT-1 checkpoint has an untrained SO-100 adapter. "
            "Use an adapted checkpoint, or pass --public-zero-shot only for a native-prior diagnostic."
        )
    if report["format"] == "so100_adapted" and (report["mismatched"] or report["missing"]):
        raise RuntimeError(f"Adapted checkpoint is incomplete: {report}")
    if args.public_zero_shot and report["format"] != "public_rt1":
        raise RuntimeError("--public-zero-shot requires the public RT-1 checkpoint")
    if args.public_zero_shot:
        print("[IRASim] PUBLIC ZERO-SHOT: SO-100 adapter is exactly zero; action fidelity is not evaluated")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(
        device=device, dtype=torch.bfloat16
    ).eval()
    scheduler = PNDMScheduler.from_pretrained(
        str(IRASIM_SRC / "pretrained_models/scheduler"),
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="learned_range",
    )
    pipeline = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

    selected = ids(root)[:args.limit]
    if args.action_mode != "absolute" and args.source_state_root is None:
        raise ValueError(
            f"--action-mode {args.action_mode} requires --source-state-root. "
            "Do not substitute action[0] for the missing source state."
        )
    prepared_actions = [
        prepare_eval_actions(
            np.load(root / "actions" / f"{sample_id}.npy"),
            args.action_mode,
            (
                np.load(args.source_state_root / f"{sample_id}.npy")
                if args.source_state_root is not None
                else None
            ),
        )
        for sample_id in selected
    ]
    if args.action_ablation == "batch-roll" and len(prepared_actions) < 2:
        raise ValueError("batch-roll ablation requires --limit 2 or greater")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for index, sample_id in enumerate(selected):
        path = output / f"{sample_id}.mp4"
        if path.exists() and not args.overwrite:
            continue
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        latent = source_latent(
            root / "images" / f"{sample_id}.png", vae, device, generator
        )
        actions = prepared_actions[index].clone()
        if args.action_ablation == "zero":
            actions.zero_()
        elif args.action_ablation == "reverse-time":
            actions = actions.flip(0)
        elif args.action_ablation == "batch-roll":
            actions = prepared_actions[(index + 1) % len(prepared_actions)]
        with torch.no_grad():
            video, _ = pipeline(
                actions.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                mask_x=latent,
                video_length=16,
                height=256,
                width=320,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,
                generator=generator,
                device=device,
                return_dict=False,
                output_type="both",
            )
        save_competition_video(
            video,
            root / "images" / f"{sample_id}.png",
            path,
        )
        print(f"[IRASim generate] {index + 1}/{len(selected)}")

    seconds = time.perf_counter() - start
    if args.benchmark_json:
        Path(args.benchmark_json).write_text(json.dumps({
            "samples": len(selected),
            "seconds": seconds,
            "seconds_per_sample": seconds / max(1, len(selected)),
            "num_inference_steps": args.num_inference_steps,
            "frames_saved": 16,
            "frame_layout": "source_0_plus_generated_1_to_15",
            "weight_branch": args.weight_branch,
            "peak_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if torch.cuda.is_available() else 0.0
            ),
        }, indent=2))


if __name__ == "__main__":
    main()
