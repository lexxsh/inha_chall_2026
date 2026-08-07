"""A/B gate: can vanilla Wan2.1 I2V animate a challenge image at all?

This intentionally loads no DreamZero weights and supplies no robot action/state.
It is a generator-capability diagnostic, not a competition prediction method.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402


DEFAULT_PROMPT = (
    "A fixed-camera photorealistic video. The black tabletop robot arm clearly and "
    "smoothly moves its gripper down and to the right toward the white charger, then "
    "closes the gripper. The robot keeps the same identity and rigid mechanical shape. "
    "The white charger, paper, table, mat, background, lighting, and camera remain static."
)

DEFAULT_NEGATIVE_PROMPT = (
    "static, still image, no motion, frozen robot, camera motion, camera shake, zoom, "
    "blurry, low quality, deformation, warped robot, extra robot arm, extra gripper, "
    "melting, morphing, newly appearing objects, disappearing objects, scene change"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P"),
    )
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--sample-id", default="val_000009")
    parser.add_argument(
        "--output-root", default=str(REPO / "diagnostics/wan21_vanilla_motion")
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    model = Path(args.model_path).resolve()
    source = Path(args.challenge_root).resolve() / "images" / f"{args.sample_id}.png"
    required = [
        model / "models_t5_umt5-xxl-enc-bf16.pth",
        model / "Wan2.1_VAE.pth",
        model / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        model / "google/umt5-xxl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    shards = sorted(model.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        missing.append(str(model / "diffusion_pytorch_model-*.safetensors"))
    if not source.exists():
        missing.append(str(source))
    if missing:
        raise FileNotFoundError("Missing required local files:\n  " + "\n  ".join(missing))
    if args.num_frames < 2 or (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num-frames must be 4k+1 (17 is the competition diagnostic).")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16.")
    return model, source


def sharpness(frame: np.ndarray) -> float:
    gray = frame.astype(np.float32).mean(axis=2)
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:] - gray[:-1]).mean()
    return float((gx + gy) / 2)


def frame_metrics(source: Image.Image, frames: list[Image.Image]) -> dict:
    src = np.asarray(source, dtype=np.float32)
    arrays = [np.asarray(frame.convert("RGB"), dtype=np.float32) for frame in frames]
    first = arrays[0]
    mse0 = float(np.mean((first - src) ** 2))
    psnr0 = float(20 * np.log10(255.0 / np.sqrt(max(mse0, 1e-12))))
    temporal_mae = [float(np.mean(np.abs(frame - first))) for frame in arrays]
    consecutive_mae = [
        float(np.mean(np.abs(arrays[i] - arrays[i - 1]))) for i in range(1, len(arrays))
    ]
    edit_fraction = [
        float(np.mean(np.max(np.abs(frame - first), axis=2) >= 12.0)) for frame in arrays
    ]
    sharpness_values = [sharpness(frame) for frame in arrays]
    return {
        "generated_frame0_psnr_to_input": psnr0,
        "temporal_mae_from_generated_frame0": temporal_mae,
        "consecutive_frame_mae": consecutive_mae,
        "edit_fraction_from_generated_frame0_at_12_255": edit_fraction,
        "sharpness": sharpness_values,
        "last_minus_first_mae": temporal_mae[-1],
        "max_temporal_mae": max(temporal_mae),
        "mean_consecutive_mae": float(np.mean(consecutive_mae)),
        "last_edit_fraction": edit_fraction[-1],
        "last_sharpness_ratio": sharpness_values[-1] / max(sharpness_values[0], 1e-12),
    }


def save_contact_sheet(frames: list[Image.Image], target: Path) -> None:
    indices = sorted(set([0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]))
    thumbs = [frames[index].convert("RGB") for index in indices]
    width, height = thumbs[0].size
    label_height = 28
    sheet = Image.new("RGB", (width * len(thumbs), height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (index, frame) in enumerate(zip(indices, thumbs)):
        sheet.paste(frame, (column * width, label_height))
        draw.text((column * width + 8, 7), f"frame {index}", fill="black")
    sheet.save(target)


def main() -> None:
    args = parse_args()
    model, source_path = validate_args(args)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / f"{args.sample_id}.mp4"
    report_path = output / "vanilla_benchmark.json"
    sheet_path = output / f"{args.sample_id}_contact_sheet.jpg"
    if video_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {video_path}; pass --overwrite to replace it.")

    shards = sorted(str(path) for path in model.glob("diffusion_pytorch_model-*.safetensors"))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=shards),
            ModelConfig(path=str(model / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(model / "Wan2.1_VAE.pth")),
            ModelConfig(path=str(model / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(model / "google/umt5-xxl")),
        redirect_common_files=False,
    )
    if pipe.dit is None or pipe.vae is None or pipe.text_encoder is None or pipe.image_encoder is None:
        raise RuntimeError("Wan2.1 I2V did not load DiT, VAE, T5, and CLIP completely.")

    source = Image.open(source_path).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=source,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        tiled=True,
    )
    elapsed = time.perf_counter() - started
    frames = [frame.convert("RGB") for frame in frames]
    if len(frames) != args.num_frames:
        raise RuntimeError(f"Wan returned {len(frames)} frames, expected {args.num_frames}.")

    # Save the raw model result. Do not replace frame 0 with the source image.
    save_video(frames[:16], str(video_path), fps=args.fps, quality=5)
    save_contact_sheet(frames[:16], sheet_path)
    report = {
        "diagnostic": "vanilla_wan21_i2v_no_lora_no_action_no_state",
        "model_path": str(model),
        "source": str(source_path),
        "sample_id": args.sample_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "frames_generated": len(frames),
        "frames_saved": 16,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "sigma_shift": args.sigma_shift,
        "seconds": elapsed,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "video": str(video_path),
        "contact_sheet": str(sheet_path),
        "metrics": frame_metrics(source, frames[:16]),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved -> {video_path}")
    print(f"saved -> {sheet_path}")
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
