"""Native DreamZero-SO101 video-prior smoke test for the SO-100 challenge.

This deliberately runs the released model in its original direction:

    first RGB + normalized current state + text -> imagined video + predicted action

It does *not* claim to condition on the challenge's supplied future actions.  The
native smoke test is the gate before implementing action inpainting.  DreamZero-
SO101 was trained on a three-camera 2x2 canvas, so the single challenge image is
placed in the front-camera quadrant and the missing views are black.  Two causal
latent chunks decode to 17 RGB frames; frame 0 plus frames 1..15 are saved in the
competition's 16-frame format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
DREAMZERO = REPO / "third_party/dreamzero"
if str(DREAMZERO) not in sys.path:
    sys.path.insert(0, str(DREAMZERO))

from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig  # noqa: E402


NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, "
    "artwork, painting, image, still, grayscale, dull, worst quality, low quality, "
    "JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, "
    "disfigured, mutated limbs, fused fingers, stagnant image, cluttered background."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the released DreamZero-SO101 LoRA without challenge-action conditioning."
    )
    parser.add_argument(
        "--base-model",
        default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P"),
    )
    parser.add_argument(
        "--lora",
        default=str(REPO / "checkpoints/dreamzero-so101-lora"),
    )
    parser.add_argument(
        "--metadata",
        default=str(
            REPO
            / "checkpoints/dreamzero-so101-checkpoints-meta/lora-100k/checkpoint-20000/experiment_cfg/metadata.json"
        ),
    )
    parser.add_argument("--tokenizer", default=str(REPO / "checkpoints/umt5-xxl"))
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument(
        "--prediction-root", default=str(REPO / "diagnostics/dreamzero_so101_native")
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--sample-id",
        help="Generate one explicit sample instead of taking the first --limit IDs.",
    )
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--action-conditioning",
        choices=("native", "given"),
        default="native",
        help=(
            "native jointly samples DreamZero's action and video. given converts each "
            "8-step challenge segment to the released model's 24-step relative-action "
            "space and inpaints that known action while denoising video."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="An SO-101 robot arm performs the manipulation task.",
    )
    parser.add_argument(
        "--state-mode",
        choices=("metadata-center", "action0-q99"),
        default="metadata-center",
        help=(
            "metadata-center is the safe native-prior diagnostic. action0-q99 treats the "
            "first challenge target as current SO-101 state, but may saturate because SO-100 "
            "and SO-101 absolute calibration conventions differ."
        ),
    )
    parser.add_argument(
        "--view-mode",
        choices=("front-only", "replicate-three"),
        default="front-only",
        help=(
            "front-only matches the challenge's available observation. replicate-three "
            "is a missing-camera ablation and is not a claim of true multi-view input."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json")
    parser.add_argument(
        "--compile-encoders",
        action="store_true",
        help="Use DreamZero's torch.compile post-initialization path; slower on the first smoke run.",
    )
    return parser.parse_args()


def sample_ids(root: Path) -> list[str]:
    image_dir = root / "images"
    ids = sorted(path.stem for path in image_dir.glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"No PNG inputs found under {image_dir}")
    return ids


def validate_assets(args: argparse.Namespace) -> None:
    base = Path(args.base_model)
    lora = Path(args.lora)
    required = [
        base / "diffusion_pytorch_model.safetensors.index.json",
        base / "models_t5_umt5-xxl-enc-bf16.pth",
        base / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        base / "Wan2.1_VAE.pth",
        lora / "config.json",
        lora / "model.safetensors",
        Path(args.metadata),
        Path(args.tokenizer) / "spiece.model",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DreamZero assets:\n" + "\n".join(missing))
    shards = sorted(base.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 7:
        raise RuntimeError(f"Expected 7 Wan2.1 DiT shards, found {len(shards)} in {base}")


def patched_config(lora_dir: Path, base_dir: Path) -> VLAConfig:
    config_dict = json.loads((lora_dir / "config.json").read_text())
    head = config_dict["action_head_cfg"]["config"]
    head["diffusion_model_cfg"]["diffusion_model_pretrained_path"] = str(base_dir)
    head["text_encoder_cfg"]["text_encoder_pretrained_path"] = str(
        base_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    head["image_encoder_cfg"]["image_encoder_pretrained_path"] = str(
        base_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    head["vae_cfg"]["vae_pretrained_path"] = str(base_dir / "Wan2.1_VAE.pth")
    head["defer_lora_injection"] = False
    head["skip_component_loading"] = False
    return VLAConfig(**config_dict)


def load_model(args: argparse.Namespace) -> VLA:
    config = patched_config(Path(args.lora), Path(args.base_model))
    model = VLA(config)
    state_dict = load_file(str(Path(args.lora) / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict
    # Missing keys are the frozen Wan/T5/CLIP/VAE base and are expected for a LoRA-only file.
    lora_missing = [key for key in missing if "lora_" in key]
    if lora_missing or unexpected:
        raise RuntimeError(
            f"LoRA load mismatch: missing_lora={lora_missing[:8]}, unexpected={unexpected[:8]}"
        )
    print(
        f"[DreamZero] LoRA loaded: expected_base_missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )

    model.eval().requires_grad_(False)
    if model.action_head.train_architecture == "lora":
        model.action_head.model = model.action_head.model.merge_and_unload()

    model.action_head.seed = args.seed
    model.action_head.cfg_scale = args.cfg_scale
    model.action_head.num_inference_steps = args.num_inference_steps
    # The repository defaults to a 16-step acceleration mask.  A published 4-step
    # SO-101 run must evaluate all four requested DiT steps, not reuse step 3.
    model.action_head.dit_step_mask = [True] * args.num_inference_steps

    if args.compile_encoders:
        model.post_initialize()
    else:
        model.to(device="cuda", dtype=torch.bfloat16)
        model.action_head._device = "cuda"
    return model


def center_crop_resize(image: np.ndarray, out_h: int = 176, out_w: int = 320) -> np.ndarray:
    """Match SO-101 eval: 0.95 center crop followed by bilinear resize."""
    height, width = image.shape[:2]
    crop_h, crop_w = int(height * 0.95), int(width * 0.95)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = image[top : top + crop_h, left : left + crop_w]
    return np.asarray(
        Image.fromarray(cropped).resize((out_w, out_h), Image.Resampling.BILINEAR)
    )


def make_three_view_canvas(
    image: np.ndarray, mode: str = "front-only"
) -> np.ndarray:
    """Build the SO-101 2x2 camera canvas.

    ``front-only`` is the faithful challenge observation: front TL and the two
    unavailable cameras black. ``replicate-three`` is an explicit OOD ablation
    that copies the only view into the gripper/top slots. It is useful only for
    checking whether missing-view black tiles suppress the released prior.
    """
    front = center_crop_resize(image)
    canvas = np.zeros((352, 640, 3), dtype=np.uint8)
    canvas[:176, :320] = front
    if mode == "replicate-three":
        canvas[176:, :320] = front
        canvas[:176, 320:] = front
    elif mode != "front-only":
        raise ValueError(f"Unknown DreamZero canvas mode: {mode!r}")
    return canvas


def q99_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    normalized = 2.0 * (values - q01) / np.maximum(q99 - q01, 1e-8) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def prepare_state(
    action: np.ndarray, metadata_path: Path, mode: str
) -> tuple[torch.Tensor, dict[str, object]]:
    metadata = json.loads(metadata_path.read_text())["so101"]
    stats = metadata["statistics"]["state"]["joint_pos"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if mode == "metadata-center":
        normalized = np.zeros(6, dtype=np.float32)
        raw = (q01 + q99) / 2.0
    else:
        raw = action[0].astype(np.float32)
        normalized = q99_normalize(raw, q01, q99)
    padded = np.zeros((1, 64), dtype=np.float32)
    padded[0, :6] = normalized
    report = {
        "mode": mode,
        "raw_state": raw.tolist(),
        "normalized_state": normalized.tolist(),
        "saturation_fraction": float(np.mean(np.abs(normalized) >= 0.999)),
    }
    return torch.from_numpy(padded)[None].to("cuda", dtype=torch.bfloat16), report


def convert_given_action_chunks(
    action: np.ndarray,
    metadata_path: Path,
    anchor_states: np.ndarray | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Convert 16 absolute SO-100 targets into two 24-token relative chunks.

    DreamZero-SO101 samples eight video frames at action offsets
    ``[0, 3, ..., 21]`` and pairs them with 24 consecutive action targets. The
    challenge provides one action per output frame, so each eight-frame segment
    is linearly expanded to 24 tokens. During training, all 24 targets are made
    relative to the observation state at that segment's anchor. If oracle states
    are unavailable, the first action remains the explicitly reported proxy.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (16, 6):
        raise ValueError(f"Expected action shape (16, 6), got {action.shape}")
    if anchor_states is not None:
        anchor_states = np.asarray(anchor_states, dtype=np.float32)
        if anchor_states.shape == (2, 6):
            anchors = anchor_states
        elif anchor_states.ndim == 2 and anchor_states.shape[0] >= 9 and anchor_states.shape[1] == 6:
            anchors = anchor_states[[0, 8]]
        else:
            raise ValueError(
                "anchor_states must have shape (2, 6) or at least (9, 6), "
                f"got {anchor_states.shape}"
            )
        anchor_source = "observation_state_at_chunk_anchor"
    else:
        anchors = action[[0, 8]]
        anchor_source = "first_action_proxy"

    metadata = json.loads(metadata_path.read_text())["so101"]
    stats = metadata["statistics"]["action"]["joint_pos"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    chunks: list[np.ndarray] = []
    reports = []
    for chunk_index, start in enumerate((0, 8)):
        absolute = action[start : start + 8].astype(np.float32)
        relative = absolute - anchors[chunk_index]
        source_t = np.arange(8, dtype=np.float32)
        target_t = np.linspace(0.0, 7.0, 24, dtype=np.float32)
        relative_24 = np.stack(
            [np.interp(target_t, source_t, relative[:, dim]) for dim in range(6)],
            axis=-1,
        ).astype(np.float32)
        normalized = q99_normalize(relative_24, q01, q99)
        padded = np.zeros((1, 24, 32), dtype=np.float32)
        padded[0, :, :6] = normalized
        chunks.append(padded)
        reports.append(
            {
                "start": start,
                "anchor_state": anchors[chunk_index].tolist(),
                "relative_first": relative[0].tolist(),
                "relative_last": relative[-1].tolist(),
                "normalized_min": normalized.min(axis=0).tolist(),
                "normalized_max": normalized.max(axis=0).tolist(),
                "saturation_fraction": float(np.mean(np.abs(normalized) >= 0.999)),
            }
        )
    return chunks, {
        "representation": "state_anchored_relative_q99",
        "anchor_source": anchor_source,
        "temporal_mapping": "8 challenge steps linearly expanded to 24 DreamZero action tokens",
        "chunks": reports,
    }


def prepare_given_action_chunks(
    action: np.ndarray,
    metadata_path: Path,
    anchor_states: np.ndarray | None = None,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    chunks, report = convert_given_action_chunks(action, metadata_path, anchor_states)
    tensors = [
        torch.from_numpy(chunk).to("cuda", dtype=torch.bfloat16) for chunk in chunks
    ]
    return tensors, report


def tokenize(tokenizer, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [prompt], padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )
    return encoded.input_ids.to("cuda"), encoded.attention_mask.to("cuda")


def decode_latents(model: VLA, latents: torch.Tensor) -> np.ndarray:
    head = model.action_head
    with torch.inference_mode():
        decoded = head.vae.decode(
            latents,
            tiled=head.tiled,
            tile_size=(head.tile_size_height, head.tile_size_width),
            tile_stride=(head.tile_stride_height, head.tile_stride_width),
        )
    frames = decoded[0].permute(1, 2, 3, 0).float()
    return ((frames + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()


def save_video(frames: np.ndarray, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def front_view_competition_frames(
    composite: np.ndarray, source: np.ndarray, count: int = 16
) -> np.ndarray:
    if len(composite) < count:
        raise RuntimeError(f"DreamZero decoded only {len(composite)} frames; need {count}")
    front = composite[:count, :176, :320]
    restored = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((640, 480), Image.Resampling.BILINEAR))
            for frame in front
        ]
    )
    restored[0] = np.asarray(
        Image.fromarray(source).resize((640, 480), Image.Resampling.BILINEAR)
    )
    return restored


def model_inputs(
    canvas: np.ndarray,
    state: torch.Tensor,
    prompt_tokens: tuple[torch.Tensor, torch.Tensor],
    negative_tokens: tuple[torch.Tensor, torch.Tensor],
    num_input_frames: int,
) -> dict[str, torch.Tensor]:
    text, text_mask = prompt_tokens
    negative, negative_mask = negative_tokens
    images = np.repeat(canvas[None, None], num_input_frames, axis=1)
    return {
        "images": torch.from_numpy(images).to("cuda"),
        "state": state,
        # The released adapter contains one embodiment category.  The DiT maps
        # this SO-101-only checkpoint to internal category 0.
        "embodiment_id": torch.zeros((1,), device="cuda", dtype=torch.long),
        "text": text,
        "text_attention_mask": text_mask,
        "text_negative": negative,
        "text_attention_mask_negative": negative_mask,
    }


def generate_one(
    model: VLA,
    tokenizer,
    source: np.ndarray,
    action: np.ndarray,
    args: argparse.Namespace,
    *,
    conditioned_action: np.ndarray | None = None,
    action_anchor_states: np.ndarray | None = None,
    view_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    head = model.action_head
    head.current_start_frame = 0
    head.language = None
    head.clip_feas = None
    head.ys = None
    head.kv_cache1 = head.kv_cache_neg = None
    head.crossattn_cache = head.crossattn_cache_neg = None

    state, state_report = prepare_state(action, Path(args.metadata), args.state_mode)
    if args.action_conditioning == "given":
        action_chunks, action_report = prepare_given_action_chunks(
            action if conditioned_action is None else conditioned_action,
            Path(args.metadata),
            action_anchor_states,
        )
    else:
        action_chunks = [None, None]
        action_report = {"representation": "native_joint_sampling"}
    resolved_view_mode = view_mode or getattr(args, "view_mode", "front-only")
    canvas = make_three_view_canvas(source, resolved_view_mode)
    positive = tokenize(tokenizer, args.prompt)
    negative = tokenize(tokenizer, NEGATIVE_PROMPT)

    first_inputs = model_inputs(canvas, state, positive, negative, num_input_frames=1)
    with torch.inference_mode():
        first = model.lazy_joint_video_action_causal(
            first_inputs, condition_action=action_chunks[0]
        )
        first_latents = first["video_pred"]

        # Keep the causal cache alive.  A 9-frame-shaped observation prevents the
        # repository from resetting the sequence; latent_video supplies the exact
        # previous latent context.  3 + 2 latent frames decode to 17 RGB frames.
        second_inputs = model_inputs(canvas, state, positive, negative, num_input_frames=9)
        second = model.lazy_joint_video_action_causal(
            second_inputs,
            latent_video=first_latents,
            condition_action=action_chunks[1],
        )
        all_latents = torch.cat([first_latents, second["video_pred"]], dim=2)
        composite = decode_latents(model, all_latents)

    frames = front_view_competition_frames(composite, source, count=16)
    report = {
        "native_only": args.action_conditioning == "native",
        "uses_challenge_future_actions": args.action_conditioning == "given",
        "action_conditioning": action_report,
        "decoded_frames": int(len(composite)),
        "saved_frames": 16,
        "latent_frames": int(all_latents.shape[2]),
        "state": state_report,
        "view_mode": resolved_view_mode,
        "predicted_action_abs_mean": float(first["action_pred"].float().abs().mean().cpu()),
    }
    return frames, composite[:16], report


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DreamZero-SO101 14B inference requires an H100-class CUDA GPU")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    validate_assets(args)

    root = Path(args.challenge_root)
    output = Path(args.prediction_root)
    output.mkdir(parents=True, exist_ok=True)
    composite_dir = output / "_composite"

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29621")
        dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = load_model(args)
    available = sample_ids(root)
    if args.sample_id is not None:
        if args.sample_id not in available:
            raise ValueError(f"Unknown --sample-id {args.sample_id!r} under {root}")
        selected = [args.sample_id]
    else:
        selected = available[: args.limit]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    reports: dict[str, object] = {}

    try:
        for index, sample_id in enumerate(selected):
            path = output / f"{sample_id}.mp4"
            if path.exists() and not args.overwrite:
                print(f"[DreamZero native] skip existing {path}")
                continue
            source = np.asarray(Image.open(root / "images" / f"{sample_id}.png").convert("RGB"))
            action = np.load(root / "actions" / f"{sample_id}.npy")
            if action.shape != (16, 6):
                raise ValueError(f"Expected {sample_id} action shape (16, 6), got {action.shape}")
            frames, composite, report = generate_one(model, tokenizer, source, action, args)
            save_video(frames, path, args.fps)
            save_video(composite, composite_dir / f"{sample_id}.mp4", args.fps)
            reports[sample_id] = report
            print(f"[DreamZero native] {index + 1}/{len(selected)} -> {path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    seconds = time.perf_counter() - started
    benchmark = {
        "mode": (
            "native_video_prior_gate"
            if args.action_conditioning == "native"
            else "given_action_inpainting_gate"
        ),
        "warning": (
            "This run does not condition on the supplied future action sequence."
            if args.action_conditioning == "native"
            else "Known relative action is inpainted; video remains generative."
        ),
        "samples": len(reports),
        "seconds": seconds,
        "seconds_per_sample": seconds / max(1, len(reports)),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "num_inference_steps_per_chunk": args.num_inference_steps,
        "causal_chunks": 2,
        "cfg_scale": args.cfg_scale,
        "prompt": args.prompt,
        "reports": reports,
    }
    benchmark_path = (
        Path(args.benchmark_json)
        if args.benchmark_json
        else output / "native_benchmark.json"
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(benchmark, indent=2))
    print(json.dumps(benchmark, indent=2))
    print(f"saved -> {benchmark_path}")


if __name__ == "__main__":
    main()
