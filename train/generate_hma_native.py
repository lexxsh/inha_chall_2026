"""Generate a no-action native-prior screen with released HMA-MagViT.

The challenge provides one image while the released HMA checkpoint expects four
prompt frames.  We repeat the source image exactly for those four slots.  This
script intentionally supplies no robot action: borrowing a 6-D action stem from
an unrelated Open-X robot would be a semantic mismatch, not a zero-shot test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1]
HMA_ROOT = REPO / "third_party/HMA"
if str(HMA_ROOT) not in sys.path:
    sys.path.insert(0, str(HMA_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=str(REPO / "models/HMA/hma-base-disc")
    )
    parser.add_argument(
        "--tokenizer", default=str(REPO / "models/HMA/magvit/magvit2.ckpt")
    )
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument(
        "--prediction-root", default=str(REPO / "diagnostics/hma_native_neutral")
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--maskgit-steps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--action-mode", choices=("neutral", "stress", "none"), default="neutral",
        help=(
            "neutral keeps the checkpoint's trained 64 action tokens using the mean "
            "action of its only 6-D domain; stress supplies an in-distribution, time-varying "
            "action; none is an intentionally OOD ablation"
        ),
    )
    parser.add_argument("--neutral-domain", default="berkeley_fanuc_manipulation")
    parser.add_argument(
        "--stress-scale", type=float, default=0.75,
        help="Amplitude in the selected public domain's action standard deviations.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    return parser.parse_args()


def audit_contract(checkpoint: Path, tokenizer: Path, challenge_root: Path) -> dict:
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete HMA checkpoint at {checkpoint}")
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    config = json.loads(config_path.read_text())
    required = {
        "T": 12,
        "S": 256,
        "image_vocab_size": 262_144,
        "factored_vocab_size": 512,
        "num_factored_vocabs": 2,
        "num_prompt_frames": 4,
        "use_actions": True,
    }
    mismatch = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required.items()
        if config.get(key) != value
    }
    if mismatch:
        raise ValueError(f"HMA checkpoint contract mismatch: {mismatch}")
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        required_keys = {
            "pos_embed_TSC": (1, 12, 320, 256),
            "token_embed.factored_embeds.0.weight": (512, 256),
            "token_embed.factored_embeds.1.weight": (512, 256),
            "token_embed.mask_token_embed": (1, 256),
        }
        bad_keys = {
            key: {
                "expected": shape,
                "actual": tuple(handle.get_slice(key).get_shape()) if key in keys else None,
            }
            for key, shape in required_keys.items()
            if key not in keys or tuple(handle.get_slice(key).get_shape()) != shape
        }
        if bad_keys:
            raise ValueError(f"HMA tensor contract mismatch: {bad_keys}")
    images = sorted((challenge_root / "images").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No challenge images under {challenge_root / 'images'}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": weights_path.stat().st_size,
        "checkpoint_tensors": len(keys),
        "tokenizer": str(tokenizer.resolve()),
        "window_frames": 12,
        "prompt_frames": 4,
        "generated_per_window": 8,
        "rollout_windows": 2,
        "output_frames": 16,
        "maskgit_token_grid": [16, 16],
        "prompt_policy": "repeat the single source frame four times",
        "challenge_images": len(images),
    }


def crop_geometry(height: int, width: int) -> tuple[int, int, int, int]:
    """Return the source-space square corresponding to HMA's center crop."""
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return top, left, side, side


def official_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height < width:
        new_height, new_width = 256, int(256 * width / height)
    else:
        new_height, new_width = int(256 * height / width), 256
    resized = cv2.resize(image, (new_width, new_height))
    top = (new_height - 256) // 2
    left = (new_width - 256) // 2
    return resized[top : top + 256, left : left + 256]


def load_tokenizer(path: Path, device: torch.device, dtype: torch.dtype):
    from external.magvit2.config import VQConfig
    from external.magvit2.models.lfqgan import VQModel

    model = VQModel(
        VQConfig(), ckpt_path=str(path), inference_only=True, use_ema=True
    )
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def encode_tokens(model, crop: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = torch.from_numpy(crop.copy()).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32).div(127.5).sub(1).to(dtype)
    _, _, indices, _ = model.encode(image, flip=True)
    if indices.numel() != 256:
        raise RuntimeError(f"Expected 256 HMA tokens, got {indices.numel()}")
    return indices.reshape(1, 16, 16).long()


def decode_tokens(model, tokens: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    quant = model.quantize.get_codebook_entry(
        rearrange(tokens, "b t h w -> (b t) (h w)"),
        bhwc=(tokens.shape[0] * tokens.shape[1], 16, 16, model.quantize.codebook_dim),
    ).flip(1)
    decoded = model.decode(quant.to(device=tokens.device, dtype=dtype)).float().clamp(-1, 1)
    decoded = decoded.add(1).mul(127.5).round().clamp(0, 255).byte()
    return decoded.permute(0, 2, 3, 1).cpu().numpy()


def generate_window(
    model,
    prompt_tokens: torch.Tensor,
    args: argparse.Namespace,
    action_ids: torch.Tensor | None,
    domain: list[str] | None,
) -> torch.Tensor:
    if prompt_tokens.shape != (1, 4, 16, 16):
        raise ValueError(f"Expected [1,4,16,16] prompt, got {tuple(prompt_tokens.shape)}")
    sequence = torch.full(
        (1, 12, 16, 16), model.mask_token_id,
        dtype=torch.long, device=prompt_tokens.device,
    )
    sequence[:, :4] = prompt_tokens
    for timestep in range(4, 12):
        sample, _, _ = model.maskgit_generate(
            sequence,
            out_t=timestep,
            maskgit_steps=args.maskgit_steps,
            temperature=args.temperature,
            action_ids=action_ids,
            domain=domain,
        )
        sequence[:, timestep] = sample
    return sequence


def composite_source_anchor(source: np.ndarray, crops: np.ndarray) -> np.ndarray:
    top, left, height, width = crop_geometry(*source.shape[:2])
    output = np.repeat(source[None], len(crops), axis=0)
    for index, crop in enumerate(crops):
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
        output[index, top : top + height, left : left + width] = resized
    # Frame zero is the exact supplied PNG, not a tokenizer reconstruction.
    output[0] = source
    return output


def save_video(path: Path, frames: np.ndarray) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=6)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def frame_metrics(source: np.ndarray, frames: np.ndarray) -> dict:
    source_f = source.astype(np.float32) / 255.0
    future = frames[1:].astype(np.float32) / 255.0
    difference = np.abs(future - source_f).mean(axis=(1, 2, 3))
    gray = future.mean(axis=3)
    source_gray = source_f.mean(axis=2)
    edge = (
        np.square(np.diff(gray, axis=1)).mean(axis=(1, 2))
        + np.square(np.diff(gray, axis=2)).mean(axis=(1, 2))
    )
    source_edge = (
        np.square(np.diff(source_gray, axis=0)).mean()
        + np.square(np.diff(source_gray, axis=1)).mean()
    )
    return {
        "mean_abs_change": float(difference.mean()),
        "last_abs_change": float(difference[-1]),
        "temporal_std": float(future.std(axis=0).mean()),
        "median_sharpness_ratio": float(np.median(edge / max(source_edge, 1e-12))),
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    challenge_root = Path(args.challenge_root)
    contract = audit_contract(checkpoint, tokenizer_path, challenge_root)
    if args.structure_only:
        print(json.dumps({**contract, "verdict": "PASS_STRUCTURE"}, indent=2))
        return

    if args.maskgit_steps < 1:
        raise ValueError("--maskgit-steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --structure-only for a CPU audit.")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    cuda_index = None
    if device.type == "cuda":
        # Some PyTorch/CUDA builds reject a torch.device object in the memory
        # statistics API even though tensor placement accepts it.  Resolve and
        # select the integer index once so CUDA_VISIBLE_DEVICES remapping also
        # behaves correctly.
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    from hma.model.st_mask_git import STMaskGIT

    started = time.perf_counter()
    dynamics = STMaskGIT.from_pretrained(str(checkpoint)).to(device=device, dtype=dtype).eval()
    dynamics.requires_grad_(False)
    tokenizer = load_tokenizer(tokenizer_path, device, dtype)
    action_ids = None
    domain = None
    if args.action_mode in ("neutral", "stress"):
        if args.neutral_domain not in dynamics.action_preprocessor:
            raise ValueError(
                f"Unknown neutral domain {args.neutral_domain!r}; available domains are "
                f"{list(dynamics.action_preprocessor.keys())}"
            )
        stats = dynamics.action_preprocessor[args.neutral_domain]
        raw_dim = int(stats.mean.numel())
        configured_dim = dynamics.config.d_actions[
            dynamics.config.action_domains.index(args.neutral_domain)
        ]
        if configured_dim != raw_dim:
            raise ValueError(
                f"Neutral gate requires stride-one actions, got configured={configured_dim}, "
                f"raw={raw_dim} for {args.neutral_domain}"
            )
        # ActionStat subtracts this mean, so the action MLP receives exact zero
        # while all 64 trained action tokens and per-layer modulation remain.
        action_ids = stats.mean.to(device=device, dtype=dtype).view(1, 1, raw_dim)
        action_ids = action_ids.expand(1, dynamics.config.T, raw_dim).contiguous()
        if args.action_mode == "stress":
            # A smooth, bounded action trajectory in the public domain's own
            # normalization.  Different phase/frequency per dimension prevents
            # the test from degenerating to a single global scalar direction.
            position = torch.linspace(0, 1, dynamics.config.T, device=device, dtype=dtype)
            dimension = torch.arange(raw_dim, device=device, dtype=dtype)
            phase = dimension * (torch.pi / max(raw_dim, 1))
            frequency = 1.0 + (dimension % 3) * 0.5
            normalized = torch.sin(2 * torch.pi * position[:, None] * frequency + phase)
            normalized = normalized * args.stress_scale
            std = stats.std.to(device=device, dtype=dtype)
            action_ids = action_ids + normalized[None] * std.view(1, 1, raw_dim)
        domain = [args.neutral_domain]
    load_seconds = time.perf_counter() - started
    output_root = Path(args.prediction_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted((challenge_root / "images").glob("*.png"))[: args.limit]
    rows = []

    with torch.inference_mode():
        for number, image_path in enumerate(image_paths, start=1):
            output_path = output_root / f"{image_path.stem}.mp4"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"{output_path} exists; pass --overwrite")
            sample_started = time.perf_counter()
            source = np.asarray(Image.open(image_path).convert("RGB")).copy()
            source_token = encode_tokens(tokenizer, official_crop(source), device, dtype)
            first_prompt = source_token[:, None].expand(-1, 4, -1, -1).contiguous()
            first_window = generate_window(dynamics, first_prompt, args, action_ids, domain)
            second_window = generate_window(
                dynamics, first_window[:, -4:].contiguous(), args, action_ids, domain
            )
            # Exact source + eight first-window futures + seven continuation futures.
            output_tokens = torch.cat(
                [source_token[:, None], first_window[:, 4:], second_window[:, 4:11]], dim=1
            )
            if output_tokens.shape[1] != 16:
                raise RuntimeError(f"Bad rollout length {output_tokens.shape}")
            decoded_crops = decode_tokens(tokenizer, output_tokens, dtype)
            frames = composite_source_anchor(source, decoded_crops)
            save_video(output_path, frames)
            metrics = frame_metrics(source, frames)
            row = {
                "sample_id": image_path.stem,
                "seconds": time.perf_counter() - sample_started,
                **metrics,
            }
            rows.append(row)
            print(
                f"[HMA native] {number}/{len(image_paths)} {image_path.stem} "
                f"{row['seconds']:.1f}s change={row['mean_abs_change']:.4f} "
                f"sharp={row['median_sharpness_ratio']:.3f}"
            )

    total_seconds = time.perf_counter() - started
    result = {
        **contract,
        "samples": rows,
        "load_seconds": load_seconds,
        "total_seconds": total_seconds,
        "seconds_per_sample_excluding_load": (
            sum(row["seconds"] for row in rows) / max(len(rows), 1)
        ),
        "maskgit_steps": args.maskgit_steps,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "action_mode": args.action_mode,
        "neutral_domain": (
            args.neutral_domain if args.action_mode in ("neutral", "stress") else None
        ),
        "normalized_action": (
            "exact zero" if args.action_mode == "neutral" else
            f"bounded sinusoid, amplitude={args.stress_scale} std" if args.action_mode == "stress" else
            None
        ),
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated(cuda_index) / 2**30
            if cuda_index is not None else 0.0
        ),
        "scope_warning": (
            "No challenge action is used. Neutral mode preserves the checkpoint's trained "
            "action-token structure but borrows no Fanuc command semantics. This is only a "
            "native robot-video prior screen, not a submission candidate."
        ),
        "verdict": "NEEDS_VISUAL_REVIEW",
    }
    benchmark = output_root / "native_benchmark.json"
    benchmark.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"saved -> {benchmark.resolve()}")


if __name__ == "__main__":
    main()
