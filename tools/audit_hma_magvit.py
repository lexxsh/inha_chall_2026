"""Gate HMA's released MagViT tokenizer before adapting its dynamics model.

This is deliberately a reconstruction-only test.  It uses train-only holdout
videos and future pixels only as reconstruction targets, never as a deployable
model input.  A failed tokenizer gate means that no action model operating on
these discrete tokens can recover the missing robot detail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
HMA_ROOT = REPO / "third_party/HMA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "models/HMA/magvit/magvit2.ckpt"),
    )
    parser.add_argument(
        "--valset",
        default=str(REPO / "diagnostics/spatial_control_gate/valset"),
        help="Train-only group holdout made by prepare_spatial_control_gate.py.",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", default=str(REPO / "results/hma_magvit_gate.json")
    )
    parser.add_argument(
        "--diagnostic-root", default=str(REPO / "diagnostics/hma_magvit_gate")
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Check files/checkpoint contracts on CPU without constructing the model.",
    )
    return parser.parse_args()


def checkpoint_contract(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(f"{path} has no Lightning state_dict")
    state = payload["state_dict"]
    required = {
        "encoder.conv_in.weight": (128, 3, 3, 3),
        "encoder.conv_out.weight": (18, 512, 1, 1),
        "decoder.conv_in.weight": (512, 18, 3, 3),
        "decoder.conv_out.weight": (3, 128, 3, 3),
    }
    bad = {
        key: {"expected": shape, "actual": tuple(state[key].shape) if key in state else None}
        for key, shape in required.items()
        if key not in state or tuple(state[key].shape) != shape
    }
    if bad:
        raise ValueError(f"HMA MagViT checkpoint contract mismatch: {bad}")
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_tensors": len(state),
        "encoder_tensors": sum(key.startswith("encoder.") for key in state),
        "decoder_tensors": sum(key.startswith("decoder.") for key in state),
        "ema_tensors": sum(key.startswith("model_ema.") for key in state),
        "latent_channels": 18,
        "token_grid": [16, 16],
        "codebook_size": 262_144,
    }


def official_preprocess(frames: np.ndarray) -> np.ndarray:
    """Reproduce HMA datasets.utils.resize_image for a batch of RGB frames."""
    output = []
    for image in frames:
        height, width = image.shape[:2]
        if height < width:
            new_height, new_width = 256, int(256 * width / height)
        else:
            new_height, new_width = int(256 * height / width), 256
        resized = cv2.resize(image, (new_width, new_height))
        top = (new_height - 256) // 2
        left = (new_width - 256) // 2
        output.append(resized[top : top + 256, left : left + 256])
    return np.stack(output)


def load_model(checkpoint: Path, device: torch.device):
    if str(HMA_ROOT) not in sys.path:
        sys.path.insert(0, str(HMA_ROOT))
    from external.magvit2.config import VQConfig
    from external.magvit2.models.lfqgan import VQModel

    # inference_only avoids constructing the training discriminator/LPIPS and
    # therefore avoids an unrelated VGG download.  This does not change the
    # released encoder, LFQ quantizer, or decoder.
    model = VQModel(
        VQConfig(), ckpt_path=str(checkpoint), inference_only=True, use_ema=True
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    return model, dtype


def token_roundtrip(model, batch: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Use the exact HMA index convention, not an unquantized VAE shortcut."""
    from einops import rearrange

    _, _, indices, _ = model.encode(batch.to(dtype=dtype), flip=True)
    side = int(round((indices.numel() / batch.shape[0]) ** 0.5))
    if side * side * batch.shape[0] != indices.numel():
        raise RuntimeError(f"Unexpected flattened HMA token count: {indices.shape}")
    indices = indices.reshape(batch.shape[0], side, side)
    quant = model.quantize.get_codebook_entry(
        rearrange(indices, "b h w -> b (h w)"),
        bhwc=indices.shape + (model.quantize.codebook_dim,),
    ).flip(1)
    return model.decode(quant.to(device=batch.device, dtype=dtype)).float().clamp(-1, 1)


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    mse = (reference.float() - prediction.float()).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))


def gradient_energy(image: torch.Tensor) -> torch.Tensor:
    gray = image.float().mean(dim=1)
    dx = (gray[:, :, 1:] - gray[:, :, :-1]).square().mean(dim=(1, 2))
    dy = (gray[:, 1:, :] - gray[:, :-1, :]).square().mean(dim=(1, 2))
    return dx + dy


def save_side_by_side(path: Path, original: np.ndarray, reconstructed: np.ndarray) -> None:
    frames = np.concatenate([original, reconstructed], axis=2)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=6)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def discover_samples(valset: Path, limit: int) -> list[tuple[str, Path]]:
    video_root = valset / "gt_videos"
    samples = [(path.stem, path) for path in sorted(video_root.glob("*.npy"))[:limit]]
    if not samples:
        raise FileNotFoundError(f"No train-only GT videos under {video_root}")
    return samples


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    valset = Path(args.valset)
    contract = checkpoint_contract(checkpoint)
    samples = discover_samples(valset, args.limit)
    structure = {
        **contract,
        "valset": str(valset.resolve()),
        "sample_count": len(samples),
        "scope_warning": (
            "Train-only future frames are reconstruction targets. This gate validates only "
            "HMA MagViT; it does not validate action conditioning or generated dynamics."
        ),
    }
    if args.structure_only:
        structure["verdict"] = "PASS_STRUCTURE"
        print(json.dumps(structure, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run --structure-only or choose --device cpu.")
    model, dtype = load_model(checkpoint, device)
    diagnostic = Path(args.diagnostic_root)
    diagnostic.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for number, (sample_id, path) in enumerate(samples, start=1):
            frames = np.load(path)
            if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                raise ValueError(f"Expected uint8 [T,H,W,3] at {path}, got {frames.shape} {frames.dtype}")
            cropped = official_preprocess(frames)
            source = torch.from_numpy(cropped.copy()).permute(0, 3, 1, 2).float()
            source = source.div(127.5).sub(1).to(device)
            chunks = []
            for start in range(0, len(source), args.batch_size):
                chunks.append(token_roundtrip(model, source[start : start + args.batch_size], dtype))
            reconstruction = torch.cat(chunks)
            values = psnr(source, reconstruction)
            edge_ratio = gradient_energy(reconstruction) / gradient_energy(source).clamp_min(1e-12)
            mae = (source - reconstruction).abs().mean(dim=(1, 2, 3)) * 127.5
            row = {
                "sample_id": sample_id,
                "frames": int(len(frames)),
                "median_psnr": float(values.median()),
                "p10_psnr": float(torch.quantile(values, 0.10)),
                "median_mae_255": float(mae.median()),
                "median_gradient_energy_ratio": float(edge_ratio.median()),
            }
            rows.append(row)
            reconstructed = (
                reconstruction.add(1).mul(127.5).round().clamp(0, 255)
                .byte().permute(0, 2, 3, 1).cpu().numpy()
            )
            save_side_by_side(diagnostic / f"{sample_id}.mp4", cropped, reconstructed)
            print(
                f"[HMA MagViT] {number}/{len(samples)} {sample_id} "
                f"PSNR={row['median_psnr']:.2f} p10={row['p10_psnr']:.2f} "
                f"edge={row['median_gradient_energy_ratio']:.3f}"
            )

    median_psnr = float(np.median([row["median_psnr"] for row in rows]))
    p10_psnr = float(np.median([row["p10_psnr"] for row in rows]))
    median_edge = float(np.median([row["median_gradient_energy_ratio"] for row in rows]))
    # These are representation gates, not leaderboard estimates.  In addition
    # to the median, p10 prevents a few easy/static clips from hiding severe
    # articulation failures.
    passed = median_psnr >= 24.0 and p10_psnr >= 20.0 and 0.70 <= median_edge <= 1.30
    result = {
        **structure,
        "preprocess": "official HMA short-side-256 plus 256x256 center crop",
        "decode": "quantized LFQ indices through released MagViT decoder",
        "samples": rows,
        "median_psnr": median_psnr,
        "median_p10_psnr": p10_psnr,
        "median_mae_255": float(np.median([row["median_mae_255"] for row in rows])),
        "median_gradient_energy_ratio": median_edge,
        "thresholds": {
            "median_psnr_min": 24.0,
            "median_p10_psnr_min": 20.0,
            "gradient_energy_ratio": [0.70, 1.30],
        },
        "verdict": "PASS_HMA_MAGVIT" if passed else "REJECT_HMA_MAGVIT",
        "next_step": (
            "Audit and adapt the released HMA action dynamics checkpoint."
            if passed else
            "Stop HMA: its tokenizer already loses too much SO-100 detail."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
