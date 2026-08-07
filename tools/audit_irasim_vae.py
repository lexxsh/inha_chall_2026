"""Measure the SDXL VAE ceiling independently of IRASim denoising."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.models import AutoencoderKL

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from train.irasim_so100 import preprocess_irasim_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", default=str(REPO / "models/IRASim/sdxl-base"))
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(REPO / "results/irasim_vae_gate.json"))
    parser.add_argument("--diagnostic-root", default=str(REPO / "diagnostics/irasim_vae"))
    return parser.parse_args()


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"))[None].copy()
    return preprocess_irasim_frames(array).permute(1, 0, 2, 3).to(device)


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = (reference.float() - prediction.float()).square().mean().clamp(min=1e-12)
    return float(10.0 * torch.log10(4.0 / mse))


def gradient_energy(image: torch.Tensor) -> float:
    gray = image.float().mean(dim=1)
    dx = (gray[:, :, 1:] - gray[:, :, :-1]).square().mean()
    dy = (gray[:, 1:, :] - gray[:, :-1, :]).square().mean()
    return float(dx + dy)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    root = Path(args.challenge_root)
    ids = sorted(path.stem for path in (root / "images").glob("*.png"))[: args.limit]
    diagnostic = Path(args.diagnostic_root)
    diagnostic.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample_id in enumerate(ids):
        source = image_tensor(root / "images" / f"{sample_id}.png", device)
        with torch.no_grad():
            posterior = vae.encode(source).latent_dist
            mode_latent = posterior.mode() * vae.config.scaling_factor
            mode_reconstruction = vae.decode(
                mode_latent / vae.config.scaling_factor
            ).sample
            generator = torch.Generator(device=device).manual_seed(3407 + index)
            sample_latent = posterior.sample(generator=generator) * vae.config.scaling_factor
            reconstruction = vae.decode(
                sample_latent / vae.config.scaling_factor
            ).sample
        row = {
            "sample_id": sample_id,
            "mode_psnr": psnr(source, mode_reconstruction),
            "psnr": psnr(source, reconstruction),
            "mae_255": float((source - reconstruction).abs().mean() * 127.5),
            "gradient_energy_ratio": gradient_energy(reconstruction) / max(gradient_energy(source), 1e-12),
        }
        rows.append(row)
        src = ((source[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        rec = ((reconstruction[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.concatenate([src, rec], axis=1)).save(diagnostic / f"{sample_id}.png")
        print(
            f"[VAE] {index + 1}/{len(ids)} {sample_id} "
            f"official-sample PSNR={row['psnr']:.2f} mode={row['mode_psnr']:.2f}"
        )

    median_psnr = float(np.median([row["psnr"] for row in rows]))
    median_mae = float(np.median([row["mae_255"] for row in rows]))
    median_edge = float(np.median([row["gradient_energy_ratio"] for row in rows]))
    result = {
        "samples": rows,
        "median_psnr": median_psnr,
        "median_mode_psnr": float(np.median([row["mode_psnr"] for row in rows])),
        "median_mae_255": median_mae,
        "median_gradient_energy_ratio": median_edge,
        # This is a diagnostic threshold, not a leaderboard proxy.  It asks
        # whether the VAE alone preserves ordinary image detail well enough
        # that a later collapse must be attributed to denoising.
        "verdict": "PASS_VAE" if median_psnr >= 25.0 and median_edge >= 0.70 else "REJECT_VAE",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
