"""Render and score all 16 frames of the frozen FOMM oracle clip."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from source_anchored_fomm import SourceAnchoredFOMM, motion_target  # noqa: E402
from train_fomm_oracle import FixedOracleClip  # noqa: E402


def save_video(path: Path, frames: torch.Tensor, fps: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = frames.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = array.shape[2]
        stream.height = array.shape[1]
        stream.pix_fmt = "yuv420p"
        for image in array:
            stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24"))
        for packet in stream.encode():
            container.mux(packet)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--output-root", default=str(REPO / "diagnostics/fomm_oracle"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text())
    dataset = FixedOracleClip(
        args.data_root,
        config["dataset_index"],
        config["clip_index"],
        config["height"],
        config["width"],
        config["seed"],
    )
    device = torch.device(args.device)
    model = SourceAnchoredFOMM(config["num_kp"], config["model_size"])
    state = load_file(str(checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    target = dataset.video.to(device)
    source = target[:1].expand(16, -1, -1, -1)
    with torch.inference_mode():
        output = model(source, driving=target)
    prediction = torch.cat([target[:1], output["prediction"][1:]], dim=0)
    static = source
    per_frame_psnr = psnr(prediction, target)
    static_psnr = psnr(static, target)
    expected_edit = motion_target(source, target)
    background_error = (
        (prediction - source).abs() * (1.0 - expected_edit)
    ).sum() / ((1.0 - expected_edit).sum() * 3.0).clamp_min(1.0)
    gradient_prediction = (
        prediction[..., 1:, :] - prediction[..., :-1, :]
    ).abs().mean()
    gradient_target = (target[..., 1:, :] - target[..., :-1, :]).abs().mean()

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "dataset": dataset.dataset_path,
        "start_idx": dataset.start_idx,
        "median_psnr": float(per_frame_psnr.median().cpu()),
        "median_static_psnr": float(static_psnr.median().cpu()),
        "psnr_gain_over_static": float((per_frame_psnr - static_psnr).median().cpu()),
        "background_mae": float(background_error.cpu()),
        "gradient_energy_ratio": float((gradient_prediction / gradient_target.clamp_min(1e-8)).cpu()),
        "mean_edit_fraction": float(output["edit_mask"].mean().cpu()),
    }
    result["verdict"] = (
        "PASS_RENDERER_ORACLE"
        if result["median_psnr"] >= 25.0
        and result["psnr_gain_over_static"] >= 3.0
        and result["background_mae"] <= 0.02
        and 0.70 <= result["gradient_energy_ratio"] <= 1.35
        else "REJECT_RENDERER_ORACLE"
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    save_video(output_root / "prediction.mp4", prediction)
    save_video(output_root / "target.mp4", target)
    save_video(output_root / "static.mp4", static)
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {output_root.resolve()}")


if __name__ == "__main__":
    main()
