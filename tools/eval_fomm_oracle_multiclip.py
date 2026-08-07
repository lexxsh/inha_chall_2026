"""Held-episode oracle gate for a multi-clip FOMM renderer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from source_anchored_fomm import SourceAnchoredFOMM, motion_target  # noqa: E402
from train_fomm_oracle import MultiOracleClips  # noqa: E402


def save_video(path: Path, frames: torch.Tensor) -> None:
    array = frames.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=6)
        stream.width, stream.height, stream.pix_fmt = array.shape[2], array.shape[1], "yuv420p"
        for image in array:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default=str(REPO / "diagnostics/fomm_oracle_multiclip"))
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation expects one CUDA GPU.")
    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text())
    if not config.get("multi_clip"):
        raise ValueError("Checkpoint config is not a multi-clip renderer.")
    dataset = MultiOracleClips(
        config["data_root"], config["dataset_index"], config["train_clips"],
        config["height"], config["width"], config["seed"]
    )
    device = torch.device("cuda")
    model = SourceAnchoredFOMM(config["num_kp"], config["model_size"])
    model.load_state_dict(load_file(str(checkpoint), device="cpu"))
    model.to(device).eval()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    all_psnr, all_static, all_background, all_gradient, all_motion_gradient = [], [], [], [], []
    held_start = dataset.train_clips
    held_count = min(args.limit, len(dataset.base) - held_start)
    if held_count <= 0:
        raise ValueError("No held episodes remain; reduce --train-clips.")
    for number, clip_index in enumerate(range(held_start, held_start + held_count)):
        sample = dataset.base[clip_index]
        target = sample["video"].permute(1, 0, 2, 3).add(1).mul(0.5).to(device)
        source = target[:1].expand_as(target)
        with torch.inference_mode():
            prediction = model(source, driving=target)["prediction"]
        prediction = torch.cat([target[:1], prediction[1:]], dim=0)
        mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
        static_mse = (source - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
        all_psnr.append(-10 * torch.log10(mse)[1:])
        all_static.append(-10 * torch.log10(static_mse)[1:])
        edit = motion_target(source, target)
        background = ((prediction - source).abs() * (1 - edit)).sum() / ((1 - edit).sum() * 3).clamp_min(1)
        all_background.append(background[None])
        pred_grad = (prediction[..., 1:, :] - prediction[..., :-1, :]).abs().mean()
        target_grad = (target[..., 1:, :] - target[..., :-1, :]).abs().mean()
        all_gradient.append((pred_grad / target_grad.clamp_min(1e-8))[None])
        pred_dy = (prediction[..., 1:, :] - prediction[..., :-1, :]).abs().mean(dim=1, keepdim=True)
        target_dy = (target[..., 1:, :] - target[..., :-1, :]).abs().mean(dim=1, keepdim=True)
        pred_dx = (prediction[..., :, 1:] - prediction[..., :, :-1]).abs().mean(dim=1, keepdim=True)
        target_dx = (target[..., :, 1:] - target[..., :, :-1]).abs().mean(dim=1, keepdim=True)
        mask_y = torch.maximum(edit[..., 1:, :], edit[..., :-1, :])
        mask_x = torch.maximum(edit[..., :, 1:], edit[..., :, :-1])
        pred_motion_grad = (pred_dy * mask_y).sum() + (pred_dx * mask_x).sum()
        target_motion_grad = (target_dy * mask_y).sum() + (target_dx * mask_x).sum()
        all_motion_gradient.append((pred_motion_grad / target_motion_grad.clamp_min(1e-8))[None])
        if number < 4:
            save_video(output / f"clip{number:02d}_prediction.mp4", prediction)
            save_video(output / f"clip{number:02d}_target.mp4", target)

    psnr = torch.cat(all_psnr)
    static = torch.cat(all_static)
    result = {
        "held_clips": held_count,
        "median_psnr": float(psnr.median()),
        "median_static_psnr": float(static.median()),
        "median_psnr_gain": float((psnr - static).median()),
        "median_background_mae": float(torch.cat(all_background).median()),
        "median_gradient_energy_ratio": float(torch.cat(all_gradient).median()),
        "median_motion_gradient_ratio": float(torch.cat(all_motion_gradient).median()),
    }
    result["verdict"] = (
        "PASS_MULTICLIP_RENDERER_ORACLE"
        if result["median_psnr"] >= 22
        and result["median_psnr_gain"] >= 4
        and result["median_background_mae"] <= 0.02
        and 0.65 <= result["median_gradient_energy_ratio"] <= 1.35
        and 0.60 <= result["median_motion_gradient_ratio"] <= 1.40
        else "REJECT_MULTICLIP_RENDERER_ORACLE"
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
