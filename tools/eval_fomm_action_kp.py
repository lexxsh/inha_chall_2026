"""Generate normal/static/reverse videos for the action-to-FOMM-KP gate."""
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

from source_anchored_fomm import ActionKeypointPredictor, SourceAnchoredFOMM, motion_target  # noqa: E402
from train_fomm_action_kp import features, load_scales  # noqa: E402
from train_fomm_oracle import FixedOracleClip  # noqa: E402


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
    parser.add_argument("--output-root", default=str(REPO / "diagnostics/fomm_action_kp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation expects one CUDA GPU.")
    device = torch.device("cuda")
    predictor_path = Path(args.checkpoint)
    config = json.loads((predictor_path.parent / "config.json").read_text())
    renderer_path = Path(config["renderer_checkpoint"])
    renderer_config = json.loads((renderer_path.parent / "config.json").read_text())
    dataset = FixedOracleClip(
        renderer_config["data_root"], config["dataset_index"], config["clip_index"],
        config["height"], config["width"], renderer_config["seed"]
    )
    renderer = SourceAnchoredFOMM(config["num_kp"], config["model_size"])
    renderer.load_state_dict(load_file(str(renderer_path), device="cpu"))
    renderer.to(device).eval().requires_grad_(False)
    predictor = ActionKeypointPredictor(
        config["num_kp"], hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]
    )
    predictor.load_state_dict(load_file(str(predictor_path), device="cpu"))
    predictor.to(device).eval()

    target = dataset.video.to(device)
    source = target[:1]
    action = dataset.actions[None].to(device)
    variants = {
        "normal": action,
        "zero": action[:, :1].expand_as(action),
        "reverse": action.flip(1),
        "roll4": action.roll(4, dims=1),
    }
    scales = load_scales(device)
    videos = {}
    with torch.inference_mode():
        source_kp = renderer.kp_detector(source)
        source_frames = source.expand(16, -1, -1, -1)
        for name, variant in variants.items():
            kp = predictor(features(variant, scales), source_kp)
            driving_kp = {"value": kp["value"][0], "jacobian": kp["jacobian"][0]}
            video = renderer(source_frames, driving_kp=driving_kp)["prediction"]
            videos[name] = torch.cat([source, video[1:]], dim=0)

    def psnr(video: torch.Tensor) -> torch.Tensor:
        mse = (video - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
        return -10 * torch.log10(mse)

    normal_psnr = psnr(videos["normal"])
    zero_psnr = psnr(videos["zero"])
    reverse_psnr = psnr(videos["reverse"])
    static_psnr = psnr(source.expand_as(target))
    expected_edit = motion_target(source.expand_as(target), target)
    background_mae = (
        (videos["normal"] - source).abs() * (1 - expected_edit)
    ).sum() / ((1 - expected_edit).sum() * 3).clamp_min(1)
    result = {
        "normal_median_psnr": float(normal_psnr.median()),
        "static_median_psnr": float(static_psnr.median()),
        "normal_minus_static_psnr": float((normal_psnr - static_psnr).median()),
        "normal_minus_zero_psnr": float((normal_psnr - zero_psnr).median()),
        "normal_minus_reverse_psnr": float((normal_psnr - reverse_psnr).median()),
        "normal_zero_abs_difference": float((videos["normal"] - videos["zero"]).abs().mean()),
        "normal_reverse_abs_difference": float((videos["normal"] - videos["reverse"]).abs().mean()),
        "background_mae": float(background_mae),
    }
    result["verdict"] = (
        "PASS_ACTION_KP_OVERFIT"
        if result["normal_median_psnr"] >= 20
        and result["normal_minus_static_psnr"] >= 3
        and result["normal_minus_zero_psnr"] >= 1
        and result["normal_minus_reverse_psnr"] >= 1
        and result["background_mae"] <= 0.02
        else "REJECT_ACTION_KP_OVERFIT"
    )
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    for name, video in videos.items():
        save_video(output / f"{name}.mp4", video)
    save_video(output / "target.mp4", target)
    save_video(output / "static.mp4", source.expand_as(target))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
