"""Render and score the memorized clip for the native BWM preflight."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
for dependency in (
    REPO / "train",
    REPO / "third_party/boundless-world-model",
    REPO / "third_party/DiffSynth-Studio",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from diffsynth.core import ModelConfig  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402
from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline  # noqa: E402
from bwm_native_so100 import ACTION_DIM, build_action_tokens, local_wan22_model_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B"))
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    array = ((frame.float().permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
    return Image.fromarray(array.byte().cpu().numpy(), mode="RGB")


def psnr(prediction: list[Image.Image], target: list[Image.Image]) -> float:
    pred = np.stack([np.asarray(frame, dtype=np.float32) for frame in prediction])
    truth = np.stack([np.asarray(frame, dtype=np.float32) for frame in target])
    mse = float(np.mean(np.square(pred - truth)))
    return float(10.0 * np.log10((255.0**2) / max(mse, 1e-12)))


def main() -> None:
    args = parse_args()
    payload = torch.load(args.sample, map_location="cpu", weights_only=True)
    video = payload["video"].float()
    action = payload["action"].float()
    if video.ndim != 5 or video.shape[:3] != (1, 3, 17):
        raise ValueError(f"Bad overfit video: {tuple(video.shape)}")
    if action.shape != (1, 17, ACTION_DIM):
        raise ValueError(f"Bad overfit action: {tuple(action.shape)}")
    height, width = int(video.shape[-2]), int(video.shape[-1])
    target = [tensor_to_pil(video[0, :, index]) for index in range(16)]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_video(target, str(output / "ground_truth.mp4"), fps=6, quality=9)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    shards, vae = local_wan22_model_paths(args.model_root)
    pipe = build_wan_video_action_pipeline(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=shards), ModelConfig(path=vae)],
        tokenizer_config=None,
        redirect_common_files=False,
        ckpt_path=args.checkpoint,
        action_dim=ACTION_DIM,
        action_mode="adaln",
    )
    pipe.use_gradient_checkpointing = False
    pipe.use_gradient_checkpointing_offload = False
    pipe.eval()

    future = action[0, 1:]
    variants = {
        "normal": action[0],
        "zero-motion": build_action_tokens(future[:1].expand_as(future)),
        "reverse-time": build_action_tokens(future.flip(0)),
    }
    report = {"scope": "single train clip; never submit", "variants": {}}
    for name, tokens in variants.items():
        started = time.perf_counter()
        frames = pipe(
            input_video=video[:, :, :1],
            action=tokens[None],
            seed=args.seed,
            rand_device="cpu",
            height=height,
            width=width,
            num_frames=17,
            num_history_frames=1,
            cfg_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            tiled=False,
            output_type="quantized",
        )
        prediction = [target[0]] + [frame.convert("RGB") for frame in frames[1:16]]
        save_video(prediction, str(output / f"{name}.mp4"), fps=6, quality=9)
        report["variants"][name] = {
            "psnr": psnr(prediction, target),
            "seconds": time.perf_counter() - started,
        }
    normal = report["variants"]["normal"]["psnr"]
    wrong = max(
        report["variants"]["zero-motion"]["psnr"],
        report["variants"]["reverse-time"]["psnr"],
    )
    report["normal_psnr_gain_over_best_wrong"] = normal - wrong
    report["verdict"] = (
        "PASS_SINGLE_CLIP_OVERFIT"
        if normal >= 24.0 and normal - wrong >= 0.5
        else "REJECT_SINGLE_CLIP_OVERFIT"
    )
    (output / "overfit_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {(output / 'overfit_gate.json').resolve()}")


if __name__ == "__main__":
    main()
