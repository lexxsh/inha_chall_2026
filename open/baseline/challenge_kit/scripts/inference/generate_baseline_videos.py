from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import torch
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.utils.train import get_model
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.feature_csv_utils import (
    build_inference_batch,
    list_challenge_sample_ids,
    load_action_stats,
    save_video_tensor,
)


def config_value(config, key: str, default):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def load_defaults(config_path: str) -> dict:
    config = OmegaConf.load(config_path)
    return {
        "checkpoint": config_value(config, "act_cond_unet_checkpoint", "../checkpoints/baseline_diffusion.ckpt"),
        "challenge_root": config_value(config, "data.params.root", "../data/eval"),
        "prediction_root": "../../submission_kit/input_videos",
        "action_stats_path": config_value(config, "data.params.action_stats_path", "../data/train/so100_action_statistics.json"),
        "batch_size": 1,
        "fps": 6,
        "target_height": config_value(config, "data.params.target_height", 320),
        "target_width": config_value(config, "data.params.target_width", 512),
        "pad": config_value(config, "data.params.pad", True),
        "precision": 16,
    }


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def load_diffusion_model(config_path: str, checkpoint_path: str, device: torch.device):
    eval_config = OmegaConf.load(config_path)
    model_config = OmegaConf.load(eval_config.model_config_file).model
    model = get_model(model_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return eval_config, model, DDIMSampler(model)


def generate_predictions(args, sample_ids: list[str], device: torch.device) -> None:
    eval_config, model, sampler = load_diffusion_model(args.config, args.checkpoint, device)
    ddim_kwargs = OmegaConf.to_container(eval_config.ddim_kwargs, resolve=True)
    if args.ddim_steps is not None:
        ddim_kwargs["ddim_steps"] = args.ddim_steps

    action_mean, action_std = load_action_stats(args.action_stats_path)
    if action_mean is not None:
        action_mean = action_mean.to(device)
        action_std = action_std.to(device)

    amp_context = torch.cuda.amp.autocast() if args.precision == 16 and device.type == "cuda" else nullcontext()
    challenge_root = Path(args.challenge_root)
    prediction_root = Path(args.prediction_root)

    for start in range(0, len(sample_ids), args.batch_size):
        batch_ids = sample_ids[start : start + args.batch_size]
        pred_paths = [prediction_root / f"{sample_id}.mp4" for sample_id in batch_ids]
        if not args.overwrite and all(path.exists() for path in pred_paths):
            continue

        batch = build_inference_batch(
            challenge_root,
            batch_ids,
            args.target_height,
            args.target_width,
            args.pad,
            args.fps,
            action_mean,
            action_std,
            device,
        )
        z, c, uc, cond_mask, _logs, kwargs = model.prepare_batch_for_inference(batch)
        sample_kwargs = dict(ddim_kwargs)
        sample_kwargs.update(kwargs)
        sample_steps = sample_kwargs.pop("ddim_steps")
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), model.ema_scope("Baseline Video Generation"):
            with amp_context:
                samples, _ = sampler.sample(
                    sample_steps,
                    batch_size=z.shape[0],
                    shape=shape,
                    conditioning=c,
                    unconditional_conditioning=uc,
                    mask=cond_mask,
                    x0=z,
                    **sample_kwargs,
                )
            generated = model.decode_first_stage(samples)

        for sample_id, video in zip(batch_ids, generated):
            save_video_tensor(video, prediction_root / f"{sample_id}.mp4", args.fps)
        print(f"[generate] wrote predictions for {start + len(batch_ids)}/{len(sample_ids)} samples")


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/eval/inha_submission_eval_11M.yaml")
    config_args, _ = config_parser.parse_known_args()
    defaults = load_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        description="Generate baseline mp4 videos from eval images/actions.",
        parents=[config_parser],
    )
    parser.add_argument("--checkpoint", default=defaults["checkpoint"])
    parser.add_argument("--challenge-root", default=defaults["challenge_root"])
    parser.add_argument("--prediction-root", default=defaults["prediction_root"])
    parser.add_argument("--action-stats-path", default=defaults["action_stats_path"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--fps", type=int, default=defaults["fps"])
    parser.add_argument("--target-height", type=int, default=defaults["target_height"])
    parser.add_argument("--target-width", type=int, default=defaults["target_width"])
    parser.add_argument("--pad", type=str_to_bool, nargs="?", const=True, default=defaults["pad"])
    parser.add_argument("--no-pad", dest="pad", action="store_false")
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--precision", type=int, default=defaults["precision"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_ids = list_challenge_sample_ids(Path(args.challenge_root))
    generate_predictions(args, sample_ids, device)


if __name__ == "__main__":
    main()
