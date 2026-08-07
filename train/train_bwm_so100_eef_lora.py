"""Fine-tune public BWM-5B with its original 14D EEF action contract."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import accelerate

from bwm_so100_eef import BWMSO100EEFDataset, DEFAULT_STATS
from train_bwm_so100_lora import (
    REPO,
    BWMSO100TrainingModule,
    launch,
    set_seed,
)
from diffsynth.diffusion import ModelLogger


ACTION_CONTRACT = (
    "17x14 original BWM dual-arm EEF; SO100 source-anchored FK in left 7D, "
    "inactive right arm zero; original 14D input projections preserved"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS))
    parser.add_argument("--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B"))
    parser.add_argument(
        "--bwm-checkpoint",
        default=str(REPO / "checkpoints/Boundless-World-Model/step-12000.safetensors"),
    )
    parser.add_argument(
        "--output-path", default=str(REPO / "open/baseline/outputs/bwm_so100_eef14_12k")
    )
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--start-step", type=int, default=-1)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--dataset-num-workers", type=int, default=4)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--action-learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("BWM height and width must be divisible by 32")
    for required in (args.dataset_root, args.stats_path, args.model_root, args.bwm_checkpoint):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    if args.resume_checkpoint:
        if not Path(args.resume_checkpoint).is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        if args.start_step < 0:
            match = re.search(r"step-(\d+)\.safetensors$", Path(args.resume_checkpoint).name)
            if match is None:
                raise ValueError("Pass --start-step when checkpoint name is not step-N.safetensors")
            args.start_step = int(match.group(1))
    elif args.start_step not in (-1, 0):
        raise ValueError("--start-step requires --resume-checkpoint")
    else:
        args.start_step = 0
    if args.max_steps <= args.start_step:
        raise ValueError(f"max steps must exceed start step: {args.max_steps} <= {args.start_step}")

    set_seed(args.seed)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    dataset = BWMSO100EEFDataset(
        root=args.dataset_root,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        holdout_count=args.holdout_count,
        seed=args.seed,
        split="train",
        stats_path=args.stats_path,
    )
    if accelerator.is_main_process:
        print(
            f"[BWM EEF14] datasets={len(dataset.selected_paths)}, clips={len(dataset.base)}, "
            f"resolution={args.height}x{args.width}, target_steps={args.max_steps}"
        )
    model = BWMSO100TrainingModule(
        model_root=args.model_root,
        bwm_checkpoint=args.bwm_checkpoint,
        lora_rank=args.lora_rank,
        resume_checkpoint=args.resume_checkpoint,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.gradient_checkpointing_offload,
        action_dim=14,
        expand_action_inputs=False,
        action_contract=ACTION_CONTRACT,
        device=accelerator.device,
    )
    first_shapes = {
        "action_mlp1": list(model.pipe.action_encoder.action_mlp1[0].weight.shape),
        "action_mlp2": list(model.pipe.action_encoder.action_mlp2[0].weight.shape),
    }
    if first_shapes != {"action_mlp1": [3072, 14], "action_mlp2": [12288, 56]}:
        raise RuntimeError(f"Original BWM action inputs were not preserved: {first_shapes}")
    if accelerator.is_main_process:
        print(json.dumps({**model.architecture_summary, "input_shapes": first_shapes}, indent=2))
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=None)
    launch(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    main()
