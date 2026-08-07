"""Train the released Ctrl-World architecture on native SO-100 pose trajectories."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


REPO = Path(__file__).resolve().parents[1]
CTRL_WORLD = REPO / "third_party/Ctrl-World"
for dependency in (str(REPO), str(CTRL_WORLD)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

from models.ctrl_world import CrtlWorld  # noqa: E402
# This file is launched by path (``python train/train_...py``).  In that mode
# ``train/train.py`` shadows the namespace package name ``train``, so import
# the sibling module directly rather than relying on package resolution.
from ctrl_world_so100 import (  # noqa: E402
    CtrlWorldSO100LatentDataset,
    ctrl_world_args,
    load_ctrl_world_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument(
        "--cache-root", type=Path, default=REPO / "cache/ctrl_world_so100_svd"
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=REPO / "results/ctrl_world_so100_pose_stats.json",
    )
    parser.add_argument(
        "--svd-model-path",
        type=Path,
        default=REPO / "checkpoints/stable-video-diffusion-img2vid",
    )
    parser.add_argument("--ctrl-world-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO / "open/baseline/outputs/ctrl_world_so100",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=20260806)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--save-optimizer", action="store_true")
    return parser.parse_args()


def step_from_path(path: Path | None) -> int:
    if path is None:
        return 0
    match = re.search(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else 0


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    keep = {
        name: value.detach().to("cpu").contiguous()
        for name, value in state.items()
        if name.startswith("unet.") or name.startswith("action_encoder.")
    }
    if not keep:
        raise RuntimeError("No Ctrl-World UNet/action tensors found while saving")
    return keep


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    step: int,
    args: argparse.Namespace,
) -> None:
    if not accelerator.is_main_process:
        return
    output_path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    checkpoint = output_path / f"step-{step}.safetensors"
    save_file(
        trainable_state_dict(unwrapped),
        str(checkpoint),
        metadata={
            "step": str(step),
            "method": "Ctrl-World released architecture; native SO100 6D pose",
            "base": str(args.svd_model_path),
        },
    )
    if args.save_optimizer:
        torch.save(
            {"step": step, "optimizer": optimizer.state_dict()},
            output_path / f"step-{step}-optimizer.pt",
        )
    print(f"saved -> {checkpoint.resolve()}")


def main() -> None:
    args = parse_args()
    if not args.svd_model_path.is_dir():
        raise FileNotFoundError(f"Missing official SVD base: {args.svd_model_path}")
    if not args.stats_path.is_file():
        raise FileNotFoundError(f"Missing pose statistics; run PHASE=stats: {args.stats_path}")
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    model_args = ctrl_world_args(svd_model_path=args.svd_model_path)
    model = CrtlWorld(model_args)

    initialization_report = None
    selected_checkpoint = args.resume_checkpoint or args.ctrl_world_checkpoint
    if selected_checkpoint is not None:
        initialization_report = load_ctrl_world_checkpoint(model, selected_checkpoint)
        if accelerator.is_main_process:
            print(json.dumps(initialization_report, indent=2))

    # This is the paper's fine-tuning regime: the complete SVD UNet and the
    # action MLP learn; VAE/image encoder remain frozen.  No LoRA/adapter-only
    # shortcut is used.
    model.unet.requires_grad_(True)
    model.action_encoder.requires_grad_(True)
    model.vae.requires_grad_(False)
    model.image_encoder.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.resume_checkpoint is not None:
        optimizer_path = args.resume_checkpoint.with_name(
            args.resume_checkpoint.stem + "-optimizer.pt"
        )
        if optimizer_path.is_file():
            optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_payload["optimizer"])
            if accelerator.is_main_process:
                print(f"resumed optimizer -> {optimizer_path.resolve()}")

    dataset = CtrlWorldSO100LatentDataset(
        data_root=args.data_root,
        cache_root=args.cache_root,
        stats_path=args.stats_path,
        split="train",
        holdout_count=args.holdout_count,
        split_seed=args.split_seed,
        sample_seed=args.seed,
        repeat=args.repeat,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    start_step = step_from_path(args.resume_checkpoint)
    completed = start_step
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if accelerator.is_main_process:
        args.output_path.mkdir(parents=True, exist_ok=True)
        config = {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "world_size": accelerator.num_processes,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "dataset_records": len(dataset.records),
            "train_datasets": len(dataset.dataset_paths),
            "initialization": initialization_report,
        }
        (args.output_path / "config.json").write_text(json.dumps(config, indent=2))
        print(json.dumps(config, indent=2))

    progress = tqdm(
        total=max(0, args.max_steps - start_step),
        disable=not accelerator.is_local_main_process,
        desc="Ctrl-World SO100",
    )
    started = time.perf_counter()
    running_loss = 0.0
    running_count = 0
    epoch = 0
    while completed < args.max_steps:
        dataset.sample_seed = args.seed + epoch * max(1, len(dataset))
        for batch in loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, _ = model(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            gathered = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            running_loss += gathered
            running_count += 1
            if not accelerator.sync_gradients:
                continue
            completed += 1
            progress.update(1)
            if completed == 1 or completed % 20 == 0:
                progress.set_postfix(loss=running_loss / max(1, running_count))
                running_loss = 0.0
                running_count = 0
            if completed % args.save_steps == 0 or completed == args.max_steps:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, model, optimizer, args.output_path, completed, args)
            if completed >= args.max_steps:
                break
        epoch += 1

    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - started
    if accelerator.is_main_process:
        report = {
            "optimizer_steps": completed,
            "start_step": start_step,
            "seconds": elapsed,
            "seconds_per_new_step": elapsed / max(1, completed - start_step),
            "world_size": accelerator.num_processes,
            "batch_size_per_device": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size
            * args.gradient_accumulation_steps
            * accelerator.num_processes,
            "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0,
            "trainable_parameters": trainable_parameters,
        }
        (args.output_path / "train_benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    progress.close()


if __name__ == "__main__":
    main()
