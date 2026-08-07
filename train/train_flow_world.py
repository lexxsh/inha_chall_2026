"""Train the conservative flow world model on provided SO-100 videos only."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import accelerate
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    LeRobotSO100Dataset,
    discover_lerobot_so100_datasets,
)
from data_module import load_delta_scale, transform_actions  # noqa: E402
from flow_world_model import ConservativeFlowWorldModel, flow_world_loss  # noqa: E402

DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


class FlowSO100Dataset(Dataset):
    def __init__(
        self,
        root: str,
        height: int = 160,
        width: int = 256,
        holdout_count: int = 6,
        split: str = "train",
        seed: int = 0,
    ) -> None:
        paths = discover_lerobot_so100_datasets(root)
        paths = [p for p in paths if not any(p.endswith(x) for x in DEFAULT_EXCLUDE)]
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        held = set(shuffled[:holdout_count])
        selected = [p for p in paths if (p in held) == (split == "holdout")]
        stats = json.loads((Path(root) / "so100_action_statistics.json").read_text())
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=selected,
            train=True,
            traj_len=16,
            target_height=height,
            target_width=width,
            pad=True,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            action_mean=stats["mean"],
            action_std=stats["std"],
            use_all_episodes=True,
        )
        self.scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), "hybrid")
        self.selected_paths = selected

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        video = sample["video"].permute(1, 0, 2, 3).contiguous().add(1).mul(0.5)
        actions = transform_actions(sample["act"], self.scale, "hybrid", 0)
        return {"video": video, "actions": actions}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(REPO / "open/data/train"))
    p.add_argument("--output", default=str(REPO / "open/baseline/outputs/flow_world_4k"))
    p.add_argument("--height", type=int, default=160)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ranking-margin", type=float, default=0.01)
    p.add_argument("--no-pretrained-encoder", action="store_true")
    return p.parse_args()


def save_checkpoint(accelerator, model, output: Path, step: int) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    state = accelerator.get_state_dict(model)
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    output.mkdir(parents=True, exist_ok=True)
    save_file(state, output / f"step-{step}.safetensors")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    dataset = FlowSO100Dataset(args.data_root, args.height, args.width, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    model = ConservativeFlowWorldModel(pretrained_encoder=not args.no_pretrained_encoder)
    raft_weights = Raft_Small_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    flow_teacher = raft_small(weights=raft_weights, progress=False).eval()
    flow_teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    flow_teacher = flow_teacher.to(accelerator.device)
    teacher_indices = torch.tensor([0, 7, 14], device=accelerator.device)
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(json.dumps({
            "datasets": len(dataset.selected_paths),
            "clips": len(dataset),
            "trainable_parameters": trainable,
            "effective_batch": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
        }, indent=2))

    output = Path(args.output)
    completed = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    progress = tqdm(total=args.max_steps, disable=not accelerator.is_local_main_process)
    model.train()
    while completed < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                video = batch["video"]
                actions = batch["actions"]
                source = video[:, 0]
                teacher_future = video[:, 1:].index_select(1, teacher_indices)
                source_teacher = source[:, None].expand_as(teacher_future)
                teacher_shape = teacher_future.shape
                teacher_future = teacher_future.reshape(-1, *teacher_shape[2:]).float()
                source_teacher = source_teacher.reshape(-1, *teacher_shape[2:]).float()
                teacher_future, source_teacher = raft_transforms(
                    teacher_future, source_teacher
                )
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    # RAFT(image1, image2) predicts image1 -> image2 flow. Using
                    # future as image1 yields the backward sampling field needed
                    # to reconstruct the future from source pixels.
                    teacher_flow_px = flow_teacher(teacher_future, source_teacher)[-1]
                teacher_flow_px[:, 0] *= 2.0 / max(video.shape[-1] - 1, 1)
                teacher_flow_px[:, 1] *= 2.0 / max(video.shape[-2] - 1, 1)
                teacher_flow = teacher_flow_px.reshape(
                    video.shape[0], len(teacher_indices), 2, video.shape[-2], video.shape[-1]
                )
                wrong_actions = actions.roll(1, dims=0) if actions.shape[0] > 1 else actions.flip(1)
                # A single DDP forward is required here. Calling the wrapped
                # model twice before backward makes DDP broadcast BatchNorm
                # buffers again, modifying values retained by the first graph
                # in-place and causing a version-counter failure.
                paired_source = torch.cat([source, source], dim=0)
                paired_actions = torch.cat([actions, wrong_actions], dim=0)
                paired_prediction, paired_aux = model(
                    paired_source, paired_actions, return_aux=True
                )
                local_batch = source.shape[0]
                prediction = paired_prediction[:local_batch]
                wrong_prediction = paired_prediction[local_batch:]
                aux = {key: value[:local_batch] for key, value in paired_aux.items()}
                loss, metrics = flow_world_loss(
                    prediction,
                    video,
                    aux,
                    wrong_prediction,
                    ranking_margin=args.ranking_margin,
                    teacher_flow=teacher_flow,
                    teacher_indices=teacher_indices,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.detach().float().item():.4f}",
                    rec=f"{metrics['reconstruction'].float().item():.4f}",
                    rank=f"{metrics['ranking'].float().item():.4f}",
                    mask=f"{metrics['mask'].float().item():.3f}",
                    flow=f"{metrics['flow_supervision'].float().item():.3f}",
                )
                if completed % args.save_steps == 0 or completed == args.max_steps:
                    save_checkpoint(accelerator, model, output, completed)
                if completed >= args.max_steps:
                    break
    progress.close()
    elapsed = time.perf_counter() - started
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report = {
            "optimizer_steps": completed,
            "seconds": elapsed,
            "seconds_per_step": elapsed / completed,
            "world_size": accelerator.num_processes,
            "batch_size_per_rank": args.batch_size,
            "effective_batch": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
            "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
            "training_resolution": [args.height, args.width],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_benchmark.json").write_text(json.dumps(report, indent=2))
        (output / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
