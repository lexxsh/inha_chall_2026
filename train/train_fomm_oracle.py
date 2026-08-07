"""Overfit the source-anchored FOMM renderer before learning action motion."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models import VGG19_Weights, vgg19

REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset, discover_lerobot_so100_datasets  # noqa: E402
from source_anchored_fomm import SourceAnchoredFOMM, motion_target, oracle_renderer_loss  # noqa: E402
from modules.model import Transform  # noqa: E402

DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


class VGG19Perceptual(nn.Module):
    """Frozen ImageNet VGG features used by the official FOMM recipe."""

    def __init__(self) -> None:
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30].eval()
        self.capture = {1, 6, 11, 20, 29}
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        value = (image - self.mean) / self.std
        output = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in self.capture:
                output.append(value)
        return output

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = prediction.new_zeros(())
        count = 0
        for scale in (1.0, 0.5, 0.25, 0.125):
            if scale != 1.0:
                size = (max(16, round(prediction.shape[-2] * scale)), max(16, round(prediction.shape[-1] * scale)))
                pred_scaled = F.interpolate(prediction, size=size, mode="bilinear", align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
            else:
                pred_scaled, target_scaled = prediction, target
            predicted_features = self.encode(pred_scaled)
            with torch.no_grad():
                target_features = self.encode(target_scaled)
            for predicted, expected in zip(predicted_features, target_features):
                total = total + (predicted - expected).abs().mean()
                count += 1
        return total / max(count, 1)


class FixedOracleClip(Dataset):
    """One frozen 16-frame clip; target pixels are never generator inputs."""

    def __init__(self, root: str, dataset_index: int, clip_index: int, height: int, width: int, seed: int):
        paths = discover_lerobot_so100_datasets(root)
        paths = [p for p in paths if not any(p.endswith(x) for x in DEFAULT_EXCLUDE)]
        if not 0 <= dataset_index < len(paths):
            raise IndexError(f"dataset-index {dataset_index} outside [0,{len(paths) - 1}]")
        base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=[paths[dataset_index]],
            train=True,
            traj_len=16,
            target_height=height,
            target_width=width,
            pad=True,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            use_all_episodes=True,
        )
        if not 0 <= clip_index < len(base):
            raise IndexError(f"clip-index {clip_index} outside [0,{len(base) - 1}]")
        # __getitem__ samples its start once here; every optimizer step sees the
        # same pixels, which makes this a true representation/overfit gate.
        sample = base[clip_index]
        self.video = sample["video"].permute(1, 0, 2, 3).contiguous().add(1.0).mul(0.5)
        self.actions = sample["act"].contiguous()
        self.dataset_path = paths[dataset_index]
        self.start_idx = int(sample["start_idx"])

    def __len__(self) -> int:
        return 15 * 10_000

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_index = index % 15 + 1
        return {
            "source": self.video[0],
            "target": self.video[frame_index],
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }


class MultiOracleClips(Dataset):
    """Random 16-frame windows from several episodes of one camera/domain."""

    def __init__(
        self, root: str, dataset_index: int, train_clips: int, height: int, width: int, seed: int
    ) -> None:
        paths = discover_lerobot_so100_datasets(root)
        paths = [p for p in paths if not any(p.endswith(x) for x in DEFAULT_EXCLUDE)]
        if not 0 <= dataset_index < len(paths):
            raise IndexError(f"dataset-index {dataset_index} outside [0,{len(paths) - 1}]")
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=[paths[dataset_index]],
            train=True,
            traj_len=16,
            target_height=height,
            target_width=width,
            pad=True,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            use_all_episodes=True,
        )
        self.train_clips = min(train_clips, max(len(self.base) - 1, 1))
        self.dataset_path = paths[dataset_index]
        self.start_idx = -1

    def __len__(self) -> int:
        # Plenty for short screens without constructing a multi-million item
        # RandomSampler permutation at every epoch.
        return self.train_clips * 15 * 100

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip_index = index % self.train_clips
        frame_index = (index // self.train_clips) % 15 + 1
        sample = self.base[clip_index]
        video = sample["video"].permute(1, 0, 2, 3).contiguous().add(1.0).mul(0.5)
        return {
            "source": video[0],
            "target": video[frame_index],
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--output", default=str(REPO / "open/baseline/outputs/fomm_oracle_clip0"))
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--multi-clip", action="store_true")
    parser.add_argument("--train-clips", type=int, default=32)
    parser.add_argument("--resume")
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--equivariance-value-weight", type=float, default=0.0)
    parser.add_argument("--equivariance-jacobian-weight", type=float, default=0.0)
    parser.add_argument("--equivariance-sigma-affine", type=float, default=0.05)
    parser.add_argument("--equivariance-sigma-tps", type=float, default=0.005)
    parser.add_argument("--equivariance-points-tps", type=int, default=5)
    # Official five-block KP hourglass needs H,W divisible by 128. 256x384
    # preserves the full 4:3 camera view through letterboxing without a crop.
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-kp", type=int, default=10)
    parser.add_argument("--model-size", choices=("tiny", "base"), default="base")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_checkpoint(accelerator, model, output: Path, step: int, args: argparse.Namespace) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    state = accelerator.get_state_dict(model)
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    output.mkdir(parents=True, exist_ok=True)
    save_file(state, output / f"step-{step}.safetensors")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    dataset = (
        MultiOracleClips(
            args.data_root, args.dataset_index, args.train_clips, args.height, args.width, args.seed
        )
        if args.multi_clip
        else FixedOracleClip(
            args.data_root, args.dataset_index, args.clip_index, args.height, args.width, args.seed
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    model = SourceAnchoredFOMM(num_kp=args.num_kp, model_size=args.model_size)
    if args.resume:
        missing, unexpected = model.load_state_dict(load_file(args.resume, device="cpu"), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"resume mismatch missing={missing} unexpected={unexpected}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    perceptual = None
    if args.perceptual_weight > 0:
        perceptual = VGG19Perceptual().to(accelerator.device)

    if accelerator.is_main_process:
        print(json.dumps({
            "dataset": dataset.dataset_path,
            "start_idx": dataset.start_idx,
            "multi_clip": args.multi_clip,
            "train_clips": getattr(dataset, "train_clips", 1),
            "frames": 16,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "effective_batch": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
        }, indent=2))

    output = Path(args.output)
    completed = 0
    started = time.perf_counter()
    progress = tqdm(total=args.max_steps, disable=not accelerator.is_local_main_process)
    model.train()
    while completed < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                source = batch["source"]
                target = batch["target"]
                use_equivariance = (
                    args.equivariance_value_weight > 0
                    or args.equivariance_jacobian_weight > 0
                )
                transform = None
                transformed_target = None
                if use_equivariance:
                    transform = Transform(
                        source.shape[0],
                        sigma_affine=args.equivariance_sigma_affine,
                        sigma_tps=args.equivariance_sigma_tps,
                        points_tps=args.equivariance_points_tps,
                    )
                    transformed_target = transform.transform_frame(target)
                # One wrapped forward avoids extra DDP buffer broadcasts. The
                # second group pins identity. The optional third group extracts
                # keypoints from FOMM's official affine/TPS transform.
                source_groups = [source, source]
                driving_groups = [target, source]
                if transformed_target is not None:
                    source_groups.append(source)
                    driving_groups.append(transformed_target)
                paired_source = torch.cat(source_groups, dim=0)
                paired_driving = torch.cat(driving_groups, dim=0)
                paired = model(paired_source, driving=paired_driving)
                local_batch = source.shape[0]
                output_target = {
                    key: (
                        {subkey: value[:local_batch] for subkey, value in item.items()}
                        if isinstance(item, dict)
                        else item[:local_batch]
                    )
                    for key, item in paired.items()
                }
                identity_prediction = paired["prediction"][local_batch : 2 * local_batch]
                loss, metrics = oracle_renderer_loss(
                    output_target, source, target, identity_prediction
                )
                perceptual_value = loss.new_zeros(())
                if perceptual is not None:
                    # Replace the unchanged background with detached target
                    # pixels so VGG capacity is spent on robot/object detail.
                    edit = motion_target(source, target).detach()
                    focused_prediction = output_target["prediction"] * edit + target.detach() * (1 - edit)
                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=torch.bfloat16,
                        enabled=accelerator.device.type == "cuda",
                    ):
                        perceptual_value = perceptual(focused_prediction, target)
                    loss = loss + args.perceptual_weight * perceptual_value
                equivariance_value = loss.new_zeros(())
                equivariance_jacobian = loss.new_zeros(())
                if transform is not None:
                    transformed_kp = {
                        key: value[2 * local_batch : 3 * local_batch]
                        for key, value in paired["kp_driving"].items()
                    }
                    if args.equivariance_value_weight > 0:
                        expected_kp = transform.warp_coordinates(transformed_kp["value"])
                        target_kp = paired["kp_driving"]["value"][:local_batch]
                        equivariance_value = (target_kp - expected_kp).abs().mean()
                        loss = loss + args.equivariance_value_weight * equivariance_value
                    if args.equivariance_jacobian_weight > 0:
                        transformed_jacobian = torch.matmul(
                            transform.jacobian(transformed_kp["value"]),
                            transformed_kp["jacobian"],
                        ).float()
                        relative = torch.matmul(
                            torch.inverse(
                                paired["kp_driving"]["jacobian"][:local_batch]
                                .float()
                            ),
                            transformed_jacobian,
                        )
                        identity = torch.eye(2, device=relative.device, dtype=relative.dtype)
                        equivariance_jacobian = (
                            relative - identity.view(1, 1, 2, 2)
                        ).abs().mean()
                        loss = loss + args.equivariance_jacobian_weight * equivariance_jacobian
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.detach().float().item():.4f}",
                    rec=f"{metrics['reconstruction'].float().item():.4f}",
                    identity=f"{metrics['identity'].float().item():.4f}",
                    edit=f"{metrics['edit_fraction'].float().item():.3f}",
                    perc=f"{perceptual_value.detach().float().item():.3f}",
                    eqv=f"{equivariance_value.detach().float().item():.3f}",
                    eqj=f"{equivariance_jacobian.detach().float().item():.3f}",
                )
                if completed % args.save_steps == 0 or completed == args.max_steps:
                    save_checkpoint(accelerator, model, output, completed, args)
                if completed >= args.max_steps:
                    break
    progress.close()
    elapsed = time.perf_counter() - started
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report = {
            "optimizer_steps": completed,
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed, 1),
            "world_size": accelerator.num_processes,
            "model_size": args.model_size,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
