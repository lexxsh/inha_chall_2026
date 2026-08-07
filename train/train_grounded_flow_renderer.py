"""Train a source-preserving flow renderer from a frozen grounding probe.

The renderer never has to infer the SO-100 action semantics.  A grounding
checkpoint converts the aligned command trajectory into a visual residual
token; this script learns only where and how to move source pixels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
SUBMISSION_KIT = REPO / "open/submission_kit"
for path in (REPO, REPO / "train", SUBMISSION_KIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_csv_utils import preprocess_video  # noqa: E402
from flow_world_model import ConservativeFlowWorldModel, flow_world_loss  # noqa: E402
from tools.cache_so100_grounding_features import load_manifest, load_window  # noqa: E402
from train_so100_grounding_probe import GroundingProbe  # noqa: E402


class CachedGroundedVideoDataset(Dataset):
    """Decode the exact train windows represented by the DINO grounding cache."""

    def __init__(
        self,
        features: Path,
        manifest: Path,
        data_root: Path,
        height: int,
        width: int,
    ) -> None:
        self.payload = torch.load(features, map_location="cpu", weights_only=False)
        self.indices = torch.nonzero(self.payload["split"] == 0, as_tuple=False).flatten()
        records = {record["id"]: record for record in load_manifest(manifest)}
        self.records = []
        for index in self.indices.tolist():
            record_id = self.payload["ids"][index]
            if record_id not in records:
                raise KeyError(f"cached record {record_id!r} is absent from {manifest}")
            self.records.append(records[record_id])
        self.data_root = data_root
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        cache_index = int(self.indices[item])
        video, actions, _ = load_window(self.data_root, self.records[item])
        # Official aspect-preserving letterbox, returned as [C,T,H,W] in [-1,1].
        video_tensor = preprocess_video(video, self.height, self.width, pad=True)
        video_tensor = video_tensor.permute(1, 0, 2, 3).add(1).mul(0.5)
        cached_actions = self.payload["actions"][cache_index].float()
        loaded_actions = torch.from_numpy(actions)
        if not torch.allclose(cached_actions, loaded_actions, atol=1e-5, rtol=0):
            raise ValueError(f"action cache mismatch for {self.records[item]['id']}")
        return {
            "video": video_tensor,
            "actions": cached_actions,
            "source_dino": self.payload["dino"][cache_index, 0].float(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=REPO / "results/so100_grounding_features.pt"
    )
    parser.add_argument(
        "--manifest", type=Path, default=REPO / "results/so100_contract_manifest.jsonl"
    )
    parser.add_argument("--data-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument("--grounding-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=REPO / "open/baseline/outputs/grounded_flow_renderer"
    )
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--ranking-margin", type=float, default=0.01)
    parser.add_argument("--reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--flow-supervision-weight", type=float, default=5.0)
    parser.add_argument("--flow-motion-boost", type=float, default=20.0)
    parser.add_argument("--mask-supervision-weight", type=float, default=0.5)
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def load_grounder(path: Path, dino_dim: int, device: torch.device) -> GroundingProbe:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = GroundingProbe(
        dino_dim=dino_dim,
        hidden_dim=int(saved_args.get("hidden_dim", 256)),
        layers=int(saved_args.get("layers", 3)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def motion_tokens(
    grounder: GroundingProbe,
    source_dino: torch.Tensor,
    normalized_actions: torch.Tensor,
) -> torch.Tensor:
    prediction = grounder(source_dino, normalized_actions)["future_dino"]
    source = F.normalize(source_dino.float(), dim=-1)[:, None]
    return prediction.float() - source


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
    dataset = CachedGroundedVideoDataset(
        args.features, args.manifest, args.data_root, args.height, args.width
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    dino_dim = int(dataset.payload["dino"].shape[-1])
    grounder = load_grounder(args.grounding_checkpoint, dino_dim, accelerator.device)
    stats = json.loads((args.data_root / "so100_action_statistics.json").read_text())
    action_mean = torch.tensor(stats["mean"], device=accelerator.device)
    action_std = torch.tensor(stats["std"], device=accelerator.device).clamp_min(1e-6)

    model = ConservativeFlowWorldModel(
        grounding_dim=dino_dim,
        pretrained_encoder=not args.no_pretrained_encoder,
    )
    # A frozen spatial encoder retains ImageNet localization and keeps this
    # short screen focused on the motion decoder.
    for module in (model.image_stem, model.image_layer1, model.image_layer2):
        module.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    raft_weights = Raft_Small_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    flow_teacher = raft_small(weights=raft_weights, progress=False).eval().requires_grad_(False)

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
        print(
            json.dumps(
                {
                    "records": len(dataset),
                    "grounding_checkpoint": str(args.grounding_checkpoint.resolve()),
                    "grounding_step": torch.load(
                        args.grounding_checkpoint, map_location="cpu", weights_only=False
                    )["gate"]["step"],
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                    ),
                    "world_size": accelerator.num_processes,
                    "effective_batch": args.batch_size
                    * accelerator.num_processes
                    * args.gradient_accumulation_steps,
                },
                indent=2,
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.perf_counter()
    progress = tqdm(total=args.max_steps, disable=not accelerator.is_local_main_process)
    model.train()
    unwrapped = accelerator.unwrap_model(model)
    for module in (unwrapped.image_stem, unwrapped.image_layer1, unwrapped.image_layer2):
        module.eval()
    iterator = iter(loader)
    while completed < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        with accelerator.accumulate(model):
            video = batch["video"]
            source = video[:, 0]
            actions = batch["actions"]
            source_dino = batch["source_dino"]
            normalized_actions = (actions - action_mean) / action_std
            wrong_actions = normalized_actions.roll(1, dims=0)
            with torch.no_grad():
                correct_motion = motion_tokens(grounder, source_dino, normalized_actions)
                wrong_motion = motion_tokens(grounder, source_dino, wrong_actions)

                teacher_future = video[:, 1:].index_select(1, teacher_indices)
                source_teacher = source[:, None].expand_as(teacher_future)
                teacher_shape = teacher_future.shape
                teacher_future = teacher_future.reshape(-1, *teacher_shape[2:]).float()
                source_teacher = source_teacher.reshape(-1, *teacher_shape[2:]).float()
                teacher_future, source_teacher = raft_transforms(teacher_future, source_teacher)
                with torch.autocast("cuda", enabled=False):
                    teacher_flow_px = flow_teacher(teacher_future, source_teacher)[-1]
                teacher_flow_px[:, 0] *= 2.0 / max(video.shape[-1] - 1, 1)
                teacher_flow_px[:, 1] *= 2.0 / max(video.shape[-2] - 1, 1)
                teacher_flow = teacher_flow_px.reshape(
                    video.shape[0], len(teacher_indices), 2, video.shape[-2], video.shape[-1]
                )

            paired_prediction, paired_aux = model(
                torch.cat([source, source], dim=0),
                motion_tokens=torch.cat([correct_motion, wrong_motion], dim=0),
                return_aux=True,
            )
            local_batch = source.shape[0]
            prediction = paired_prediction[:local_batch]
            wrong_prediction = paired_prediction[local_batch:]
            aux = {key: value[:local_batch] for key, value in paired_aux.items()}
            loss, metrics = flow_world_loss(
                prediction,
                video,
                aux,
                wrong_prediction=wrong_prediction,
                ranking_margin=args.ranking_margin,
                teacher_flow=teacher_flow,
                teacher_indices=teacher_indices,
                reconstruction_weight=args.reconstruction_weight,
                flow_supervision_weight=args.flow_supervision_weight,
                flow_motion_boost=args.flow_motion_boost,
                mask_supervision_weight=args.mask_supervision_weight,
            )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
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
                flow=f"{metrics['flow_supervision'].float().item():.4f}",
            )
            if completed % args.save_steps == 0 or completed == args.max_steps:
                save_checkpoint(accelerator, model, args.output, completed)
    progress.close()
    elapsed = time.perf_counter() - started
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        config = {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        }
        config.update({"grounding_dim": dino_dim, "grounding_contract": "v2 visual residual"})
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        report = {
            "optimizer_steps": completed,
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed, 1),
            "world_size": accelerator.num_processes,
            "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
        }
        (args.output / "train_benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
