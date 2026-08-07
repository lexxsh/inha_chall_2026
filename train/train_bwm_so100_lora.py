"""Fine-tune BWM-5B on SO-100 with full action MLPs and all-block DiT LoRA."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import accelerate
import numpy as np
import torch
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
DIFFSYNTH = REPO / "third_party/DiffSynth-Studio"
BWM_REPO = REPO / "third_party/boundless-world-model"
for dependency in (str(BWM_REPO), str(DIFFSYNTH)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from diffsynth.core import ModelConfig, load_state_dict  # noqa: E402
from diffsynth.diffusion import DiffusionTrainingModule, FlowMatchSFTLoss, ModelLogger  # noqa: E402
from wan_video_action.pipelines.wan_video_action import (  # noqa: E402
    build_wan_video_action_pipeline,
)

from bwm_so100 import (  # noqa: E402
    BWM_ACTION_DIM,
    BWM_ACTION_FRAMES,
    BWMSO100Dataset,
    bwm_trainable_summary,
    expand_bwm_action_encoder,
    local_wan22_model_paths,
)


LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"


class BWMSO100TrainingModule(DiffusionTrainingModule):
    """BWM public model initialized before SO100-specific trainable layers."""

    def __init__(
        self,
        *,
        model_root: str,
        bwm_checkpoint: str,
        lora_rank: int = 32,
        resume_checkpoint: str | None = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        action_dim: int = BWM_ACTION_DIM,
        expand_action_inputs: bool = True,
        action_contract: str = "hybrid18",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        shards, vae = local_wan22_model_paths(model_root)
        model_configs = [ModelConfig(path=shards), ModelConfig(path=vae)]
        self.pipe = build_wan_video_action_pipeline(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=None,
            redirect_common_files=False,
            ckpt_path=bwm_checkpoint,
            action_dim=14,
            action_mode="adaln",
        )
        if expand_action_inputs:
            expand_bwm_action_encoder(self.pipe.action_encoder, action_dim)
        elif action_dim != 14 or int(self.pipe.action_encoder.action_dim) != 14:
            raise ValueError("Preserved BWM action input requires action_dim=14")
        self.action_dim = int(action_dim)
        self.action_contract = action_contract

        # SFT keeps all preprocessing units. Freeze the base, train the complete
        # BWM action MLP, and patch every matching linear in all DiT blocks.
        self.pipe = self.split_pipeline_units(
            "sft", self.pipe, trainable_models="action_encoder", lora_base_model="dit"
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models="action_encoder",
            lora_base_model="dit",
            lora_target_modules=LORA_TARGETS,
            lora_rank=lora_rank,
            lora_checkpoint=None,
            task="sft",
        )
        # action_embedding belongs to an older/noise path and is not called by
        # encode_ti2v2. Training it would trigger DDP unused-parameter failures.
        self.pipe.action_encoder.action_embedding.requires_grad_(False)
        self.pipe.action_encoder.action_embedding.eval()

        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        # The public BWM patched inference ``__call__`` reads these attributes
        # from the pipeline, but its builder does not initialize them. Keep the
        # module and pipeline values synchronized for both training and direct
        # inference calls.
        self.pipe.use_gradient_checkpointing = self.use_gradient_checkpointing
        self.pipe.use_gradient_checkpointing_offload = (
            self.use_gradient_checkpointing_offload
        )
        if resume_checkpoint:
            self.load_trainable_checkpoint(resume_checkpoint)

        summary = bwm_trainable_summary(self)
        expected_blocks = list(range(len(self.pipe.dit.blocks)))
        if summary["unexpected_trainables"]:
            raise RuntimeError(
                "Unexpected trainable BWM parameters: "
                + ", ".join(summary["unexpected_trainables"][:20])
            )
        if not summary["action_parameter_tensors"]:
            raise RuntimeError("BWM action MLP is not trainable")
        if summary["lora_blocks"] != expected_blocks:
            raise RuntimeError(
                f"LoRA does not cover every DiT block: got {summary['lora_blocks']}, "
                f"expected {expected_blocks}"
            )
        self.architecture_summary = summary

    def load_trainable_checkpoint(self, path: str) -> None:
        state = load_state_dict(path, torch_dtype=self.pipe.torch_dtype, device="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys in SO100 BWM checkpoint: {unexpected[:20]}")
        current = set(self.state_dict())
        if not state or not set(state).issubset(current):
            raise ValueError(f"Checkpoint did not map cleanly to this model: {path}")
        print(
            f"[BWM resume] loaded {len(state)} trainable tensors from {path}; "
            f"frozen/missing tensors={len(missing)}"
        )

    def get_pipeline_inputs(self, data: dict) -> tuple[dict, dict, dict]:
        video = data["video"]
        action = data["action"]
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[2] != BWM_ACTION_FRAMES:
            raise ValueError(f"BWM video must be [1,3,17,H,W], got {tuple(video.shape)}")
        if action.shape != (1, BWM_ACTION_FRAMES, self.action_dim):
            raise ValueError(
                f"BWM action must be [1,17,{self.action_dim}], got {tuple(action.shape)}"
            )
        shared = {
            "input_video": video,
            "action": action,
            "height": int(video.shape[-2]),
            "width": int(video.shape[-1]),
            "num_frames": BWM_ACTION_FRAMES,
            "num_history_frames": 1,
            "num_views": 1,
            "cfg_scale": 1.0,
            "tiled": False,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "rand_device": self.pipe.device,
            "seed": None,
            "vace_reference_image": None,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
        }
        return shared, {}, {}

    def forward(self, data: dict) -> torch.Tensor:
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return FlowMatchSFTLoss(self.pipe, **inputs[0], **inputs[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument(
        "--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B")
    )
    parser.add_argument(
        "--bwm-checkpoint",
        default=str(REPO / "checkpoints/Boundless-World-Model/step-12000.safetensors"),
    )
    parser.add_argument(
        "--output-path", default=str(REPO / "open/baseline/outputs/bwm_so100_lora_10k")
    )
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--start-step",
        type=int,
        default=-1,
        help="Absolute step already completed; -1 infers it from step-N.safetensors.",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--dataset-num-workers", type=int, default=4)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--action-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing-offload", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def launch(
    accelerator: accelerate.Accelerator,
    dataset: BWMSO100Dataset,
    model: BWMSO100TrainingModule,
    logger: ModelLogger,
    args: argparse.Namespace,
) -> None:
    action_parameters: list[torch.nn.Parameter] = []
    lora_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".action_mlp1." in name or ".action_mlp2." in name:
            action_parameters.append(parameter)
        elif "lora_" in name:
            lora_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
    if not action_parameters or not lora_parameters:
        raise RuntimeError("Both the BWM action MLP and DiT LoRA must be trainable")

    optimizer = torch.optim.AdamW(
        [
            {"params": action_parameters, "lr": args.action_learning_rate},
            {"params": lora_parameters, "lr": args.lora_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
        pin_memory=True,
        persistent_workers=args.dataset_num_workers > 0,
    )
    model.to(accelerator.device)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    completed = int(args.start_step)
    steps_this_run = args.max_steps - completed
    logger.num_steps = completed
    started = time.perf_counter()
    progress = tqdm(
        total=steps_this_run,
        disable=not accelerator.is_local_main_process,
        desc=f"BWM steps {completed}->{args.max_steps}",
    )
    while completed < args.max_steps:
        for data in loader:
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed += 1
                logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().float().item():.5f}")
                if completed >= args.max_steps:
                    break
    progress.close()
    logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.wait_for_everyone()

    elapsed = time.perf_counter() - started
    if accelerator.is_main_process:
        report = {
            "optimizer_steps": completed,
            "start_step": args.start_step,
            "steps_this_run": steps_this_run,
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(steps_this_run, 1),
            "peak_gpu_gib_rank0": (
                torch.cuda.max_memory_allocated(accelerator.device) / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            "world_size": accelerator.num_processes,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "height": args.height,
            "width": args.width,
            "train_datasets": len(dataset.selected_paths),
            "train_clips": len(dataset.base),
            "bwm_checkpoint": str(Path(args.bwm_checkpoint).resolve()),
            "action_contract": accelerator.unwrap_model(model).action_contract,
            **accelerator.unwrap_model(model).architecture_summary,
        }
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("BWM height and width must be divisible by 32")
    for required in (args.model_root, args.bwm_checkpoint, args.dataset_root):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    if args.resume_checkpoint:
        if not Path(args.resume_checkpoint).is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        if args.start_step < 0:
            match = re.search(r"step-(\d+)\.safetensors$", Path(args.resume_checkpoint).name)
            if match is None:
                raise ValueError(
                    "Cannot infer --start-step from resume checkpoint name; pass it explicitly"
                )
            args.start_step = int(match.group(1))
    elif args.start_step not in (-1, 0):
        raise ValueError("--start-step requires --resume-checkpoint")
    else:
        args.start_step = 0
    if args.max_steps <= args.start_step:
        raise ValueError(
            f"--max-steps is the absolute target and must exceed start step: "
            f"{args.max_steps} <= {args.start_step}"
        )
    set_seed(args.seed)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )
    dataset = BWMSO100Dataset(
        root=args.dataset_root,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        holdout_count=args.holdout_count,
        seed=args.seed,
        split="train",
    )
    if accelerator.is_main_process:
        print(
            f"[BWM SO100] datasets={len(dataset.selected_paths)}, clips={len(dataset.base)}, "
            f"resolution={args.height}x{args.width}, target_steps={args.max_steps}"
        )
    model = BWMSO100TrainingModule(
        model_root=args.model_root,
        bwm_checkpoint=args.bwm_checkpoint,
        lora_rank=args.lora_rank,
        resume_checkpoint=args.resume_checkpoint,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.gradient_checkpointing_offload,
        device=accelerator.device,
    )
    if accelerator.is_main_process:
        print(json.dumps(model.architecture_summary, indent=2))
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=None)
    launch(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    main()
