"""Native BWM-style full post-training of Wan2.2-TI2V-5B on SO100.

Unlike the older ``bwm_so100_lora`` experiment, this starts from the vanilla
Wan2.2 base (not the public BWM robot checkpoint), creates a fresh 6D action
encoder, and updates the real DiT weights together with that encoder.  Use the
provided FSDP launcher; ordinary 8-GPU DDP replicates the 5B optimizer state.
"""
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
for dependency in (str(BWM_REPO), str(DIFFSYNTH), str(REPO / "train")):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diffsynth.core import ModelConfig, load_state_dict  # noqa: E402
from diffsynth.diffusion import DiffusionTrainingModule, FlowMatchSFTLoss, ModelLogger  # noqa: E402
from wan_video_action.pipelines.wan_video_action import (  # noqa: E402
    build_wan_video_action_pipeline,
)
from bwm_native_so100 import (  # noqa: E402
    ACTION_DIM,
    RGB_FRAMES,
    BWMNativeSO100Dataset,
    CachedSingleClip,
    local_wan22_model_paths,
)


class BWMNativeTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        *,
        model_root: str,
        resume_checkpoint: str | None = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        shards, vae = local_wan22_model_paths(model_root)
        self.pipe = build_wan_video_action_pipeline(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[ModelConfig(path=shards), ModelConfig(path=vae)],
            tokenizer_config=None,
            redirect_common_files=False,
            ckpt_path=None,
            action_dim=ACTION_DIM,
            action_mode="adaln",
        )
        self.pipe = self.split_pipeline_units(
            "sft", self.pipe, trainable_models="dit,action_encoder", lora_base_model=None
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models="dit,action_encoder",
            lora_base_model=None,
            task="sft",
        )

        # These paths are absent from the BWM forward when text is disabled.
        self.pipe.dit.text_embedding.requires_grad_(False).eval()
        for block in self.pipe.dit.blocks:
            for module in (block.cross_attn.k, block.cross_attn.v, block.cross_attn.norm_k):
                module.requires_grad_(False).eval()
        # Legacy flattened-action path; encode_ti2v2 uses only mlp1/mlp2.
        self.pipe.action_encoder.action_embedding.requires_grad_(False).eval()

        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self.pipe.use_gradient_checkpointing = self.use_gradient_checkpointing
        self.pipe.use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload
        if resume_checkpoint:
            self.load_native_checkpoint(resume_checkpoint)

        self.architecture_summary = self._architecture_summary()

    def _architecture_summary(self) -> dict:
        action, dit, unexpected = 0, 0, []
        tensors = {"action": 0, "dit": 0}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("pipe.action_encoder.action_mlp"):
                action += parameter.numel()
                tensors["action"] += 1
            elif name.startswith("pipe.dit."):
                dit += parameter.numel()
                tensors["dit"] += 1
            else:
                unexpected.append(name)
        if unexpected or action == 0 or dit == 0:
            raise RuntimeError(
                f"Bad native BWM trainables: action={action}, dit={dit}, unexpected={unexpected[:20]}"
            )
        return {
            "initialization": "vanilla Wan2.2-TI2V-5B + fresh 6D BWM action encoder",
            "action_parameters": action,
            "dit_parameters": dit,
            "trainable_parameters": action + dit,
            "action_parameter_tensors": tensors["action"],
            "dit_parameter_tensors": tensors["dit"],
            "frozen_text_and_shared_action_kv": True,
            "lora": False,
        }

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """Write a complete BWM-compatible DiT/action checkpoint.

        Under FSDP, ``named_parameters()`` contains wrapper-qualified block
        names while ``accelerator.get_state_dict()`` has restored canonical
        names.  Filtering one with the other silently dropped every wrapped
        DiTBlock.  Select canonical state-dict prefixes directly instead.
        """
        exported = {}
        for key, value in state_dict.items():
            if key.startswith("pipe.dit."):
                exported[key[len("pipe.dit.") :]] = value
            elif key.startswith("pipe.action_encoder."):
                exported[key] = value
        block_ids = {
            int(key.split(".", 2)[1])
            for key in exported
            if key.startswith("blocks.") and key.split(".", 2)[1].isdigit()
        }
        if block_ids != set(range(len(self.pipe.dit.blocks))):
            raise RuntimeError(
                f"Refusing incomplete native BWM checkpoint: blocks={sorted(block_ids)}"
            )
        if not any(key.startswith("pipe.action_encoder.action_mlp1.") for key in exported):
            raise RuntimeError("Refusing checkpoint without the native BWM action encoder")
        return exported

    def load_native_checkpoint(self, path: str) -> None:
        raw = load_state_dict(path, torch_dtype=self.pipe.torch_dtype, device="cpu")
        mapped = {
            (key if key.startswith("pipe.action_encoder.") else f"pipe.dit.{key}"): value
            for key, value in raw.items()
        }
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected native BWM checkpoint keys: {unexpected[:20]}")
        expected = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        absent = sorted(expected - set(mapped))
        if absent:
            raise ValueError(f"Native BWM checkpoint is incomplete; missing {absent[:20]}")
        print(f"[BWM native resume] loaded {len(raw)} tensors from {path}; frozen missing={len(missing)}")

    def get_pipeline_inputs(self, data: dict) -> tuple[dict, dict, dict]:
        video, action = data["video"], data["action"]
        if video.ndim != 5 or video.shape[:3] != (1, 3, RGB_FRAMES):
            raise ValueError(f"Video must be [1,3,17,H,W], got {tuple(video.shape)}")
        if action.shape != (1, RGB_FRAMES, ACTION_DIM):
            raise ValueError(f"Action must be [1,17,6], got {tuple(action.shape)}")
        shared = {
            "input_video": video,
            "action": action,
            "height": int(video.shape[-2]),
            "width": int(video.shape[-1]),
            "num_frames": RGB_FRAMES,
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
        inputs = self.transfer_data_to_device(
            self.get_pipeline_inputs(data), self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return FlowMatchSFTLoss(self.pipe, **inputs[0], **inputs[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B"))
    parser.add_argument("--output-path", default=str(REPO / "open/baseline/outputs/bwm_native_so100_10k"))
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--start-step", type=int, default=-1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--dataset-repeat", type=int, default=2)
    parser.add_argument("--dataset-num-workers", type=int, default=2)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-learning-rate", type=float, default=5e-5)
    parser.add_argument("--dit-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing-offload", action="store_true")
    parser.add_argument("--single-clip-overfit", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_start_step(args: argparse.Namespace) -> None:
    if args.resume_checkpoint:
        if not Path(args.resume_checkpoint).is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        if args.start_step < 0:
            match = re.search(r"step-(\d+)\.safetensors$", Path(args.resume_checkpoint).name)
            if match is None:
                raise ValueError("Pass --start-step for a non step-N checkpoint")
            args.start_step = int(match.group(1))
    else:
        if args.start_step not in (-1, 0):
            raise ValueError("--start-step requires --resume-checkpoint")
        args.start_step = 0
    if args.max_steps <= args.start_step:
        raise ValueError(f"max_steps must exceed start_step: {args.max_steps} <= {args.start_step}")


def launch(accelerator, dataset, model, logger, args) -> None:
    action_parameters, dit_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("pipe.action_encoder.action_mlp"):
            action_parameters.append(parameter)
        elif name.startswith("pipe.dit."):
            dit_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
    optimizer = torch.optim.AdamW(
        [
            {"params": action_parameters, "lr": args.action_learning_rate},
            {"params": dit_parameters, "lr": args.dit_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=not args.single_clip_overfit,
        batch_size=1,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
        pin_memory=True,
        persistent_workers=args.dataset_num_workers > 0,
    )
    model.to(accelerator.device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    logger.num_steps = args.start_step
    completed = args.start_step
    started = time.perf_counter()
    progress = tqdm(total=args.max_steps, initial=completed, disable=not accelerator.is_main_process)
    while completed < args.max_steps:
        for data in loader:
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            completed += 1
            progress.update(1)
            progress.set_postfix(loss=f"{float(loss.detach()):.5f}")
            logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
            if completed >= args.max_steps:
                break
    progress.close()
    logger.on_training_end(accelerator, model, args.save_steps)
    elapsed = time.perf_counter() - started
    if accelerator.is_main_process:
        report = {
            "optimizer_steps": completed,
            "start_step": args.start_step,
            "steps_this_run": completed - args.start_step,
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed - args.start_step, 1),
            "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
            "world_size": accelerator.num_processes,
            "height": args.height,
            "width": args.width,
            "single_clip_overfit": args.single_clip_overfit,
            **accelerator.unwrap_model(model).architecture_summary,
        }
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_benchmark.json").write_text(json.dumps(report, indent=2))
        (output / "training_args.json").write_text(json.dumps(vars(args), indent=2))
        print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("height and width must be divisible by 32")
    resolve_start_step(args)
    set_seed(args.seed)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    dataset = BWMNativeSO100Dataset(
        root=args.dataset_root,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        holdout_count=args.holdout_count,
        seed=args.seed,
        split="train",
    )
    if args.single_clip_overfit:
        dataset = CachedSingleClip(
            dataset, repeats=(args.max_steps - args.start_step) * accelerator.num_processes * 2
        )
        if accelerator.is_main_process:
            output = Path(args.output_path)
            output.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "video": dataset.sample["video"].cpu(),
                    "action": dataset.sample["action"].cpu(),
                    "height": args.height,
                    "width": args.width,
                    "contract": "BWM native single-clip preflight; never submit",
                },
                output / "overfit_sample.pt",
            )
        accelerator.wait_for_everyone()
    model = BWMNativeTrainingModule(
        model_root=args.model_root,
        resume_checkpoint=args.resume_checkpoint,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.gradient_checkpointing_offload,
        device=accelerator.device,
    )
    if accelerator.is_main_process:
        print(json.dumps(model.architecture_summary, indent=2))
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=None)
    try:
        launch(accelerator, dataset, model, logger, args)
    finally:
        # Avoid the ProcessGroupNCCL watchdog warning on otherwise clean exits.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
