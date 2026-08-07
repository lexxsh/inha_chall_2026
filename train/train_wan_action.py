"""LoRA + temporal action AdaLN-Zero training for Wan2.2-TI2V-5B."""
from __future__ import annotations

import os
import json
import sys
import time
import warnings
from pathlib import Path

import accelerate
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from diffsynth.diffusion import ModelLogger  # noqa: E402
from examples.wanvideo.model_training.train import WanTrainingModule, wan_parser  # noqa: E402
from wan_action_dataset import WanSO100Dataset  # noqa: E402
from data_module import action_dims_for  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanActionTrainingModule(WanTrainingModule):
    def __init__(
        self, action_dim=6, action_hidden_dim=512, action_conditioner_version="v1", **kwargs
    ):
        resume = kwargs.pop("resume_from_checkpoint", None)
        extra = kwargs.get("extra_inputs") or ""
        extras = [x for x in extra.split(",") if x]
        for key in ("input_image", "actions"):
            if key not in extras:
                extras.append(key)
        kwargs["extra_inputs"] = ",".join(extras)
        # The adapter must exist before loading a combined LoRA+action resume.
        super().__init__(resume_from_checkpoint=None, **kwargs)
        conditioner = self.pipe.dit.enable_action_conditioning(
            action_dim, action_hidden_dim, version=action_conditioner_version
        )
        reference = next(self.pipe.dit.parameters())
        conditioner.to(device=reference.device, dtype=reference.dtype)
        conditioner.requires_grad_(True)
        if resume is not None:
            self.resume_from_checkpoint(resume, kwargs.get("remove_prefix_in_ckpt"))


def launch_max_steps(accelerator, dataset, model, logger, args):
    action_params = []
    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "action_conditioner" in name or "action_token_encoder" in name or ".action_" in name:
            action_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)
    if not action_params or not lora_params:
        raise RuntimeError(
            f"Expected trainable action and LoRA parameters, got action={len(action_params)}, "
            f"lora={len(lora_params)}, other={len(other_params)}"
        )
    param_groups = [
        {"params": action_params, "lr": args.action_learning_rate},
        {"params": lora_params, "lr": args.lora_learning_rate},
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
    )
    if args.enable_model_cpu_offload:
        raise ValueError("First Wan gate intentionally requires --enable_model_cpu_offload=false.")
    model.to(accelerator.device)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    completed = 0
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    started = time.perf_counter()
    progress = tqdm(total=args.max_steps, disable=not accelerator.is_local_main_process)
    while completed < args.max_steps:
        for data in loader:
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed, 1),
            "peak_gpu_gib_rank0": (
                torch.cuda.max_memory_allocated(accelerator.device) / 2**30
                if torch.cuda.is_available() else 0.0
            ),
            "world_size": accelerator.num_processes,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "action_learning_rate": args.action_learning_rate,
            "lora_learning_rate": args.lora_learning_rate,
        }
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "train_benchmark.json"), "w") as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))


def main():
    parser = wan_parser()
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--action_dim", type=int, default=0, help="0 derives it from action_mode")
    parser.add_argument("--action_hidden_dim", type=int, default=512)
    parser.add_argument("--action_conditioner_version", choices=("v1", "v2", "xattn"), default="v1")
    parser.add_argument("--action_learning_rate", type=float)
    parser.add_argument("--lora_learning_rate", type=float)
    parser.add_argument("--action_mode", default="delta")
    parser.add_argument("--action_shift", type=int, default=0)
    parser.add_argument("--prompt", default="A fixed-camera video of a robot arm manipulating objects.")
    parser.add_argument("--holdout_count", type=int, default=6)
    args = parser.parse_args()
    derived_action_dim = action_dims_for(args.action_mode)
    if args.action_dim not in (0, derived_action_dim):
        raise ValueError(
            f"action_dim={args.action_dim} conflicts with {args.action_mode} ({derived_action_dim})"
        )
    args.action_dim = derived_action_dim
    if args.action_learning_rate is None:
        args.action_learning_rate = args.learning_rate
    if args.lora_learning_rate is None:
        args.lora_learning_rate = args.learning_rate
    if args.num_frames != 17:
        raise ValueError("Wan action alignment is defined for --num_frames 17.")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    dataset = WanSO100Dataset(
        root=args.dataset_base_path,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        holdout_count=args.holdout_count,
        action_mode=args.action_mode,
        action_shift=args.action_shift,
        prompt=args.prompt,
    )
    if accelerator.is_main_process:
        print(f"[Wan data] datasets={len(dataset.selected_paths)}, clips={len(dataset.base)}, repeat={dataset.repeat}")

    model = WanActionTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        action_dim=args.action_dim,
        action_hidden_dim=args.action_hidden_dim,
        action_conditioner_version=args.action_conditioner_version,
    )
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch_max_steps(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
