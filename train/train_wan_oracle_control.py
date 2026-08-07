"""Cheap Wan gate: can a zero-init spatial adapter obey oracle point tracks?"""
from __future__ import annotations

import json
import os
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
from wan_oracle_track_dataset import WanOracleTrackDataset  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def paired_track_flow_loss(
    pipe,
    inputs_shared,
    inputs_posi,
    margin: float,
    weight: float,
) -> torch.Tensor:
    """Correct-vs-wrong control with identical latent, noise, and timestep."""
    inputs = {**inputs_shared, **inputs_posi}
    max_id = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_id = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_id, max_id, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(inputs["input_latents"])
    noisy = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    if "first_frame_latents" in inputs:
        noisy[:, :, :1] = inputs["first_frame_latents"]

    control = inputs["track_control"]
    wrong_control = inputs.pop("wrong_track_control", None)
    if wrong_control is None:
        # v1 compatibility: preserve frame zero and reverse only future motion.
        time_dim = control.ndim - 3
        clean_control = control.narrow(time_dim, 0, 1)
        future_control = control.narrow(time_dim, 1, control.shape[time_dim] - 1).flip(time_dim)
        wrong_control = torch.cat([clean_control, future_control], dim=time_dim)
    if control.ndim == 4:
        control = control.unsqueeze(0)
        wrong_control = wrong_control.unsqueeze(0)

    # One paired forward avoids DDP/re-entrant-checkpoint hooks seeing the same
    # parameters in two forward graphs before backward.
    batch = noisy.shape[0]
    paired_inputs = {}
    for key, value in inputs.items():
        if key == "track_control":
            paired_inputs[key] = torch.cat([control, wrong_control], dim=0)
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
            paired_inputs[key] = torch.cat([value, value], dim=0)
        else:
            paired_inputs[key] = value
    paired_inputs["latents"] = torch.cat([noisy, noisy], dim=0)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    # Wan2.2 TI2V expands one shared diffusion timestep over its clean/future
    # token map internally.  Keep shape [1]; t_mod broadcasts over paired B=2.
    paired = pipe.model_fn(**models, **paired_inputs, timestep=timestep)
    correct, wrong = paired[:batch], paired[batch:]
    if "first_frame_latents" in inputs:
        correct, wrong, target = correct[:, :, 1:], wrong[:, :, 1:], target[:, :, 1:]

    reduce_dims = tuple(range(1, correct.ndim))
    correct_per = (correct.float() - target.float()).square().mean(dim=reduce_dims)
    wrong_per = (wrong.float() - target.float()).square().mean(dim=reduce_dims)
    ranking = torch.relu(margin + correct_per - wrong_per).mean()
    return (correct_per.mean() + weight * ranking) * pipe.scheduler.training_weight(timestep)


class WanOracleControlTrainingModule(WanTrainingModule):
    def __init__(
        self,
        track_hidden_dim: int = 128,
        control_rank_margin: float = 0.005,
        control_rank_weight: float = 0.1,
        structural_zero: bool = False,
        adapter_only: bool = False,
        wrong_control_mode: str = "reverse",
        **kwargs,
    ):
        resume = kwargs.pop("resume_from_checkpoint", None)
        extra = [item for item in (kwargs.get("extra_inputs") or "").split(",") if item]
        keys = ["input_image", "track_control"]
        if wrong_control_mode == "cross_clip":
            keys.append("wrong_track_control")
        for key in keys:
            if key not in extra:
                extra.append(key)
        kwargs["extra_inputs"] = ",".join(extra)
        super().__init__(resume_from_checkpoint=None, **kwargs)
        adapter = self.pipe.dit.enable_track_conditioning(
            input_dim=3, hidden_dim=track_hidden_dim, structural_zero=structural_zero
        )
        if adapter_only:
            self.pipe.dit.requires_grad_(False)
        reference = next(self.pipe.dit.parameters())
        adapter.to(device=reference.device, dtype=reference.dtype).requires_grad_(True)
        self.task_to_loss["sft"] = lambda pipe, shared, posi, nega: paired_track_flow_loss(
            pipe, shared, posi, control_rank_margin, control_rank_weight
        )
        if resume is not None:
            self.resume_from_checkpoint(resume, kwargs.get("remove_prefix_in_ckpt"))


def launch(accelerator, dataset, model, logger, args) -> None:
    adapter_params, lora_params, other_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "track_adapter" in name:
            adapter_params.append(parameter)
        elif "lora_" in name:
            lora_params.append(parameter)
        else:
            other_params.append(parameter)
    if not adapter_params or (not args.adapter_only and not lora_params):
        raise RuntimeError(
            f"Expected track adapter and LoRA parameters, got adapter={len(adapter_params)}, "
            f"lora={len(lora_params)}, other={len(other_params)}"
        )
    groups = [{"params": adapter_params, "lr": args.track_learning_rate}]
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lora_learning_rate})
    if other_params:
        groups.append({"params": other_params, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
    )
    if args.enable_model_cpu_offload:
        raise ValueError("Oracle gate requires --enable_model_cpu_offload=false")
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
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed, 1),
            "peak_gpu_gib_rank0": (
                torch.cuda.max_memory_allocated(accelerator.device) / 2**30
                if torch.cuda.is_available() else 0.0
            ),
            "world_size": accelerator.num_processes,
            "train_clips": len(dataset.records),
            "track_learning_rate": args.track_learning_rate,
            "lora_learning_rate": args.lora_learning_rate,
            "adapter_only": args.adapter_only,
            "structural_zero": args.structural_zero,
            "wrong_control_mode": args.wrong_control_mode,
        }
        os.makedirs(args.output_path, exist_ok=True)
        Path(args.output_path, "train_benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


def main() -> None:
    parser = wan_parser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--track_hidden_dim", type=int, default=128)
    parser.add_argument("--track_learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_learning_rate", type=float, default=2e-5)
    parser.add_argument("--control_rank_margin", type=float, default=0.005)
    parser.add_argument("--control_rank_weight", type=float, default=0.1)
    parser.add_argument("--structural_zero", action="store_true")
    parser.add_argument("--adapter_only", action="store_true")
    parser.add_argument(
        "--wrong_control_mode", choices=("reverse", "cross_clip"), default="reverse"
    )
    parser.add_argument("--prompt", default="A fixed-camera video of a robot arm manipulating objects.")
    args = parser.parse_args()
    if args.num_frames != 17:
        raise ValueError("Oracle track gate is defined for 17 frames")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    dataset = WanOracleTrackDataset(
        manifest=args.manifest,
        data_root=args.dataset_base_path,
        split="train",
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        prompt=args.prompt,
    )
    if accelerator.is_main_process:
        print(f"[oracle data] fixed train clips={len(dataset.records)}, repeat={dataset.repeat}")
    model = WanOracleControlTrainingModule(
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
        track_hidden_dim=args.track_hidden_dim,
        control_rank_margin=args.control_rank_margin,
        control_rank_weight=args.control_rank_weight,
        structural_zero=args.structural_zero,
        adapter_only=args.adapter_only,
        wrong_control_mode=args.wrong_control_mode,
    )
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
