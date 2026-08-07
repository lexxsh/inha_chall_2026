"""Train a source-conditioned SO100 spatial control branch on Wan2.1 I2V 14B."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from diffsynth.diffusion import ModelLogger  # noqa: E402
from examples.wanvideo.model_training.train import WanTrainingModule, wan_parser  # noqa: E402
try:
    from wan_spatial_action_dataset import WanSO100SpatialActionDataset  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    from train.wan_spatial_action_dataset import WanSO100SpatialActionDataset  # noqa: E402


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")


def local_model_paths(model_root: Path) -> tuple[str, str]:
    model_root = model_root.resolve()
    shards = sorted(str(path) for path in model_root.glob("diffusion_pytorch_model-*.safetensors"))
    components: list[object] = [
        shards,
        str(model_root / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(model_root / "Wan2.1_VAE.pth"),
        str(model_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]
    tokenizer = model_root / "google/umt5-xxl"
    missing = []
    if not shards:
        missing.append(str(model_root / "diffusion_pytorch_model-*.safetensors"))
    missing.extend(str(path) for path in components[1:] if not Path(path).exists())
    if not tokenizer.is_dir():
        missing.append(str(tokenizer))
    if missing:
        raise FileNotFoundError("Missing local Wan2.1 files:\n  " + "\n  ".join(missing))
    return json.dumps(components), str(tokenizer)


def grouped_motion_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Map RGB masks [B,1,17,H,W] to Wan latents [B,1,5,h,w]."""
    if mask.ndim == 4:
        mask = mask.unsqueeze(0)
    if mask.ndim != 5 or mask.shape[1] != 1 or mask.shape[2] != 17:
        raise ValueError(f"motion_mask must be [B,1,17,H,W], got {tuple(mask.shape)}")
    groups = [mask[:, :, :1]]
    for group in range(4):
        begin = 1 + 4 * group
        groups.append(mask[:, :, begin : begin + 4].amax(dim=2, keepdim=True))
    latent_mask = torch.cat(groups, dim=2)
    return F.interpolate(
        latent_mask.float(),
        size=target.shape[2:],
        mode="trilinear",
        align_corners=False,
    )


def _batchify(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1).expand(batch)
    if tensor.shape[0] == batch:
        return tensor
    if batch == 1:
        return tensor.unsqueeze(0)
    raise ValueError(f"Cannot infer batch dimension for shape {tuple(tensor.shape)}, B={batch}")


def paired_spatial_action_flow_loss(
    pipe,
    inputs_shared: dict,
    inputs_posi: dict,
    margin: float,
    rank_weight: float,
    motion_weight: float,
    metrics_sink=None,
) -> torch.Tensor:
    """Correct-vs-counterfactual denoising with identical noise and timestep."""
    inputs = {**inputs_shared, **inputs_posi}
    wrong_actions = inputs.pop("wrong_actions")
    raw_mask = inputs.pop("motion_mask")
    ranking_valid = inputs.pop("ranking_valid")

    max_id = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_id = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_id, max_id, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    clean = inputs["input_latents"]
    batch = clean.shape[0]
    noise = torch.randn_like(clean)
    noisy = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)
    if "first_frame_latents" in inputs:
        noisy[:, :, :1] = inputs["first_frame_latents"]

    correct_actions = _batchify(inputs["actions"], batch)
    wrong_actions = _batchify(wrong_actions, batch)
    paired_inputs = {}
    for key, value in inputs.items():
        if key == "actions":
            paired_inputs[key] = torch.cat([correct_actions, wrong_actions], dim=0)
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
            paired_inputs[key] = torch.cat([value, value], dim=0)
        else:
            paired_inputs[key] = value
    paired_inputs["latents"] = torch.cat([noisy, noisy], dim=0)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **paired_inputs, timestep=timestep)
    correct, wrong = prediction[:batch], prediction[batch:]
    if "first_frame_latents" in inputs:
        correct, wrong, target = correct[:, :, 1:], wrong[:, :, 1:], target[:, :, 1:]

    mask = grouped_motion_mask(raw_mask, prediction).to(device=target.device)
    if target.shape[2] != mask.shape[2]:
        # Only TI2V fused models expose first_frame_latents; keep this path
        # correct if the adapter is later audited on such a model.
        mask = mask[:, :, -target.shape[2] :]
    weight = 1.0 + motion_weight * mask
    squared_correct = (correct.float() - target.float()).square()
    squared_wrong = (wrong.float() - target.float()).square()
    denominator = weight.sum(dim=(1, 2, 3, 4)) * target.shape[1]
    correct_per = (squared_correct * weight).sum(dim=(1, 2, 3, 4)) / denominator
    wrong_per = (squared_wrong * weight).sum(dim=(1, 2, 3, 4)) / denominator

    valid = _batchify(ranking_valid, batch).float().to(correct_per.device)
    rank_per = torch.relu(margin + correct_per - wrong_per) * valid
    rank_loss = rank_per.sum() / valid.sum().clamp(min=1.0)
    reconstruction = correct_per.mean()
    scheduler_weight = pipe.scheduler.training_weight(timestep)
    loss = (reconstruction + rank_weight * rank_loss) * scheduler_weight

    if metrics_sink is not None:
        metrics_sink.last_loss_metrics = {
            "reconstruction": float(reconstruction.detach()),
            "ranking": float(rank_loss.detach()),
            "correct_minus_wrong": float((correct_per - wrong_per).mean().detach()),
            "motion_fraction": float(mask.mean().detach()),
        }
    return loss


class WanSpatialActionTrainingModule(WanTrainingModule):
    def __init__(
        self,
        base_model_path: str,
        action_dim: int = 18,
        control_hidden_dim: int = 192,
        injection_layers: str = "0,10,20,30",
        control_rank_margin: float = 0.002,
        control_rank_weight: float = 0.5,
        motion_loss_weight: float = 2.0,
        adapter_only: bool = False,
        **kwargs,
    ) -> None:
        resume = kwargs.pop("resume_from_checkpoint", None)
        model_paths, tokenizer_path = local_model_paths(Path(base_model_path))
        kwargs["model_paths"] = model_paths
        kwargs["model_id_with_origin_paths"] = None
        kwargs["tokenizer_path"] = tokenizer_path
        extra = [item for item in (kwargs.get("extra_inputs") or "").split(",") if item]
        for key in ("input_image", "actions", "wrong_actions", "motion_mask", "ranking_valid"):
            if key not in extra:
                extra.append(key)
        kwargs["extra_inputs"] = ",".join(extra)

        super().__init__(resume_from_checkpoint=None, **kwargs)
        layers = tuple(int(value) for value in injection_layers.split(",") if value.strip())
        adapter = self.pipe.dit.enable_spatial_action_conditioning(
            action_dim=action_dim,
            hidden_dim=control_hidden_dim,
            source_dim=16,
            injection_layers=layers,
        )
        if adapter_only:
            self.pipe.dit.requires_grad_(False)
        reference = next(self.pipe.dit.parameters())
        adapter.to(device=reference.device, dtype=reference.dtype).requires_grad_(True)
        self.last_loss_metrics: dict[str, float] = {}
        self.task_to_loss["sft"] = lambda pipe, shared, posi, nega: paired_spatial_action_flow_loss(
            pipe,
            shared,
            posi,
            margin=control_rank_margin,
            rank_weight=control_rank_weight,
            motion_weight=motion_loss_weight,
            metrics_sink=self,
        )
        if resume is not None:
            self.resume_from_checkpoint(resume, kwargs.get("remove_prefix_in_ckpt"))


def launch(accelerator, dataset, model, logger, args) -> None:
    adapter_params, lora_params, other_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "spatial_action_adapter" in name:
            adapter_params.append(parameter)
        elif "lora_" in name:
            lora_params.append(parameter)
        else:
            other_params.append(parameter)
    if not adapter_params:
        raise RuntimeError("No trainable spatial action adapter parameters")
    if not args.adapter_only and not lora_params:
        raise RuntimeError("LoRA was requested but no trainable LoRA parameters were found")
    if other_params:
        names = [name for name, p in model.named_parameters() if p.requires_grad and "spatial_action_adapter" not in name and "lora_" not in name]
        raise RuntimeError(f"Unexpected trainable parameters: {names[:20]}")

    if accelerator.is_main_process:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        control_config = {
            "format": "wan21_so100_spatial_action_v1",
            "base_model_path": str(Path(args.base_model_path).resolve()),
            "action_mode": args.action_mode,
            "action_dim": dataset.action_dim,
            "control_hidden_dim": args.control_hidden_dim,
            "injection_layers": list(model.pipe.dit.spatial_action_adapter.injection_layers),
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "lora_rank": args.lora_rank,
            "lora_target_modules": args.lora_target_modules,
        }
        (output_path / "control_config.json").write_text(json.dumps(control_config, indent=2))
        (output_path / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    accelerator.wait_for_everyone()

    groups = [{"params": adapter_params, "lr": args.control_learning_rate}]
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lora_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
        persistent_workers=args.dataset_num_workers > 0,
    )
    if args.enable_model_cpu_offload:
        raise ValueError("This DDP gate does not support layerwise model CPU offload")
    model.to(accelerator.device)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    completed = 0
    metric_history: list[dict[str, float | int]] = []
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
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed += 1
                logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                progress.update(1)
                unwrapped = accelerator.unwrap_model(model)
                metrics = unwrapped.last_loss_metrics
                metric_values = torch.tensor(
                    [
                        loss.detach().float().item(),
                        metrics.get("reconstruction", 0.0),
                        metrics.get("ranking", 0.0),
                        metrics.get("correct_minus_wrong", 0.0),
                        metrics.get("motion_fraction", 0.0),
                    ],
                    dtype=torch.float32,
                    device=accelerator.device,
                )
                metric_values = accelerator.gather(metric_values).reshape(
                    accelerator.num_processes, -1
                ).mean(dim=0)
                aggregate = {
                    "step": completed,
                    "loss": float(metric_values[0]),
                    "reconstruction": float(metric_values[1]),
                    "ranking": float(metric_values[2]),
                    "correct_minus_wrong": float(metric_values[3]),
                    "motion_fraction": float(metric_values[4]),
                    "adapter_lr": float(optimizer.param_groups[0]["lr"]),
                    "lora_lr": float(optimizer.param_groups[-1]["lr"] if lora_params else 0.0),
                }
                if accelerator.is_main_process:
                    metric_history.append(aggregate)
                progress.set_postfix(
                    loss=f"{aggregate['loss']:.5f}",
                    rank=f"{aggregate['ranking']:.4f}",
                    delta=f"{aggregate['correct_minus_wrong']:+.4f}",
                )
                if completed >= args.max_steps:
                    break
    progress.close()
    logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - started
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        adapter = unwrapped.pipe.dit.spatial_action_adapter
        report = {
            "optimizer_steps": completed,
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed, 1),
            "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
            "world_size": accelerator.num_processes,
            "train_dataset_groups": len(dataset.selected_paths),
            "base_clips": len(dataset.base),
            "action_mode": args.action_mode,
            "action_dim": dataset.action_dim,
            "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
            "lora_parameters": sum(parameter.numel() for parameter in lora_params),
            "injection_layers": list(adapter.injection_layers),
            "control_learning_rate": args.control_learning_rate,
            "lora_learning_rate": args.lora_learning_rate,
            "control_rank_margin": args.control_rank_margin,
            "control_rank_weight": args.control_rank_weight,
            "motion_loss_weight": args.motion_loss_weight,
            "static_probability": args.static_probability,
            "last_loss_metrics": metric_history[-1] if metric_history else {},
        }
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        Path(args.output_path, "train_benchmark.json").write_text(json.dumps(report, indent=2))
        Path(args.output_path, "training_metrics.json").write_text(
            json.dumps(metric_history, indent=2)
        )
        print(json.dumps(report, indent=2))


def main() -> None:
    parser = wan_parser()
    parser.add_argument(
        "--base_model_path",
        default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P"),
    )
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--action_mode", default="hybrid")
    parser.add_argument("--action_shift", type=int, default=0)
    parser.add_argument("--holdout_count", type=int, default=6)
    parser.add_argument("--static_probability", type=float, default=0.15)
    parser.add_argument("--motion_mask_threshold", type=float, default=0.06)
    parser.add_argument("--motion_mask_dilation", type=int, default=9)
    parser.add_argument("--control_hidden_dim", type=int, default=192)
    parser.add_argument("--injection_layers", default="0,10,20,30")
    parser.add_argument("--control_learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_learning_rate", type=float, default=1e-5)
    parser.add_argument("--control_rank_margin", type=float, default=0.002)
    parser.add_argument("--control_rank_weight", type=float, default=0.5)
    parser.add_argument("--motion_loss_weight", type=float, default=2.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adapter_only", action="store_true")
    parser.add_argument(
        "--prompt",
        default="A fixed-camera video of a tabletop robot arm manipulating objects.",
    )
    args = parser.parse_args()
    if args.task != "sft":
        raise ValueError("The paired spatial-action objective currently supports only --task sft")
    if args.enable_model_cpu_offload:
        raise ValueError("Wan21 spatial DDP does not support --enable_model_cpu_offload")
    if args.num_frames != 17:
        raise ValueError("The SO100 spatial action model requires --num_frames 17")
    if args.height % 16 or args.width % 16:
        raise ValueError("Training height and width must be divisible by 16")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )
    dataset = WanSO100SpatialActionDataset(
        root=args.dataset_base_path,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        holdout_count=args.holdout_count,
        action_mode=args.action_mode,
        action_shift=args.action_shift,
        prompt=args.prompt,
        split="train",
        static_probability=args.static_probability,
        motion_mask_threshold=args.motion_mask_threshold,
        motion_mask_dilation=args.motion_mask_dilation,
    )
    if accelerator.is_main_process:
        print(
            f"[Wan21 spatial data] groups={len(dataset.selected_paths)}, "
            f"clips={len(dataset.base)}, repeat={dataset.repeat}, "
            f"action={dataset.action_mode}/{dataset.action_dim}D"
        )

    model = WanSpatialActionTrainingModule(
        base_model_path=args.base_model_path,
        action_dim=dataset.action_dim,
        control_hidden_dim=args.control_hidden_dim,
        injection_layers=args.injection_layers,
        control_rank_margin=args.control_rank_margin,
        control_rank_weight=args.control_rank_weight,
        motion_loss_weight=args.motion_loss_weight,
        adapter_only=args.adapter_only,
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
        task=args.task,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
