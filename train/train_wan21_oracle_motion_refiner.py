"""Train a frozen-Wan2.1 oracle dense-motion refiner on train-only SO100 clips."""
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
    from wan_oracle_motion_field_dataset import WanOracleMotionFieldDataset  # type: ignore # noqa: E402
except ModuleNotFoundError:
    from train.wan_oracle_motion_field_dataset import WanOracleMotionFieldDataset  # noqa: E402


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
    if mask.ndim == 4:
        mask = mask.unsqueeze(0)
    if mask.ndim != 5 or mask.shape[1:3] != (1, 17):
        raise ValueError(f"motion_mask must be [B,1,17,H,W], got {tuple(mask.shape)}")
    groups = [mask[:, :, :1]]
    for group in range(4):
        begin = 1 + 4 * group
        groups.append(mask[:, :, begin : begin + 4].amax(dim=2, keepdim=True))
    mask = torch.cat(groups, dim=2)
    return F.interpolate(mask.float(), size=target.shape[2:], mode="trilinear", align_corners=False)


def _batchify(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1).expand(batch)
    if tensor.shape[0] == batch:
        return tensor
    if batch == 1:
        return tensor.unsqueeze(0)
    raise ValueError(f"Cannot infer batch dimension for {tuple(tensor.shape)}, B={batch}")


def paired_oracle_refiner_loss(
    pipe,
    inputs_shared: dict,
    inputs_posi: dict,
    margin: float,
    rank_weight: float,
    motion_weight: float,
    metrics_sink=None,
) -> torch.Tensor:
    inputs = {**inputs_shared, **inputs_posi}
    wrong_control = inputs.pop("wrong_motion_field_control")
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

    correct_control = _batchify(inputs["motion_field_control"], batch)
    wrong_control = _batchify(wrong_control, batch)
    paired_inputs = {}
    for key, value in inputs.items():
        if key == "motion_field_control":
            paired_inputs[key] = torch.cat([correct_control, wrong_control], dim=0)
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
            paired_inputs[key] = torch.cat([value, value], dim=0)
        else:
            paired_inputs[key] = value
    paired_inputs["latents"] = torch.cat([noisy, noisy], dim=0)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **paired_inputs, timestep=timestep)
    correct, wrong = prediction[:batch], prediction[batch:]

    mask = grouped_motion_mask(raw_mask, target).to(target.device)
    weight = 1.0 + motion_weight * mask
    correct_sq = (correct.float() - target.float()).square()
    wrong_sq = (wrong.float() - target.float()).square()
    denominator = weight.sum(dim=(1, 2, 3, 4)) * target.shape[1]
    correct_per = (correct_sq * weight).sum(dim=(1, 2, 3, 4)) / denominator
    wrong_per = (wrong_sq * weight).sum(dim=(1, 2, 3, 4)) / denominator
    valid = _batchify(ranking_valid, batch).float().to(correct_per.device)
    rank_per = torch.relu(margin + correct_per - wrong_per) * valid
    rank_loss = rank_per.sum() / valid.sum().clamp_min(1.0)
    reconstruction = correct_per.mean()
    loss = (reconstruction + rank_weight * rank_loss) * pipe.scheduler.training_weight(timestep)
    if metrics_sink is not None:
        metrics_sink.last_loss_metrics = {
            "reconstruction": float(reconstruction.detach()),
            "ranking": float(rank_loss.detach()),
            "correct_minus_wrong": float((correct_per - wrong_per).mean().detach()),
            "motion_fraction": float(mask.mean().detach()),
        }
    return loss


class WanOracleMotionRefinerTrainingModule(WanTrainingModule):
    def __init__(
        self,
        base_model_path: str,
        control_dim: int = 7,
        control_hidden_dim: int = 192,
        injection_layers: str = "0,10,20,30",
        control_rank_margin: float = 0.002,
        control_rank_weight: float = 0.5,
        motion_loss_weight: float = 2.0,
        **kwargs,
    ) -> None:
        resume = kwargs.pop("resume_from_checkpoint", None)
        model_paths, tokenizer_path = local_model_paths(Path(base_model_path))
        kwargs["model_paths"] = model_paths
        kwargs["model_id_with_origin_paths"] = None
        kwargs["tokenizer_path"] = tokenizer_path
        extra = [item for item in (kwargs.get("extra_inputs") or "").split(",") if item]
        for key in (
            "input_image", "motion_field_control", "wrong_motion_field_control",
            "motion_mask", "ranking_valid",
        ):
            if key not in extra:
                extra.append(key)
        kwargs["extra_inputs"] = ",".join(extra)
        super().__init__(resume_from_checkpoint=None, **kwargs)

        layers = tuple(int(value) for value in injection_layers.split(",") if value.strip())
        adapter = self.pipe.dit.enable_oracle_motion_field_conditioning(
            control_dim=control_dim,
            hidden_dim=control_hidden_dim,
            source_dim=16,
            injection_layers=layers,
        )
        # The base I2V prior remains bit-for-bit frozen; only the new branch is trained.
        self.pipe.dit.requires_grad_(False)
        reference = next(self.pipe.dit.parameters())
        adapter.to(device=reference.device, dtype=reference.dtype).requires_grad_(True)
        self.last_loss_metrics: dict[str, float] = {}
        self.task_to_loss["sft"] = lambda pipe, shared, posi, nega: paired_oracle_refiner_loss(
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
    adapter_params = []
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "oracle_motion_field_adapter" in name:
            adapter_params.append(parameter)
        else:
            unexpected.append(name)
    if not adapter_params:
        raise RuntimeError("No trainable oracle motion adapter parameters")
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:20]}")

    if accelerator.is_main_process:
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        config = {
            "format": "wan21_oracle_motion_refiner_v1",
            "scope": "train-only oracle; not valid competition inference",
            "base_model_path": str(Path(args.base_model_path).resolve()),
            "control_dim": 7,
            "control_hidden_dim": args.control_hidden_dim,
            "injection_layers": list(model.pipe.dit.oracle_motion_field_adapter.injection_layers),
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
        }
        (output / "control_config.json").write_text(json.dumps(config, indent=2))
        (output / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    accelerator.wait_for_everyone()

    optimizer = torch.optim.AdamW(
        adapter_params, lr=args.control_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
        persistent_workers=args.dataset_num_workers > 0,
    )
    model.to(accelerator.device)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()
    completed = 0
    history = []
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
                    accelerator.clip_grad_norm_(adapter_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed += 1
                logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                unwrapped = accelerator.unwrap_model(model)
                values = torch.tensor(
                    [
                        float(loss.detach()),
                        unwrapped.last_loss_metrics.get("reconstruction", 0.0),
                        unwrapped.last_loss_metrics.get("ranking", 0.0),
                        unwrapped.last_loss_metrics.get("correct_minus_wrong", 0.0),
                        unwrapped.last_loss_metrics.get("motion_fraction", 0.0),
                    ],
                    device=accelerator.device,
                )
                values = accelerator.gather(values).reshape(accelerator.num_processes, -1).mean(0)
                row = {
                    "step": completed,
                    "loss": float(values[0]),
                    "reconstruction": float(values[1]),
                    "ranking": float(values[2]),
                    "correct_minus_wrong": float(values[3]),
                    "motion_fraction": float(values[4]),
                }
                if accelerator.is_main_process:
                    history.append(row)
                progress.update(1)
                progress.set_postfix(loss=f"{row['loss']:.5f}", delta=f"{row['correct_minus_wrong']:+.4f}")
                if completed >= args.max_steps:
                    break
    progress.close()
    logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - started
    if accelerator.is_main_process:
        adapter = accelerator.unwrap_model(model).pipe.dit.oracle_motion_field_adapter
        report = {
            "optimizer_steps": completed,
            "seconds": elapsed,
            "seconds_per_step": elapsed / max(completed, 1),
            "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
            "world_size": accelerator.num_processes,
            "train_records": len(dataset.records),
            "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
            "injection_layers": list(adapter.injection_layers),
            "base_model_frozen": True,
            "last_loss_metrics": history[-1] if history else {},
        }
        Path(args.output_path, "train_benchmark.json").write_text(json.dumps(report, indent=2))
        Path(args.output_path, "training_metrics.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(report, indent=2))


def main() -> None:
    parser = wan_parser()
    parser.add_argument(
        "--base_model_path", default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P")
    )
    parser.add_argument(
        "--oracle_manifest",
        default=str(REPO / "diagnostics/oracle_motion_field_gate/manifest.json"),
    )
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--control_hidden_dim", type=int, default=192)
    parser.add_argument("--injection_layers", default="0,10,20,30")
    parser.add_argument("--control_learning_rate", type=float, default=1e-4)
    parser.add_argument("--control_rank_margin", type=float, default=0.002)
    parser.add_argument("--control_rank_weight", type=float, default=0.5)
    parser.add_argument("--motion_loss_weight", type=float, default=2.0)
    parser.add_argument("--motion_mask_threshold", type=float, default=0.04)
    parser.add_argument("--motion_mask_dilation", type=int, default=7)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--prompt", default="A fixed-camera video of a tabletop robot arm manipulating objects."
    )
    args = parser.parse_args()
    if args.task != "sft" or args.num_frames != 17:
        raise ValueError("Oracle refiner supports only --task sft --num_frames 17")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    if args.enable_model_cpu_offload:
        raise ValueError("DDP oracle training does not support model CPU offload")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    dataset = WanOracleMotionFieldDataset(
        manifest=args.oracle_manifest,
        data_root=args.dataset_base_path,
        split="train",
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        prompt=args.prompt,
        motion_mask_threshold=args.motion_mask_threshold,
        motion_mask_dilation=args.motion_mask_dilation,
    )
    if accelerator.is_main_process:
        print(f"[Wan oracle refiner] records={len(dataset.records)} repeat={dataset.repeat}")
    model = WanOracleMotionRefinerTrainingModule(
        base_model_path=args.base_model_path,
        control_hidden_dim=args.control_hidden_dim,
        injection_layers=args.injection_layers,
        control_rank_margin=args.control_rank_margin,
        control_rank_weight=args.control_rank_weight,
        motion_loss_weight=args.motion_loss_weight,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=0,
        lora_checkpoint=None,
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
