"""Fine-tune the released IRASim Frame-Ada model on SO-100 clips."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import accelerate
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
IRASIM_SRC = REPO / "third_party/IRASim"
if str(IRASIM_SRC) not in sys.path:
    sys.path.insert(0, str(IRASIM_SRC))

from diffusers.models import AutoencoderKL  # noqa: E402
from diffusion import create_mask_diffusion  # noqa: E402
from irasim_so100 import IRASimSO100Dataset, build_irasim, load_irasim_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True)
    p.add_argument("--vae", default=str(REPO / "models/IRASim/sdxl-base"))
    p.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    p.add_argument("--output", default=str(REPO / "open/baseline/outputs/irasim_action_500"))
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dataset-repeat", type=int, default=100)
    p.add_argument("--holdout-count", type=int, default=6)
    p.add_argument(
        "--action-mode",
        choices=("absolute", "delta", "delta_step"),
        default="absolute",
        help="Relative modes use train observation.state as source state and are not deployable "
        "until a source-image state estimator is supplied at inference.",
    )
    # Official RT-1 Frame-Ada trains the full model with one constant 1e-4 LR.
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--mixed-precision", choices=("no", "bf16"), default="no")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def encode_video(vae, video: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    batch, frames = video.shape[:2]
    flat = video.flatten(0, 1)
    encoded = []
    with torch.no_grad():
        for start in range(0, len(flat), chunk):
            posterior = vae.encode(flat[start:start + chunk]).latent_dist
            encoded.append(posterior.sample() * vae.config.scaling_factor)
    return torch.cat(encoded).unflatten(0, (batch, frames))


@torch.no_grad()
def update_ema(ema_model, model, decay: float) -> None:
    """IRASim's official parameter EMA, including the SO-100 adapter."""
    source = dict(model.named_parameters())
    for name, parameter in ema_model.named_parameters():
        parameter.mul_(decay).add_(source[name].detach(), alpha=1.0 - decay)


def main() -> None:
    args = parse_args()
    accelerator = accelerate.Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    torch.manual_seed(args.seed + accelerator.process_index)

    dataset = IRASimSO100Dataset(
        root=args.dataset_root,
        repeat=args.dataset_repeat,
        holdout_count=args.holdout_count,
        seed=args.seed,
        action_mode=args.action_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    model = build_irasim(action_dim=dataset.action_dim, num_frames=16)
    report = load_irasim_checkpoint(model, args.pretrained)
    if report["format"] != "public_rt1" or report["mismatched"] or report["unexpected"]:
        raise RuntimeError(f"Expected an exact public RT-1 backbone: {report}")
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    ema_model.to(accelerator.device)
    vae_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(
        accelerator.device, dtype=vae_dtype
    )
    vae.requires_grad_(False).eval()
    diffusion = create_mask_diffusion(timestep_respacing="", learn_sigma=False)

    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        print(f"[IRASim] datasets={len(dataset.selected_paths)}, clips={len(dataset.base)}")
        print(f"[IRASim] public backbone exact tensors={report['matched']}")
        print("[IRASim] only new tensor=action_adapter.weight (zero initialized)")
        print(f"[IRASim] parameters={sum(p.numel() for p in model.parameters()):,}")

    start_time = time.perf_counter()
    peak = 0.0
    step = 0
    model.train()
    while step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                video = batch["video"].to(accelerator.device, dtype=vae_dtype)
                actions = batch["actions"].to(accelerator.device, dtype=vae_dtype)
                latents = encode_video(vae, video)
                timesteps = torch.randint(
                    0, diffusion.num_timesteps, (latents.shape[0],), device=latents.device
                )
                with accelerator.autocast():
                    losses = diffusion.training_losses(
                        model, latents, timesteps,
                        {"actions": actions, "mask_frame_num": 1},
                    )
                    loss = losses["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                update_ema(
                    ema_model,
                    accelerator.unwrap_model(model),
                    decay=args.ema_decay,
                )
                step += 1
                if torch.cuda.is_available():
                    peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                if accelerator.is_main_process and (step == 1 or step % 20 == 0):
                    print(f"step={step}/{args.max_steps} loss={loss.item():.6f}", flush=True)
                if step % args.save_every == 0 or step == args.max_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        state = accelerator.get_state_dict(model)
                        ema_state = {
                            key: value.detach().cpu()
                            for key, value in ema_model.state_dict().items()
                        }
                        torch.save(
                            {"model": state, "ema": ema_state, "args": vars(args)},
                            output / f"step-{step}.pt",
                        )
                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        seconds = time.perf_counter() - start_time
        benchmark = {
            "optimizer_steps": step,
            "seconds": seconds,
            "seconds_per_step": seconds / step,
            "peak_gpu_gib_rank0": peak,
            "world_size": accelerator.num_processes,
            "action_mode": args.action_mode,
            "public_backbone_matched_tensors": report["matched"],
            "new_tensors": ["action_adapter.weight"],
            "ema_decay": args.ema_decay,
            "mixed_precision": args.mixed_precision,
        }
        (output / "train_benchmark.json").write_text(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
