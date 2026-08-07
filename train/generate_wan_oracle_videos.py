"""Generate train-only holdout videos for the Wan oracle-track control gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "models"))
os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("HF_HOME", str(REPO / "models/.hf_cache"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402
from wan_oracle_track_dataset import resolve_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--valset", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--ablation", choices=("none", "zero", "batch-roll"), default="none")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track-hidden-dim", type=int, default=128)
    parser.add_argument("--structural-zero", action="store_true")
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json")
    return parser.parse_args()


def load_checkpoint(
    pipe: WanVideoPipeline,
    checkpoint: str,
    hidden_dim: int,
    structural_zero: bool,
    adapter_only: bool,
) -> None:
    state = load_state_dict(checkpoint)
    adapter = pipe.dit.enable_track_conditioning(
        input_dim=3, hidden_dim=hidden_dim, structural_zero=structural_zero
    )
    adapter_state = {
        key.removeprefix("track_adapter."): value
        for key, value in state.items()
        if key.startswith("track_adapter.")
    }
    if not adapter_state:
        raise ValueError(f"No track_adapter weights in {checkpoint}")
    adapter.load_state_dict(adapter_state, strict=True)
    reference = next(pipe.dit.parameters())
    adapter.to(device=reference.device, dtype=reference.dtype).eval()
    lora_state = {key: value for key, value in state.items() if "lora_" in key}
    if not lora_state and not adapter_only:
        raise ValueError(f"No LoRA weights in {checkpoint}")
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=1.0)
    print(f"[checkpoint] track adapter={len(adapter_state)} tensors, LoRA={len(lora_state)} tensors")


def main() -> None:
    args = parse_args()
    output = Path(args.prediction_root)
    output.mkdir(parents=True, exist_ok=True)
    records = [row for row in json.loads(Path(args.manifest).read_text()) if row["split"] == "holdout"]
    records = records[: args.limit]
    if not records:
        raise ValueError("No holdout records in oracle manifest")
    valset = Path(args.valset)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
        ),
        redirect_common_files=False,
    )
    load_checkpoint(
        pipe,
        args.checkpoint,
        args.track_hidden_dim,
        structural_zero=args.structural_zero,
        adapter_only=args.adapter_only,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter_effects = []
    for index, record in enumerate(records):
        target = output / f"{record['sample_id']}.mp4"
        if target.exists() and not args.overwrite:
            continue
        image = ImageOps.pad(
            Image.open(valset / "images" / f"{record['sample_id']}.png").convert("RGB"),
            (args.width, args.height),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0),
        )
        control_record = record
        if args.ablation == "batch-roll":
            control_record = records[(index + 1) % len(records)]
        with np.load(resolve_artifact(control_record["artifact"])) as artifact:
            control = torch.from_numpy(artifact["control"].astype(np.float32))
        if args.ablation == "zero":
            control.zero_()
        with torch.no_grad():
            reference = next(pipe.dit.parameters())
            effect = pipe.dit.track_adapter(
                control.unsqueeze(0).to(device=reference.device, dtype=reference.dtype),
                (5, args.height // 16, args.width // 16),
            )
            adapter_effects.append(float(effect.float().abs().mean().item()))
        frames = pipe(
            prompt="A fixed-camera video of a robot arm manipulating objects.",
            negative_prompt="",
            input_image=image,
            track_control=control,
            seed=args.seed + index,
            height=args.height,
            width=args.width,
            num_frames=17,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            tiled=True,
        )
        if len(frames) != 17:
            raise RuntimeError(f"Wan returned {len(frames)} frames, expected 17")
        save_video(frames[:16], str(target), fps=6, quality=5)
        print(f"[oracle generate] {index + 1}/{len(records)}")

    elapsed = time.perf_counter() - started
    report = {
        "samples": len(records),
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(len(records), 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "ablation": args.ablation,
        "adapter_output_abs_mean": float(np.mean(adapter_effects)) if adapter_effects else 0.0,
        "structural_zero": args.structural_zero,
        "adapter_only": args.adapter_only,
    }
    print(json.dumps(report, indent=2))
    if args.benchmark_json:
        path = Path(args.benchmark_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
