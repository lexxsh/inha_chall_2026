"""Generate the train-only Wan2.1 oracle motion-field refiner gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402


PROMPT = "A fixed-camera video of a tabletop robot arm manipulating objects."
NEGATIVE = (
    "camera motion, camera shake, zoom, pan, tilt, scene change, blurry, low quality, "
    "deformed robot, warped robot, extra robot arm, melting, morphing"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-model-path", default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P")
    )
    parser.add_argument(
        "--oracle-root", default=str(REPO / "diagnostics/oracle_motion_field_gate")
    )
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE)
    parser.add_argument("--ablation", choices=("none", "zero", "batch-roll", "all"), default="all")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--video-quality", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-json")
    return parser.parse_args()


def local_model_configs(model_root: Path) -> tuple[list[ModelConfig], ModelConfig]:
    model_root = model_root.resolve()
    shards = sorted(str(path) for path in model_root.glob("diffusion_pytorch_model-*.safetensors"))
    paths: list[object] = [
        shards,
        str(model_root / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(model_root / "Wan2.1_VAE.pth"),
        str(model_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]
    tokenizer = model_root / "google/umt5-xxl"
    missing = []
    if not shards:
        missing.append(str(model_root / "diffusion_pytorch_model-*.safetensors"))
    missing.extend(path for path in paths[1:] if not Path(path).exists())
    if not tokenizer.is_dir():
        missing.append(str(tokenizer))
    if missing:
        raise FileNotFoundError("Missing local Wan2.1 files:\n  " + "\n  ".join(missing))
    return [ModelConfig(path=path) for path in paths], ModelConfig(path=str(tokenizer))


def checkpoint_config(checkpoint: Path) -> dict:
    path = checkpoint.resolve().parent / "control_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_adapter(pipe: WanVideoPipeline, checkpoint: Path) -> dict:
    state = load_state_dict(str(checkpoint))
    prefix = "oracle_motion_field_adapter."
    adapter_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    if not adapter_state:
        raise ValueError(f"{checkpoint} has no {prefix} tensors")
    hidden = int(adapter_state["control_encoder.0.weight"].shape[0])
    control_dim = int(adapter_state["control_encoder.0.weight"].shape[1])
    output_indices = sorted(
        {int(key.split(".")[1]) for key in adapter_state if key.startswith("outputs.")}
    )
    config = checkpoint_config(checkpoint)
    layers = tuple(config.get("injection_layers", (0, 10, 20, 30)))
    if len(layers) != len(output_indices):
        raise ValueError(
            f"Checkpoint has {len(output_indices)} outputs but config declares layers {layers}"
        )
    adapter = pipe.dit.enable_oracle_motion_field_conditioning(
        control_dim=control_dim,
        hidden_dim=hidden,
        source_dim=16,
        injection_layers=layers,
    )
    adapter.load_state_dict(adapter_state, strict=True)
    reference = next(pipe.dit.parameters())
    adapter.to(device=reference.device, dtype=reference.dtype).eval()
    print(
        f"[checkpoint] oracle adapter={sum(p.numel() for p in adapter.parameters()):,} "
        f"params control={control_dim}D hidden={hidden} layers={layers}"
    )
    return {"control_dim": control_dim, "hidden_dim": hidden, "injection_layers": layers}


def load_records(root: Path, limit: int) -> list[dict]:
    records = [
        record for record in json.loads((root / "manifest.json").read_text())
        if record["split"] == "holdout"
    ]
    records.sort(key=lambda record: record["sample_id"])
    return records[:limit] if limit else records


def load_control(record: dict) -> torch.Tensor:
    path = Path(record["motion_artifact"])
    if not path.is_absolute():
        path = REPO / path
    with np.load(path) as artifact:
        return torch.from_numpy(artifact["control"].astype(np.float32))


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    root = Path(args.oracle_root).resolve()
    valset = root / "valset"
    records = load_records(root, args.limit)
    if not records:
        raise ValueError(f"No oracle holdout records in {root}")
    output = Path(args.prediction_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assigned = records[rank::world_size]
    if not assigned:
        print(f"[oracle generate] rank {rank}/{world_size}: no samples")
        return
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    model_configs, tokenizer = local_model_configs(Path(args.base_model_path))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer,
        redirect_common_files=False,
    )
    adapter_info = load_adapter(pipe, Path(args.checkpoint).resolve())
    controls = {record["sample_id"]: load_control(record) for record in records}
    modes = ("none", "zero", "batch-roll") if args.ablation == "all" else (args.ablation,)
    started = time.perf_counter()
    total = 0
    reports = {}
    torch.cuda.reset_peak_memory_stats(local_rank)
    for mode in modes:
        mode_output = output / mode if args.ablation == "all" else output
        mode_output.mkdir(parents=True, exist_ok=True)
        mode_started = time.perf_counter()
        count = 0
        for record in assigned:
            sid = record["sample_id"]
            target = mode_output / f"{sid}.mp4"
            if target.exists() and not args.overwrite:
                count += 1
                total += 1
                continue
            control = controls[sid]
            if mode == "zero":
                control = torch.zeros_like(control)
            elif mode == "batch-roll":
                index = next(i for i, item in enumerate(records) if item["sample_id"] == sid)
                control = controls[records[(index + 1) % len(records)]["sample_id"]]
            source = Image.open(valset / "images" / f"{sid}.png").convert("RGB")
            source = source.resize((args.width, args.height), Image.Resampling.LANCZOS)
            frames = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                input_image=source,
                motion_field_control=control,
                seed=args.seed + next(i for i, item in enumerate(records) if item["sample_id"] == sid),
                height=args.height,
                width=args.width,
                num_frames=17,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                sigma_shift=args.sigma_shift,
                tiled=True,
            )
            if len(frames) != 17:
                raise RuntimeError(f"Wan returned {len(frames)} frames for {sid}")
            submission = [source] + [frame.convert("RGB") for frame in frames[1:16]]
            save_video(submission, str(target), fps=args.fps, quality=args.video_quality)
            count += 1
            total += 1
            print(
                f"[oracle {mode}] {count}/{len(assigned)} "
                f"{(time.perf_counter() - mode_started) / count:.1f}s/sample"
            )
        reports[mode] = {
            "samples": count,
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(mode_output),
        }
    elapsed = time.perf_counter() - started
    report = {
        "scope": "train-only oracle; invalid for competition submission",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "rank": rank,
        "world_size": world_size,
        "samples": total,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(total, 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        **adapter_info,
        "ablation": args.ablation,
        "modes": reports,
        "num_inference_steps": args.num_inference_steps,
        "source_frame_zero_replaced": True,
    }
    print(json.dumps(report, indent=2))
    report_path = Path(args.benchmark_json) if args.benchmark_json else output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(f"{report_path.stem}.rank{rank}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
