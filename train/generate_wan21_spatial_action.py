"""Generate competition videos with Wan2.1 and the SO100 spatial action branch."""
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

os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
os.environ.setdefault("DIFFSYNTH_REDIRECT_COMMON_FILES", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402

try:
    from data_module import load_delta_scale, transform_actions  # type: ignore
except ModuleNotFoundError:
    from train.data_module import load_delta_scale, transform_actions


PROMPT = "A fixed-camera video of a tabletop robot arm manipulating objects."
NEGATIVE_PROMPT = (
    "camera motion, camera shake, zoom, pan, tilt, scene change, blurry, low quality, "
    "deformed robot, warped robot, extra robot arm, extra gripper, melting, morphing"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-model-path",
        default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P"),
    )
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument(
        "--action-mode",
        default=None,
        help="Defaults to the checkpoint's control_config.json (or hybrid for legacy files).",
    )
    parser.add_argument(
        "--action-ablation",
        choices=("none", "zero-motion", "reverse-time", "batch-roll", "all"),
        default="none",
    )
    parser.add_argument("--control-hidden-dim", type=int, default=0)
    parser.add_argument(
        "--injection-layers",
        default=None,
        help="Defaults to the checkpoint's control_config.json (or 0,10,20,30).",
    )
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


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(path.stem for path in (root / "images").glob(f"{prefix}*.png"))


def infer_adapter_shape(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    key = "spatial_action_adapter.action_encoder.1.weight"
    if key not in state:
        raise ValueError(f"Checkpoint has no {key}; it is not a spatial-action checkpoint")
    hidden_dim, feature_dim = state[key].shape
    if (feature_dim - 2) % 3:
        raise ValueError(f"Cannot infer action dimension from {key} shape {tuple(state[key].shape)}")
    action_dim = (feature_dim - 2) // 3
    return int(action_dim), int(hidden_dim)


def checkpoint_control_config(checkpoint: Path) -> dict:
    config_path = checkpoint.resolve().parent / "control_config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Malformed spatial-action config: {config_path}")
    return config


def load_control_checkpoint(
    pipe: WanVideoPipeline,
    checkpoint: str,
    hidden_override: int,
    injection_layers: tuple[int, ...],
) -> tuple[int, int]:
    state = load_state_dict(checkpoint)
    action_dim, hidden_dim = infer_adapter_shape(state)
    if hidden_override and hidden_override != hidden_dim:
        raise ValueError(
            f"--control-hidden-dim={hidden_override} conflicts with checkpoint hidden={hidden_dim}"
        )
    adapter = pipe.dit.enable_spatial_action_conditioning(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        source_dim=16,
        injection_layers=injection_layers,
    )
    adapter_state = {
        key.removeprefix("spatial_action_adapter."): value
        for key, value in state.items()
        if key.startswith("spatial_action_adapter.")
    }
    missing, unexpected = adapter.load_state_dict(adapter_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Adapter checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    reference = next(pipe.dit.parameters())
    adapter.to(device=reference.device, dtype=reference.dtype).eval()

    lora_state = {key: value for key, value in state.items() if "lora_" in key}
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=1.0)
    print(
        f"[checkpoint] spatial adapter={len(adapter_state)} tensors, "
        f"LoRA={len(lora_state)} tensors, action={action_dim}D, hidden={hidden_dim}"
    )
    return action_dim, hidden_dim


def normalized_raw_actions(root: Path, sid: str) -> torch.Tensor:
    stats = json.loads((REPO / "open/data/train/so100_action_statistics.json").read_text())
    raw = torch.from_numpy(np.load(root / "actions" / f"{sid}.npy").astype(np.float32))
    if raw.shape != (16, 6):
        raise ValueError(f"Expected {sid} actions [16,6], got {tuple(raw.shape)}")
    return (raw - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])


def prepare_actions(raw: torch.Tensor, mode: str) -> torch.Tensor:
    scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), mode)
    return transform_actions(raw, scale, mode, shift=0)


def main() -> None:
    args = parse_args()
    if args.num_frames != 17:
        raise ValueError("Competition generation requires --num-frames 17")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    if not 0 <= args.video_quality <= 10:
        raise ValueError("--video-quality must be in [0,10]")
    checkpoint = Path(args.checkpoint).resolve()
    saved_config = checkpoint_control_config(checkpoint)
    action_mode = args.action_mode or saved_config.get("action_mode", "hybrid")
    if (
        args.action_mode is not None
        and saved_config.get("action_mode") is not None
        and args.action_mode != saved_config["action_mode"]
    ):
        raise ValueError(
            f"Requested action mode {args.action_mode!r} conflicts with checkpoint config "
            f"{saved_config['action_mode']!r}"
        )
    saved_layers = saved_config.get("injection_layers", [0, 10, 20, 30])
    layer_text = args.injection_layers or ",".join(str(value) for value in saved_layers)
    root = Path(args.challenge_root).resolve()
    output = Path(args.prediction_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids_all = sample_ids(root)
    selected_ids = ids_all[: args.limit] if args.limit else ids_all
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid generation shard rank/world: {rank}/{world_size}")
    ids = selected_ids[rank::world_size]
    if args.action_ablation in ("batch-roll", "all") and len(selected_ids) < 2:
        raise ValueError("batch-roll requires at least two challenge samples")

    # torchrun is used only as an embarrassingly parallel sample launcher; the
    # 14B model is independently loaded once per GPU and samples never overlap.
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if not ids:
        print(f"[Wan21 spatial] rank {rank}/{world_size} has no assigned samples")
        return

    model_configs, tokenizer_config = local_model_configs(Path(args.base_model_path))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    layers = tuple(int(value) for value in layer_text.split(",") if value.strip())
    if args.injection_layers is not None and saved_config:
        expected_layers = tuple(int(value) for value in saved_layers)
        if layers != expected_layers:
            raise ValueError(
                f"Requested injection layers {layers} conflict with checkpoint config "
                f"{expected_layers}"
            )
    action_dim, hidden_dim = load_control_checkpoint(
        pipe, args.checkpoint, args.control_hidden_dim, layers
    )

    raw_by_id = {sid: normalized_raw_actions(root, sid) for sid in selected_ids}
    torch.cuda.reset_peak_memory_stats(local_rank)
    started = time.perf_counter()
    modes = (
        ("none", "zero-motion", "batch-roll")
        if args.action_ablation == "all"
        else (args.action_ablation,)
    )
    generated = 0
    mode_reports = {}
    for mode in modes:
        mode_started = time.perf_counter()
        mode_generated = 0
        mode_output = output / mode if args.action_ablation == "all" else output
        mode_output.mkdir(parents=True, exist_ok=True)
        for local_index, sid in enumerate(ids):
            global_index = selected_ids.index(sid)
            target = mode_output / f"{sid}.mp4"
            if target.exists() and not args.overwrite:
                mode_generated += 1
                generated += 1
                continue
            raw = raw_by_id[sid]
            if mode == "zero-motion":
                raw = raw[:1].expand_as(raw).clone()
            elif mode == "reverse-time":
                raw = raw.flip(0)
            elif mode == "batch-roll":
                source_index = (global_index + 1) % len(selected_ids)
                raw = raw_by_id[selected_ids[source_index]]
            actions = prepare_actions(raw, action_mode)
            if actions.shape[-1] != action_dim:
                raise ValueError(
                    f"action mode {action_mode!r} produced {actions.shape[-1]}D, "
                    f"checkpoint expects {action_dim}D"
                )

            source = Image.open(root / "images" / f"{sid}.png").convert("RGB")
            source = source.resize((args.width, args.height), Image.Resampling.LANCZOS)
            frames = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                input_image=source,
                actions=actions,
                seed=args.seed + global_index,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                sigma_shift=args.sigma_shift,
                tiled=True,
            )
            if len(frames) != 17:
                raise RuntimeError(f"Wan returned {len(frames)} frames for {sid}, expected 17")
            # Challenge frame zero is known exactly.  Keep generated frames 1..15.
            submission_frames = [source] + [frame.convert("RGB") for frame in frames[1:16]]
            save_video(
                submission_frames,
                str(target),
                fps=args.fps,
                quality=args.video_quality,
            )
            mode_generated += 1
            generated += 1
            print(
                f"[Wan21 spatial {mode}] {mode_generated}/{len(ids)}, "
                f"{(time.perf_counter() - mode_started) / mode_generated:.1f}s/sample"
            )
        mode_reports[mode] = {
            "samples": mode_generated,
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(mode_output),
        }

    elapsed = time.perf_counter() - started
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_model": str(Path(args.base_model_path).resolve()),
        "rank": rank,
        "world_size": world_size,
        "assigned_ids": ids,
        "samples": generated,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(generated, 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
        "action_mode": action_mode,
        "action_dim": action_dim,
        "control_hidden_dim": hidden_dim,
        "injection_layers": list(layers),
        "action_ablation": args.action_ablation,
        "modes": mode_reports,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "source_frame_zero_replaced": True,
        "video_quality": args.video_quality,
    }
    print(json.dumps(report, indent=2))
    report_path = Path(args.benchmark_json) if args.benchmark_json else output / "benchmark.json"
    if world_size > 1:
        report_path = report_path.with_name(
            f"{report_path.stem}.rank{rank}{report_path.suffix or '.json'}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
