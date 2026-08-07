"""Generate competition-format 16-frame videos with Wan2.2-TI2V-5B."""
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
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HOME", str(REPO / "models/.hf_cache"))
os.environ.setdefault("HF_XET_CACHE", str(REPO / "models/.hf_cache/xet"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402
try:
    from data_module import action_dims_for, load_delta_scale, transform_actions  # type: ignore # noqa: E402
except ModuleNotFoundError:
    from train.data_module import action_dims_for, load_delta_scale, transform_actions  # noqa: E402


PROMPT = "A fixed-camera video of a robot arm manipulating objects."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    p.add_argument("--prediction-root", default=str(REPO / "diagnostics/wan_action"))
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=17)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--sigma-shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--action-mode", default="delta")
    p.add_argument("--action-shift", type=int, default=0)
    p.add_argument("--action-ablation", choices=("none", "zero", "reverse-time", "batch-roll"), default="none")
    p.add_argument("--action-hidden-dim", type=int, default=512)
    p.add_argument("--action-conditioner-version", choices=("v1", "v2", "xattn"), default="v1")
    p.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    p.add_argument("--vanilla", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--benchmark-json")
    return p.parse_args()


def sample_ids(root: Path) -> list[str]:
    prefix = "sample_" if (root / "images/sample_000000.png").exists() else "val_"
    return sorted(p.stem for p in (root / "images").glob(f"{prefix}*.png"))


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, method=Image.Resampling.BILINEAR, color=(0, 0, 0))


def prepare_action(raw: np.ndarray, mode: str, shift: int) -> torch.Tensor:
    stats = json.loads((REPO / "open/data/train/so100_action_statistics.json").read_text())
    action = torch.from_numpy(raw.astype(np.float32))
    action = (action - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])
    scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), mode)
    return transform_actions(action, scale, mode, shift)


def load_adapter(
    pipe: WanVideoPipeline, checkpoint: str, hidden_dim: int, action_dim: int, version: str
):
    state = load_state_dict(checkpoint)
    conditioner = pipe.dit.enable_action_conditioning(
        action_dim=action_dim, hidden_dim=hidden_dim, version=version
    )
    reference = next(pipe.dit.parameters())
    if version == "xattn":
        # xattn weights live both in the shared encoder and in per-block
        # projections/gates (blocks.N.action_*), all under the dit prefix.
        dit_keys = set(pipe.dit.state_dict().keys())
        xattn_keys = {
            key for key in dit_keys
            if key.startswith("action_token_encoder.")
            or ".action_q." in key or ".action_o." in key
            or ".action_norm" in key or key.endswith(".action_gate")
        }
        action_state = {key: state[key] for key in xattn_keys if key in state}
        provided = set(action_state)
        if provided != xattn_keys:
            raise ValueError(
                f"xattn checkpoint mismatch: missing={sorted(xattn_keys - provided)}"
            )
        pipe.dit.load_state_dict(action_state, strict=False)
        pipe.dit.action_token_encoder.to(
            device=reference.device, dtype=reference.dtype
        ).eval()
        lora_state = {key: value for key, value in state.items() if "lora_" in key}
        if not lora_state:
            raise ValueError(f"No LoRA weights in {checkpoint}")
        pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=1.0)
        print(f"[checkpoint] action(xattn)={len(action_state)} tensors, LoRA={len(lora_state)} tensors")
        return
    action_state = {
        key.removeprefix("action_conditioner."): value
        for key, value in state.items() if key.startswith("action_conditioner.")
    }
    if not action_state:
        raise ValueError(f"No action_conditioner weights in {checkpoint}")
    missing, unexpected = conditioner.load_state_dict(action_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Action checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    conditioner.to(device=reference.device, dtype=reference.dtype).eval()
    lora_state = {key: value for key, value in state.items() if "lora_" in key}
    if not lora_state:
        raise ValueError(f"No LoRA weights in {checkpoint}")
    pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=1.0)
    print(f"[checkpoint] action={len(action_state)} tensors, LoRA={len(lora_state)} tensors")


def main():
    args = parse_args()
    if args.num_frames != 17:
        raise ValueError("TI2V competition alignment requires 17 decode frames and saves frames 0..15.")
    root = Path(args.challenge_root)
    out = Path(args.prediction_root)
    out.mkdir(parents=True, exist_ok=True)

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
    if not args.vanilla:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --vanilla is set")
        load_adapter(
            pipe,
            args.checkpoint,
            args.action_hidden_dim,
            action_dims_for(args.action_mode),
            args.action_conditioner_version,
        )

    all_ids = sample_ids(root)
    # limit <= 0 means "all samples" (matches the submission convention).
    ids = all_ids if args.limit <= 0 else all_ids[: args.limit]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    done = 0
    for index, sid in enumerate(ids):
        path = out / f"{sid}.mp4"
        if path.exists() and not args.overwrite:
            done += 1
            continue
        image = letterbox(Image.open(root / "images" / f"{sid}.png"), (args.width, args.height))
        actions = None
        if not args.vanilla:
            action_sid = sid
            if args.action_ablation == "batch-roll":
                action_sid = all_ids[(all_ids.index(sid) + 1) % len(all_ids)]
            actions = prepare_action(
                np.load(root / "actions" / f"{action_sid}.npy"), args.action_mode, args.action_shift
            )
            if args.action_ablation == "zero":
                actions.zero_()
            elif args.action_ablation == "reverse-time":
                actions = actions.flip(0)

        frames = pipe(
            prompt=args.prompt,
            negative_prompt="",
            input_image=image,
            actions=actions,
            seed=args.seed + index,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            tiled=True,
        )
        if len(frames) != 17:
            raise RuntimeError(f"Wan returned {len(frames)} frames, expected 17")
        save_video(frames[:16], str(path), fps=6, quality=5)
        done += 1
        elapsed = time.perf_counter() - started
        print(f"[Wan generate] {done}/{len(ids)}, {elapsed / done:.1f}s/sample")

    elapsed = time.perf_counter() - started
    projected = elapsed / max(done, 1) * len(all_ids)
    report = {
        "samples": done,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(done, 1),
        "projected_total_samples": len(all_ids),
        "projected_seconds": projected,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "num_inference_steps": args.num_inference_steps,
        "frames_generated": 17,
        "frames_saved": 16,
    }
    print(json.dumps(report, indent=2))
    if args.benchmark_json:
        target = Path(args.benchmark_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
