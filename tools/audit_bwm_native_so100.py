"""CPU-only structural/data audit for native BWM SO100 post-training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
for dependency in (
    REPO / "train",
    REPO / "third_party/boundless-world-model",
    REPO / "third_party/DiffSynth-Studio",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from bwm_native_so100 import (  # noqa: E402
    ACTION_DIM,
    BWMNativeSO100Dataset,
    build_action_tokens,
    local_wan22_model_paths,
)
from wan_video_action.models.wan_video_action_encoder import WanVideoActionEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B"))
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--output", default=str(REPO / "results/bwm_native_so100_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards, vae = local_wan22_model_paths(args.model_root)
    future = torch.linspace(-1.0, 1.0, 16 * ACTION_DIM).reshape(16, ACTION_DIM)
    tokens = build_action_tokens(future)
    zero = build_action_tokens(future[:1].expand_as(future))
    roll = build_action_tokens(future.flip(0))

    # Small dimensions exercise the exact official encode_ti2v2 grouping on CPU.
    encoder = WanVideoActionEncoder(action_dim=ACTION_DIM, dim=32, num_action_per_chunk=17)
    context, modulation = encoder.encode_ti2v2(tokens[None])
    grouped = torch.cat([tokens[:1].repeat(3, 1), tokens], dim=0).reshape(5, 24)
    grouping_gate = (
        context.shape == (1, 17, 32)
        and modulation.shape == (1, 5, 32)
        and torch.equal(grouped[0].reshape(4, 6), future[:1].repeat(4, 1))
        and torch.equal(grouped[1].reshape(4, 6), future[:4])
        and torch.equal(grouped[-1].reshape(4, 6), future[12:16])
    )

    dataset = BWMNativeSO100Dataset(
        root=args.dataset_root,
        height=args.height,
        width=args.width,
        repeat=1,
        holdout_count=8,
        seed=42,
        split="train",
    )
    sample = dataset[0]
    data_gate = (
        sample["video"].shape == (1, 3, 17, args.height, args.width)
        and sample["action"].shape == (1, 17, 6)
        and torch.isfinite(sample["video"]).all()
        and torch.isfinite(sample["action"]).all()
        and float(sample["action"].abs().max()) <= 1.0
    )
    action_gate = (
        torch.equal(tokens[0], tokens[1])
        and not torch.allclose(tokens, zero)
        and not torch.allclose(tokens, roll)
    )
    assets_gate = len(shards) == 3 and Path(vae).is_file()
    criteria = {
        "vanilla_wan22_assets": assets_gate,
        "dataset_contract": bool(data_gate),
        "action_contract": bool(action_gate),
        "exact_17rgb_to_5latent_grouping": bool(grouping_gate),
        "public_bwm_checkpoint_not_required": True,
    }
    report = {
        "scope": "CPU-only; no GPU/model weights loaded",
        "initialization": "vanilla Wan2.2-TI2V-5B + fresh 6D action encoder",
        "base_model_shards": shards,
        "vae": vae,
        "train_datasets": len(dataset.selected_paths),
        "train_clips": len(dataset.base),
        "sample_video_shape": list(sample["video"].shape),
        "sample_action_shape": list(sample["action"].shape),
        "action_context_shape": list(context.shape),
        "action_modulation_shape": list(modulation.shape),
        "normal_minus_zero_abs_mean": float((tokens - zero).abs().mean()),
        "normal_minus_reverse_abs_mean": float((tokens - roll).abs().mean()),
        "criteria": criteria,
        "verdict": "PASS_BWM_NATIVE_SO100" if all(criteria.values()) else "REJECT_BWM_NATIVE_SO100",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {output.resolve()}")
    if report["verdict"].startswith("REJECT"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
