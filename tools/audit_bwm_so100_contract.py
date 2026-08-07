"""CPU-only preflight for the BWM-5B/SO100 training contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1]
for dependency in (
    REPO / "train",
    REPO / "third_party/boundless-world-model",
    REPO / "third_party/DiffSynth-Studio",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from bwm_so100 import (  # noqa: E402
    BWM_ACTION_DIM,
    BWM_ACTION_FRAMES,
    BWMSO100Dataset,
    build_bwm_action_tokens,
    expand_bwm_action_encoder,
    load_delta_scale,
    local_wan22_model_paths,
)
from wan_video_action.models.wan_video_action_encoder import WanVideoActionEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument(
        "--model-root", default=str(REPO / "models/Wan-AI/Wan2.2-TI2V-5B")
    )
    parser.add_argument(
        "--bwm-checkpoint",
        default=str(REPO / "checkpoints/Boundless-World-Model/step-12000.safetensors"),
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument(
        "--output", default=str(REPO / "results/bwm_so100_contract_audit.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards, vae = local_wan22_model_paths(args.model_root)
    checkpoint = Path(args.bwm_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        action_shapes = {
            key: list(handle.get_slice(key).get_shape())
            for key in keys
            if key.startswith("pipe.action_encoder.")
        }
        block_ids = sorted(
            {
                int(key.split(".", 2)[1])
                for key in keys
                if key.startswith("blocks.") and key.split(".", 2)[1].isdigit()
            }
        )

    expected_action_shapes = {
        "pipe.action_encoder.action_mlp1.0.bias": [3072],
        "pipe.action_encoder.action_mlp1.0.weight": [3072, 14],
        "pipe.action_encoder.action_mlp1.2.bias": [3072],
        "pipe.action_encoder.action_mlp1.2.weight": [3072, 3072],
        "pipe.action_encoder.action_mlp2.0.bias": [12288],
        "pipe.action_encoder.action_mlp2.0.weight": [12288, 56],
        "pipe.action_encoder.action_mlp2.2.bias": [3072],
        "pipe.action_encoder.action_mlp2.2.weight": [3072, 12288],
    }

    scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), "hybrid")
    future = torch.linspace(-1.0, 1.0, 16 * 6).reshape(16, 6)
    normal = build_bwm_action_tokens(future, scale)
    zero = build_bwm_action_tokens(future[:1].expand_as(future), scale)
    rolled = build_bwm_action_tokens(future.flip(0), scale, source_action=future[0])

    # Exercise the 14D->18D replacement without allocating the real 5B model.
    tiny = WanVideoActionEncoder(action_dim=14, dim=16, num_action_per_chunk=4)
    downstream_before = tiny.action_mlp1[2].weight.detach().clone()
    expand_bwm_action_encoder(tiny, BWM_ACTION_DIM)
    expansion_gate = (
        tiny.action_mlp1[0].in_features == 18
        and tiny.action_mlp2[0].in_features == 72
        and torch.count_nonzero(tiny.action_mlp1[0].weight) == 0
        and torch.count_nonzero(tiny.action_mlp2[0].weight) == 0
        and torch.equal(downstream_before, tiny.action_mlp1[2].weight)
    )

    dataset = BWMSO100Dataset(
        root=args.dataset_root,
        height=args.height,
        width=args.width,
        repeat=1,
        holdout_count=args.holdout_count,
        seed=42,
        split="train",
    )
    sample = dataset[0]
    data_gate = (
        tuple(sample["video"].shape) == (1, 3, 17, args.height, args.width)
        and tuple(sample["action"].shape) == (1, 17, 18)
        and torch.isfinite(sample["video"]).all()
        and torch.isfinite(sample["action"]).all()
    )
    action_gate = (
        tuple(normal.shape) == (BWM_ACTION_FRAMES, BWM_ACTION_DIM)
        and torch.count_nonzero(normal[0, 6:]) == 0
        and torch.allclose(normal[0, :6], future[0].clamp(-3, 3) / 3)
        and not torch.allclose(normal, zero)
        and not torch.allclose(normal, rolled)
    )
    checkpoint_gate = action_shapes == expected_action_shapes and block_ids == list(range(30))
    assets_gate = len(shards) == 3 and Path(vae).is_file() and checkpoint.stat().st_size > 9_000_000_000
    verdict = all((assets_gate, checkpoint_gate, expansion_gate, action_gate, data_gate))

    report = {
        "bwm_checkpoint": str(checkpoint.resolve()),
        "bwm_checkpoint_bytes": checkpoint.stat().st_size,
        "bwm_checkpoint_tensors": len(keys),
        "bwm_dit_blocks": block_ids,
        "public_action_shapes": action_shapes,
        "base_model_shards": shards,
        "vae": vae,
        "train_datasets": len(dataset.selected_paths),
        "train_clips": len(dataset.base),
        "sample_video_shape": list(sample["video"].shape),
        "sample_action_shape": list(sample["action"].shape),
        "normal_minus_zero_abs_mean": float((normal - zero).abs().mean()),
        "normal_minus_roll_abs_mean": float((normal - rolled).abs().mean()),
        "action_abs_max": float(normal.abs().max()),
        "criteria": {
            "assets_gate": assets_gate,
            "public_checkpoint_gate": checkpoint_gate,
            "action_encoder_expansion_gate": bool(expansion_gate),
            "action_contract_gate": bool(action_gate),
            "dataset_gate": bool(data_gate),
        },
        "verdict": "PASS_BWM_SO100_CONTRACT" if verdict else "REJECT_BWM_SO100_CONTRACT",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {output.resolve()}")
    if not verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
