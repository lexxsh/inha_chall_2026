#!/usr/bin/env python3
"""Cheap checkpoint gate for the SO-100 joint adapter (no Cosmos model load)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from cosmos_so100.adapter import SO100JointAdapter
from cosmos_so100.dataset import CosmosSO100Dataset


PREFIX = "net.joint_adapter."


def load_adapter(checkpoint: Path) -> SO100JointAdapter:
    model_dir = checkpoint / "model" if (checkpoint / "model").is_dir() else checkpoint
    metadata = FileSystemReader(str(model_dir)).read_metadata().state_dict_metadata
    adapter_meta = {key: value for key, value in metadata.items() if key.startswith(PREFIX)}
    if not adapter_meta:
        raise RuntimeError(f"No {PREFIX} tensors in {model_dir}")

    flat_state = {
        key: torch.empty(tuple(value.size), dtype=torch.float32)
        for key, value in adapter_meta.items()
    }
    dcp.load(flat_state, checkpoint_id=str(model_dir))
    has_bias = PREFIX + "fc1.bias" in flat_state
    input_dim = flat_state[PREFIX + "fc1.weight"].shape[1]
    adapter = SO100JointAdapter(input_dim=input_dim, bias=has_bias)
    adapter.load_state_dict(
        {key.removeprefix(PREFIX): value for key, value in flat_state.items()},
        strict=True,
    )
    return adapter.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("open/data/train"))
    parser.add_argument(
        "--action-mode",
        choices=["absolute", "delta", "delta_step", "hybrid_step"],
        default="hybrid_step",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    adapter = load_adapter(args.checkpoint)
    dataset = CosmosSO100Dataset(
        root=str(args.data_root),
        split="holdout",
        height=256,
        width=320,
        num_frames=17,
        action_mode=args.action_mode,
        action_layout="so1006",
    )

    zero = torch.zeros(1, 16, adapter.fc1.in_features)
    with torch.no_grad():
        zero_output = adapter(zero)
        per_sample = []
        for index in range(min(args.num_samples, len(dataset))):
            action = dataset[index]["action"].unsqueeze(0)
            normal_output = adapter(action)
            reversed_output = adapter(torch.flip(action, dims=(1,)))
            per_sample.append(
                {
                    "action_abs_mean": action.abs().mean().item(),
                    "adapter_abs_mean": normal_output.abs().mean().item(),
                    "normal_minus_zero_abs_mean": (normal_output - zero_output).abs().mean().item(),
                    "normal_minus_reverse_abs_mean": (normal_output - reversed_output).abs().mean().item(),
                    "temporal_std": normal_output.std(dim=1).mean().item(),
                    "saturation_fraction": (normal_output.abs() > 0.475).float().mean().item(),
                }
            )

    def average(key: str) -> float:
        return sum(sample[key] for sample in per_sample) / len(per_sample)

    zero_abs_mean = zero_output.abs().mean().item()
    normal_minus_reverse = average("normal_minus_reverse_abs_mean")
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "bias": adapter.fc1.bias is not None,
        "parameter_count": sum(p.numel() for p in adapter.parameters()),
        "zero_output_abs_mean": zero_abs_mean,
        "adapter_output_abs_mean": average("adapter_abs_mean"),
        "normal_minus_zero_abs_mean": average("normal_minus_zero_abs_mean"),
        "normal_minus_reverse_abs_mean": normal_minus_reverse,
        "adapter_temporal_std": average("temporal_std"),
        "saturation_fraction": average("saturation_fraction"),
        "num_samples": len(per_sample),
        "structural_zero_gate": zero_abs_mean < 1e-8,
        "adapter_sensitivity_gate": normal_minus_reverse > 1e-3,
    }
    result["verdict"] = (
        "PASS_ADAPTER_ONLY"
        if result["structural_zero_gate"] and result["adapter_sensitivity_gate"]
        else "REJECT_ADAPTER"
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
