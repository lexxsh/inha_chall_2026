"""Reject incomplete FSDP exports before expensive BWM generation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        parameters = sum(math.prod(handle.get_slice(key).get_shape()) for key in keys)
    blocks = sorted(
        {
            int(key.split(".", 2)[1])
            for key in keys
            if key.startswith("blocks.") and key.split(".", 2)[1].isdigit()
        }
    )
    action_keys = [key for key in keys if key.startswith("pipe.action_encoder.action_mlp")]
    criteria = {
        "all_30_dit_blocks": blocks == list(range(30)),
        "native_action_encoder": len(action_keys) == 8,
        "at_least_4b_parameters": parameters >= 4_000_000_000,
    }
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "tensors": len(keys),
        "parameters": parameters,
        "dit_blocks": blocks,
        "action_tensors": len(action_keys),
        "criteria": criteria,
        "verdict": "PASS_BWM_NATIVE_CHECKPOINT" if all(criteria.values()) else "REJECT_INCOMPLETE_FSDP_EXPORT",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"saved -> {output.resolve()}")
    if report["verdict"].startswith("REJECT"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
