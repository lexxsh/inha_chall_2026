"""CPU-only audit for the faithful BWM EEF14 transfer contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from bwm_so100_eef import (  # noqa: E402
    BWMSO100EEFDataset,
    CANONICAL_Q,
    DEFAULT_STATS,
    SO100EEFConverter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS))
    parser.add_argument(
        "--bwm-checkpoint",
        default=str(REPO / "checkpoints/Boundless-World-Model/step-12000.safetensors"),
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", default=str(REPO / "results/bwm_so100_eef14_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = json.loads(Path(args.stats_path).read_text())
    converter = SO100EEFConverter(args.stats_path)
    challenge = Path(args.challenge_root)
    ids = sorted(path.stem for path in (challenge / "actions").glob("*.npy"))[: args.limit]
    if not ids:
        raise FileNotFoundError(f"No challenge actions under {challenge}")

    normals = []
    zeros = []
    rolls = []
    raw = {}
    for sid in ids:
        action = np.load(challenge / "actions" / f"{sid}.npy").astype(np.float32)
        if action.shape != (16, 6):
            raise ValueError(f"{sid}: expected [16,6], got {action.shape}")
        raw[sid] = action
    for index, sid in enumerate(ids):
        action = raw[sid]
        rolled = raw[ids[(index + 1) % len(ids)]]
        normals.append(converter.build_tokens(action, source_action=action[0]).numpy())
        stationary = np.repeat(action[:1], 16, axis=0)
        zeros.append(converter.build_tokens(stationary, source_action=stationary[0]).numpy())
        rolls.append(converter.build_tokens(rolled, source_action=rolled[0]).numpy())
    normal = np.stack(normals)
    zero = np.stack(zeros)
    roll = np.stack(rolls)

    public_shapes = {}
    with safe_open(args.bwm_checkpoint, framework="pt", device="cpu") as handle:
        for name in (
            "pipe.action_encoder.action_mlp1.0.weight",
            "pipe.action_encoder.action_mlp2.0.weight",
        ):
            public_shapes[name] = list(handle.get_slice(name).get_shape())

    # Decode one real train clip to include the complete data boundary in the audit.
    dataset = BWMSO100EEFDataset(
        root=args.dataset_root,
        height=64,
        width=96,
        holdout_count=8,
        seed=42,
        split="train",
        stats_path=args.stats_path,
    )
    sample = dataset[0]
    violations = stats["joint_limit_violation_fraction"]
    criteria = {
        "source_anchored_stats": stats.get("contract", "").startswith("source-anchored"),
        "original_bwm_input_shapes": public_shapes
        == {
            "pipe.action_encoder.action_mlp1.0.weight": [3072, 14],
            "pipe.action_encoder.action_mlp2.0.weight": [12288, 56],
        },
        "dataset_contract": list(sample["video"].shape) == [1, 3, 17, 64, 96]
        and list(sample["action"].shape) == [1, 17, 14],
        "finite_and_bounded": bool(np.isfinite(normal).all() and np.abs(normal).max() <= 1.0),
        "inactive_arm_is_zero": bool(np.abs(normal[..., 7:]).max() == 0.0),
        "canonical_source_is_stable": bool(np.abs(normal[:, 0] - normal[:1, 0]).max() < 1e-6),
        "normal_differs_from_zero": bool(np.mean(np.abs(normal - zero)) > 0.01),
        "normal_differs_from_roll": bool(np.mean(np.abs(normal - roll)) > 0.01),
        "preclip_limit_violations_below_10pct": bool(max(violations.values()) < 0.10),
    }
    report = {
        "scope": "CPU contract audit; no generator or GPU executed",
        "contract": stats["contract"],
        "canonical_q": CANONICAL_Q.tolist(),
        "stats_count": stats["count"],
        "sampled_windows": stats["sampled_windows"],
        "joint_limit_violation_fraction_before_clip": violations,
        "public_checkpoint_input_shapes": public_shapes,
        "dataset_video_shape": list(sample["video"].shape),
        "dataset_action_shape": list(sample["action"].shape),
        "token_abs_max": float(np.abs(normal).max()),
        "inactive_arm_abs_max": float(np.abs(normal[..., 7:]).max()),
        "normal_minus_zero_abs_mean": float(np.mean(np.abs(normal - zero))),
        "normal_minus_batch_roll_abs_mean": float(np.mean(np.abs(normal - roll))),
        "normal_temporal_abs_mean": float(np.mean(np.abs(np.diff(normal, axis=1)))),
        "criteria": criteria,
        "verdict": "PASS_EEF14_CONTRACT" if all(criteria.values()) else "REJECT_EEF14_CONTRACT",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
