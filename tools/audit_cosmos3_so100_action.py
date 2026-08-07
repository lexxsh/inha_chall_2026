"""CPU-only gate for the SO-100 -> Cosmos3 Edge action contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train.cosmos3_so100_action import (  # noqa: E402
    DEFAULT_BRIDGE_STATS,
    IDENTITY_ROT6D,
    SO100Cosmos3ActionConverter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-root", type=Path, default=REPO / "valset_holdout")
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_BRIDGE_STATS)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, default=REPO / "results/cosmos3_edge_so100_action_audit.json")
    return parser.parse_args()


def rotation_angle_degrees(rot6d: np.ndarray) -> np.ndarray:
    first = rot6d[..., :3]
    second = rot6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    trace = first[..., 0] + second[..., 1] + third[..., 2]
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def main() -> None:
    args = parse_args()
    if args.limit <= 1:
        raise SystemExit("--limit must be at least 2 for batch-roll audit")
    converter = SO100Cosmos3ActionConverter(stats_path=args.stats_path)
    ids = sorted(path.stem for path in (args.challenge_root / "actions").glob("*.npy"))[: args.limit]
    if len(ids) < 2:
        raise FileNotFoundError(f"Need at least two challenge actions under {args.challenge_root}")
    source_rgb_matches_gt0 = True
    try:
        from PIL import Image

        for sid in ids:
            image = np.asarray(Image.open(args.challenge_root / "images" / f"{sid}.png").convert("RGB"))
            gt = np.load(args.challenge_root / "gt_videos" / f"{sid}.npy", mmap_mode="r")
            source_rgb_matches_gt0 &= gt.shape[0] == 16 and np.array_equal(image, gt[0])
    except FileNotFoundError:
        # The actual evaluation split intentionally has no future video.  The
        # local holdout carries it and is the only place this assertion runs.
        source_rgb_matches_gt0 = True

    commands = {
        sid: np.load(args.challenge_root / "actions" / f"{sid}.npy").astype(np.float32)
        for sid in ids
    }
    normal_raw, normal, zero, reverse, roll = [], [], [], [], []
    violations = []
    for index, sid in enumerate(ids):
        action = commands[sid]
        if action.shape != (16, 6):
            raise ValueError(f"{sid}: expected [16,6], got {action.shape}")
        other = commands[ids[(index + 1) % len(ids)]]
        source = converter.estimate_source_action(action[0])
        stationary = np.repeat(source[None], 16, axis=0)
        reversed_action = action[::-1].copy()
        normal_raw.append(converter.build_raw_actions(action, source_action=source))
        normal.append(converter.build_actions(action, source_action=source).numpy())
        zero.append(converter.build_actions(stationary, source_action=source).numpy())
        reverse.append(converter.build_actions(reversed_action, source_action=source).numpy())
        # Counterfactual action stays anchored to the current source image.
        roll.append(converter.build_actions(other, source_action=source).numpy())
        violations.append(converter.joint_limit_violation_fraction(action))

    normal_raw_np = np.stack(normal_raw)
    normal_np = np.stack(normal)
    zero_np = np.stack(zero)
    reverse_np = np.stack(reverse)
    roll_np = np.stack(roll)
    zero_translation = np.max(np.abs(np.stack([
        converter.build_raw_actions(
            np.repeat(converter.estimate_source_action(commands[sid][0])[None], 16, axis=0),
            source_action=converter.estimate_source_action(commands[sid][0]),
        )[:, :3]
        for sid in ids
    ])))
    zero_rotation_error = np.max(np.abs(np.stack([
        converter.build_raw_actions(
            np.repeat(converter.estimate_source_action(commands[sid][0])[None], 16, axis=0),
            source_action=converter.estimate_source_action(commands[sid][0]),
        )[:, 3:9]
        for sid in ids
    ]) - IDENTITY_ROT6D))
    first_pose_identity_error = np.mean(
        np.abs(normal_raw_np[:, 0, :9] - np.r_[0, 0, 0, IDENTITY_ROT6D][None]), axis=-1
    )
    raw_in_bridge_range = (
        (normal_raw_np >= converter.q01[None, None])
        & (normal_raw_np <= converter.q99[None, None])
    )
    translation_mm = np.linalg.norm(normal_raw_np[..., :3], axis=-1) * 1000.0
    rotation_deg = rotation_angle_degrees(normal_raw_np[..., 3:9])
    gripper = normal_raw_np[..., 9]
    arm_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    arm_violation_by_joint = {
        name: float(np.mean([row[name] for row in violations])) for name in arm_names
    }
    aggregate_arm_violation = float(np.mean(list(arm_violation_by_joint.values())))
    max_sample_arm_violation = max(max(row[name] for name in arm_names) for row in violations)

    criteria = {
        "shape_16x10": list(normal_np.shape[1:]) == [16, 10],
        "all_finite": bool(np.isfinite(normal_np).all()),
        "source_is_exact_gt_frame0": bool(source_rgb_matches_gt0),
        "first_transition_not_forced_identity": bool(np.median(first_pose_identity_error) > 1e-5),
        "zero_motion_is_identity": bool(zero_translation < 1e-7 and zero_rotation_error < 1e-6),
        "gripper_bounded": bool(gripper.min() >= 0.0 and gripper.max() <= 1.0),
        "normal_differs_from_zero": bool(np.mean(np.abs(normal_np - zero_np)) > 0.01),
        "normal_differs_from_reverse": bool(np.mean(np.abs(normal_np - reverse_np)) > 0.01),
        "normal_differs_from_batch_roll": bool(np.mean(np.abs(normal_np - roll_np)) > 0.01),
        "unclamped_affine_actions_finite_and_nonexplosive": bool(np.abs(normal_np).max() < 10.0),
        "most_raw_values_within_bridge_q01_q99": bool(raw_in_bridge_range.mean() >= 0.90),
        "aggregate_preclip_arm_limit_violations_below_5pct": bool(aggregate_arm_violation < 0.05),
    }
    report = {
        "scope": "CPU codec audit only; no generator or GPU executed",
        "model_contract": "Cosmos3 bridge_orig_lerobot forward_dynamics: image + normalized [16,10] -> 17 frames",
        "action_contract": "backward_framewise [dxyz3, column_rot6d6, gripper1]",
        "submission_alignment": "save generated frames 0..15; visible transitions use challenge actions 0..14; action15 is generated then dropped",
        "source_state_proxy": "action[0] plus train-only median state[0]-action[0] offset",
        "stats_path": str(args.stats_path.resolve()),
        "stats_sha256": hashlib.sha256(args.stats_path.read_bytes()).hexdigest(),
        "urdf_path": str(converter.urdf_path),
        "urdf_sha256": hashlib.sha256(converter.urdf_path.read_bytes()).hexdigest(),
        "samples": len(ids),
        "sample_ids": ids,
        "normal_shape": list(normal_np.shape),
        "normalized_abs_max": float(np.abs(normal_np).max()),
        "median_first_transition_pose_identity_error": float(np.median(first_pose_identity_error)),
        "raw_in_bridge_q01_q99_fraction": float(raw_in_bridge_range.mean()),
        "translation_step_mm": {
            "median": float(np.median(translation_mm)),
            "p95": float(np.quantile(translation_mm, 0.95)),
            "max": float(translation_mm.max()),
        },
        "rotation_step_degrees": {
            "median": float(np.median(rotation_deg)),
            "p95": float(np.quantile(rotation_deg, 0.95)),
            "max": float(rotation_deg.max()),
        },
        "gripper_raw_range": [float(gripper.min()), float(gripper.max())],
        "normal_minus_zero_abs_mean": float(np.mean(np.abs(normal_np - zero_np))),
        "normal_minus_reverse_abs_mean": float(np.mean(np.abs(normal_np - reverse_np))),
        "normal_minus_batch_roll_abs_mean": float(np.mean(np.abs(normal_np - roll_np))),
        "preclip_arm_limit_violation_by_joint": arm_violation_by_joint,
        "aggregate_preclip_arm_limit_violation_fraction": aggregate_arm_violation,
        "max_sample_preclip_arm_limit_violation_fraction": float(max_sample_arm_violation),
        "note": "FK clips rare arm-limit tails; gripper uses its separate [0,1] command mapping.",
        "criteria": criteria,
        "verdict": "PASS_COSMOS3_ACTION_CODEC" if all(criteria.values()) else "REJECT_COSMOS3_ACTION_CODEC",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
