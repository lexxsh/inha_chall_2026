#!/usr/bin/env python3
"""Render a SO100 visual-action prompt from challenge-format actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from so100_renderer import Camera, JOINT_NAMES, SO100Model  # noqa: E402


DEFAULT_URDF = REPO / "third_party/ManiSkill/mani_skill/assets/robots/so100/so100.urdf"


def legacy_action_to_q(actions: np.ndarray) -> np.ndarray:
    """Initial SO100-v2 heuristic; alignment fitting must refine q[0].

    The first five recorded dimensions are degree-calibrated and the gripper is
    linear.  The constants describe the common legacy neutral convention.  The
    important invariant for later fitting is that temporal movement is computed
    from action deltas, so per-robot servo offsets cancel.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"actions must have shape (T,6), got {actions.shape}")
    q = np.empty_like(actions)
    q[:, 0] = np.deg2rad(-actions[:, 0])
    q[:, 1] = np.deg2rad(actions[:, 1] - 180.0)
    q[:, 2] = np.deg2rad(180.0 - actions[:, 2])
    q[:, 3] = np.deg2rad(actions[:, 3])
    q[:, 4] = np.deg2rad(actions[:, 4])
    q[:, 5] = -1.1 + 2.2 * actions[:, 5] / 100.0
    return q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, help="Challenge-format (T,6) .npy")
    parser.add_argument("--output", type=Path, default=REPO / "diagnostics/so100_prompt.mp4")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--eye", type=float, nargs=3, default=(0.45, 0.35, 0.35))
    parser.add_argument("--target", type=float, nargs=3, default=(0.0, 0.10, 0.12))
    parser.add_argument("--fov", type=float, default=52.0)
    parser.add_argument("--skeleton", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actions is None:
        actions = np.array(
            [
                [0, 180, 180, 38, 0, 0],
                [0, 150, 150, 20, 0, 25],
                [20, 120, 120, 0, 45, 50],
            ],
            dtype=np.float32,
        )
    else:
        actions = np.load(args.actions)
    q_trajectory = legacy_action_to_q(actions)
    model = SO100Model(args.urdf)
    camera = Camera(np.asarray(args.eye), np.asarray(args.target), args.fov)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output}")
    try:
        for q in q_trajectory:
            frame, _ = model.render(
                q, camera, width=args.width, height=args.height, skeleton=args.skeleton
            )
            writer.write(frame)
    finally:
        writer.release()
    limits = {
        joint.name: [joint.lower, joint.upper]
        for joint in model.joints
        if joint.name in JOINT_NAMES
    }
    violations = {
        name: int(((q_trajectory[:, index] < limits[name][0]) | (q_trajectory[:, index] > limits[name][1])).sum())
        for index, name in enumerate(JOINT_NAMES)
    }
    print(f"saved -> {args.output}")
    print(f"frames={len(q_trajectory)}, joint_limit_violations={violations}")


if __name__ == "__main__":
    main()
