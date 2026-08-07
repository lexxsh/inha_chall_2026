#!/usr/bin/env python3
"""Oracle gate for SO100 action -> camera-aligned visual prompts.

The fit uses train-only RAFT tracks and therefore is not an inference method.
Its purpose is narrower: reject URDF/sign conventions that cannot explain real
pixel trajectories before downloading or running a large video generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from so100_renderer import JOINT_NAMES, SO100Model  # noqa: E402


URDF = REPO / "third_party/ManiSkill/mani_skill/assets/robots/so100/so100.urdf"
ARTIFACT_ROOT = REPO / "diagnostics/spatial_control_gate"
VALSET_ROOT = ARTIFACT_ROOT / "valset"

# Derivatives only.  Per-robot servo offsets are absorbed by fitted q0.
ACTION_TO_Q_SCALE = np.array(
    [-np.pi / 180, np.pi / 180, -np.pi / 180, np.pi / 180, np.pi / 180, 2.2 / 100],
    dtype=np.float64,
)


def euler_matrix(angles: np.ndarray) -> np.ndarray:
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rz @ ry @ rx


def q_trajectory(q0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return q0[None] + (actions - actions[:1]) * ACTION_TO_Q_SCALE[None]


def chain_points(model: SO100Model, q: np.ndarray, samples_per_link: int = 5) -> np.ndarray:
    positions = model.joint_positions(q)
    anchors = np.asarray(
        [positions[model.root_link]] + [positions[name] for name in JOINT_NAMES], dtype=np.float64
    )
    points = []
    for start, end in zip(anchors[:-1], anchors[1:]):
        for fraction in np.linspace(0.0, 1.0, samples_per_link, endpoint=False):
            points.append(start * (1.0 - fraction) + end * fraction)
    points.append(anchors[-1])
    return np.asarray(points)


def project_trajectory(
    model: SO100Model, q_values: np.ndarray, params: np.ndarray
) -> np.ndarray:
    rotation = euler_matrix(params[6:9])
    scale = np.exp(params[9])
    translation = params[10:12]
    result = []
    for q in q_values:
        points = chain_points(model, q)
        rotated = (rotation @ points.T).T
        projected = rotated[:, :2] * scale + translation
        result.append(projected)
    return np.asarray(result)


def pair_costs(
    projected: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    time_mask: np.ndarray,
) -> np.ndarray:
    track_count = tracks.shape[1]
    point_count = projected.shape[1]
    costs = np.full((track_count, point_count), 1e3, dtype=np.float64)
    for track_index in range(track_count):
        valid = visible[:, track_index].astype(bool) & time_mask
        if valid.sum() < 3:
            continue
        observed = tracks[valid, track_index]
        observed_delta = observed - observed[:1]
        for point_index in range(point_count):
            predicted = projected[valid, point_index]
            predicted_delta = predicted - predicted[:1]
            absolute = np.sqrt(np.mean(np.sum((predicted - observed) ** 2, axis=1)))
            motion = np.sqrt(np.mean(np.sum((predicted_delta - observed_delta) ** 2, axis=1)))
            costs[track_index, point_index] = 0.35 * absolute + 0.65 * motion
    return costs


def robust_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rows, cols = linear_sum_assignment(costs)
    values = costs[rows, cols]
    valid = values < 999
    rows, cols, values = rows[valid], cols[valid], values[valid]
    if not len(values):
        return rows, cols, 1e3
    keep_count = max(3, int(np.ceil(len(values) * 0.6)))
    keep = np.argsort(values)[:keep_count]
    return rows[keep], cols[keep], float(np.mean(values[keep]))


def score_fixed_pairs(
    projected: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    time_mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> float:
    errors = []
    for row, col in zip(rows, cols):
        valid = visible[:, row].astype(bool) & time_mask
        if valid.sum() < 2:
            continue
        observed = tracks[valid, row]
        predicted = projected[valid, col]
        observed_delta = observed - observed[:1]
        predicted_delta = predicted - predicted[:1]
        errors.append(np.sqrt(np.mean(np.sum((predicted_delta - observed_delta) ** 2, axis=1))))
    return float(np.median(errors)) if errors else 1e3


def fit_one(
    model: SO100Model,
    actions: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    seed: int,
    maxiter: int,
) -> dict:
    fit_mask = np.arange(len(actions)) % 2 == 0
    eval_mask = ~fit_mask
    limits = np.asarray(
        [[model.joint_map[name].lower, model.joint_map[name].upper] for name in JOINT_NAMES]
    )
    # Avoid exact boundaries because action deltas still need room to move.
    margin = np.array((0.02, 0.02, 0.02, 0.02, 0.02, 0.01))
    q_bounds = [(float(lo + m), float(hi - m)) for (lo, hi), m in zip(limits, margin)]
    bounds = q_bounds + [
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (np.log(120.0), np.log(1800.0)),
        (-32.0, 288.0),
        (-24.0, 184.0),
    ]

    def objective(params: np.ndarray) -> float:
        q_values = q_trajectory(params[:6], actions)
        below = np.maximum(limits[:, 0][None] - q_values, 0.0)
        above = np.maximum(q_values - limits[:, 1][None], 0.0)
        limit_penalty = float((below + above).sum()) * 100.0
        projected = project_trajectory(model, q_values, params)
        _, _, cost = robust_assignment(pair_costs(projected, tracks, visible, fit_mask))
        # Discourage solutions that explain tracks with a robot almost entirely
        # outside the image while retaining freedom for genuinely cropped views.
        center = projected[0].mean(axis=0)
        frame_penalty = (
            max(-64.0 - center[0], 0.0)
            + max(center[0] - 320.0, 0.0)
            + max(-40.0 - center[1], 0.0)
            + max(center[1] - 200.0, 0.0)
        )
        return cost + limit_penalty + frame_penalty

    solution = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=7,
        polish=True,
        workers=1,
        updating="immediate",
        tol=1e-4,
    )
    params = solution.x
    q_values = q_trajectory(params[:6], actions)
    projected = project_trajectory(model, q_values, params)
    rows, cols, fit_score = robust_assignment(pair_costs(projected, tracks, visible, fit_mask))
    eval_score = score_fixed_pairs(projected, tracks, visible, eval_mask, rows, cols)
    return {
        "params": params,
        "q_values": q_values,
        "projected": projected,
        "rows": rows,
        "cols": cols,
        "fit_score": fit_score,
        "eval_score": eval_score,
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
    }


def variant_score(
    model: SO100Model,
    fitted: dict,
    actions: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
) -> float:
    params = fitted["params"]
    projected = project_trajectory(model, q_trajectory(params[:6], actions), params)
    eval_mask = np.arange(len(actions)) % 2 == 1
    return score_fixed_pairs(
        projected, tracks, visible, eval_mask, fitted["rows"], fitted["cols"]
    )


def save_overlay(
    sample_id: str,
    projected: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    output: Path,
) -> None:
    first = cv2.imread(str(VALSET_ROOT / "images" / f"{sample_id}.png"))
    future_rgb = np.load(VALSET_ROOT / "gt_videos" / f"{sample_id}.npy")
    if first is None or future_rgb.ndim != 4:
        return
    first = cv2.resize(first, (future_rgb.shape[2], future_rgb.shape[1]))
    frames = [first] + [frame[..., ::-1].copy() for frame in future_rgb]
    panels = []
    for time_index in (0, 4, 8, 12, 16):
        frame = frames[min(time_index, len(frames) - 1)].copy()
        sx, sy = frame.shape[1] / 256.0, frame.shape[0] / 160.0
        points = projected[time_index] * np.array((sx, sy))
        for point_index in range(len(points) - 1):
            cv2.line(
                frame,
                tuple(np.round(points[point_index]).astype(int)),
                tuple(np.round(points[point_index + 1]).astype(int)),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for row, col in zip(rows, cols):
            if visible[time_index, row]:
                observed = tracks[time_index, row] * np.array((sx, sy))
                predicted = points[col]
                cv2.circle(frame, tuple(np.round(observed).astype(int)), 3, (255, 0, 255), -1)
                cv2.line(
                    frame,
                    tuple(np.round(observed).astype(int)),
                    tuple(np.round(predicted).astype(int)),
                    (0, 180, 0),
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(frame, f"t={time_index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panels.append(cv2.resize(frame, (256, 160)))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.hstack(panels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=REPO / "results/so100_track_alignment_gate.json"
    )
    parser.add_argument(
        "--overlay-root", type=Path, default=REPO / "diagnostics/so100_alignment"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SO100Model(URDF)
    rows = []
    for index in range(args.num_samples):
        sample_id = f"holdout_{index:04d}"
        artifact = np.load(ARTIFACT_ROOT / f"{sample_id}.npz")
        actions = artifact["actions"].astype(np.float64)
        tracks = artifact["tracks"].astype(np.float64)
        visible = artifact["visible"].astype(bool)
        if tracks.shape[1] < 3 or np.max(np.abs(actions - actions[:1])) < 1e-4:
            rows.append({"sample_id": sample_id, "status": "SKIP_UNIDENTIFIABLE"})
            print(f"[{sample_id}] skip: tracks={tracks.shape[1]}, action_motion=0")
            continue
        fitted = fit_one(model, actions, tracks, visible, args.seed + index, args.maxiter)
        zero = np.repeat(actions[:1], len(actions), axis=0)
        reverse = actions[::-1].copy()
        other_index = (index + 1) % args.num_samples
        other = np.load(ARTIFACT_ROOT / f"holdout_{other_index:04d}.npz")["actions"].astype(np.float64)
        if len(other) != len(actions):
            pick = np.linspace(0, len(other) - 1, len(actions)).round().astype(int)
            other = other[pick]
        normal_score = fitted["eval_score"]
        zero_score = variant_score(model, fitted, zero, tracks, visible)
        reverse_score = variant_score(model, fitted, reverse, tracks, visible)
        batch_roll_score = variant_score(model, fitted, other, tracks, visible)
        record = {
            "sample_id": sample_id,
            "status": "OK",
            "matched_tracks": int(len(fitted["rows"])),
            "fit_even_rmse": float(fitted["fit_score"]),
            "normal_odd_motion_rmse": normal_score,
            "zero_odd_motion_rmse": zero_score,
            "reverse_odd_motion_rmse": reverse_score,
            "batch_roll_odd_motion_rmse": batch_roll_score,
            "normal_minus_zero": normal_score - zero_score,
            "normal_minus_reverse": normal_score - reverse_score,
            "normal_minus_batch_roll": normal_score - batch_roll_score,
            "q0_radians": fitted["params"][:6].tolist(),
            "camera_ortho": {
                "euler": fitted["params"][6:9].tolist(),
                "pixels_per_meter": float(np.exp(fitted["params"][9])),
                "translation": fitted["params"][10:12].tolist(),
            },
        }
        rows.append(record)
        save_overlay(
            sample_id,
            fitted["projected"],
            fitted["rows"],
            fitted["cols"],
            tracks,
            visible,
            args.overlay_root / f"{sample_id}.jpg",
        )
        print(
            f"[{sample_id}] normal={normal_score:.2f} zero={zero_score:.2f} "
            f"reverse={reverse_score:.2f} roll={batch_roll_score:.2f}"
        )

    valid = [row for row in rows if row["status"] == "OK"]
    comparisons = ("normal_minus_zero", "normal_minus_reverse", "normal_minus_batch_roll")
    medians = {
        name: float(np.median([row[name] for row in valid])) if valid else float("nan")
        for name in comparisons
    }
    # Lower RMSE is better. Require normal to beat all counterfactuals in the
    # median and at least 60% of identifiable clips.
    win_rates = {
        name: float(np.mean([row[name] < 0 for row in valid])) if valid else 0.0
        for name in comparisons
    }
    passed = bool(valid) and all(medians[name] < 0 and win_rates[name] >= 0.6 for name in comparisons)
    result = {
        "method": "train-only oracle camera/q0 fit; not deployable inference",
        "identifiable_samples": len(valid),
        "median_differences": medians,
        "win_rates": win_rates,
        "verdict": "PASS_ACTION_TO_VISUAL_PROMPT_GEOMETRY" if passed else "REJECT_URDF_ALIGNMENT",
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
