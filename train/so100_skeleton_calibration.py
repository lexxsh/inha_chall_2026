"""Calibration methods for SO-100 action-to-OSCAR skeleton conversion.

The three methods intentionally share one camera/action representation:

``multiframe``
    RobotArena-style oracle teacher.  It uses train-video states and motion
    tracks to estimate joint zeroes and a weak-perspective camera.
``singleframe``
    RoboPose-style deployable fit.  It uses only the source RGB image and a
    zero-shot robot bounding box.
``mask``
    EasyHeC/CtRNet-style deployable refinement.  It replaces generic image
    edges with a source-frame robot mask/centerline.

Only the latter two are legal at challenge inference.  The multiframe result
is a training target and an upper-bound diagnostic, never an eval input.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment, minimize

from so100_renderer import JOINT_NAMES, SO100Model


# Dataset degrees/linear gripper -> ManiSkill SO-100 URDF radians.  Absolute
# offsets are deliberately absent and are represented by q0.
ACTION_TO_Q_SCALE = np.asarray(
    (-np.pi / 180, np.pi / 180, -np.pi / 180, np.pi / 180, np.pi / 180, 2.2 / 100),
    dtype=np.float64,
)


@dataclass
class Calibration:
    params: np.ndarray
    method: str
    deployable: bool
    objective: float

    @property
    def q0(self) -> np.ndarray:
        return self.params[:6]

    @property
    def euler(self) -> np.ndarray:
        return self.params[6:9]

    @property
    def pixels_per_meter(self) -> float:
        return float(np.exp(self.params[9]))

    @property
    def translation(self) -> np.ndarray:
        return self.params[10:12]

    def to_json(self) -> dict:
        return {
            "method": self.method,
            "deployable": self.deployable,
            "objective": float(self.objective),
            "q0_radians": self.q0.tolist(),
            "camera_ortho": {
                "euler": self.euler.tolist(),
                "pixels_per_meter": self.pixels_per_meter,
                "translation": self.translation.tolist(),
            },
        }


def euler_matrix(angles: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(angles, dtype=np.float64)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def legacy_action_q0(action: np.ndarray) -> np.ndarray:
    """A bounded initialization only; never treated as calibrated state."""
    action = np.asarray(action, dtype=np.float64)
    q = np.empty(6, dtype=np.float64)
    q[0] = np.deg2rad(-action[0])
    q[1] = np.deg2rad(action[1] - 180.0)
    q[2] = np.deg2rad(180.0 - action[2])
    q[3] = np.deg2rad(action[3])
    q[4] = np.deg2rad(action[4])
    q[5] = -1.1 + 2.2 * action[5] / 100.0
    return q


def q_trajectory(
    q0: np.ndarray,
    signal: np.ndarray,
    reference: np.ndarray | None = None,
    response: float = 1.0,
) -> np.ndarray:
    """Convert raw state/action values to URDF positions around fitted q0.

    ``reference`` is the measured source-frame state during train calibration.
    At challenge inference it is unavailable and defaults to the first target
    action.  ``response`` optionally models a first-order servo lag.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 6:
        raise ValueError(f"signal must have shape (T,6), got {signal.shape}")
    origin = signal[0] if reference is None else np.asarray(reference, dtype=np.float64)
    target = np.asarray(q0, dtype=np.float64)[None] + (signal - origin[None]) * ACTION_TO_Q_SCALE
    alpha = float(np.clip(response, 1e-3, 1.0))
    if alpha >= 1.0:
        return target
    filtered = target.copy()
    for index in range(1, len(filtered)):
        filtered[index] = (1.0 - alpha) * filtered[index - 1] + alpha * target[index]
    return filtered


def skeleton_condition_trajectory(
    q0: np.ndarray,
    actions: np.ndarray,
    source_state: np.ndarray | None = None,
    response: float = 1.0,
) -> np.ndarray:
    """Build the 16 skeleton poses aligned with the submitted RGB clip.

    Frame zero is the supplied observation.  Frames 1..15 correspond to the
    response to actions 0..14 because LeRobot action[t] targets state/frame
    t+1.  Action 15 is context beyond the saved horizon.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"actions must have shape (T,6), got {actions.shape}")
    if not len(actions):
        return np.asarray(q0, dtype=np.float64)[None]
    origin = actions[0] if source_state is None else np.asarray(source_state, dtype=np.float64)
    targets = np.asarray(q0, dtype=np.float64)[None] + (
        actions[:-1] - origin[None]
    ) * ACTION_TO_Q_SCALE
    alpha = float(np.clip(response, 1e-3, 1.0))
    poses = [np.asarray(q0, dtype=np.float64).copy()]
    current = poses[0]
    for target in targets:
        current = (1.0 - alpha) * current + alpha * target
        poses.append(current.copy())
    return np.asarray(poses)


def chain_points(model: SO100Model, q: np.ndarray, samples_per_link: int = 7) -> np.ndarray:
    positions = model.joint_positions(q)
    anchors = np.asarray(
        [positions[model.root_link]] + [positions[name] for name in JOINT_NAMES],
        dtype=np.float64,
    )
    points: list[np.ndarray] = []
    for start, end in zip(anchors[:-1], anchors[1:]):
        for fraction in np.linspace(0.0, 1.0, samples_per_link, endpoint=False):
            points.append(start * (1.0 - fraction) + end * fraction)
    points.append(anchors[-1])
    return np.asarray(points)


def project_points(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    rotation = euler_matrix(params[6:9])
    camera = (rotation @ np.asarray(points, dtype=np.float64).T).T
    return camera[:, :2] * np.exp(params[9]) + params[10:12]


def project_trajectory(model: SO100Model, q_values: np.ndarray, params: np.ndarray) -> np.ndarray:
    return np.asarray([project_points(chain_points(model, q), params) for q in q_values])


def _joint_limits(model: SO100Model) -> np.ndarray:
    return np.asarray(
        [[model.joint_map[name].lower, model.joint_map[name].upper] for name in JOINT_NAMES],
        dtype=np.float64,
    )


def _pair_costs(
    projected: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    time_mask: np.ndarray,
) -> np.ndarray:
    costs = np.full((tracks.shape[1], projected.shape[1]), 1e3, dtype=np.float64)
    for track_index in range(tracks.shape[1]):
        valid = visible[:, track_index].astype(bool) & time_mask
        if valid.sum() < 3:
            continue
        observed = tracks[valid, track_index]
        observed_delta = observed - observed[:1]
        for point_index in range(projected.shape[1]):
            predicted = projected[valid, point_index]
            predicted_delta = predicted - predicted[:1]
            absolute = np.sqrt(np.mean(np.sum((predicted - observed) ** 2, axis=1)))
            motion = np.sqrt(np.mean(np.sum((predicted_delta - observed_delta) ** 2, axis=1)))
            costs[track_index, point_index] = 0.25 * absolute + 0.75 * motion
    return costs


def _robust_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rows, cols = linear_sum_assignment(costs)
    values = costs[rows, cols]
    valid = values < 999
    rows, cols, values = rows[valid], cols[valid], values[valid]
    if not len(values):
        return rows, cols, 1e3
    keep_count = max(3, int(np.ceil(len(values) * 0.65)))
    keep = np.argsort(values)[:keep_count]
    return rows[keep], cols[keep], float(np.mean(values[keep]))


def motion_track_score(
    projected: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    time_mask: np.ndarray | None = None,
) -> float:
    if time_mask is None:
        time_mask = np.ones(len(projected), dtype=bool)
    _, _, score = _robust_assignment(_pair_costs(projected, tracks, visible, time_mask))
    return float(score)


def fit_multiframe(
    model: SO100Model,
    states: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    source_box: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    seed: int = 0,
    maxiter: int = 60,
) -> Calibration:
    """RobotArena-style train-only calibration from realized states + tracks."""
    states = np.asarray(states, dtype=np.float64)
    limits = _joint_limits(model)
    margin = np.asarray((0.02, 0.02, 0.02, 0.02, 0.02, 0.01))
    bounds = [
        (float(lo + pad), float(hi - pad))
        for (lo, hi), pad in zip(limits, margin)
    ] + [
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (np.log(120.0), np.log(1800.0)),
        (-64.0, 320.0),
        (-40.0, 200.0),
    ]
    fit_mask = np.arange(len(states)) % 2 == 0
    target_box = None if source_box is None else np.asarray(source_box, dtype=np.float64)
    centerline_distance = None
    binary_mask = None
    if source_mask is not None:
        binary_mask = (np.asarray(source_mask) > 0).astype(np.uint8) * 255
        target_skeleton = morphological_skeleton(binary_mask)
        centerline_distance = cv2.distanceTransform(
            (target_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        if target_box is None:
            ys, xs = np.nonzero(binary_mask)
            if len(xs):
                target_box = np.asarray(
                    (xs.min(), ys.min(), xs.max(), ys.max()), dtype=np.float64
                )

    def objective(params: np.ndarray) -> float:
        q_values = q_trajectory(params[:6], states, reference=states[0])
        below = np.maximum(limits[:, 0][None] - q_values, 0.0)
        above = np.maximum(q_values - limits[:, 1][None], 0.0)
        limit_penalty = float((below + above).sum()) * 100.0
        projected = project_trajectory(model, q_values, params)
        _, _, cost = _robust_assignment(_pair_costs(projected, tracks, visible, fit_mask))
        center = projected[0].mean(axis=0)
        frame_penalty = (
            max(-32.0 - center[0], 0.0)
            + max(center[0] - 288.0, 0.0)
            + max(-24.0 - center[1], 0.0)
            + max(center[1] - 184.0, 0.0)
        )
        source_points = projected[0]
        alignment_penalty = 0.0
        if target_box is not None:
            alignment_penalty += 45.0 * _bbox_loss(
                _bbox(source_points), target_box, 256, 160
            )
        if centerline_distance is not None and binary_mask is not None:
            center_cost, inside_frame = _sample_distance(
                centerline_distance, source_points
            )
            rounded = np.round(source_points).astype(np.int32)
            valid = (
                (rounded[:, 0] >= 0)
                & (rounded[:, 0] < binary_mask.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < binary_mask.shape[0])
            )
            inside_mask = 0.0
            if valid.any():
                dilated = cv2.dilate(binary_mask, np.ones((7, 7), np.uint8))
                inside_mask = float(
                    np.mean(dilated[rounded[valid, 1], rounded[valid, 0]] > 0)
                )
            alignment_penalty += (
                0.5 * center_cost
                + 18.0 * (1.0 - inside_frame)
                + 18.0 * (1.0 - inside_mask)
            )
        return cost + limit_penalty + frame_penalty + alignment_penalty

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
    return Calibration(solution.x, "multiframe_state_track", False, float(solution.fun))


def _bbox(points: np.ndarray) -> np.ndarray:
    return np.asarray(
        (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()),
        dtype=np.float64,
    )


def _bbox_loss(predicted: np.ndarray, target: np.ndarray, width: int, height: int) -> float:
    normalizer = np.asarray((width, height, width, height), dtype=np.float64)
    return float(np.mean(np.abs((predicted - target) / normalizer)))


def _sample_distance(distance: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    height, width = distance.shape
    xy = np.round(points).astype(np.int32)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    if not inside.any():
        return float(max(width, height)), 0.0
    values = distance[xy[inside, 1], xy[inside, 0]]
    return float(np.mean(np.clip(values, 0, 32))), float(inside.mean())


def _singleframe_bounds(model: SO100Model, action0: np.ndarray, box: np.ndarray) -> list[tuple[float, float]]:
    limits = _joint_limits(model)
    initial = np.clip(legacy_action_q0(action0), limits[:, 0], limits[:, 1])
    q_bounds = [
        (max(float(lo), float(value - 1.0)), min(float(hi), float(value + 1.0)))
        for value, (lo, hi) in zip(initial, limits)
    ]
    x0, y0, x1, y1 = box
    return q_bounds + [
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (np.log(120.0), np.log(1800.0)),
        (float(x0 - 96), float(x1 + 96)),
        (float(y0 - 72), float(y1 + 72)),
    ]


def fit_singleframe_edges(
    model: SO100Model,
    image_rgb: np.ndarray,
    action0: np.ndarray,
    robot_box: np.ndarray,
    seed: int = 0,
    maxiter: int = 35,
) -> Calibration:
    """RoboPose-style source-only fit using a detection box and RGB edges."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    box = np.asarray(robot_box, dtype=np.float64)
    bounds = _singleframe_bounds(model, action0, box)
    limits = _joint_limits(model)

    def objective(params: np.ndarray) -> float:
        q = params[:6]
        points = project_points(chain_points(model, q, samples_per_link=10), params)
        edge_cost, inside = _sample_distance(distance, points)
        box_cost = _bbox_loss(_bbox(points), box, width, height)
        limit_cost = float(
            np.maximum(limits[:, 0] - q, 0.0).sum()
            + np.maximum(q - limits[:, 1], 0.0).sum()
        )
        return edge_cost + 45.0 * box_cost + 30.0 * (1.0 - inside) + 100.0 * limit_cost

    solution = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=6,
        polish=True,
        workers=1,
        updating="immediate",
        tol=1e-4,
    )
    return Calibration(solution.x, "singleframe_render_compare", True, float(solution.fun))


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    current = (np.asarray(mask) > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(current)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(current):
        eroded = cv2.erode(current, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded
    return skeleton


def fit_singleframe_mask(
    model: SO100Model,
    image_rgb: np.ndarray,
    action0: np.ndarray,
    robot_mask: np.ndarray,
    initial: Calibration,
    maxiter: int = 180,
) -> Calibration:
    """EasyHeC-style source-only mask/centerline refinement."""
    del image_rgb  # Kept in the interface for parity with future feature losses.
    mask = (np.asarray(robot_mask) > 0).astype(np.uint8) * 255
    height, width = mask.shape
    target_skeleton = morphological_skeleton(mask)
    distance = cv2.distanceTransform(
        (target_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("robot_mask is empty")
    target_box = np.asarray((xs.min(), ys.min(), xs.max(), ys.max()), dtype=np.float64)
    base_bounds = _singleframe_bounds(model, action0, target_box)
    local_radius = np.asarray((0.45,) * 6 + (0.55,) * 3 + (0.45, 40.0, 32.0))
    bounds = []
    for value, radius, (lower, upper) in zip(initial.params, local_radius, base_bounds):
        bounds.append((max(lower, float(value - radius)), min(upper, float(value + radius))))

    def objective(params: np.ndarray) -> float:
        points = project_points(chain_points(model, params[:6], samples_per_link=12), params)
        center_cost, inside_frame = _sample_distance(distance, points)
        rounded = np.round(points).astype(np.int32)
        valid = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        inside_mask = 0.0
        if valid.any():
            inside_mask = float(
                np.mean(mask[rounded[valid, 1], rounded[valid, 0]] > 0)
            )
        box_cost = _bbox_loss(_bbox(points), target_box, width, height)
        return center_cost + 35.0 * box_cost + 20.0 * (1.0 - inside_mask) + 20.0 * (1.0 - inside_frame)

    solution = minimize(
        objective,
        initial.params,
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-4},
    )
    return Calibration(solution.x, "singleframe_mask_refine", True, float(solution.fun))


def calibration_projection(
    model: SO100Model,
    calibration: Calibration,
    actions: np.ndarray,
    source_state: np.ndarray | None = None,
    response: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    q_values = skeleton_condition_trajectory(
        calibration.q0, actions, source_state=source_state, response=response
    )
    return q_values, project_trajectory(model, q_values, calibration.params)
