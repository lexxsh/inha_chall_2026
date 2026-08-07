"""SO-100 joint commands converted to Cosmos3 Bridge forward-dynamics actions.

Cosmos3's released single-arm forward-dynamics contract is a 10-D action per
visual transition::

    [body-frame translation delta (3), column rot6d delta (6), gripper (1)]

The pose delta is ``T_i^-1 @ T_{i+1}`` (``backward_framewise`` in the
Cosmos3 codebase).  The temporal contract is ``state[0] -> action[0] ->
frame[1]``: training uses the measured source state while challenge inference
uses a train-only median ``state[0] - action[0]`` offset.  This avoids the old
bug that duplicated action 0 as the source state and silently made the first
transition an identity.

This module deliberately does not invent a new action adapter.  It maps into
the released ``bridge_orig_lerobot`` action/domain contract and applies the
official Bridge quantile statistics used by Cosmos3 inference.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO / "third_party/ManiSkill/mani_skill/assets/robots/so100/so100.urdf"
DEFAULT_BRIDGE_STATS = (
    REPO
    / "third_party/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats"
    / "bridge_orig_lerobot_stats.json"
)
ACTION_STEPS = 16
SO100_ACTION_DIM = 6
COSMOS3_ACTION_DIM = 10
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_TO_Q_SCALE = np.asarray(
    (-np.pi / 180, np.pi / 180, -np.pi / 180, np.pi / 180, np.pi / 180, 2.2 / 100),
    dtype=np.float64,
)
IDENTITY_ROT6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
# The prior BWM converter used a bent pose to avoid Euler gimbal lock.
# Cosmos3 uses rot6d, so a symmetric neutral is valid and preserves the
# maximum range in both directions for source-relative command changes.
COSMOS3_CANONICAL_Q = np.zeros(SO100_ACTION_DIM, dtype=np.float64)
# Train-only robust estimate from SO-100 parquet state/action pairs.  Evaluation
# exposes the source image and future commands but not observation.state.
DEFAULT_SOURCE_STATE_OFFSET = np.asarray(
    (-0.2197265625, -0.703125, 0.6591796875, -0.087890625, 0.1318359375, 0.5254341065883636),
    dtype=np.float64,
)


def _numbers(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class _SO100KinematicModel:
    """Minimal dependency-free SO-100 URDF graph used only for FK."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.links = tuple(link.get("name", "") for link in root.findall("link"))
        self.joints: list[_Joint] = []
        for joint in root.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            parent = joint.find("parent")
            child = joint.find("child")
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            if parent is None or child is None:
                continue
            self.joints.append(
                _Joint(
                    name=joint.get("name", ""),
                    parent=parent.get("link", ""),
                    child=child.get("link", ""),
                    origin=_transform(
                        _numbers(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
                        _numbers(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
                    ),
                    axis=_numbers(axis.get("xyz") if axis is not None else None, (1, 0, 0)),
                    lower=float(limit.get("lower", "-inf")) if limit is not None else -np.inf,
                    upper=float(limit.get("upper", "inf")) if limit is not None else np.inf,
                )
            )
        self.joint_map = {joint.name: joint for joint in self.joints}
        missing = [name for name in JOINT_NAMES if name not in self.joint_map]
        if missing:
            raise ValueError(f"URDF is missing SO-100 joints: {missing}")
        children = {joint.child for joint in self.joints}
        roots = [name for name in self.links if name not in children]
        if not roots:
            raise ValueError("Could not find SO-100 URDF root link")
        self.root_link = roots[0]


def _axis_rotation_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    angles = np.asarray(angles, dtype=np.float64)
    c, s = np.cos(angles), np.sin(angles)
    one = 1.0 - c
    rotation = np.empty((len(angles), 4, 4), dtype=np.float64)
    rotation[:] = np.eye(4, dtype=np.float64)
    rotation[:, 0, 0] = c + x * x * one
    rotation[:, 0, 1] = x * y * one - z * s
    rotation[:, 0, 2] = x * z * one + y * s
    rotation[:, 1, 0] = y * x * one + z * s
    rotation[:, 1, 1] = c + y * y * one
    rotation[:, 1, 2] = y * z * one - x * s
    rotation[:, 2, 0] = z * x * one - y * s
    rotation[:, 2, 1] = z * y * one + x * s
    rotation[:, 2, 2] = c + z * z * one
    return rotation


def matrix_to_column_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Encode the first two rotation-matrix columns in Cosmos3 order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in [3,3], got {matrix.shape}")
    return matrix[..., :, :2].swapaxes(-1, -2).reshape(*matrix.shape[:-2], 6).astype(np.float32)


class SO100Cosmos3ActionConverter:
    """Faithful SO-100 FK -> Cosmos3 Bridge 10-D action conversion."""

    def __init__(
        self,
        *,
        stats_path: str | Path = DEFAULT_BRIDGE_STATS,
        urdf_path: str | Path = DEFAULT_URDF,
        endpoint_link: str = "Fixed_Jaw",
    ) -> None:
        self.stats_path = Path(stats_path).resolve()
        if not self.stats_path.is_file():
            raise FileNotFoundError(
                f"Missing official Cosmos3 Bridge stats: {self.stats_path}. "
                "Clone NVIDIA/cosmos-framework under third_party/cosmos-framework first."
            )
        payload = json.loads(self.stats_path.read_text())
        self.q01 = np.asarray(payload["q01"], dtype=np.float64)
        self.q99 = np.asarray(payload["q99"], dtype=np.float64)
        if self.q01.shape != (COSMOS3_ACTION_DIM,) or self.q99.shape != (COSMOS3_ACTION_DIM,):
            raise ValueError(f"Bridge statistics must be 10-D, got {self.q01.shape} and {self.q99.shape}")
        if np.any(self.q99 - self.q01 <= 1e-8):
            raise ValueError("Bridge statistics contain a degenerate channel")

        self.model = _SO100KinematicModel(urdf_path)
        if endpoint_link not in self.model.links:
            raise ValueError(f"URDF has no endpoint link {endpoint_link!r}")
        self.endpoint_link = endpoint_link
        self.joint_limits = np.asarray(
            [[self.model.joint_map[name].lower, self.model.joint_map[name].upper] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        self.urdf_path = self.model.urdf_path

    def raw_to_poses(
        self,
        raw_actions: np.ndarray | torch.Tensor,
        *,
        source_action: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Return endpoint absolute FK poses for a source-anchored command path."""
        raw = np.asarray(
            raw_actions.detach().cpu().numpy() if isinstance(raw_actions, torch.Tensor) else raw_actions,
            dtype=np.float64,
        )
        source = np.asarray(
            source_action.detach().cpu().numpy() if isinstance(source_action, torch.Tensor) else source_action,
            dtype=np.float64,
        )
        if raw.ndim != 2 or raw.shape[1] != SO100_ACTION_DIM:
            raise ValueError(f"SO-100 command path must be [T,6], got {raw.shape}")
        if source.shape != (SO100_ACTION_DIM,):
            raise ValueError(f"source action must be [6], got {source.shape}")

        q = COSMOS3_CANONICAL_Q + (raw - source[None]) * ACTION_TO_Q_SCALE
        q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1]).reshape(-1, SO100_ACTION_DIM)
        count = len(q)
        identity = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
        transforms: dict[str, np.ndarray] = {self.model.root_link: identity}
        q_by_name = {name: q[:, index] for index, name in enumerate(JOINT_NAMES)}
        remaining = list(self.model.joints)
        while remaining:
            progressed = False
            for joint in remaining[:]:
                if joint.parent not in transforms:
                    continue
                angles = q_by_name.get(joint.name, np.zeros(count, dtype=np.float64))
                motion = _axis_rotation_batch(joint.axis, angles)
                origin = np.broadcast_to(joint.origin, (count, 4, 4))
                transforms[joint.child] = transforms[joint.parent] @ origin @ motion
                remaining.remove(joint)
                progressed = True
            if not progressed:
                raise RuntimeError("Disconnected SO-100 URDF graph")
        poses = transforms[self.endpoint_link]
        if not np.isfinite(poses).all():
            raise RuntimeError("Non-finite SO-100 FK pose")
        return poses

    @staticmethod
    def _gripper_command(
        future_actions: np.ndarray,
        source_action: np.ndarray,
    ) -> np.ndarray:
        """Map the absolute SO-100 gripper command to Bridge's [0,1] channel."""
        del source_action
        # SO-100 URDF conversion is q = raw * 2.2 / 100 with limits [-1.1,1.1].
        # Mapping that joint interval to [0,1] gives exactly 0.5 + raw/100.
        gripper = 0.5 + future_actions[:, 5] / 100.0
        return np.clip(gripper, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def estimate_source_action(
        first_command: np.ndarray | torch.Tensor,
        *,
        offset: np.ndarray | torch.Tensor = DEFAULT_SOURCE_STATE_OFFSET,
    ) -> np.ndarray:
        """Estimate hidden source joint state from the first future command."""
        command = np.asarray(
            first_command.detach().cpu().numpy() if isinstance(first_command, torch.Tensor) else first_command,
            dtype=np.float64,
        )
        delta = np.asarray(offset.detach().cpu().numpy() if isinstance(offset, torch.Tensor) else offset, dtype=np.float64)
        if command.shape != (SO100_ACTION_DIM,) or delta.shape != (SO100_ACTION_DIM,):
            raise ValueError(f"Expected command/offset [6], got {command.shape} and {delta.shape}")
        return command + delta

    def build_raw_actions(
        self,
        future_actions: np.ndarray | torch.Tensor,
        *,
        source_action: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Build unnormalized Cosmos3 actions with shape ``[16,10]``."""
        future = np.asarray(
            future_actions.detach().cpu().numpy()
            if isinstance(future_actions, torch.Tensor)
            else future_actions,
            dtype=np.float64,
        )
        if future.shape != (ACTION_STEPS, SO100_ACTION_DIM):
            raise ValueError(f"Expected SO-100 actions [16,6], got {future.shape}")
        if source_action is None:
            source = self.estimate_source_action(future[0])
        else:
            source = np.asarray(
                source_action.detach().cpu().numpy()
                if isinstance(source_action, torch.Tensor)
                else source_action,
                dtype=np.float64,
            )
        if source.shape != (SO100_ACTION_DIM,):
            raise ValueError(f"Expected source action [6], got {source.shape}")

        # 17 endpoint poses define the 16 source/command transitions.
        command_path = np.concatenate([source[None], future], axis=0)
        poses = self.raw_to_poses(command_path, source_action=source)
        relative = np.linalg.inv(poses[:-1]) @ poses[1:]
        translation = relative[:, :3, 3].astype(np.float32)
        rotation = matrix_to_column_rot6d(relative[:, :3, :3])
        gripper = self._gripper_command(future, source)[:, None]
        action = np.concatenate([translation, rotation, gripper], axis=-1)
        if action.shape != (ACTION_STEPS, COSMOS3_ACTION_DIM) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid Cosmos3 raw action: {action.shape}")
        return action

    def normalize(
        self,
        raw_actions: np.ndarray | torch.Tensor,
        *,
        clamp: bool = False,
    ) -> np.ndarray:
        """Apply official affine Bridge q01/q99 normalization.

        Official ActionProcessor normalization is unclamped.  ``clamp=True``
        remains available only for reproducing an older diagnostic.
        """
        raw = np.asarray(
            raw_actions.detach().cpu().numpy() if isinstance(raw_actions, torch.Tensor) else raw_actions,
            dtype=np.float64,
        )
        if raw.shape[-1] != COSMOS3_ACTION_DIM:
            raise ValueError(f"Cosmos3 action must end in 10 channels, got {raw.shape}")
        normalized = 2.0 * (raw - self.q01) / (self.q99 - self.q01) - 1.0
        if clamp:
            normalized = np.clip(normalized, -1.0, 1.0)
        if not np.isfinite(normalized).all():
            raise RuntimeError("Non-finite normalized Cosmos3 action")
        return normalized.astype(np.float32)

    def build_actions(
        self,
        future_actions: np.ndarray | torch.Tensor,
        *,
        source_action: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build normalized Cosmos3 Bridge-domain actions ``[16,10]``."""
        raw = self.build_raw_actions(future_actions, source_action=source_action)
        return torch.from_numpy(self.normalize(raw))

    def joint_limit_violation_fraction(self, raw_actions: np.ndarray) -> dict[str, float]:
        """Report pre-clip violations; clipping is only a final safety bound."""
        action = np.asarray(raw_actions, dtype=np.float64)
        unbounded = COSMOS3_CANONICAL_Q + (action - action[:1]) * ACTION_TO_Q_SCALE
        result = {}
        for index, name in enumerate(JOINT_NAMES):
            lo, hi = self.joint_limits[index]
            result[name] = float(np.mean((unbounded[:, index] < lo) | (unbounded[:, index] > hi)))
        return result


__all__ = [
    "ACTION_STEPS",
    "COSMOS3_ACTION_DIM",
    "COSMOS3_CANONICAL_Q",
    "DEFAULT_SOURCE_STATE_OFFSET",
    "DEFAULT_BRIDGE_STATS",
    "IDENTITY_ROT6D",
    "SO100Cosmos3ActionConverter",
    "matrix_to_column_rot6d",
]
