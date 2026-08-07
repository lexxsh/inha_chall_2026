"""Faithful SO100 joint-target to BWM dual-arm EEF action conversion.

BWM was pretrained with 14 values per frame: xyz, Euler rotation and gripper
for each of two arms.  This module preserves that contract and the public
14-dimensional action input weights.  SO100's single arm occupies one 7D slot;
the inactive slot is the normalized neutral value (zero).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    LeRobotSO100Dataset,
    discover_lerobot_so100_datasets,
)

try:
    from so100_renderer import JOINT_NAMES, SO100Model  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    from train.so100_renderer import JOINT_NAMES, SO100Model  # noqa: E402


DEFAULT_URDF = REPO / "third_party/ManiSkill/mani_skill/assets/robots/so100/so100.urdf"
DEFAULT_STATS = REPO / "train/so100_eef_statistics.json"
DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)
ACTION_FRAMES = 17
FUTURE_ACTIONS = 16
EEF_DIM = 7
BWM_ACTION_DIM = 14


def selected_so100_paths(
    root: str | Path,
    *,
    split: str,
    holdout_count: int = 8,
    seed: int = 42,
) -> list[str]:
    paths = discover_lerobot_so100_datasets(str(root))
    paths = [path for path in paths if not any(path.endswith(x) for x in DEFAULT_EXCLUDE)]
    order = list(paths)
    random.Random(seed).shuffle(order)
    held = set(order[:holdout_count])
    selected = [path for path in paths if (path in held) == (split == "holdout")]
    if not selected:
        raise ValueError(f"No SO100 datasets selected for split={split!r}")
    return selected


ACTION_TO_Q_SCALE = np.asarray(
    (-np.pi / 180, np.pi / 180, -np.pi / 180, np.pi / 180, np.pi / 180, 2.2 / 100),
    dtype=np.float64,
)
# A feasible, non-singular SO100 pose.  The all-zero URDF pose places the EEF
# at Euler pitch=pi/2 (gimbal lock), which turns smooth joint motion into
# discontinuous roll/yaw targets.  This reference stays comfortably inside
# the joint limits and near the public BWM EEF orientation distribution.
CANONICAL_Q = np.asarray((0.0, -0.5, 0.5, 0.5, 0.0, 0.0), dtype=np.float64)


def source_anchored_action_to_q(
    raw_actions: np.ndarray,
    source_action: np.ndarray | None = None,
    *,
    limits: np.ndarray | None = None,
) -> np.ndarray:
    """Map calibration-variant SO100 targets to a bounded relative URDF path.

    Absolute servo zeroes differ across the collected robots.  Their temporal
    derivatives share the degree convention, so the source action is aligned
    to the URDF neutral pose and only the calibrated derivatives are applied.
    The source RGB remains the authority for the actual initial visual pose.
    """
    actions = np.asarray(raw_actions, dtype=np.float64)
    if actions.ndim < 2 or actions.shape[-1] != 6:
        raise ValueError(f"SO100 trajectory must end in [T,6], got {actions.shape}")
    if source_action is None:
        source = actions[..., :1, :]
    else:
        source = np.asarray(source_action, dtype=np.float64)
        if source.shape == actions.shape[:-2] + (6,):
            source = source[..., None, :]
        try:
            source = np.broadcast_to(source, actions.shape[:-2] + (1, 6))
        except ValueError as error:
            raise ValueError(
                f"source action {source.shape} cannot anchor trajectory {actions.shape}"
            ) from error
    q = CANONICAL_Q + (actions - source) * ACTION_TO_Q_SCALE
    if limits is not None:
        bounds = np.asarray(limits, dtype=np.float64)
        q = np.clip(q, bounds[:, 0], bounds[:, 1])
    return q


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


def _matrix_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Inverse of Rz(yaw) @ Ry(pitch) @ Rx(roll), wrapped to [-pi, pi]."""
    rotation = np.asarray(rotation, dtype=np.float64)
    sy = np.sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
    singular = sy < 1e-7
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = np.arctan2(-rotation[:, 2, 0], sy)
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    roll[singular] = np.arctan2(
        -rotation[singular, 1, 2], rotation[singular, 1, 1]
    )
    yaw[singular] = 0.0
    return np.stack([roll, pitch, yaw], axis=-1)


class SO100EEFConverter:
    def __init__(
        self,
        stats_path: str | Path | None = DEFAULT_STATS,
        urdf_path: str | Path = DEFAULT_URDF,
        endpoint_link: str = "Fixed_Jaw",
    ) -> None:
        self.stats_path = Path(stats_path).resolve() if stats_path is not None else None
        self.urdf_path = Path(urdf_path).resolve()
        self.endpoint_link = endpoint_link
        self.model = SO100Model(self.urdf_path)
        if endpoint_link not in self.model.links:
            raise ValueError(f"URDF has no endpoint link {endpoint_link!r}")
        self.p01 = None
        self.p99 = None
        self.joint_limits = np.asarray(
            [
                [self.model.joint_map[name].lower, self.model.joint_map[name].upper]
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )
        if self.stats_path is None:
            return
        if not self.stats_path.is_file():
            raise FileNotFoundError(
                f"Missing SO100 EEF statistics: {self.stats_path}. "
                "Run tools/compute_so100_eef_stats.py first."
            )
        payload = json.loads(self.stats_path.read_text())
        self.p01 = np.asarray(payload["p01"], dtype=np.float64)
        self.p99 = np.asarray(payload["p99"], dtype=np.float64)
        if self.p01.shape != (EEF_DIM,) or self.p99.shape != (EEF_DIM,):
            raise ValueError(f"Malformed EEF stats in {self.stats_path}")
        if np.any(self.p99 - self.p01 <= 1e-6):
            raise ValueError("SO100 EEF statistics contain a degenerate dimension")
        expected_hash = payload.get("urdf_sha256")
        current_hash = hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
        if expected_hash and expected_hash != current_hash:
            raise ValueError("EEF statistics were computed with a different URDF")

    def raw_to_eef(
        self,
        raw_actions: np.ndarray | torch.Tensor,
        source_action: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        raw = np.asarray(
            raw_actions.detach().cpu().numpy() if isinstance(raw_actions, torch.Tensor) else raw_actions,
            dtype=np.float64,
        )
        original_shape = raw.shape[:-1]
        source = (
            source_action.detach().cpu().numpy()
            if isinstance(source_action, torch.Tensor)
            else source_action
        )
        q_shaped = source_anchored_action_to_q(
            raw, source, limits=self.joint_limits
        )
        q = q_shaped.reshape(-1, 6)
        raw = raw.reshape(-1, 6)
        count = len(q)
        identity = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
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
                raise RuntimeError("Disconnected SO100 URDF graph")
        endpoint = transforms[self.endpoint_link]
        xyz = endpoint[:, :3, 3]
        euler = _matrix_to_euler_xyz(endpoint[:, :3, :3])
        # BWM's EEF gripper field is an openness scalar in [0,1].  Align the
        # unknown source opening to 0.5 and retain the SO100 command change.
        source_open = np.clip(0.5 + (q[:, 5:6] - CANONICAL_Q[5]) / 2.2, 0.0, 1.0)
        eef = np.concatenate([xyz, euler, source_open], axis=-1)
        return eef.reshape(*original_shape, EEF_DIM)

    def normalize_eef(self, eef: np.ndarray) -> np.ndarray:
        if self.p01 is None or self.p99 is None:
            raise RuntimeError("EEF normalization requires a statistics file")
        normalized = 2.0 * (np.asarray(eef, dtype=np.float64) - self.p01) / (
            self.p99 - self.p01
        ) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def build_tokens(
        self,
        future_actions: np.ndarray | torch.Tensor,
        *,
        source_action: np.ndarray | torch.Tensor | None = None,
        active_arm: str = "left",
    ) -> torch.Tensor:
        future = np.asarray(
            future_actions.detach().cpu().numpy()
            if isinstance(future_actions, torch.Tensor)
            else future_actions,
            dtype=np.float64,
        )
        if future.shape != (FUTURE_ACTIONS, 6):
            raise ValueError(f"Expected future action [16,6], got {future.shape}")
        if source_action is None:
            source = future[0]
        else:
            source = np.asarray(
                source_action.detach().cpu().numpy()
                if isinstance(source_action, torch.Tensor)
                else source_action,
                dtype=np.float64,
            )
        if source.shape != (6,):
            raise ValueError(f"Expected source action [6], got {source.shape}")
        trajectory = np.concatenate([source[None], future], axis=0)
        active = self.normalize_eef(
            self.raw_to_eef(trajectory, source_action=source)
        )
        neutral = np.zeros_like(active, dtype=np.float32)
        if active_arm == "left":
            tokens = np.concatenate([active, neutral], axis=-1)
        elif active_arm == "right":
            tokens = np.concatenate([neutral, active], axis=-1)
        else:
            raise ValueError("active_arm must be 'left' or 'right'")
        if tokens.shape != (ACTION_FRAMES, BWM_ACTION_DIM) or not np.isfinite(tokens).all():
            raise RuntimeError(f"Invalid BWM EEF tokens: {tokens.shape}")
        return torch.from_numpy(tokens)

    def joint_limit_violation_fraction(self, raw_actions: np.ndarray) -> dict[str, float]:
        unbounded = source_anchored_action_to_q(raw_actions).reshape(-1, 6)
        result = {}
        for index, name in enumerate(JOINT_NAMES):
            joint = self.model.joint_map[name]
            result[name] = float(
                np.mean(
                    (unbounded[:, index] < joint.lower)
                    | (unbounded[:, index] > joint.upper)
                )
            )
        return result


class BWMSO100EEFDataset(Dataset):
    """17-frame SO100 clips with the original BWM 14D EEF input contract."""

    load_from_cache = False

    def __init__(
        self,
        root: str,
        height: int = 384,
        width: int = 512,
        repeat: int = 1,
        holdout_count: int = 8,
        seed: int = 42,
        split: str = "train",
        stats_path: str | Path = DEFAULT_STATS,
        active_arm: str = "left",
    ) -> None:
        selected = selected_so100_paths(
            root, split=split, holdout_count=holdout_count, seed=seed
        )
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=selected,
            train=True,
            traj_len=ACTION_FRAMES,
            target_height=height,
            target_width=width,
            pad=True,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            action_mean=None,
            action_std=None,
            use_all_episodes=True,
        )
        self.converter = SO100EEFConverter(stats_path=stats_path)
        self.active_arm = active_arm
        self.selected_paths = selected
        self.repeat = int(repeat)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index % len(self.base)]
        video = sample["video"].float()
        if video.shape != (3, ACTION_FRAMES, self.height, self.width):
            raise ValueError(f"Unexpected SO100 video shape: {tuple(video.shape)}")
        raw = sample["act"][:FUTURE_ACTIONS].float()
        tokens = self.converter.build_tokens(raw, active_arm=self.active_arm)
        return {"video": video.unsqueeze(0), "action": tokens.unsqueeze(0)}
