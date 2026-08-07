"""LeRobot SO-100 clips in the format expected by Cosmos-Predict2.5."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[2]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    LeRobotSO100Dataset,
    discover_lerobot_so100_datasets,
)

DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


def _load_action_scale(path: Path, mode: str) -> torch.Tensor:
    if mode == "absolute":
        return torch.ones(6, dtype=torch.float32)
    stats = json.loads(path.read_text())
    by_mode = stats.get("scale_by_mode", {})
    if mode == "hybrid_step":
        return torch.tensor(by_mode["delta_step"], dtype=torch.float32)
    if mode in by_mode:
        return torch.tensor(by_mode[mode], dtype=torch.float32)
    if mode == "delta":
        return torch.tensor(stats["delta_std"], dtype=torch.float32)
    raise ValueError(
        "Cosmos adapter currently supports absolute/delta/delta_step/hybrid_step, "
        f"got {mode!r}"
    )


def _transform_actions(
    actions: torch.Tensor,
    scale: torch.Tensor,
    mode: str,
    shift: int,
) -> torch.Tensor:
    """Lightweight equivalent of train.data_module.transform_actions.

    It is kept local because importing the Lightning data module would pull a
    second training framework into the isolated Cosmos environment.
    """
    if shift:
        indices = torch.arange(actions.shape[0], device=actions.device).add(shift)
        actions = actions.index_select(0, indices.clamp(0, actions.shape[0] - 1))
    if mode == "absolute":
        return actions
    anchor = actions[:1]
    if mode == "hybrid_step":
        # The public Cosmos action is a local EE displacement plus an absolute
        # gripper command.  A joint->EE map needs both the current joint target
        # (for configuration/Jacobian and gripper state) and the local step.
        step = actions - torch.cat([anchor, actions[:-1]], dim=0)
        step = step / scale.to(actions.device).clamp(min=1e-6)
        return torch.cat([actions, step], dim=-1)
    if mode == "delta":
        delta = actions - anchor
    elif mode == "delta_step":
        delta = actions - torch.cat([anchor, actions[:-1]], dim=0)
    else:
        raise ValueError(f"Unsupported Cosmos action mode: {mode!r}")
    return delta / scale.to(actions.device).clamp(min=1e-6)


def so100_to_cosmos7(actions: torch.Tensor) -> torch.Tensor:
    """Embed SO-100's 5 joints + gripper into Cosmos Bridge's 7D layout.

    Cosmos' public robot checkpoint expects six motion coordinates followed by
    a gripper coordinate. We retain its pretrained input shapes by using the
    first five motion slots for SO-100 joints, leaving motion slot 5 at zero,
    and moving the SO-100 gripper to slot 6.
    """
    if actions.ndim != 2 or actions.shape[-1] != 6:
        raise ValueError(f"Expected [T,6] actions, got {tuple(actions.shape)}")
    mapped = actions.new_zeros(actions.shape[0], 7)
    mapped[:, :5] = actions[:, :5]
    mapped[:, 6] = actions[:, 5]
    return mapped


class CosmosSO100Dataset(Dataset):
    """Return 17 frames and 16 action commands for Cosmos' 4x temporal VAE.

    Seventeen frames become five latent frames (one conditioned source plus
    four generated latents). The challenge keeps output frames 0..15; frame 16
    exists during training to keep all four action chunks aligned.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        height: int = 256,
        width: int = 320,
        num_frames: int = 17,
        holdout_count: int = 6,
        seed: int = 0,
        action_mode: str = "delta",
        action_shift: int = 0,
        action_variant: str = "normal",
        action_layout: str = "cosmos7",
        repeat: int = 1,
    ) -> None:
        if num_frames < 5 or (num_frames - 1) % 4:
            raise ValueError(
                f"num_frames must be 4k+1 for the temporal VAE, got {num_frames}"
            )
        if action_variant not in {"normal", "zero", "reverse"}:
            raise ValueError(
                "action_variant must be normal/zero/reverse, "
                f"got {action_variant!r}"
            )
        if action_layout not in {"cosmos7", "so1006"}:
            raise ValueError(
                f"action_layout must be 'cosmos7' or 'so1006', got {action_layout!r}"
            )
        paths = discover_lerobot_so100_datasets(root)
        paths = [str(path) for path in paths]
        paths = [path for path in paths if not any(path.rstrip("/").endswith(x) for x in DEFAULT_EXCLUDE)]
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        held = set(shuffled[:holdout_count])
        selected = [path for path in paths if (path in held) == (split == "holdout")]
        if split not in {"train", "holdout"}:
            raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")
        if not selected:
            raise ValueError(f"No datasets selected for split={split!r}")

        stats = json.loads((Path(root) / "so100_action_statistics.json").read_text())
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=selected,
            train=True,
            traj_len=num_frames,
            target_height=height,
            target_width=width,
            pad=True,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            action_mean=stats["mean"],
            action_std=stats["std"],
            use_all_episodes=True,
        )
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.repeat = repeat
        self.action_mode = action_mode
        self.action_shift = action_shift
        self.action_variant = action_variant
        self.action_layout = action_layout
        self.delta_scale = _load_action_scale(
            REPO / "train/delta_action_stats.json", action_mode
        )
        self.selected_paths = selected

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index % len(self.base)]
        video = sample["video"].add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
        actions6 = _transform_actions(
            sample["act"][: self.num_frames - 1],
            self.delta_scale,
            self.action_mode,
            self.action_shift,
        )
        actions_out = (
            so100_to_cosmos7(actions6)
            if self.action_layout == "cosmos7"
            else actions6
        )
        if self.action_variant == "zero":
            actions_out.zero_()
        elif self.action_variant == "reverse":
            actions_out = torch.flip(actions_out, dims=(0,))
        key = f"{self.selected_paths[index % len(self.selected_paths)]}:{int(sample['start_idx'])}"
        return {
            "video": video.contiguous(),
            "action": actions_out.float().contiguous(),
            "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": torch.tensor(6, dtype=torch.int64),
            "image_size": torch.tensor(
                [self.height, self.width, self.height, self.width], dtype=torch.int64
            ),
            "num_frames": self.num_frames,
            "padding_mask": torch.zeros(1, self.height, self.width, dtype=torch.bool),
            "ai_caption": "",
            "annotation_file": key,
            "__key__": key,
        }
