"""SO100 clips for source-conditioned spatial action training on Wan2.1 I2V."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
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
    from data_module import action_dims_for, load_delta_scale, transform_actions  # type: ignore
except ModuleNotFoundError:
    from train.data_module import action_dims_for, load_delta_scale, transform_actions


DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


def to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert C,T,H,W in [-1,1] to the PIL list expected by DiffSynth."""
    frames = ((video.permute(1, 2, 3, 0) + 1.0) * 127.5).round().clamp(0, 255)
    return [Image.fromarray(frame.byte().numpy(), mode="RGB") for frame in frames]


def motion_mask(video: torch.Tensor, threshold: float = 0.06, dilation: int = 9) -> torch.Tensor:
    """Conservative moving-region mask, including a margin around robot edges."""
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"video must be [3,T,H,W], got {tuple(video.shape)}")
    difference = (video - video[:, :1]).abs().amax(dim=0, keepdim=True)
    mask = (difference >= threshold).to(video.dtype)
    if dilation > 1:
        if dilation % 2 == 0:
            raise ValueError("motion-mask dilation must be odd")
        time = mask.shape[1]
        mask = F.max_pool2d(
            mask.transpose(0, 1),
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        ).transpose(0, 1)
        if mask.shape[1] != time:
            raise RuntimeError("motion-mask temporal shape changed unexpectedly")
    mask[:, 0].zero_()
    return mask


class WanSO100SpatialActionDataset(Dataset):
    """Yield true/static clips with correct and counterfactual SO100 controls.

    A fixed fraction of examples are changed to exact first-frame repeats and
    paired with a constant command.  This teaches zero motion explicitly and
    counters the vanilla Wan camera-zoom prior without requiring a second
    diffusion forward.  Dynamic examples use time-reversed actions as the
    counterfactual for the paired ranking loss.
    """

    load_from_cache = False

    def __init__(
        self,
        root: str,
        height: int = 320,
        width: int = 432,
        repeat: int = 100,
        holdout_count: int = 6,
        seed: int = 0,
        action_mode: str = "hybrid",
        action_shift: int = 0,
        prompt: str = "A fixed-camera video of a tabletop robot arm manipulating objects.",
        split: str = "train",
        static_probability: float = 0.15,
        motion_mask_threshold: float = 0.06,
        motion_mask_dilation: int = 9,
    ) -> None:
        if not 0.0 <= static_probability < 1.0:
            raise ValueError("static_probability must be in [0,1)")
        paths = discover_lerobot_so100_datasets(root)
        paths = [path for path in paths if not any(path.endswith(x) for x in DEFAULT_EXCLUDE)]
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        held = set(shuffled[:holdout_count])
        selected = [path for path in paths if (path in held) == (split == "holdout")]
        if not selected:
            raise ValueError(f"No datasets selected for split={split!r}")

        stats = json.loads((Path(root) / "so100_action_statistics.json").read_text())
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=selected,
            train=True,
            traj_len=17,
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
        self.repeat = repeat
        self.seed = seed
        self.action_mode = action_mode
        self.action_shift = action_shift
        self.action_dim = action_dims_for(action_mode)
        self.delta_scale = load_delta_scale(
            str(REPO / "train/delta_action_stats.json"), action_mode
        )
        self.prompt = prompt
        self.static_probability = static_probability
        self.motion_mask_threshold = motion_mask_threshold
        self.motion_mask_dilation = motion_mask_dilation
        self.selected_paths = selected

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def _is_static_augmentation(self, index: int) -> bool:
        # Deterministic across DataLoader workers and restarts.
        bucket = (index * 2654435761 + self.seed * 40503) % 10000
        return bucket < round(self.static_probability * 10000)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index % len(self.base)]
        video = sample["video"]
        raw = sample["act"][:16]
        if raw.shape != (16, 6):
            raise ValueError(f"Expected raw actions [16,6], got {tuple(raw.shape)}")

        if self._is_static_augmentation(index):
            video = video[:, :1].expand(-1, 17, -1, -1).clone()
            correct_raw = raw[:1].expand_as(raw).clone()
            wrong_raw = raw
            sample_kind = "static"
        else:
            correct_raw = raw
            wrong_raw = raw.flip(0)
            sample_kind = "dynamic"

        actions = transform_actions(
            correct_raw, self.delta_scale, self.action_mode, self.action_shift
        )
        wrong_actions = transform_actions(
            wrong_raw, self.delta_scale, self.action_mode, self.action_shift
        )
        difference = (actions - wrong_actions).abs().mean()
        ranking_valid = torch.tensor(float(difference >= 1e-3), dtype=torch.float32)

        return {
            "video": to_pil_frames(video),
            "actions": actions,
            "wrong_actions": wrong_actions,
            "motion_mask": motion_mask(
                video,
                threshold=self.motion_mask_threshold,
                dilation=self.motion_mask_dilation,
            ),
            "ranking_valid": ranking_valid,
            "prompt": self.prompt,
            "sample_kind": sample_kind,
        }
