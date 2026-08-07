"""SO100 data contract for native BWM-style Wan2.2 post-training.

The challenge supplies 16 commands for the 16 future RGB transitions.  BWM's
Wan2.2 path consumes one token per RGB frame, including the observed history
frame.  We therefore prepend ``action[0]`` as a history-only proxy and keep
all 16 commands in order.  BWM then groups them as::

    history: [a0, a0, a0, a0]
    future : [a0..a3], [a4..a7], [a8..a11], [a12..a15]

This is the exact 17-RGB -> 5-latent temporal layout of Wan's VAE.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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


ACTION_DIM = 6
FUTURE_ACTIONS = 16
RGB_FRAMES = 17
DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


def action_statistics(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads((Path(root) / "so100_action_statistics.json").read_text())
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32).clamp(min=1e-6)
    if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,):
        raise ValueError(f"Expected SO100 6D statistics, got {mean.shape}/{std.shape}")
    return mean, std


def normalize_actions(raw: torch.Tensor, root: str | Path) -> torch.Tensor:
    """Map raw commands to the bounded scale used by BWM action inputs."""
    mean, std = action_statistics(root)
    normalized = (torch.as_tensor(raw).float() - mean) / std
    # BWM's released loaders use bounded percentile normalization.  The
    # challenge publishes mean/std, so clipped three-sigma scaling is the
    # deterministic train/eval analogue and avoids unbounded action outliers.
    return normalized.clamp(-3.0, 3.0) / 3.0


def build_action_tokens(normalized_future: torch.Tensor) -> torch.Tensor:
    normalized_future = torch.as_tensor(normalized_future, dtype=torch.float32)
    if normalized_future.shape != (FUTURE_ACTIONS, ACTION_DIM):
        raise ValueError(
            f"Expected normalized future actions [{FUTURE_ACTIONS},{ACTION_DIM}], "
            f"got {tuple(normalized_future.shape)}"
        )
    future = normalized_future.clamp(-1.0, 1.0)
    tokens = torch.cat([future[:1], future], dim=0)
    if tokens.shape != (RGB_FRAMES, ACTION_DIM) or not torch.isfinite(tokens).all():
        raise RuntimeError(f"Invalid native BWM action tokens: {tuple(tokens.shape)}")
    return tokens


def selected_dataset_paths(
    root: str | Path,
    *,
    split: str,
    holdout_count: int = 8,
    seed: int = 42,
) -> list[str]:
    paths = discover_lerobot_so100_datasets(root)
    paths = [path for path in paths if not any(path.endswith(x) for x in DEFAULT_EXCLUDE)]
    order = list(paths)
    random.Random(seed).shuffle(order)
    held = set(order[:holdout_count])
    selected = [path for path in paths if (path in held) == (split == "holdout")]
    if not selected:
        raise ValueError(f"No SO100 datasets selected for split={split!r}")
    return selected


class BWMNativeSO100Dataset(Dataset):
    """17 RGB frames plus the native 17x6 BWM action sequence."""

    load_from_cache = False

    def __init__(
        self,
        root: str,
        height: int = 480,
        width: int = 640,
        repeat: int = 1,
        holdout_count: int = 8,
        seed: int = 42,
        split: str = "train",
    ) -> None:
        paths = selected_dataset_paths(
            root, split=split, holdout_count=holdout_count, seed=seed
        )
        mean, std = action_statistics(root)
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=paths,
            train=True,
            traj_len=RGB_FRAMES,
            target_height=height,
            target_width=width,
            pad=True,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            action_mean=mean.tolist(),
            action_std=std.tolist(),
            use_all_episodes=True,
        )
        self.selected_paths = paths
        self.repeat = int(repeat)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index % len(self.base)]
        video = sample["video"].float()
        if video.shape != (3, RGB_FRAMES, self.height, self.width):
            raise ValueError(f"Unexpected SO100 video shape: {tuple(video.shape)}")
        # Dataset output is z-scored.  Convert it to the bounded BWM scale.
        future = sample["act"][:FUTURE_ACTIONS].float().clamp(-3.0, 3.0) / 3.0
        action = build_action_tokens(future)
        return {"video": video.unsqueeze(0), "action": action.unsqueeze(0)}


class CachedSingleClip(Dataset):
    """Repeat one materialized clip for a strict memorization preflight."""

    load_from_cache = False

    def __init__(self, dataset: Dataset, repeats: int) -> None:
        self.sample = dataset[0]
        self.repeats = max(1, int(repeats))
        self.selected_paths = getattr(dataset, "selected_paths", [])[:1]
        self.base = self

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.sample.items()}


def local_wan22_model_paths(model_root: str | Path) -> tuple[list[str], str]:
    root = Path(model_root).resolve()
    shards = sorted(str(path) for path in root.glob("diffusion_pytorch_model-*.safetensors"))
    vae = root / "Wan2.2_VAE.pth"
    missing: list[str] = []
    if len(shards) != 3:
        missing.append(f"{root}/diffusion_pytorch_model-*.safetensors (found {len(shards)})")
    if not vae.is_file():
        missing.append(str(vae))
    if missing:
        raise FileNotFoundError("Missing local Wan2.2 assets:\n  " + "\n  ".join(missing))
    return shards, str(vae)
