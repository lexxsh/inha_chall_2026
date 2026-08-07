"""SO-100 clips adapted to Wan2.2-TI2V's 4n+1 temporal layout."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
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
try:  # Script execution and package-style imports both occur in local tools.
    from data_module import load_delta_scale, transform_actions  # type: ignore # noqa: E402
except ModuleNotFoundError:
    from train.data_module import load_delta_scale, transform_actions  # noqa: E402


DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


def _to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert C,T,H,W in [-1,1] to the PIL list expected by DiffSynth."""
    frames = ((video.permute(1, 2, 3, 0) + 1.0) * 127.5).round().clamp(0, 255).byte().numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in frames]


class WanSO100Dataset(Dataset):
    """Yield 17 video frames and 16 actions for TI2V-5B.

    The first latent is the clean input image.  The remaining 16 RGB frames and
    16 actions compress to four future latents, avoiding action truncation when
    the competition submission keeps output frames 0..15.
    """

    load_from_cache = False

    def __init__(
        self,
        root: str,
        height: int = 320,
        width: int = 512,
        repeat: int = 100,
        holdout_count: int = 6,
        seed: int = 0,
        action_mode: str = "delta",
        action_shift: int = 0,
        prompt: str = "A fixed-camera video of a robot arm manipulating objects.",
        split: str = "train",
    ):
        paths = discover_lerobot_so100_datasets(root)
        paths = [p for p in paths if not any(p.endswith(x) for x in DEFAULT_EXCLUDE)]
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        held = set(shuffled[:holdout_count])
        selected = [p for p in paths if (p in held) == (split == "holdout")]
        if not selected:
            raise ValueError(f"No datasets selected for split={split!r}.")

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
        self.action_mode = action_mode
        self.action_shift = action_shift
        self.delta_scale = load_delta_scale(
            str(REPO / "train/delta_action_stats.json"), action_mode
        )
        self.prompt = prompt
        self.selected_paths = selected

    def __len__(self):
        return len(self.base) * self.repeat

    def __getitem__(self, index: int):
        sample = self.base[index % len(self.base)]
        # action[0:16] controls the four future latent groups. Frame 16 is kept
        # during Wan training/decoding solely to satisfy the 4n+1 VAE layout.
        actions = transform_actions(
            sample["act"][:16], self.delta_scale, self.action_mode, self.action_shift
        )
        return {
            "video": _to_pil_frames(sample["video"]),
            "actions": actions,
            "prompt": self.prompt,
        }
