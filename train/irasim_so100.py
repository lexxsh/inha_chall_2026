"""Faithful IRASim Frame-Ada adaptation for the SO-100 challenge.

The released IRASim RT-1 backbone is kept byte-for-byte compatible: 16 video
frames, 15 frame-level actions and the original 7D action MLP.  A separate,
zero-initialized 6D->7D adapter is the only robot-specific model component.
This is deliberately preferable to changing ``embed_state.fc1`` and silently
discarding a public checkpoint tensor.
"""
from __future__ import annotations

import json
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
IRASIM_SRC = REPO / "third_party/IRASim"
for path in (str(IRASIM_SRC), str(KIT_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    LeRobotSO100Dataset,
    discover_lerobot_so100_datasets,
    preprocess_video,
)
# Import the released model file directly. Importing IRASim's ``models``
# package also imports unrelated VDM baselines and their optional dependencies.
_spec = importlib.util.spec_from_file_location(
    "irasim_official_model", IRASIM_SRC / "models/irasim.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load the official IRASim model module")
_irasim_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_irasim_module)
IRASim_models = _irasim_module.IRASim_models

try:
    from data_module import action_dims_for, load_delta_scale  # type: ignore # noqa: E402
except ModuleNotFoundError:
    from train.data_module import action_dims_for, load_delta_scale  # noqa: E402
try:
    from so100_transition_contract import irasim_transition_actions  # type: ignore # noqa: E402
except ModuleNotFoundError:
    from train.so100_transition_contract import irasim_transition_actions  # noqa: E402


DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


class IRASimSO100Dataset(Dataset):
    """Return 16 frames and the 15 deployable actions for frames 1..15.

    The challenge image is exactly ground-truth frame 0.  IRASim prepends a
    learned mask embedding for that source frame, so its action tensor must be
    one element shorter than the video.  Absolute commands are the deployable
    default because challenge inference does not provide source proprioception.
    Relative modes are useful only with an oracle or learned source-state input.
    """

    def __init__(
        self,
        root: str,
        height: int = 256,
        width: int = 320,
        repeat: int = 100,
        holdout_count: int = 6,
        seed: int = 0,
        action_mode: str = "absolute",
        split: str = "train",
    ) -> None:
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
            traj_len=16,
            target_height=height,
            target_width=width,
            # Official IRASim resizes directly to 256x320.  Keeping this exact
            # also preserves the public checkpoint's 32x40 latent geometry.
            pad=None,
            camera_key="auto",
            val_fraction=0.0,
            seed=seed,
            fps=6,
            action_mean=stats["mean"],
            action_std=stats["std"],
            use_all_episodes=True,
            return_state=True,
        )
        self.repeat = repeat
        self.action_mode = action_mode
        self.action_dim = action_dims_for(action_mode)
        self.delta_scale = load_delta_scale(
            str(REPO / "train/delta_action_stats.json"), action_mode
        )
        self.selected_paths = selected

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index % len(self.base)]
        source_state = sample["state"][0]
        if self.base.action_mean is not None and self.base.action_std is not None:
            source_state = (source_state - self.base.action_mean) / self.base.action_std
        actions = irasim_transition_actions(
            sample["act"], source_state, self.action_mode, self.delta_scale
        )
        if actions.shape != (15, self.action_dim):
            raise RuntimeError(f"Expected 15x{self.action_dim} actions, got {tuple(actions.shape)}")
        return {
            "video": sample["video"].permute(1, 0, 2, 3),
            "actions": actions,
            "source_state": sample["state"][0],
        }


def preprocess_irasim_frames(video) -> torch.Tensor:
    """Apply the exact direct 256x320 resize used by the RT-1 recipe."""
    return preprocess_video(video, target_height=256, target_width=320, pad=None)


class SO100IRASim(torch.nn.Module):
    """Official RT-1 IRASim plus a minimal embodiment adapter."""

    def __init__(self, action_dim: int = 6, num_frames: int = 16) -> None:
        super().__init__()
        if num_frames != 16:
            raise ValueError("The released RT-1 checkpoint is a 16-frame model.")
        args = SimpleNamespace(dataset="rt1", final_frame_ada=False)
        self.backbone = IRASim_models["IRASim-XL/2"](
            input_size=[32, 40],
            num_frames=16,
            learn_sigma=False,
            extras=3,
            attention_mode="math",
            args=args,
        )
        self.action_dim = int(action_dim)
        self.action_adapter = torch.nn.Linear(self.action_dim, 7, bias=False)
        # Step zero preserves a single, public-checkpoint-compatible control
        # point.  Gradients through the public action MLP train this projection.
        torch.nn.init.zeros_(self.action_adapter.weight)

    def forward(self, x, t, actions=None, **kwargs):
        if actions is None:
            raise ValueError("SO100IRASim requires actions")
        if actions.ndim != 3 or actions.shape[1:] != (15, self.action_dim):
            raise ValueError(
                f"Expected actions [B,15,{self.action_dim}], got {tuple(actions.shape)}"
            )
        mapped = self.action_adapter(actions)
        return self.backbone(x, t, actions=mapped, **kwargs)

    @property
    def in_channels(self) -> int:
        return self.backbone.in_channels


def build_irasim(action_dim: int = 6, num_frames: int = 16) -> SO100IRASim:
    return SO100IRASim(action_dim=action_dim, num_frames=num_frames)


def checkpoint_state(
    path: str | Path,
    branch: str = "auto",
) -> dict[str, torch.Tensor]:
    # The released checkpoint also stores optimizer state and is ~11 GiB.
    # mmap prevents every DDP rank from eagerly materializing those unused
    # optimizer tensors in host RAM while we copy only model/EMA weights.
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if branch not in {"auto", "ema", "model"}:
        raise ValueError(f"branch must be auto, ema, or model; got {branch!r}")
    selected = "ema" if branch == "auto" and "ema" in raw else branch
    if selected == "auto":
        selected = "model" if "model" in raw else "raw"
    if selected in {"ema", "model"}:
        if selected not in raw:
            raise KeyError(f"Checkpoint has no {selected!r} branch")
        raw = raw[selected]
    return {key.removeprefix("module."): value for key, value in raw.items()}


def load_irasim_checkpoint(
    model: SO100IRASim,
    path: str | Path,
    branch: str = "auto",
) -> dict[str, object]:
    """Load either an exact public backbone or a strict adapted checkpoint."""
    source = checkpoint_state(path, branch=branch)
    if any(key.startswith("backbone.") for key in source):
        missing, unexpected = model.load_state_dict(source, strict=False)
        mismatched: list[str] = []
        target = model.state_dict()
        for key, value in source.items():
            if key in target and value.shape != target[key].shape:
                mismatched.append(key)
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "Adapted IRASim checkpoint must load strictly: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
                f"mismatched={sorted(mismatched)}"
            )
        return {
            "format": "so100_adapted",
            "branch": "ema" if branch == "auto" else branch,
            "matched": len(source),
            "missing": [],
            "unexpected": [],
            "mismatched": [],
        }

    target = model.backbone.state_dict()
    missing = sorted(key for key in target if key not in source)
    unexpected = sorted(key for key in source if key not in target)
    mismatched = sorted(
        key for key in source if key in target and source[key].shape != target[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Public IRASim backbone is not architecture-exact: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    model.backbone.load_state_dict(source, strict=True)
    return {
        "format": "public_rt1",
        "branch": "ema" if branch == "auto" else branch,
        "matched": len(source),
        "missing": ["action_adapter.weight"],
        "unexpected": [],
        "mismatched": [],
    }


def prepare_eval_actions(
    raw,
    action_mode: str = "absolute",
    source_state=None,
) -> torch.Tensor:
    stats = json.loads((REPO / "open/data/train/so100_action_statistics.json").read_text())
    actions = torch.as_tensor(raw, dtype=torch.float32)
    actions = (actions - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])
    scale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), action_mode)
    normalized_source = None
    if source_state is not None:
        normalized_source = (
            torch.as_tensor(source_state, dtype=torch.float32) - torch.tensor(stats["mean"])
        ) / torch.tensor(stats["std"])
    transformed = irasim_transition_actions(
        actions, normalized_source, action_mode, scale
    )
    if transformed.shape[0] != 15:
        raise ValueError(f"Expected 16 raw actions -> 15 IRASim actions, got {tuple(transformed.shape)}")
    return transformed
