"""SO-100 data contract and model helpers for Boundless World Model (BWM).

This module deliberately keeps the competition-specific action conversion in
one place.  BWM consumes one action token per RGB frame (17 tokens for a
17-frame Wan clip), while the challenge provides 16 future actions with six
normalized SO-100 joint coordinates.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
DIFFSYNTH = REPO / "third_party/DiffSynth-Studio"
BWM_REPO = REPO / "third_party/boundless-world-model"
for dependency in (str(BWM_REPO), str(DIFFSYNTH), str(KIT_SRC)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    LeRobotSO100Dataset,
    discover_lerobot_so100_datasets,
)

try:  # Works both as ``python train/foo.py`` and as a package import.
    from data_module import load_delta_scale  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    from train.data_module import load_delta_scale  # noqa: E402


DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)
BWM_ACTION_DIM = 18
BWM_ACTION_FRAMES = 17
SO100_FUTURE_ACTIONS = 16
BWM_INPUT_CLIP_STD = 3.0


def action_statistics(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    stats = json.loads((Path(root) / "so100_action_statistics.json").read_text())
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32).clamp(min=1e-6)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(f"Expected six-dimensional SO100 statistics, got {mean.shape}/{std.shape}")
    return mean, std


def normalize_so100_actions(raw: torch.Tensor, root: str | Path) -> torch.Tensor:
    """Normalize raw joint targets exactly as the official training dataset does."""
    mean, std = action_statistics(root)
    return (raw.float() - mean.to(raw.device)) / std.to(raw.device)


def build_bwm_action_tokens(
    future_actions: torch.Tensor,
    delta_scale: torch.Tensor,
    *,
    source_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert normalized ``[16,6]`` SO100 actions to BWM ``[17,18]`` tokens.

    Each token is ``[absolute joint target, source-relative delta, step delta]``.
    Token zero describes the observed source frame and therefore has zero
    deltas.  Tokens 1..16 describe the supplied future actions.  Keeping the
    source action separate matters for the batch-roll counterfactual: the
    source proxy remains tied to the source image while only future control is
    swapped.
    """
    future_actions = torch.as_tensor(future_actions, dtype=torch.float32)
    if future_actions.shape != (SO100_FUTURE_ACTIONS, 6):
        raise ValueError(
            f"Expected future actions [{SO100_FUTURE_ACTIONS},6], got {tuple(future_actions.shape)}"
        )
    if source_action is None:
        source_action = future_actions[0]
    source_action = torch.as_tensor(
        source_action, dtype=future_actions.dtype, device=future_actions.device
    )
    if source_action.shape != (6,):
        raise ValueError(f"Expected source action [6], got {tuple(source_action.shape)}")
    delta_scale = torch.as_tensor(
        delta_scale, dtype=future_actions.dtype, device=future_actions.device
    )
    if delta_scale.shape != (12,):
        raise ValueError(f"Expected hybrid delta scale [12], got {tuple(delta_scale.shape)}")

    anchor_scale = delta_scale[:6].clamp(min=1e-6)
    step_scale = delta_scale[6:].clamp(min=1e-6)
    anchor_delta = (future_actions - source_action) / anchor_scale
    previous = torch.cat([source_action[None], future_actions[:-1]], dim=0)
    step_delta = (future_actions - previous) / step_scale
    # Public BWM uses percentile normalization into [-1, 1]. SO100's official
    # dataset instead supplies z-scores, so use the stable three-sigma analogue
    # rather than feeding unbounded 5+ sigma values into BWM's action MLP.
    future_tokens = torch.cat([future_actions, anchor_delta, step_delta], dim=-1)
    future_tokens = future_tokens.clamp(-BWM_INPUT_CLIP_STD, BWM_INPUT_CLIP_STD)
    future_tokens = future_tokens / BWM_INPUT_CLIP_STD
    source_absolute = source_action.clamp(-BWM_INPUT_CLIP_STD, BWM_INPUT_CLIP_STD)
    source_absolute = source_absolute / BWM_INPUT_CLIP_STD
    source_token = torch.cat([source_absolute, source_action.new_zeros(12)], dim=-1)
    tokens = torch.cat([source_token[None], future_tokens], dim=0)
    if tokens.shape != (BWM_ACTION_FRAMES, BWM_ACTION_DIM) or not torch.isfinite(tokens).all():
        raise RuntimeError(f"Invalid BWM action tokens: {tuple(tokens.shape)}, finite={torch.isfinite(tokens).all()}")
    return tokens


class BWMSO100Dataset(Dataset):
    """Dataset-level holdout and exact BWM tensor/action contract."""

    load_from_cache = False

    def __init__(
        self,
        root: str,
        height: int = 384,
        width: int = 512,
        repeat: int = 1,
        holdout_count: int = 8,
        seed: int = 0,
        split: str = "train",
    ) -> None:
        paths = discover_lerobot_so100_datasets(root)
        paths = [p for p in paths if not any(p.endswith(x) for x in DEFAULT_EXCLUDE)]
        order = list(paths)
        random.Random(seed).shuffle(order)
        held = set(order[:holdout_count])
        selected = [p for p in paths if (p in held) == (split == "holdout")]
        if not selected:
            raise ValueError(f"No SO100 datasets selected for split={split!r}")

        mean, std = action_statistics(root)
        self.base = LeRobotSO100Dataset(
            root=root,
            dataset_paths=selected,
            train=True,
            traj_len=BWM_ACTION_FRAMES,
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
        self.repeat = int(repeat)
        self.delta_scale = load_delta_scale(
            str(REPO / "train/delta_action_stats.json"), "hybrid"
        )
        self.selected_paths = selected
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index % len(self.base)]
        video = sample["video"].float()
        if video.shape != (3, BWM_ACTION_FRAMES, self.height, self.width):
            raise ValueError(f"Unexpected SO100 video shape: {tuple(video.shape)}")
        actions = sample["act"][:SO100_FUTURE_ACTIONS].float()
        tokens = build_bwm_action_tokens(actions, self.delta_scale)
        # BWM handles views explicitly: V,C,T,H,W.  The challenge has one view.
        return {"video": video.unsqueeze(0), "action": tokens.unsqueeze(0)}


def expand_bwm_action_encoder(action_encoder: nn.Module, action_dim: int = BWM_ACTION_DIM) -> None:
    """Replace only BWM's two action input projections (14D -> SO100 18D).

    The public checkpoint's deeper action layers are retained.  New input
    weights start at zero so the architecture cannot inject random motion at
    step zero; the pretrained biases and downstream layers remain intact and
    the complete action MLP is then optimized.
    """
    old_dim = int(action_encoder.action_dim)
    if old_dim == action_dim:
        return
    if old_dim != 14:
        raise ValueError(f"Expected public BWM action_dim=14 before expansion, got {old_dim}")

    def replace_linear(container: nn.Sequential, new_in_features: int) -> None:
        old = container[0]
        if not isinstance(old, nn.Linear):
            raise TypeError(f"Expected first action layer to be Linear, got {type(old)}")
        new = nn.Linear(
            new_in_features,
            old.out_features,
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            new.weight.zero_()
            if old.bias is not None:
                new.bias.copy_(old.bias)
        container[0] = new

    replace_linear(action_encoder.action_mlp1, action_dim)
    replace_linear(action_encoder.action_mlp2, action_dim * 4)
    action_encoder.action_dim = int(action_dim)


def local_wan22_model_paths(model_root: str | Path) -> tuple[list[str], str]:
    root = Path(model_root).resolve()
    shards = sorted(str(path) for path in root.glob("diffusion_pytorch_model-*.safetensors"))
    vae = root / "Wan2.2_VAE.pth"
    missing: list[str] = []
    if len(shards) != 3:
        missing.append(f"{root}/diffusion_pytorch_model-*.safetensors (found {len(shards)}, expected 3)")
    if not vae.is_file():
        missing.append(str(vae))
    if missing:
        raise FileNotFoundError("Missing local Wan2.2 assets:\n  " + "\n  ".join(missing))
    return shards, str(vae)


def bwm_trainable_summary(module: nn.Module) -> dict:
    action_names: list[str] = []
    lora_names: list[str] = []
    unexpected: list[str] = []
    blocks: set[int] = set()
    total = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        total += parameter.numel()
        if ".action_mlp1." in name or ".action_mlp2." in name:
            action_names.append(name)
        elif "lora_" in name:
            lora_names.append(name)
            marker = ".blocks."
            if marker in name:
                suffix = name.split(marker, 1)[1]
                try:
                    blocks.add(int(suffix.split(".", 1)[0]))
                except ValueError:
                    pass
        else:
            unexpected.append(name)
    return {
        "trainable_parameters": total,
        "action_parameter_tensors": len(action_names),
        "lora_parameter_tensors": len(lora_names),
        "lora_blocks": sorted(blocks),
        "unexpected_trainables": unexpected,
    }
