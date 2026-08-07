"""Faithful SO-100 adaptation of the released Ctrl-World data contract.

Ctrl-World does *not* condition its video backbone on arbitrary action tokens.
It conditions every frame on the robot pose associated with that frame.  The
released DROID recipe uses six sparse history frames followed by the current
frame and four future frames.  This module preserves that 6+5 layout and only
changes the embodiment representation from a 7-D Franka Cartesian pose to the
native 6-D SO-100 joint state/target.

The heavyweight network and diffusion loss remain in the official
``third_party/Ctrl-World`` implementation.  Keeping the data contract here
makes the unavoidable competition-specific changes explicit and auditable.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[1]
CTRL_WORLD = REPO / "third_party/Ctrl-World"
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
for dependency in (str(CTRL_WORLD), str(KIT_SRC)):
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    _format_lerobot_path,
    _read_json,
    _read_jsonl,
    _select_video_keys,
    discover_lerobot_so100_datasets,
)


HISTORY_FRAMES = 6
PREDICTION_FRAMES = 5  # current reconstruction + four genuinely future frames
TOKENS_PER_SAMPLE = HISTORY_FRAMES + PREDICTION_FRAMES
ACTION_DIM = 6
FUTURE_PER_CHUNK = PREDICTION_FRAMES - 1
SUBMISSION_FRAMES = 16
SUBMISSION_ACTIONS = 16
DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


def safe_component(value: str) -> str:
    """Create a stable readable filesystem component for a LeRobot camera key."""
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "camera"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def cache_path(
    cache_root: str | Path,
    dataset_path: str,
    camera_key: str,
    episode_index: int,
) -> Path:
    return (
        Path(cache_root)
        / dataset_path
        / safe_component(camera_key)
        / f"episode_{episode_index:06d}.pt"
    )


@dataclass(frozen=True)
class EpisodeRecord:
    dataset_path: str
    episode_index: int
    camera_key: str
    length: int
    task: str
    data_path: Path
    video_path: Path
    latent_path: Path


def split_dataset_paths(
    data_root: str | Path,
    *,
    holdout_count: int = 8,
    seed: int = 20260806,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    holdout_manifest: str | Path | None = REPO / "valset_holdout/manifest.json",
) -> tuple[list[str], list[str]]:
    """Dataset-level split used by both training and all local gates.

    The frozen local validation manifest takes precedence over random splitting;
    otherwise a candidate can silently train on the very uploader/dataset used
    by the gate.  Additional datasets are deterministically held out until
    ``holdout_count`` is reached.
    """
    paths = discover_lerobot_so100_datasets(data_root)
    paths = [path for path in paths if not any(path.endswith(item) for item in exclude)]
    fixed: set[str] = set()
    if holdout_manifest is not None and Path(holdout_manifest).is_file():
        payload = json.loads(Path(holdout_manifest).read_text())
        fixed = {str(row["dataset"]) for row in payload}
        missing = fixed - set(paths)
        if missing:
            raise ValueError(f"Holdout manifest references undiscovered datasets: {sorted(missing)}")
    order = [path for path in paths if path not in fixed]
    random.Random(seed).shuffle(order)
    needed = max(0, holdout_count - len(fixed))
    held = fixed | set(order[:needed])
    return [path for path in paths if path not in held], [path for path in paths if path in held]


def discover_episode_records(
    data_root: str | Path,
    cache_root: str | Path,
    dataset_paths: Iterable[str],
    *,
    camera_key: str = "auto",
) -> list[EpisodeRecord]:
    root = Path(data_root)
    records: list[EpisodeRecord] = []
    for dataset_path in dataset_paths:
        dataset_root = root / dataset_path
        info = _read_json(dataset_root / "meta/info.json")
        episodes = _read_jsonl(dataset_root / "meta/episodes.jsonl")
        cameras = _select_video_keys(info["features"], camera_key)
        chunks_size = int(info.get("chunks_size", 1000))
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            task = episode.get("tasks", [""])[0]
            data_rel = _format_lerobot_path(
                info["data_path"], episode_index, chunks_size
            )
            for selected_camera in cameras:
                video_rel = _format_lerobot_path(
                    info["video_path"], episode_index, chunks_size, selected_camera
                )
                records.append(
                    EpisodeRecord(
                        dataset_path=dataset_path,
                        episode_index=episode_index,
                        camera_key=selected_camera,
                        length=int(episode["length"]),
                        task=task,
                        data_path=dataset_root / data_rel,
                        video_path=dataset_root / video_rel,
                        latent_path=cache_path(
                            cache_root, dataset_path, selected_camera, episode_index
                        ),
                    )
                )
    return records


def load_pose_statistics(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(Path(path).read_text())
    low = torch.tensor(payload["state_p01"], dtype=torch.float32)
    high = torch.tensor(payload["state_p99"], dtype=torch.float32)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError(f"Expected six-dimensional percentiles, got {low.shape}/{high.shape}")
    if not torch.all(high > low):
        raise ValueError("Every state_p99 entry must be greater than state_p01")
    return low, high


def normalize_pose(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Exact percentile-to-[-1,1] normalization used by released Ctrl-World."""
    value = torch.as_tensor(value, dtype=torch.float32)
    low = low.to(device=value.device, dtype=value.dtype)
    high = high.to(device=value.device, dtype=value.dtype)
    return (2.0 * (value - low) / (high - low).clamp_min(1e-8) - 1.0).clamp(-1.0, 1.0)


def ctrl_world_args(
    *,
    svd_model_path: str | Path,
    action_dim: int = ACTION_DIM,
    width: int = 320,
    height: int = 192,
) -> SimpleNamespace:
    """Arguments consumed by the official ``CrtlWorld`` implementation.

    Text is intentionally disabled: training captions exist, but competition
    inference supplies no instruction.  All other architecture/loss settings
    are the released defaults.
    """
    return SimpleNamespace(
        svd_model_path=str(Path(svd_model_path)),
        clip_model_path=None,
        action_dim=int(action_dim),
        num_history=HISTORY_FRAMES,
        num_frames=PREDICTION_FRAMES,
        text_cond=False,
        frame_level_cond=True,
        his_cond_zero=False,
        motion_bucket_id=127,
        fps=7,
        guidance_scale=1.0,
        num_inference_steps=50,
        decode_chunk_size=5,
        width=int(width),
        height=int(height),
    )


class CtrlWorldSO100LatentDataset(Dataset):
    """Cached single-view clips in the official 6-history + 5-frame layout."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        cache_root: str | Path,
        stats_path: str | Path,
        split: str = "train",
        holdout_count: int = 8,
        split_seed: int = 20260806,
        sample_seed: int = 0,
        repeat: int = 1,
        camera_key: str = "auto",
        missing: str = "error",
    ) -> None:
        train_paths, holdout_paths = split_dataset_paths(
            data_root, holdout_count=holdout_count, seed=split_seed
        )
        if split not in {"train", "holdout"}:
            raise ValueError("split must be 'train' or 'holdout'")
        selected = train_paths if split == "train" else holdout_paths
        records = discover_episode_records(
            data_root, cache_root, selected, camera_key=camera_key
        )
        self.missing_records = [record for record in records if not record.latent_path.is_file()]
        if self.missing_records and missing == "error":
            examples = "\n".join(str(row.latent_path) for row in self.missing_records[:5])
            raise FileNotFoundError(
                f"{len(self.missing_records)} Ctrl-World latent caches are missing. "
                f"Run PHASE=cache first. Examples:\n{examples}"
            )
        if missing not in {"error", "skip"}:
            raise ValueError("missing must be 'error' or 'skip'")
        self.records = [record for record in records if record.latent_path.is_file()]
        if not self.records:
            raise ValueError(f"No cached records available for split={split}")
        self.low, self.high = load_pose_statistics(stats_path)
        self.repeat = int(repeat)
        self.sample_seed = int(sample_seed)
        self.split = split
        self.dataset_paths = selected

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def _sample_indices(self, record: EpisodeRecord, index: int) -> list[int]:
        # Official loader samples future stride 1 or 2 and history stride 4x
        # larger, with 15% collapsed history for first-rollout robustness.
        rng = random.Random(self.sample_seed + index * 1_000_003)
        available_strides = [
            stride
            for stride in (1, 2)
            if record.length > FUTURE_PER_CHUNK * stride
        ]
        if not available_strides:
            raise ValueError(f"Episode too short: {record.latent_path} ({record.length})")
        future_stride = rng.choice(available_strides)
        future_extent = FUTURE_PER_CHUNK * future_stride
        current = rng.randint(0, record.length - future_extent - 1)
        history_stride = future_stride * 4
        if rng.random() < 0.15:
            history_stride = 0
        history = [current - i * history_stride for i in range(HISTORY_FRAMES, 0, -1)]
        future = [current + i * future_stride for i in range(PREDICTION_FRAMES)]
        return [max(0, min(record.length - 1, item)) for item in history + future]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # A few community videos advertise a full episode in LeRobot metadata
        # but decode to only one or two frames.  Cache preprocessing records the
        # actual decoded length.  Deterministically advance to the next usable
        # record instead of killing every DDP rank mid-epoch.
        latent = states = record = None
        for offset in range(len(self.records)):
            candidate = self.records[(index + offset) % len(self.records)]
            payload = torch.load(
                candidate.latent_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            candidate_latent = payload["latent"] if isinstance(payload, dict) else payload
            candidate_states = payload.get("state") if isinstance(payload, dict) else None
            if candidate_states is None:
                raise ValueError(f"Cache lacks SO100 states: {candidate.latent_path}")
            usable = min(len(candidate_latent), len(candidate_states), candidate.length)
            if usable <= FUTURE_PER_CHUNK:
                continue
            record = EpisodeRecord(**{**candidate.__dict__, "length": usable})
            latent = candidate_latent
            states = candidate_states
            break
        if record is None or latent is None or states is None:
            raise ValueError("No Ctrl-World cache contains current + four future frames")
        indices = self._sample_indices(record, index)
        latent = latent[indices].float()
        states = normalize_pose(states[indices].float(), self.low, self.high)
        expected_latent_prefix = (TOKENS_PER_SAMPLE, 4)
        if latent.shape[:2] != expected_latent_prefix or states.shape != (
            TOKENS_PER_SAMPLE,
            ACTION_DIM,
        ):
            raise ValueError(
                f"Bad cache contract for {record.latent_path}: "
                f"latent={tuple(latent.shape)}, state={tuple(states.shape)}"
            )
        return {
            "latent": latent,
            "action": states,
            "text": "",
            "episode": f"{record.dataset_path}:{record.episode_index}:{record.camera_key}",
            "indices": torch.tensor(indices, dtype=torch.long),
        }


def build_rollout_pose_tokens(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    source_pose: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build four released-layout pose chunks for one challenge example.

    Returns normalized ``[11,6]`` condition chunks and the four future command
    groups associated with the generated frames.  Output frame zero of every
    Ctrl-World chunk reconstructs the current observation and is not appended
    to the submission.
    """
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.shape != (SUBMISSION_ACTIONS, ACTION_DIM):
        raise ValueError(f"Expected challenge actions [16,6], got {tuple(actions.shape)}")
    normalized = normalize_pose(actions, low, high)
    source_proxy = (
        normalized[0]
        if source_pose is None
        else normalize_pose(torch.as_tensor(source_pose), low, high)
    )
    if source_proxy.shape != (ACTION_DIM,):
        raise ValueError(f"Expected source pose [6], got {tuple(source_proxy.shape)}")
    pose_history: list[torch.Tensor] = [source_proxy.clone() for _ in range(24)]
    chunks: list[torch.Tensor] = []
    futures: list[torch.Tensor] = []
    history_offsets = (0, 0, -8, -6, -4, -2)  # released replay script
    for chunk_index in range(4):
        start = chunk_index * FUTURE_PER_CHUNK
        stop = start + FUTURE_PER_CHUNK
        future = normalized[start:stop]
        current = source_proxy if chunk_index == 0 else normalized[start - 1]
        history = torch.stack([pose_history[offset] for offset in history_offsets])
        condition = torch.cat([history, current[None], future], dim=0)
        if condition.shape != (TOKENS_PER_SAMPLE, ACTION_DIM):
            raise RuntimeError(f"Internal rollout alignment error: {condition.shape}")
        chunks.append(condition)
        futures.append(future)
        pose_history.append(future[-1].clone())
    return chunks, futures


def load_ctrl_world_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    *,
    allow_action_input_mismatch: bool = True,
) -> dict:
    """Load official/full checkpoints while documenting every skipped tensor."""
    path = Path(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    own = model.state_dict()
    compatible = {}
    skipped = {}
    unexpected = []
    for name, value in state.items():
        canonical = name.removeprefix("module.")
        if canonical not in own:
            unexpected.append(canonical)
            continue
        if own[canonical].shape != value.shape:
            skipped[canonical] = {
                "checkpoint": list(value.shape),
                "model": list(own[canonical].shape),
            }
            continue
        compatible[canonical] = value
    if skipped and not allow_action_input_mismatch:
        raise ValueError(f"Checkpoint shape mismatches: {skipped}")
    illegal = [name for name in skipped if name != "action_encoder.action_encode.0.weight"]
    if illegal:
        raise ValueError(f"Unexpected checkpoint shape mismatches: {illegal}")
    missing, load_unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(path.resolve()),
        "loaded_tensors": len(compatible),
        "skipped_shape": skipped,
        "missing": list(missing),
        "unexpected": sorted(set(unexpected) | set(load_unexpected)),
    }
