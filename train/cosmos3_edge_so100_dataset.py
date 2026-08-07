"""Native Cosmos3-Edge forward-dynamics dataset for the SO-100 challenge.

This module intentionally adapts the data to NVIDIA's released Action SFT
contract instead of introducing another custom conditioning adapter:

* video: uint8 ``[3, 17, H, W]``;
* source frame: frame 0 (clean conditioning frame);
* action: 16 Bridge-domain 10-D EEF transitions;
* mode/domain/viewpoint: forward_dynamics / bridge_orig_lerobot / third-person.

The official ``ActionTransformPipeline`` owns spatial padding, text
tokenization, sequence-plan construction, affine action normalization, and
padding from 10 to the model's 64 action channels.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from cosmos3_so100_action import (  # noqa: E402
    ACTION_STEPS,
    DEFAULT_BRIDGE_STATS,
    IDENTITY_ROT6D,
    SO100Cosmos3ActionConverter,
)


NUM_VIDEO_FRAMES = ACTION_STEPS + 1
EMBODIMENT_TYPE = "bridge_orig_lerobot"
DOMAIN_ID = 7
DEFAULT_PROMPT = "A fixed-camera video of a SO-100 robot arm performing a tabletop manipulation."


def _stack_vectors(series: pd.Series, *, name: str, expected_dim: int = 6) -> np.ndarray:
    try:
        array = np.stack(series.to_numpy()).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(f"Could not stack parquet column {name!r}") from exc
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(f"Expected {name} [T,{expected_dim}], got {array.shape}")
    return array


class SO100ForwardDynamicsRawDataset(Dataset):
    """Map-style SO-100 episode windows with episode-local shuffle blocks."""

    EMBODIMENT_TYPE = EMBODIMENT_TYPE

    def __init__(
        self,
        *,
        root: str | Path,
        index_path: str | Path,
        holdout_manifest: str | Path | None = None,
        fps: float = 6.0,
        chunk_length: int = ACTION_STEPS,
        sample_stride: int = 4,
        prompt: str = DEFAULT_PROMPT,
        split: str = "train",
        viewpoint: str = "third_person_view",
        mode: str = "forward_dynamics",
    ) -> None:
        super().__init__()
        if chunk_length != ACTION_STEPS:
            raise ValueError(f"Cosmos3 SO-100 requires chunk_length=16, got {chunk_length}")
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be positive, got {sample_stride}")
        if viewpoint != "third_person_view":
            raise ValueError("SO-100 challenge cameras are fixed external cameras; use third_person_view")
        if mode != "forward_dynamics":
            raise ValueError("This dataset is intentionally forward_dynamics only")

        self.root = Path(root).resolve()
        self.index_path = Path(index_path).resolve()
        self.fps = float(fps)
        self.chunk_length = int(chunk_length)
        self.sample_stride = int(sample_stride)
        self.prompt = str(prompt)
        self.viewpoint = viewpoint
        self.mode = mode
        self.converter = SO100Cosmos3ActionConverter()

        episodes = pd.read_parquet(self.index_path)
        required = {"dataset", "episode_index", "num_frames", "parquet", "video", "split"}
        missing = required - set(episodes.columns)
        if missing:
            raise ValueError(f"Episode index is missing columns: {sorted(missing)}")
        episodes = episodes[episodes["split"] == split].copy()

        held_episodes: set[tuple[str, int]] = set()
        if holdout_manifest is not None:
            manifest_path = Path(holdout_manifest).resolve()
            if manifest_path.is_file():
                records = json.loads(manifest_path.read_text())
                held_episodes = {(str(row["dataset"]), int(row["episode"])) for row in records}
        if held_episodes:
            held_mask = [
                (str(row.dataset), int(row.episode_index)) in held_episodes
                for row in episodes.itertuples(index=False)
            ]
            episodes = episodes.loc[~np.asarray(held_mask, dtype=bool)].copy()

        counts = np.maximum(
            0,
            (episodes["num_frames"].to_numpy(dtype=np.int64) - NUM_VIDEO_FRAMES) // self.sample_stride + 1,
        )
        keep = counts > 0
        self.episodes = episodes.loc[keep].reset_index(drop=True)
        self.counts = counts[keep]
        self.cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(self.counts)])
        self.held_episodes = held_episodes
        if len(self) == 0:
            raise ValueError("No valid 17-frame SO-100 windows remain after filtering")

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [
            (int(self.cumulative[index]), int(count))
            for index, count in enumerate(self.counts)
            if int(count) > 0
        ]

    def resolve_index(self, index: int) -> tuple[pd.Series, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.cumulative, index, side="right") - 1)
        local_window = int(index - self.cumulative[episode_index])
        start = local_window * self.sample_stride
        return self.episodes.iloc[episode_index], start

    @lru_cache(maxsize=1)
    def _decode_episode(self, path_string: str) -> np.ndarray:
        frames: list[np.ndarray] = []
        with av.open(path_string) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError(f"Decoded no frames from {path_string}")
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f"Episode contains mixed frame shapes: {path_string}: {sorted(shapes)}")
        return np.stack(frames)

    @lru_cache(maxsize=1)
    def _read_episode_table(self, path_string: str) -> tuple[np.ndarray, np.ndarray]:
        table = pd.read_parquet(path_string, columns=["action", "observation.state"])
        actions = _stack_vectors(table["action"], name="action")
        states = _stack_vectors(table["observation.state"], name="observation.state")
        return actions, states

    @staticmethod
    def _idle_frames(raw_action: np.ndarray) -> int:
        translation_idle = np.linalg.norm(raw_action[:, :3], axis=-1) < 2.0e-3
        rotation_idle = np.linalg.norm(raw_action[:, 3:9] - IDENTITY_ROT6D[None], axis=-1) < 2.0e-2
        gripper = raw_action[:, 9]
        gripper_delta = np.abs(np.diff(gripper, prepend=gripper[0]))
        return int(np.sum(translation_idle & rotation_idle & (gripper_delta < 1.0e-2)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, start = self.resolve_index(index)
        video_path = self.root / str(record.video)
        parquet_path = self.root / str(record.parquet)
        episode_video = self._decode_episode(str(video_path))
        episode_actions, episode_states = self._read_episode_table(str(parquet_path))

        stop = start + NUM_VIDEO_FRAMES
        frames = episode_video[start:stop]
        if len(frames) != NUM_VIDEO_FRAMES:
            raise ValueError(f"{video_path}: expected 17 frames at {start}, got {len(frames)}")
        if stop - 1 > len(episode_actions) or start >= len(episode_states):
            raise ValueError(f"Parquet/video length mismatch for {parquet_path}")

        commands = episode_actions[start : start + ACTION_STEPS]
        source_state = episode_states[start]
        raw_action = self.converter.build_raw_actions(commands, source_action=source_state)
        video = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).contiguous()
        return {
            "ai_caption": self.prompt,
            "video": video,
            "action": torch.from_numpy(raw_action),
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.float32),
            "mode": self.mode,
            "domain_id": torch.tensor(DOMAIN_ID, dtype=torch.long),
            "viewpoint": self.viewpoint,
            "idle_frames": torch.tensor(self._idle_frames(raw_action), dtype=torch.long),
            "dataset_name": str(record.dataset),
            "episode_index": torch.tensor(int(record.episode_index), dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
        }


class _NormalizedActionSFTDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Any, resolution: str | int, normalizer: Any) -> None:
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution
        self._normalizer = normalizer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._transform(self._dataset[index], self._resolution, action_normalizer=self._normalizer)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return self._dataset.get_shuffle_blocks()


def get_action_so100_edge_sft_dataset(
    *,
    root: str,
    index_path: str,
    holdout_manifest: str | None = None,
    fps: float = 6.0,
    chunk_length: int = ACTION_STEPS,
    sample_stride: int = 4,
    embodiment_type: str = EMBODIMENT_TYPE,
    mode: str = "forward_dynamics",
    viewpoint: str = "third_person_view",
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
) -> Dataset:
    """Build the official ActionTransformPipeline around SO-100 raw windows."""
    if embodiment_type != EMBODIMENT_TYPE:
        raise ValueError(f"Only {EMBODIMENT_TYPE!r} is supported, got {embodiment_type!r}")
    from cosmos_framework.data.generator.action.action_processing import (
        ActionAffineNormalization,
    )
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
        ActionIterableShuffleDataset,
    )
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    raw_dataset = SO100ForwardDynamicsRawDataset(
        root=root,
        index_path=index_path,
        holdout_manifest=holdout_manifest,
        fps=fps,
        chunk_length=chunk_length,
        sample_stride=sample_stride,
        prompt=prompt,
        viewpoint=viewpoint,
        mode=mode,
    )
    stats = json.loads(Path(DEFAULT_BRIDGE_STATS).read_text())
    q01 = torch.tensor(stats["q01"], dtype=torch.float32)
    q99 = torch.tensor(stats["q99"], dtype=torch.float32)
    normalizer = ActionAffineNormalization(offset=(q01 + q99) / 2, scale=(q99 - q01) / 2)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        idle_frames_dropout=0.05,
        format_prompt_as_json=True,
    )
    dataset: Dataset = _NormalizedActionSFTDataset(raw_dataset, transform, resolution, normalizer)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(dataset, seed=episode_shuffle_seed)
    return dataset


get_action_so100_edge_sft_dataset.EMBODIMENT_TYPE = EMBODIMENT_TYPE


__all__ = [
    "DEFAULT_PROMPT",
    "DOMAIN_ID",
    "EMBODIMENT_TYPE",
    "NUM_VIDEO_FRAMES",
    "SO100ForwardDynamicsRawDataset",
    "get_action_so100_edge_sft_dataset",
]
