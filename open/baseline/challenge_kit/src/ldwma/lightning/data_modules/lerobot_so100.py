from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset


class SO100DataModule(LightningDataModule):
    def __init__(
        self,
        root: str | None,
        dataset_paths: Sequence[str] | str,
        batch_size: int = 8,
        target_height: int = 320,
        target_width: int = 512,
        traj_len: int = 16,
        pad: bool = True,
        camera_key: str = "auto",
        val_fraction: float = 0.05,
        seed: int = 0,
        downsample: int = 1,
        use_language: bool = False,
        fps: int | None = None,
        frame_stride: int | None = None,
        normalize_actions: bool = True,
        action_stats_path: str | None = None,
        remote: bool = False,
        repo_id: str | None = None,
        cache_dir: str | None = None,
        temporary_downloads: bool = False,
        hf_token: str | None = None,
        eval_all_episodes: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.traj_len = traj_len
        self.pad = pad
        self.camera_key = camera_key
        self.val_fraction = val_fraction
        self.seed = seed
        self.downsample = downsample
        self.use_language = use_language
        self.fps = fps
        self.frame_stride = frame_stride
        self.normalize_actions = normalize_actions
        self.action_stats_path = action_stats_path
        self.remote = remote
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.temporary_downloads = temporary_downloads
        self.hf_token = hf_token
        self.eval_all_episodes = eval_all_episodes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None

    def _default_action_stats_path(self) -> Path:
        if self.root is not None:
            return Path(self.root) / "so100_action_statistics.json"
        if self.cache_dir is not None:
            return Path(self.cache_dir) / "so100_action_statistics.json"
        return Path("so100_action_statistics.json")

    def _load_or_compute_action_stats(self) -> dict:
        stats_path = Path(self.action_stats_path) if self.action_stats_path else self._default_action_stats_path()
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        stats = self.train_dataset.compute_action_statistics()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    def setup(self, stage=None):
        self.train_dataset = LeRobotSO100Dataset(
            root=self.root,
            dataset_paths=self.dataset_paths,
            train=True,
            traj_len=self.traj_len,
            target_height=self.target_height,
            target_width=self.target_width,
            pad=self.pad,
            camera_key=self.camera_key,
            val_fraction=self.val_fraction,
            seed=self.seed,
            downsample=self.downsample,
            use_language=self.use_language,
            fps=self.fps,
            frame_stride=self.frame_stride,
            remote=self.remote,
            repo_id=self.repo_id,
            cache_dir=self.cache_dir,
            temporary_downloads=self.temporary_downloads,
            hf_token=self.hf_token,
        )
        action_stats = self._load_or_compute_action_stats() if self.normalize_actions else None
        if action_stats is not None:
            self.train_dataset.set_action_stats(action_stats["mean"], action_stats["std"])

        val_dataset_paths = self.train_dataset.dataset_paths if self.dataset_paths == "auto" else self.dataset_paths
        self.val_dataset = LeRobotSO100Dataset(
            root=self.root,
            dataset_paths=val_dataset_paths,
            train=False,
            traj_len=self.traj_len,
            target_height=self.target_height,
            target_width=self.target_width,
            pad=self.pad,
            camera_key=self.camera_key,
            val_fraction=self.val_fraction,
            seed=self.seed,
            downsample=self.downsample,
            use_language=self.use_language,
            fps=self.fps,
            frame_stride=self.frame_stride,
            action_mean=action_stats["mean"] if action_stats is not None else None,
            action_std=action_stats["std"] if action_stats is not None else None,
            remote=self.remote,
            repo_id=self.repo_id,
            cache_dir=self.cache_dir,
            temporary_downloads=self.temporary_downloads,
            hf_token=self.hf_token,
            use_all_episodes=self.eval_all_episodes,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
