from __future__ import annotations

import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import av
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import HfFolder, hf_hub_download, hf_hub_url, list_repo_files
from torch.utils.data import Dataset


EXCLUDED_AUTO_CAMERA_NAME_PARTS = ("wrist", "gripper", "arm")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_lerobot_path(template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    chunk_index = episode_index // chunks_size
    return template.format(
        episode_index=episode_index,
        episode_chunk=chunk_index,
        chunk_index=chunk_index,
        file_index=episode_index,
        video_key=video_key,
    )


def _get_hf_token(token: str | None = None) -> str | None:
    return token or HfFolder.get_token()


@contextmanager
def _materialize_hf_file(
    repo_id: str,
    filename: str,
    cache_dir: str | None = None,
    token: str | None = None,
    temporary_downloads: bool = False,
) -> Iterator[Path]:
    if not temporary_downloads:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=_get_hf_token(token),
        )
        yield Path(path)
        return

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    resolved_token = _get_hf_token(token)
    headers = {"authorization": f"Bearer {resolved_token}"} if resolved_token else None
    fd, temp_path = tempfile.mkstemp(suffix=Path(filename).suffix)
    os.close(fd)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        yield Path(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass




def _is_excluded_auto_camera_key(camera_key: str) -> bool:
    camera_name = camera_key[len("observation.") :] if camera_key.startswith("observation.") else camera_key
    camera_name = camera_name.lower()
    return any(part in camera_name for part in EXCLUDED_AUTO_CAMERA_NAME_PARTS)


def _select_video_keys(features: dict, requested_key: str | None) -> list[str]:
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    if not video_keys:
        raise ValueError("No video feature found in LeRobot metadata.")
    if requested_key and requested_key != "auto":
        requested_keys = [key.strip() for key in requested_key.split(",") if key.strip()]
        missing_keys = [key for key in requested_keys if key not in video_keys]
        if missing_keys:
            raise ValueError(f"Requested camera_key={missing_keys!r}, available video keys={video_keys}.")
        return requested_keys

    selected_keys = [key for key in video_keys if not _is_excluded_auto_camera_key(key)]
    if not selected_keys:
        raise ValueError(f"No non-wrist/gripper/arm video keys available. video keys={video_keys}.")
    return selected_keys


def _select_video_key(features: dict, requested_key: str | None) -> str:
    return _select_video_keys(features, requested_key)[0]

def discover_lerobot_so100_datasets(root: str | Path) -> list[str]:
    root = Path(root)
    dataset_paths = []
    for info_path in root.glob("*/*/meta/info.json"):
        rel_dataset_path = info_path.parent.parent.relative_to(root)
        try:
            info = _read_json(info_path)
            action_shape = info.get("features", {}).get("action", {}).get("shape")
            _select_video_key(info.get("features", {}), "auto")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if action_shape == [6]:
            dataset_paths.append(rel_dataset_path.as_posix())
    if not dataset_paths:
        raise ValueError(f"No SO-100 LeRobot datasets with 6D actions found under {root}.")
    return sorted(dataset_paths)


def discover_remote_lerobot_so100_datasets(
    repo_id: str,
    cache_dir: str | None = None,
    token: str | None = None,
) -> list[str]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=_get_hf_token(token))
    info_files = [filename for filename in files if filename.endswith("/meta/info.json")]
    dataset_paths = []
    for info_filename in info_files:
        rel_dataset_path = info_filename[: -len("/meta/info.json")]
        try:
            with _materialize_hf_file(repo_id, info_filename, cache_dir=cache_dir, token=token) as info_path:
                info = _read_json(info_path)
            action_shape = info.get("features", {}).get("action", {}).get("shape")
            has_video = any(value.get("dtype") == "video" for value in info.get("features", {}).values())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if action_shape == [6] and has_video:
            dataset_paths.append(rel_dataset_path)
    if not dataset_paths:
        raise ValueError(f"No SO-100 LeRobot datasets with 6D actions found in Hugging Face repo {repo_id}.")
    return sorted(dataset_paths)


def _decode_video_clip(video_path: Path, indices: list[int]) -> np.ndarray:
    wanted = set(indices)
    frames = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                frames[frame_idx] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break

    missing = [idx for idx in indices if idx not in frames]
    if missing:
        raise IndexError(f"Video {video_path} did not contain requested frames {missing[:5]}.")
    return np.stack([frames[idx] for idx in indices], axis=0)


def preprocess_video(
    video: np.ndarray,
    target_height: int,
    target_width: int,
    pad: bool | None,
) -> torch.Tensor:
    """Resize and normalize an RGB clip.

    ``pad=True`` preserves aspect ratio with letterboxing, ``pad=False``
    preserves aspect ratio with a center crop, and ``pad=None`` performs a
    direct resize.  The last mode matches IRASim's released RT-1 recipe.
    """
    video_tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2) / 255.0
    if pad is None:
        video_tensor = F.interpolate(
            video_tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    elif pad:
        _, _, height, width = video_tensor.shape
        scale = min(target_height / height, target_width / width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        video_tensor = F.interpolate(
            video_tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        pad_top = (target_height - resized_height) // 2
        pad_bottom = target_height - resized_height - pad_top
        pad_left = (target_width - resized_width) // 2
        pad_right = target_width - resized_width - pad_left
        video_tensor = F.pad(video_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    else:
        _, _, height, width = video_tensor.shape
        scale = max(target_height / height, target_width / width)
        resized_height = max(target_height, round(height * scale))
        resized_width = max(target_width, round(width * scale))
        video_tensor = F.interpolate(
            video_tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (resized_height - target_height) // 2
        left = (resized_width - target_width) // 2
        video_tensor = video_tensor[:, :, top : top + target_height, left : left + target_width]

    video_tensor = (video_tensor - 0.5) * 2.0
    return video_tensor.permute(1, 0, 2, 3).contiguous()


class LeRobotSO100Dataset(Dataset):
    def __init__(
        self,
        root: str | None,
        dataset_paths: Sequence[str] | str,
        train: bool,
        traj_len: int,
        target_height: int,
        target_width: int,
        pad: bool | None = True,
        camera_key: str = "auto",
        val_fraction: float = 0.05,
        seed: int = 0,
        downsample: int = 1,
        use_language: bool = False,
        fps: int | None = None,
        frame_stride: int | None = None,
        action_mean: Sequence[float] | None = None,
        action_std: Sequence[float] | None = None,
        remote: bool = False,
        repo_id: str | None = None,
        cache_dir: str | None = None,
        temporary_downloads: bool = False,
        hf_token: str | None = None,
        use_all_episodes: bool = False,
        return_state: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.remote = remote
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.temporary_downloads = temporary_downloads
        self.hf_token = hf_token
        self.traj_len = traj_len
        self.target_height = target_height
        self.target_width = target_width
        self.pad = pad
        self.camera_key = camera_key
        self.downsample = downsample
        self.use_language = use_language
        self.override_fps = fps
        self.override_frame_stride = frame_stride
        self.rng = random.Random(seed)
        self.action_mean = torch.tensor(action_mean, dtype=torch.float32) if action_mean is not None else None
        self.action_std = torch.tensor(action_std, dtype=torch.float32) if action_std is not None else None
        self.return_state = return_state

        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0.0, 1.0).")
        if self.remote and not self.repo_id:
            raise ValueError("repo_id is required when remote=True.")
        if not self.remote and self.root is None:
            raise ValueError("root is required when remote=False.")
        if dataset_paths == "auto":
            if self.remote:
                dataset_paths = discover_remote_lerobot_so100_datasets(self.repo_id, self.cache_dir, self.hf_token)
            else:
                dataset_paths = discover_lerobot_so100_datasets(self.root)
        self.dataset_paths = list(dataset_paths)

        examples = []
        for rel_dataset_path in self.dataset_paths:
            examples.extend(self._load_dataset_examples(rel_dataset_path))

        examples = [example for example in examples if example["length"] >= self.traj_len * self.downsample]
        self.rng.shuffle(examples)
        val_count = int(len(examples) * val_fraction)
        if val_fraction > 0 and val_count == 0 and len(examples) > 1:
            val_count = 1
        if use_all_episodes:
            self.examples = examples
        else:
            self.examples = examples[val_count:] if train else examples[:val_count]
        if not self.examples:
            split = "all" if use_all_episodes else ("train" if train else "val")
            raise ValueError(f"No {split} examples available. Check dataset_paths, traj_len, and val_fraction.")

    def set_action_stats(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.action_mean = torch.tensor(mean, dtype=torch.float32)
        self.action_std = torch.tensor(std, dtype=torch.float32)

    def compute_action_statistics(self) -> dict:
        count = 0
        total = np.zeros(6, dtype=np.float64)
        total_sq = np.zeros(6, dtype=np.float64)
        seen_data_refs = set()
        for example in self.examples:
            data_ref_key = str(example["data_ref"])
            if data_ref_key in seen_data_refs:
                continue
            seen_data_refs.add(data_ref_key)
            with self._materialize_file(example["data_ref"]) as data_path:
                table = pd.read_parquet(data_path, columns=["action"])
            actions = np.stack(table["action"].to_numpy()).astype(np.float64)
            if actions.shape[-1] != 6:
                raise ValueError(f"Expected SO-100 action dim 6, got {actions.shape[-1]} from {example['data_ref']}.")
            count += actions.shape[0]
            total += actions.sum(axis=0)
            total_sq += np.square(actions).sum(axis=0)
        if count == 0:
            raise ValueError("Cannot compute SO-100 action statistics from an empty training split.")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
        std = np.sqrt(variance)
        return {
            "count": int(count),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        }

    @contextmanager
    def _materialize_file(self, file_ref: str | Path) -> Iterator[Path]:
        if not self.remote:
            yield Path(file_ref)
            return
        with _materialize_hf_file(
            self.repo_id,
            str(file_ref),
            cache_dir=self.cache_dir,
            token=self.hf_token,
            temporary_downloads=self.temporary_downloads,
        ) as path:
            yield path

    def _dataset_file_ref(self, rel_dataset_path: str, filename: str) -> str | Path:
        if self.remote:
            return f"{rel_dataset_path}/{filename}"
        return self.root / rel_dataset_path / filename

    def _load_dataset_examples(self, rel_dataset_path: str) -> list[dict]:
        with self._materialize_file(self._dataset_file_ref(rel_dataset_path, "meta/info.json")) as info_path:
            info = _read_json(info_path)
        with self._materialize_file(self._dataset_file_ref(rel_dataset_path, "meta/episodes.jsonl")) as episodes_path:
            episodes = _read_jsonl(episodes_path)
        video_keys = _select_video_keys(info["features"], self.camera_key)
        chunks_size = int(info.get("chunks_size", 1000))
        fps = int(self.override_fps or info.get("fps", 30))
        frame_stride = int(self.override_frame_stride or fps)
        data_template = info["data_path"]
        video_template = info["video_path"]

        examples = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            data_ref = self._dataset_file_ref(
                rel_dataset_path,
                _format_lerobot_path(data_template, episode_index, chunks_size),
            )
            task = episode.get("tasks", [""])[0] if self.use_language else ""
            for video_key in video_keys:
                video_ref = self._dataset_file_ref(
                    rel_dataset_path,
                    _format_lerobot_path(video_template, episode_index, chunks_size, video_key),
                )
                examples.append(
                    {
                        "data_ref": data_ref,
                        "video_ref": video_ref,
                        "video_key": video_key,
                        "length": int(episode["length"]),
                        "task": task,
                        "fps": fps,
                        "frame_stride": frame_stride,
                    }
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        max_start = example["length"] - self.traj_len * self.downsample
        start_idx = self.rng.randint(0, max_start)
        frame_indices = [start_idx + i * self.downsample for i in range(self.traj_len)]

        columns = ["action", "observation.state"] if self.return_state else ["action"]
        with self._materialize_file(example["data_ref"]) as data_path:
            table = pd.read_parquet(data_path, columns=columns)
        actions = np.stack(table["action"].iloc[frame_indices].to_numpy()).astype(np.float32)
        if actions.shape[-1] != 6:
            raise ValueError(f"Expected SO-100 action dim 6, got {actions.shape[-1]} from {example['data_ref']}.")

        with self._materialize_file(example["video_ref"]) as video_path:
            video = _decode_video_clip(video_path, frame_indices)
        act = torch.from_numpy(actions)
        if self.action_mean is not None and self.action_std is not None:
            act = (act - self.action_mean) / self.action_std
        result = {
            "video": preprocess_video(video, self.target_height, self.target_width, self.pad),
            "act": act,
            "caption": example["task"],
            "fps": torch.tensor(example["fps"], dtype=torch.long),
            "frame_stride": torch.tensor(example["frame_stride"], dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
        }
        if self.return_state:
            states = np.stack(
                table["observation.state"].iloc[frame_indices].to_numpy()
            ).astype(np.float32)
            if states.shape != (self.traj_len, 6):
                raise ValueError(
                    f"Expected SO-100 state shape {(self.traj_len, 6)}, got {states.shape} "
                    f"from {example['data_ref']}."
                )
            result["state"] = torch.from_numpy(states)
        return result
