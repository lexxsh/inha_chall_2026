"""Fixed train-only clips paired with RAFT oracle point-track controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import _decode_video_clip, preprocess_video  # noqa: E402


def resolve_artifact(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    frames = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in frames]


class WanOracleTrackDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        data_root: str,
        split: str,
        height: int = 320,
        width: int = 512,
        repeat: int = 100,
        prompt: str = "A fixed-camera video of a robot arm manipulating objects.",
        quality_filter: bool = True,
    ) -> None:
        records = json.loads(Path(manifest).read_text())
        self.records = [record for record in records if record["split"] == split]
        if quality_filter and split == "train":
            self.records = [
                record
                for record in self.records
                if record["tracks_found"] >= 8
                and record["cycle_valid_fraction"] >= 0.6
                and record["motion_energy_coverage"] >= 0.2
            ]
        if not self.records:
            raise ValueError(f"No {split!r} records in {manifest}")
        self.wrong_indices = []
        for index, record in enumerate(self.records):
            candidates = [
                offset
                for offset, candidate in enumerate(self.records)
                if candidate["dataset"] != record["dataset"] and candidate["tracks_found"] >= 8
            ]
            if not candidates:
                raise ValueError(f"No cross-dataset wrong track for {record['sample_id']}")
            # Deterministic and independent of DataLoader shuffle/repeat.
            self.wrong_indices.append(candidates[index % len(candidates)])
        self.data_root = Path(data_root)
        self.height = height
        self.width = width
        self.repeat = repeat
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int) -> dict:
        record = self.records[index % len(self.records)]
        frame_indices = [record["start"] + offset for offset in range(17)]
        video = _decode_video_clip(self.data_root / record["video_ref"], frame_indices)
        video = preprocess_video(video, self.height, self.width, pad=True)
        with np.load(resolve_artifact(record["artifact"])) as artifact:
            control = torch.from_numpy(artifact["control"].astype(np.float32))
        wrong_record = self.records[self.wrong_indices[index % len(self.records)]]
        with np.load(resolve_artifact(wrong_record["artifact"])) as artifact:
            wrong_control = torch.from_numpy(artifact["control"].astype(np.float32))
        if control.shape[0] != 3 or control.shape[1] != 17:
            raise ValueError(f"Bad control shape {tuple(control.shape)} for {record['sample_id']}")
        return {
            "video": to_pil_frames(video),
            "track_control": control,
            "wrong_track_control": wrong_control,
            "prompt": self.prompt,
            "sample_id": record["sample_id"],
        }
