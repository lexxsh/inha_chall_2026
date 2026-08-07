"""Train-only SO100 clips paired with dense RAFT oracle motion fields."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import _decode_video_clip, preprocess_video  # noqa: E402


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    frames = ((video.permute(1, 2, 3, 0) + 1.0) * 127.5).round().clamp(0, 255)
    return [Image.fromarray(frame.byte().numpy(), mode="RGB") for frame in frames]


def moving_mask(video: torch.Tensor, threshold: float = 0.04, dilation: int = 7) -> torch.Tensor:
    difference = (video - video[:, :1]).abs().amax(dim=0, keepdim=True)
    mask = (difference >= threshold).to(video.dtype)
    if dilation > 1:
        mask = F.max_pool2d(
            mask.transpose(0, 1), dilation, stride=1, padding=dilation // 2
        ).transpose(0, 1)
    mask[:, 0].zero_()
    return mask


class WanOracleMotionFieldDataset(Dataset):
    """A fixed 64/8 train-only gate, not a competition-inference dataset."""

    load_from_cache = False

    def __init__(
        self,
        manifest: str,
        data_root: str,
        split: str = "train",
        height: int = 320,
        width: int = 512,
        repeat: int = 100,
        prompt: str = "A fixed-camera video of a tabletop robot arm manipulating objects.",
        motion_mask_threshold: float = 0.04,
        motion_mask_dilation: int = 7,
    ) -> None:
        records = json.loads(Path(manifest).read_text())
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"No records for split={split!r} in {manifest}")
        self.data_root = Path(data_root)
        self.height = int(height)
        self.width = int(width)
        self.repeat = int(repeat)
        self.prompt = prompt
        self.motion_mask_threshold = float(motion_mask_threshold)
        self.motion_mask_dilation = int(motion_mask_dilation)

        # Cross-dataset controls are harder counterfactuals than a mere time
        # reversal and cannot accidentally match the same robot trajectory.
        self.wrong_indices = []
        for index, record in enumerate(self.records):
            candidates = [
                candidate_index
                for candidate_index, candidate in enumerate(self.records)
                if candidate["dataset"] != record["dataset"]
            ]
            if not candidates:
                candidates = [candidate_index for candidate_index in range(len(self.records)) if candidate_index != index]
            if not candidates:
                raise ValueError("Oracle field gate requires at least two training records")
            self.wrong_indices.append(candidates[index % len(candidates)])

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    @staticmethod
    def load_control(record: dict) -> torch.Tensor:
        with np.load(resolve(record["motion_artifact"])) as artifact:
            control = torch.from_numpy(artifact["control"].astype(np.float32))
        if control.ndim != 4 or control.shape[:2] != (7, 17):
            raise ValueError(
                f"Bad oracle motion field {tuple(control.shape)} for {record['sample_id']}"
            )
        return control

    def __getitem__(self, index: int) -> dict:
        base_index = index % len(self.records)
        record = self.records[base_index]
        frame_indices = [int(record["start"]) + offset for offset in range(17)]
        video = _decode_video_clip(self.data_root / record["video_ref"], frame_indices)
        video = preprocess_video(video, self.height, self.width, pad=True)
        control = self.load_control(record)

        # Alternate exact zero and cross-dataset negatives so the renderer must
        # pass both counterfactual tests used by the gate.
        if (index // len(self.records)) % 2 == 0:
            wrong_control = torch.zeros_like(control)
            wrong_kind = "zero"
        else:
            wrong_record = self.records[self.wrong_indices[base_index]]
            wrong_control = self.load_control(wrong_record)
            wrong_kind = "batch-roll"
        ranking_valid = (control - wrong_control).abs().mean() >= 1e-5

        return {
            "video": to_pil_frames(video),
            "motion_field_control": control,
            "wrong_motion_field_control": wrong_control,
            "motion_mask": moving_mask(
                video,
                threshold=self.motion_mask_threshold,
                dilation=self.motion_mask_dilation,
            ),
            "ranking_valid": torch.tensor(float(ranking_valid), dtype=torch.float32),
            "prompt": self.prompt,
            "sample_id": record["sample_id"],
            "wrong_kind": wrong_kind,
        }
