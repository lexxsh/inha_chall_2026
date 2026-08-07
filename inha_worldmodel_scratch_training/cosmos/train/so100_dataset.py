# SO-100 17프레임 클립 Dataset: mp4에서 온더플라이 디코드 + 320x512 letterbox + 정규화 액션 (16,6)
from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_PACKAGED_TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "train"
_WORKSPACE_TRAIN_ROOT = Path(__file__).resolve().parents[3] / "open" / "data" / "train"
TRAIN_ROOT = Path(os.environ.get(
    "SO100_TRAIN_ROOT",
    str(_PACKAGED_TRAIN_ROOT if _PACKAGED_TRAIN_ROOT.exists() else _WORKSPACE_TRAIN_ROOT),
))
INDEX_PATH = Path(os.environ.get(
    "SO100_INDEX_PATH",
    str(Path(__file__).resolve().parent / "episode_index.parquet"),
))
STATS_PATH = TRAIN_ROOT / "so100_action_statistics.json"

NUM_FRAMES = 17
HEIGHT, WIDTH = 320, 512


def letterbox_batch(frames: torch.Tensor, target_h: int = HEIGHT, target_w: int = WIDTH) -> torch.Tensor:
    """(T,3,H,W) float 0..1 → (T,3,target_h,target_w), 제출킷과 동일한 scale+pad."""
    _, _, h, w = frames.shape
    scale = min(target_h / h, target_w / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    frames = F.interpolate(frames, size=(rh, rw), mode="bilinear", align_corners=False)
    pt = (target_h - rh) // 2
    pl = (target_w - rw) // 2
    return F.pad(frames, (pl, target_w - rw - pl, pt, target_h - rh - pt), value=0.0)


def read_frames(path: Path, start: int, count: int) -> np.ndarray:
    """mp4에서 [start, start+count) 프레임을 디코드. 6fps 저해상도라 앞에서부터 순차 디코드."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i >= start + count:
                break
            if i >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != count:
        raise ValueError(f"{path}: wanted {count} frames from {start}, got {len(frames)}")
    return np.stack(frames)


class SO100ClipDataset(Dataset):
    """반환: video (3,17,320,512) in [-1,1] bf16 아님(float32), act (16,6) 정규화."""

    def __init__(self, split: str = "train", index_path: Path = INDEX_PATH):
        df = pd.read_parquet(index_path)
        self.episodes = df[df.split == split].reset_index(drop=True)
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        self.act_mean = np.array(stats["mean"], dtype=np.float32)
        self.act_std = np.array(stats["std"], dtype=np.float32)
        # 클립 시작 인덱스 누적 테이블 (stride 1)
        counts = (self.episodes.num_frames - NUM_FRAMES + 1).clip(lower=0).to_numpy()
        self.cumsum = np.concatenate([[0], np.cumsum(counts)])

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def __getitem__(self, idx: int):
        ep_i = int(np.searchsorted(self.cumsum, idx, side="right") - 1)
        start = int(idx - self.cumsum[ep_i])
        row = self.episodes.iloc[ep_i]

        video_np = read_frames(TRAIN_ROOT / row.video, start, NUM_FRAMES)  # (17,H,W,3) uint8
        frames = torch.from_numpy(video_np).float().permute(0, 3, 1, 2) / 255.0
        frames = letterbox_batch(frames)  # (17,3,320,512)
        video = frames.permute(1, 0, 2, 3) * 2.0 - 1.0  # (3,17,H,W) in [-1,1]

        table = pd.read_parquet(TRAIN_ROOT / row.parquet, columns=["action"])
        actions = np.stack(table["action"].to_numpy()[start : start + NUM_FRAMES - 1]).astype(np.float32)
        actions = (actions - self.act_mean) / self.act_std  # (16,6)

        return {"video": video, "act": torch.from_numpy(actions)}


if __name__ == "__main__":
    import time

    ds = SO100ClipDataset("train")
    print("train clips:", len(ds), "| val clips:", len(SO100ClipDataset("val")))
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        t0 = time.time()
        item = ds[idx]
        dt = time.time() - t0
        v, a = item["video"], item["act"]
        print(
            f"idx {idx}: video {tuple(v.shape)} [{v.min():.2f},{v.max():.2f}] "
            f"act {tuple(a.shape)} mean {a.mean():.3f} std {a.std():.3f} ({dt * 1000:.0f}ms)"
        )
