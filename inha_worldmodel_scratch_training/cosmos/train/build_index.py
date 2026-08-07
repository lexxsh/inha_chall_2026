# 학습 데이터 인덱스 생성: 모든 LeRobot 데이터셋을 스캔해 에피소드 목록을 만든다.
# 필터: fps=6, so100 계열, 단일 카메라(observation.images.image), 프레임수>=17
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import pandas as pd

_PACKAGED_TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "train"
_WORKSPACE_TRAIN_ROOT = Path(__file__).resolve().parents[3] / "open" / "data" / "train"
TRAIN_ROOT = Path(os.environ.get(
    "SO100_TRAIN_ROOT",
    str(_PACKAGED_TRAIN_ROOT if _PACKAGED_TRAIN_ROOT.exists() else _WORKSPACE_TRAIN_ROOT),
))
OUT_PATH = Path(os.environ.get(
    "SO100_INDEX_PATH",
    str(Path(__file__).resolve().parent / "episode_index.parquet"),
))
MIN_FRAMES = 17
VAL_FRACTION = 0.02  # 에피소드 단위 held-out (결정적: episode 해시 기반)


def main() -> None:
    rows = []
    skipped = {"fps": 0, "camera": 0, "short": 0, "missing": 0}
    for info_path in sorted(TRAIN_ROOT.glob("*/*/meta/info.json")):
        ds_root = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("fps") != 6:
            skipped["fps"] += 1
            continue
        cam_keys = [k for k in info["features"] if k.startswith("observation.images")]
        if cam_keys != ["observation.images.image"]:
            skipped["camera"] += 1
            continue
        cam = cam_keys[0]
        chunks_size = info.get("chunks_size", 1000)
        episodes_meta = {}
        ep_jsonl = ds_root / "meta" / "episodes.jsonl"
        if ep_jsonl.exists():
            for line in ep_jsonl.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                episodes_meta[d["episode_index"]] = d.get("length")

        for ep_idx, length in episodes_meta.items():
            chunk = ep_idx // chunks_size
            pq = ds_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
            mp4 = ds_root / "videos" / f"chunk-{chunk:03d}" / cam / f"episode_{ep_idx:06d}.mp4"
            if not (pq.exists() and mp4.exists()):
                skipped["missing"] += 1
                continue
            if length is None or length < MIN_FRAMES:
                skipped["short"] += 1
                continue
            rel_pq = pq.relative_to(TRAIN_ROOT).as_posix()
            rel_mp4 = mp4.relative_to(TRAIN_ROOT).as_posix()
            is_val = (zlib.crc32(rel_pq.encode()) % 1000) < VAL_FRACTION * 1000
            rows.append(
                {
                    "dataset": ds_root.relative_to(TRAIN_ROOT).as_posix(),
                    "episode_index": ep_idx,
                    "num_frames": int(length),
                    "parquet": rel_pq,
                    "video": rel_mp4,
                    "split": "val" if is_val else "train",
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"episodes: {len(df)} (train {sum(df.split=='train')}, val {sum(df.split=='val')})")
    print(f"total frames: {df.num_frames.sum():,}, skipped: {skipped}")
    print(f"clips available (stride 1): {(df.num_frames - MIN_FRAMES + 1).sum():,}")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
