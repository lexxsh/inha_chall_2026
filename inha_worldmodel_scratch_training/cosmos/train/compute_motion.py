# 에피소드별 움직임 점수(정규화 액션의 프레임간 변화량 평균)를 계산해 저장.
# 용도: 학습 샘플링 가중치 (train 데이터 통계만 사용 — 평가 데이터 무관)
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "train"
INDEX_PATH = Path(__file__).resolve().parent / "episode_index.parquet"
OUT_PATH = Path(__file__).resolve().parent / "episode_motion.parquet"


def main() -> None:
    stats = json.loads((TRAIN_ROOT / "so100_action_statistics.json").read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    df = pd.read_parquet(INDEX_PATH)
    scores = []
    for i, row in enumerate(df.itertuples()):
        acts = np.stack(pd.read_parquet(TRAIN_ROOT / row.parquet, columns=["action"])["action"].to_numpy())
        acts = (acts.astype(np.float32) - mean) / std
        motion = float(np.abs(np.diff(acts, axis=0)).mean()) if len(acts) > 1 else 0.0
        scores.append(motion)
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(df)}", flush=True)

    df["motion"] = scores
    df.to_parquet(OUT_PATH, index=False)
    q = np.percentile(scores, [10, 50, 90])
    print(f"saved {OUT_PATH}  motion p10/p50/p90 = {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}")


if __name__ == "__main__":
    main()
