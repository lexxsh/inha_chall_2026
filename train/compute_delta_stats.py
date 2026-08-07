"""delta 액션의 정규화 통계를 구한다.

액션을 전역 mean/std로 정규화한 뒤 첫 스텝 대비 delta를 취하면 스케일이 작아진다(대략 std 0.3).
조건 신호의 스케일을 1 근처로 맞춰야 임베딩 학습이 안정적이므로 delta 자체의 std를 따로 구한다.
16프레임 윈도우를 데이터셋 전반에서 샘플링해 계산한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "open/data/train"
TRAJ_LEN = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=64)
    parser.add_argument("--action-stats", default=str(TRAIN / "so100_action_statistics.json"))
    parser.add_argument("--out", default=str(REPO / "train/delta_action_stats.json"))
    args = parser.parse_args()

    stats = json.loads(Path(args.action_stats).read_text())
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)

    rng = np.random.default_rng(0)
    deltas = []
    datasets = sorted(d for d in TRAIN.glob("*/*") if (d / "meta/info.json").exists())

    for n, dataset_dir in enumerate(datasets):
        parquets = sorted(dataset_dir.glob("data/chunk-*/*.parquet"))
        if not parquets:
            continue
        picks = rng.choice(len(parquets), size=min(args.windows_per_dataset, len(parquets)), replace=False)
        for pick in picks:
            try:
                actions = np.stack(
                    pd.read_parquet(parquets[int(pick)], columns=["action"])["action"].to_numpy()
                ).astype(np.float64)
            except Exception:
                continue
            if len(actions) < TRAJ_LEN:
                continue
            start = int(rng.integers(0, len(actions) - TRAJ_LEN + 1))
            window = (actions[start : start + TRAJ_LEN] - mean) / std
            deltas.append(window - window[:1])
        if (n + 1) % 32 == 0:
            print(f"진행 {n + 1}/{len(datasets)} 데이터셋, 윈도우 {len(deltas)}개")

    # 첫 프레임은 항상 0이므로 스케일 추정에서 제외한다.
    anchor = np.concatenate([d[1:] for d in deltas], axis=0)
    # step delta는 anchor delta의 시간축 차분과 같다. anchor std의 20% 수준이라
    # 같은 scale을 공유하면 delta_step 조건 진폭만 작아져 비교가 불공정해진다(METHOD.md 10 P0).
    step = np.concatenate([np.diff(d, axis=0) for d in deltas], axis=0)

    out = {
        "count": int(len(anchor)),
        # 하위 호환: 기존 키는 anchor scale을 가리킨다.
        "delta_std": anchor.std(axis=0).tolist(),
        "delta_abs_mean": np.abs(anchor).mean(axis=0).tolist(),
        "delta_p99": np.percentile(np.abs(anchor), 99, axis=0).tolist(),
        # 모드별 scale. data_module이 action_mode에 맞춰 골라 쓴다.
        "scale_by_mode": {
            "delta": anchor.std(axis=0).tolist(),
            "delta_anchor": anchor.std(axis=0).tolist(),
            "delta_step": step.std(axis=0).tolist(),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n윈도우 {len(deltas)}개 / anchor 프레임 {len(anchor)}개 / step 프레임 {len(step)}개")
    print("anchor delta std :", np.round(out["scale_by_mode"]["delta"], 4).tolist())
    print("step delta std   :", np.round(out["scale_by_mode"]["delta_step"], 4).tolist())
    ratio = np.array(out["scale_by_mode"]["delta_step"]) / np.array(out["scale_by_mode"]["delta"])
    print("step/anchor 비율  :", np.round(ratio, 3).tolist())
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
