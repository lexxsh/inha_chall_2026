"""같은 장면 안에서 '움직임 정확도'가 세 컴포넌트를 얼마나 움직이는지 측정한다.

장면(외형)과 움직임을 분리해야 어디에 노력을 쏟을지 정할 수 있다.
한 에피소드에서 여러 16프레임 구간을 뽑아 서로 교차 채점하면
외형은 고정한 채 움직임만 틀린 상황을 만들 수 있다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from scorer import Scorer, cosine_distance

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "open/data/train"
TRAJ_LEN = 16


def read_frames(video_path: Path) -> np.ndarray:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--clips-per-episode", type=int, default=4)
    args = parser.parse_args()

    scorer = Scorer()
    rng = np.random.default_rng(0)

    datasets = sorted(d for d in TRAIN.glob("*/*") if (d / "meta/info.json").exists())
    rng.shuffle(datasets)

    rows = []
    used = 0
    for dataset_dir in datasets:
        if used >= args.num_episodes:
            break
        info = json.loads((dataset_dir / "meta/info.json").read_text())
        camera = next(k for k in info["features"] if k.startswith("observation.images"))
        records = [json.loads(x) for x in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines()]
        # 여러 구간을 뽑으려면 충분히 긴 에피소드가 필요하다.
        long_eps = [r for r in records if r["length"] >= TRAJ_LEN * args.clips_per_episode]
        if not long_eps:
            continue
        record = long_eps[rng.integers(len(long_eps))]
        ep = record["episode_index"]
        chunk = ep // info["chunks_size"]
        video_path = dataset_dir / f"videos/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.mp4"
        parquet_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        if not video_path.exists() or not parquet_path.exists():
            continue

        frames = read_frames(video_path)
        actions = np.stack(pd.read_parquet(parquet_path, columns=["action"])["action"].values).astype(np.float32)
        usable = min(len(frames), len(actions))
        if usable < TRAJ_LEN * args.clips_per_episode:
            continue

        starts = np.linspace(0, usable - TRAJ_LEN, args.clips_per_episode).astype(int)
        clips = [scorer.to_eval(frames[s : s + TRAJ_LEN]) for s in starts]
        acts = [actions[s : s + TRAJ_LEN] for s in starts]

        batch = torch.stack(clips)
        feats = scorer.features(batch)

        for i in range(len(clips)):
            for j in range(len(clips)):
                # 영상 i를 '정답이 j인 문제'의 답으로 제출했을 때의 점수
                dino = float(cosine_distance(feats["dino"][i : i + 1], feats["dino"][j : j + 1]))
                vid = float(cosine_distance(feats["video"][i : i + 1], feats["video"][j : j + 1]))
                mae = float(scorer.action_mae(batch[i : i + 1], acts[j][None]))
                rows.append(
                    {
                        "dataset": str(dataset_dir.relative_to(TRAIN)),
                        "pred_clip": i,
                        "gt_clip": j,
                        "match": i == j,
                        "dino": dino,
                        "video": vid,
                        "action_mae": mae,
                    }
                )
        used += 1
        print(f"[{used}/{args.num_episodes}] {dataset_dir.relative_to(TRAIN)} ep{ep}")

    df = pd.DataFrame(rows)
    out = REPO / "results/motion_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== 같은 장면, 움직임만 다를 때 ===")
    summary = df.groupby("match")[["dino", "video", "action_mae"]].mean()
    print(summary.rename(index={True: "움직임 일치", False: "움직임 불일치"}).round(4).to_string())

    delta = summary.loc[False] - summary.loc[True]
    print("\n움직임이 틀렸을 때의 손해(불일치 - 일치):")
    print(delta.round(4).to_string())

    print("\n참고: 서로 다른 장면(외형까지 틀릴 때)")
    cross = []
    feats_by_ds = df.groupby("dataset")
    print(f"  데이터셋 {len(feats_by_ds)}개에서 측정")
    print(f"  (앞선 실험의 shuffled_gt DINO 0.68 이 이에 해당)")


if __name__ == "__main__":
    main()
