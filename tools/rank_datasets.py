"""train 데이터셋별로 '액션 관례가 eval과 맞는지'를 점수화한다.

대회 action extractor는 eval에서 첫 스텝 상관 0.94~0.97로 정확하지만
무작위 train 데이터셋에서는 무력했다. 데이터셋마다 캘리브레이션 오프셋이 달라
같은 관절각이 다른 자세를 뜻하기 때문으로 보인다.

추출기를 '관례 판별기'로 써서 eval과 같은 관례를 쓰는 데이터셋을 골라내면
학습 데이터 큐레이션에 바로 쓸 수 있다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from scorer import Scorer

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "open/data/train"
TRAJ_LEN = 16


def read_frames(video_path: Path, max_frames: int | None = None) -> np.ndarray:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-per-dataset", type=int, default=4)
    parser.add_argument("--out", default=str(REPO / "results/dataset_ranking.csv"))
    args = parser.parse_args()

    scorer = Scorer()
    rng = np.random.default_rng(0)
    datasets = sorted(d for d in TRAIN.glob("*/*") if (d / "meta/info.json").exists())

    rows = []
    for n, dataset_dir in enumerate(datasets):
        name = str(dataset_dir.relative_to(TRAIN))
        try:
            info = json.loads((dataset_dir / "meta/info.json").read_text())
            camera = next(k for k in info["features"] if k.startswith("observation.images"))
            records = [json.loads(x) for x in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines()]
            records = [r for r in records if r["length"] >= TRAJ_LEN]
            if not records:
                continue
            picks = rng.choice(len(records), size=min(args.clips_per_dataset, len(records)), replace=False)

            preds, targets = [], []
            for pick in picks:
                ep = records[int(pick)]["episode_index"]
                chunk = ep // info["chunks_size"]
                video_path = dataset_dir / f"videos/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.mp4"
                parquet_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
                if not video_path.exists() or not parquet_path.exists():
                    continue
                actions = np.stack(
                    pd.read_parquet(parquet_path, columns=["action"])["action"].values
                ).astype(np.float32)
                frames = read_frames(video_path, max_frames=len(actions))
                usable = min(len(frames), len(actions))
                if usable < TRAJ_LEN:
                    continue
                start = int(rng.integers(0, usable - TRAJ_LEN + 1))
                clip = scorer.to_eval(frames[start : start + TRAJ_LEN])[None]
                act = actions[start : start + TRAJ_LEN]

                x = clip.to(scorer.device).permute(0, 4, 1, 2, 3).float().div(255.0).mul(2.0).sub(1.0)
                with torch.no_grad():
                    preds.append(scorer.action_model(x).float().cpu()[0])
                targets.append((torch.from_numpy(act) - scorer.action_mean) / scorer.action_std)

            if not preds:
                continue
            pred = torch.stack(preds)
            target = torch.stack(targets)
            mae = float((pred - target).abs().mean())
            bias = (pred - target).mean(dim=1, keepdim=True)
            mae_debiased = float((pred - target - bias).abs().mean())
            trivial = float((target - target[:, :1]).abs().mean())
            rows.append(
                {
                    "dataset": name,
                    "n_clips": len(preds),
                    "action_mae": round(mae, 4),
                    "mae_debiased": round(mae_debiased, 4),
                    "trivial_hold_first": round(trivial, 4),
                    "episodes": info["total_episodes"],
                    "frames": info["total_frames"],
                    "height": info["features"][camera]["shape"][0],
                    "camera": camera,
                }
            )
        except Exception as exc:  # 커뮤니티 데이터라 깨진 파일이 섞일 수 있다
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
        if (n + 1) % 20 == 0:
            print(f"진행 {n + 1}/{len(datasets)}")

    df = pd.DataFrame(rows).sort_values("action_mae")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n측정된 데이터셋 {len(df)}개 (eval 정지영상 기준값 0.4305)")
    print("\n=== 액션 관례가 eval과 가장 잘 맞는 20개 ===")
    print(df.head(20).to_string(index=False))
    print("\n=== 가장 안 맞는 10개 ===")
    print(df.tail(10).to_string(index=False))
    good = df[df.action_mae < 0.8]
    print(f"\nMAE<0.8 데이터셋: {len(good)}개, 에피소드 {int(good.episodes.sum())}개, 프레임 {int(good.frames.sum())}개")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
