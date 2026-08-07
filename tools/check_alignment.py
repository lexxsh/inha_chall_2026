"""프레임과 액션의 시간 정렬을 데이터로 확인한다.

의문: LeRobot에서 action[t]는 서보에 보낸 목표값이라 실제로는 state[t+1]에 가깝다.
그렇다면 "프레임 t에 action[t]를 조건으로" 넣는 현재 정렬이 한 칸 어긋난 것 아닌가?

판정 기준은 우리 직관이 아니라 **대회 action extractor가 가정하는 정렬**이다.
Action Component는 extractor(생성영상)와 주어진 액션을 비교하므로,
extractor가 프레임 t를 action[t]로 읽는다면 우리도 그렇게 맞춰야 점수가 나온다.

방법: 정답 영상 [s, s+16)에 extractor를 돌리고,
타깃을 action[s+k, s+16+k) 로 k를 바꿔가며 MAE를 잰다. 최소가 되는 k가 extractor의 정렬이다.
캘리브레이션 오프셋이 MAE를 지배하므로 클립별 편향을 제거한 값으로 비교한다.
observation.state도 같이 재서 extractor가 실제로 무엇을 읽는지 본다.
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

# 추출기가 잘 작동하는 데이터셋 위주로 본다(results/dataset_ranking.csv 상위).
# 추출기가 무력한 데이터셋에서는 정렬 신호도 묻힌다.
GOOD_DATASETS = [
    "aractingi/push_cube_offline_data",
    "nbaron99/so100_pick_and_place4",
    "FeiYjf/new_GtoR",
    "sihyun77/mond_1",
    "sixpigs1/so100_stack_cube_error",
    "pandaRQ/pick_med_1",
    "VoicAndrei/so100_banana_to_plate_only",
    "frk2/so100largediffcam",
    "samsam0510/cube_reorientation_2",
    "356c/so100_duck_reposition_1",
]


def read_frames(path: Path, count: int | None = None) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
            if count is not None and len(frames) >= count:
                break
    return np.stack(frames)


def debiased_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """클립별 상수 오프셋을 제거한 MAE. 캘리브레이션 차이를 빼고 시간 정렬만 본다."""
    diff = pred - target
    return float((diff - diff.mean(dim=1, keepdim=True)).abs().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-per-dataset", type=int, default=6)
    parser.add_argument("--max-shift", type=int, default=2)
    args = parser.parse_args()

    scorer = Scorer()
    rng = np.random.default_rng(0)
    shifts = list(range(-args.max_shift, args.max_shift + 1))

    records = []
    for name in GOOD_DATASETS:
        dataset_dir = TRAIN / name
        info_path = dataset_dir / "meta/info.json"
        if not info_path.exists():
            print(f"  [skip] {name}")
            continue
        info = json.loads(info_path.read_text())
        camera = next(k for k in info["features"] if k.startswith("observation.images"))
        episodes = [json.loads(x) for x in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines()]
        # shift 범위를 확보하려면 여유 프레임이 필요하다.
        episodes = [e for e in episodes if e["length"] >= TRAJ_LEN + 2 * args.max_shift + 2]
        if not episodes:
            continue
        picks = rng.choice(len(episodes), size=min(args.clips_per_dataset, len(episodes)), replace=False)

        for pick in picks:
            ep = episodes[int(pick)]["episode_index"]
            chunk = ep // info["chunks_size"]
            video_path = dataset_dir / f"videos/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.mp4"
            parquet_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
            if not video_path.exists() or not parquet_path.exists():
                continue
            table = pd.read_parquet(parquet_path, columns=["action", "observation.state"])
            actions = np.stack(table["action"].to_numpy()).astype(np.float32)
            states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
            frames = read_frames(video_path, count=len(actions))
            usable = min(len(frames), len(actions))
            lo, hi = args.max_shift, usable - TRAJ_LEN - args.max_shift
            if hi <= lo:
                continue
            start = int(rng.integers(lo, hi))

            clip = scorer.to_eval(frames[start : start + TRAJ_LEN])[None]
            x = clip.to(scorer.device).permute(0, 4, 1, 2, 3).float().div(255.0).mul(2.0).sub(1.0)
            with torch.no_grad():
                pred = scorer.action_model(x).float().cpu()[0]

            row = {"dataset": name, "episode": ep, "start": start}
            for shift in shifts:
                s = start + shift
                for label, source in (("action", actions), ("state", states)):
                    target = torch.from_numpy(source[s : s + TRAJ_LEN])
                    target = (target - scorer.action_mean) / scorer.action_std
                    row[f"{label}_shift{shift:+d}"] = debiased_mae(pred, target)
            records.append(row)
        print(f"  {name}: 누적 {len(records)}클립")

    df = pd.DataFrame(records)
    out = REPO / "results/alignment_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n클립 {len(df)}개 / 데이터셋 {df.dataset.nunique()}개")
    print("\n=== 편향 제거 MAE (낮을수록 그 정렬이 맞다) ===")
    print(f"{'shift':>6} {'action 타깃':>14} {'state 타깃':>14}")
    print("  (shift=+1 은 프레임 t 를 action[t+1] 로 비교한다는 뜻)")
    for shift in shifts:
        a = df[f"action_shift{shift:+d}"].mean()
        s = df[f"state_shift{shift:+d}"].mean()
        print(f"{shift:>+6d} {a:>14.4f} {s:>14.4f}")

    best_action = min(shifts, key=lambda k: df[f"action_shift{k:+d}"].mean())
    best_state = min(shifts, key=lambda k: df[f"state_shift{k:+d}"].mean())
    best_overall = min(
        [(df[f"{l}_shift{k:+d}"].mean(), l, k) for l in ("action", "state") for k in shifts]
    )
    print(f"\naction 최적 shift: {best_action:+d}")
    print(f"state  최적 shift: {best_state:+d}")
    print(f"전체 최적: {best_overall[1]} shift {best_overall[2]:+d} (MAE {best_overall[0]:.4f})")

    base = df["action_shift+0"].mean()
    print(f"\n현재 정렬(action shift 0) 대비 개선폭: {base - best_overall[0]:+.4f}")

    # 데이터셋별로도 같은 결론인지 확인한다(우연 방지).
    print("\n=== 데이터셋별 action 최적 shift ===")
    for name, group in df.groupby("dataset"):
        best = min(shifts, key=lambda k: group[f"action_shift{k:+d}"].mean())
        vals = " ".join(f"{k:+d}:{group[f'action_shift{k:+d}'].mean():.3f}" for k in shifts)
        print(f"  {name:42s} 최적 {best:+d}   {vals}")


if __name__ == "__main__":
    main()
