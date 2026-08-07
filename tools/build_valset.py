"""학습 데이터에서 eval과 동일한 형식의 로컬 검증셋을 만든다.

eval 샘플은 (초기 이미지 640x480 png, 액션 (16,6) npy) 이고 정답은 16프레임 영상이다.
같은 구조를 train 에피소드에서 잘라내 GT 영상까지 함께 저장한다.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "open/data/train"
TRAJ_LEN = 16


def read_frames(video_path: Path) -> np.ndarray:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def episode_entries(dataset_dir: Path) -> list[tuple[int, int]]:
    meta = dataset_dir / "meta/episodes.jsonl"
    out = []
    for line in meta.read_text().splitlines():
        record = json.loads(line)
        if record["length"] >= TRAJ_LEN:
            out.append((record["episode_index"], record["length"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "valset"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-like",
        action="store_true",
        help="eval과 시각적으로 가까운 데이터셋만 사용한다. "
        "results/eval_source_match.json(find_eval_source.py 산출물)이 필요하다.",
    )
    parser.add_argument("--eval-like-top", type=int, default=12)
    parser.add_argument(
        "--training-group-holdout",
        type=int,
        default=0,
        metavar="N",
        help="train/data_module.py와 같은 규칙(seed 0, dataset 단위)으로 학습에서 빠진 N개 dataset만 사용한다.",
    )
    parser.add_argument(
        "--group-holdout-seed",
        type=int,
        default=0,
        help="data module의 dataset split seed. 현재 학습 설정의 기본값은 0이다.",
    )
    args = parser.parse_args()

    if args.eval_like and args.training_group_holdout:
        raise SystemExit("--eval-like와 --training-group-holdout은 함께 쓸 수 없다.")

    rng = random.Random(args.seed)
    out_root = Path(args.out)
    for sub in ("images", "actions", "gt_videos"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    datasets = sorted(d for d in TRAIN.glob("*/*") if (d / "meta/info.json").exists())

    if args.training_group_holdout:
        # SO100DeltaDataModule._resolve_dataset_paths/_split_datasets와 같은 순서다.
        # 10fps인 dragon-95/so100_sorting은 현재 학습 설정의 DEFAULT_EXCLUDE라 split 전에 제외한다.
        eligible = [d for d in datasets if str(d.relative_to(TRAIN)) != "dragon-95/so100_sorting"]
        order = list(eligible)
        random.Random(args.group_holdout_seed).shuffle(order)
        datasets = sorted(order[: args.training_group_holdout])
        print(
            f"학습 group holdout {len(datasets)}개(seed={args.group_holdout_seed}): "
            f"{[str(d.relative_to(TRAIN)) for d in datasets]}"
        )

    if args.eval_like:
        # 무작위 다양성 셋은 action extractor 지표가 eval과 딴판으로 나온다(EMPIRICAL.md 4절).
        # eval 이미지의 최근접으로 자주 지목된 데이터셋만 남겨 eval을 대변하게 한다.
        match_path = REPO / "results/eval_source_match.json"
        if not match_path.exists():
            raise SystemExit(f"{match_path} 가 없다. 먼저 tools/find_eval_source.py 를 실행할 것.")
        matches = json.loads(match_path.read_text())
        counts: dict[str, int] = {}
        for record in matches:
            name = record["match"].split("#")[0]
            counts[name] = counts.get(name, 0) + 1
        keep = {name for name, _ in sorted(counts.items(), key=lambda kv: -kv[1])[: args.eval_like_top]}
        datasets = [d for d in datasets if str(d.relative_to(TRAIN)) in keep]
        print(f"eval 유사 데이터셋 {len(datasets)}개로 제한: {sorted(keep)}")

    rng.shuffle(datasets)

    manifest = []
    idx = 0
    used_episodes: set[tuple[str, int]] = set()
    # 데이터셋 수가 목표 샘플 수보다 적을 수 있으므로 여러 바퀴 돈다(--eval-like 인 경우).
    rounds = max(1, -(-args.num_samples // max(1, len(datasets))))
    for dataset_dir in datasets * rounds:
        if idx >= args.num_samples:
            break
        info = json.loads((dataset_dir / "meta/info.json").read_text())
        camera = next(k for k in info["features"] if k.startswith("observation.images"))
        episodes = episode_entries(dataset_dir)
        key = str(dataset_dir.relative_to(TRAIN))
        episodes = [e for e in episodes if (key, e[0]) not in used_episodes]
        if not episodes:
            continue
        ep_index, ep_len = rng.choice(episodes)
        used_episodes.add((key, ep_index))
        chunk = ep_index // info["chunks_size"]
        video_path = dataset_dir / f"videos/chunk-{chunk:03d}/{camera}/episode_{ep_index:06d}.mp4"
        parquet_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep_index:06d}.parquet"
        if not video_path.exists() or not parquet_path.exists():
            continue

        frames = read_frames(video_path)
        actions = np.stack(pd.read_parquet(parquet_path, columns=["action"])["action"].values).astype(np.float32)
        usable = min(len(frames), len(actions))
        if usable < TRAJ_LEN:
            continue
        start = rng.randrange(0, usable - TRAJ_LEN + 1)
        clip = frames[start : start + TRAJ_LEN]
        act = actions[start : start + TRAJ_LEN]

        sample_id = f"val_{idx:06d}"
        Image.fromarray(clip[0]).save(out_root / "images" / f"{sample_id}.png")
        np.save(out_root / "actions" / f"{sample_id}.npy", act)
        np.save(out_root / "gt_videos" / f"{sample_id}.npy", clip)
        manifest.append(
            {
                "sample_id": sample_id,
                "dataset": str(dataset_dir.relative_to(TRAIN)),
                "episode": ep_index,
                "start": start,
                "camera": camera,
                "height": int(clip.shape[1]),
                "width": int(clip.shape[2]),
                "split": "training_group_holdout" if args.training_group_holdout else "probe",
            }
        )
        idx += 1

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} validation samples to {out_root}")


if __name__ == "__main__":
    main()
