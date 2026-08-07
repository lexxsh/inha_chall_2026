"""Rank deterministic SO-100 clips by how strongly they reject a static video."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset, discover_lerobot_so100_datasets  # noqa: E402

EXCLUDE = ("dragon-95/so100_sorting",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", default=str(REPO / "results/fomm_oracle_clip_ranking.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_lerobot_so100_datasets(args.data_root)
    paths = [path for path in paths if not any(path.endswith(item) for item in EXCLUDE)]
    rows = []
    for dataset_index in range(0, len(paths), args.stride):
        path = paths[dataset_index]
        try:
            base = LeRobotSO100Dataset(
                root=args.data_root,
                dataset_paths=[path],
                train=True,
                traj_len=16,
                target_height=160,
                target_width=256,
                pad=True,
                camera_key="auto",
                val_fraction=0.0,
                seed=0,
                fps=6,
                use_all_episodes=True,
            )
            if args.clip_index >= len(base):
                continue
            sample = base[args.clip_index]
            video = sample["video"].permute(1, 0, 2, 3).add(1).mul(0.5)
            source = video[:1]
            mse = (video[1:] - source).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
            static_psnr = -10.0 * torch.log10(mse)
            action = sample["act"]
            rows.append({
                "dataset_index": dataset_index,
                "dataset": path,
                "clip_index": args.clip_index,
                "start_idx": int(sample["start_idx"]),
                "median_static_psnr": float(static_psnr.median()),
                "last_static_psnr": float(static_psnr[-1]),
                "mean_pixel_change": float((video[1:] - source).abs().mean()),
                "action_travel_l1": float((action[1:] - action[:-1]).abs().sum(dim=0).mean()),
                "action_range_mean": float((action.max(dim=0).values - action.min(dim=0).values).mean()),
            })
        except Exception as error:
            print(f"[skip] {dataset_index} {path}: {type(error).__name__}: {error}")
        if (dataset_index + 1) % 20 == 0:
            print(f"scanned {dataset_index + 1}/{len(paths)}")

    rows.sort(key=lambda row: row["median_static_psnr"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:20], indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
