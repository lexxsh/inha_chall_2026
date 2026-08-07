"""Compute train-only SO100 EEF percentile statistics for faithful BWM input."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "train") not in sys.path:
    sys.path.insert(0, str(REPO / "train"))

from bwm_so100_eef import (  # noqa: E402
    DEFAULT_STATS,
    DEFAULT_URDF,
    SO100EEFConverter,
    selected_so100_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--output", default=str(DEFAULT_STATS))
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument(
        "--window-stride", type=int, default=4,
        help="Stride between sampled 17-frame windows inside each episode.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Diagnostics only; 0 uses all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    selected = selected_so100_paths(
        root, split="train", holdout_count=args.holdout_count, seed=args.seed
    )
    parquet_files: list[Path] = []
    for dataset in selected:
        parquet_files.extend(sorted((root / dataset / "data").glob("**/*.parquet")))
    if args.max_files:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        raise FileNotFoundError(f"No SO100 parquet files under selected datasets in {root}")

    started = time.perf_counter()
    windows: list[np.ndarray] = []
    raw_frame_count = 0
    for index, path in enumerate(parquet_files, 1):
        table = pd.read_parquet(path, columns=["action"])
        action = np.stack(table["action"].to_numpy()).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != 6:
            raise ValueError(f"Malformed action column in {path}: {action.shape}")
        raw_frame_count += len(action)
        if len(action) >= 17:
            starts = np.arange(0, len(action) - 17 + 1, args.window_stride)
            # Match challenge inference: source proxy, then 16 commanded actions.
            episode_windows = np.stack(
                [np.concatenate([action[s : s + 1], action[s : s + 16]], axis=0) for s in starts]
            )
            windows.append(episode_windows)
        if index % 1000 == 0 or index == len(parquet_files):
            print(f"[EEF stats read] {index}/{len(parquet_files)}", flush=True)
    if not windows:
        raise ValueError("No 17-frame SO100 windows available for EEF statistics")
    trajectories = np.concatenate(windows, axis=0)
    del windows

    converter = SO100EEFConverter(stats_path=None, urdf_path=DEFAULT_URDF)
    eef_parts = []
    for start in range(0, len(trajectories), args.chunk_size):
        end = min(start + args.chunk_size, len(trajectories))
        eef_parts.append(
            converter.raw_to_eef(trajectories[start:end]).astype(np.float32)
        )
        print(f"[EEF stats FK] {end}/{len(trajectories)} windows", flush=True)
    eef = np.concatenate(eef_parts, axis=0)
    violations = converter.joint_limit_violation_fraction(trajectories)
    eef_flat = eef.reshape(-1, eef.shape[-1])

    p01 = np.percentile(eef_flat, 1, axis=0)
    p99 = np.percentile(eef_flat, 99, axis=0)
    normalized = np.clip(2 * (eef_flat - p01) / np.maximum(p99 - p01, 1e-6) - 1, -1, 1)
    report = {
        "contract": "source-anchored SO100 joint-target deltas -> bounded corrected-URDF EEF14",
        "split": "train datasets only; dataset-level holdout excluded",
        "dataset_root": str(root),
        "selected_datasets": selected,
        "selected_dataset_count": len(selected),
        "parquet_files": len(parquet_files),
        "raw_frame_count": int(raw_frame_count),
        "sampled_windows": int(len(trajectories)),
        "window_stride": int(args.window_stride),
        "count": int(len(eef_flat)),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "mean": eef_flat.mean(axis=0).astype(float).tolist(),
        "std": eef_flat.std(axis=0).astype(float).tolist(),
        "min": eef_flat.min(axis=0).astype(float).tolist(),
        "max": eef_flat.max(axis=0).astype(float).tolist(),
        "p01": p01.astype(float).tolist(),
        "p99": p99.astype(float).tolist(),
        "normalized_abs_max": float(np.abs(normalized).max()),
        "joint_limit_violation_fraction": violations,
        "urdf": str(DEFAULT_URDF.resolve()),
        "urdf_sha256": hashlib.sha256(DEFAULT_URDF.read_bytes()).hexdigest(),
        "endpoint_link": converter.endpoint_link,
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
