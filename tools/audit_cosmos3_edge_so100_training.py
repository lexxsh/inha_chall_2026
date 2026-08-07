"""CPU audit for the Cosmos3-Edge SO-100 native action-SFT contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from cosmos3_edge_so100_dataset import (  # noqa: E402
    ACTION_STEPS,
    DOMAIN_ID,
    EMBODIMENT_TYPE,
    NUM_VIDEO_FRAMES,
    SO100ForwardDynamicsRawDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=REPO / "inha_worldmodel_scratch_training/cosmos/train/episode_index.parquet",
    )
    parser.add_argument("--holdout-manifest", type=Path, default=REPO / "valset_holdout/manifest.json")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=REPO / "results/cosmos3_edge_so100_native_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SO100ForwardDynamicsRawDataset(
        root=args.train_root,
        index_path=args.index_path,
        holdout_manifest=args.holdout_manifest,
        sample_stride=4,
    )
    sample_indices = np.linspace(0, len(dataset) - 1, num=max(1, args.samples), dtype=np.int64)
    reports = []
    for sample_index in sample_indices:
        sample = dataset[int(sample_index)]
        record, start = dataset.resolve_index(int(sample_index))
        actions, states = dataset._read_episode_table(str(dataset.root / str(record.parquet)))
        commands = actions[start : start + ACTION_STEPS]
        source_state = states[start]
        deploy_source = dataset.converter.estimate_source_action(commands[0])
        correct = sample["action"].numpy()
        old_identity_bug = dataset.converter.build_raw_actions(commands, source_action=commands[0])
        normalized = dataset.converter.normalize(correct, clamp=False)
        video = sample["video"].numpy()
        reports.append(
            {
                "dataset": str(record.dataset),
                "episode": int(record.episode_index),
                "start": int(start),
                "video_shape": list(video.shape),
                "action_shape": list(correct.shape),
                "frame0_to_frame1_abs_mean": float(np.abs(video[:, 1].astype(float) - video[:, 0]).mean()),
                "correct_action0_pose_abs_mean": float(np.abs(correct[0, :9] - np.r_[0, 0, 0, 1, 0, 0, 0, 1, 0]).mean()),
                "old_bug_action0_pose_abs_mean": float(
                    np.abs(old_identity_bug[0, :9] - np.r_[0, 0, 0, 1, 0, 0, 0, 1, 0]).mean()
                ),
                "deploy_source_state_mae": float(np.abs(deploy_source - source_state).mean()),
                "normalized_outside_unit_fraction": float(np.mean(np.abs(normalized) > 1.0)),
                "all_finite": bool(np.isfinite(correct).all() and np.isfinite(normalized).all()),
            }
        )

    held_overlap = sum(
        (str(row.dataset), int(row.episode_index)) in dataset.held_episodes
        for row in dataset.episodes.itertuples(index=False)
    )
    correct_first_motion = np.asarray([row["correct_action0_pose_abs_mean"] for row in reports])
    old_first_motion = np.asarray([row["old_bug_action0_pose_abs_mean"] for row in reports])
    criteria = {
        "shape_contract_17_video_16_action": all(
            row["video_shape"][1] == NUM_VIDEO_FRAMES and row["action_shape"] == [ACTION_STEPS, 10]
            for row in reports
        ),
        "all_finite": all(row["all_finite"] for row in reports),
        "holdout_episode_overlap_zero": held_overlap == 0,
        "first_transition_not_forced_identity": bool(np.median(correct_first_motion) > 1.0e-5),
        "old_identity_bug_reproduced": bool(np.max(old_first_motion) < 1.0e-6),
        "fixed_camera_viewpoint": dataset.viewpoint == "third_person_view",
        "official_bridge_domain": EMBODIMENT_TYPE == "bridge_orig_lerobot" and DOMAIN_ID == 7,
    }
    result = {
        "dataset_windows": len(dataset),
        "episodes": len(dataset.episodes),
        "sample_stride": dataset.sample_stride,
        "heldout_episodes": len(dataset.held_episodes),
        "heldout_overlap": held_overlap,
        "median_correct_action0_pose_abs_mean": float(np.median(correct_first_motion)),
        "median_old_bug_action0_pose_abs_mean": float(np.median(old_first_motion)),
        "median_deploy_source_state_mae": float(np.median([row["deploy_source_state_mae"] for row in reports])),
        "median_normalized_outside_unit_fraction": float(
            np.median([row["normalized_outside_unit_fraction"] for row in reports])
        ),
        "criteria": criteria,
        "samples": reports,
        "verdict": "PASS_NATIVE_COSMOS3_CONTRACT" if all(criteria.values()) else "REJECT_NATIVE_COSMOS3_CONTRACT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
