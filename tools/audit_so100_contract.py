"""Audit the canonical SO-100 command/state/frame mapping on train data."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train.so100_transition_contract import build_transition_contract  # noqa: E402


class Moments:
    def __init__(self, dim: int = 6) -> None:
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.square = np.zeros(dim, dtype=np.float64)

    def add(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float64).reshape(-1, len(self.total))
        self.count += len(array)
        self.total += array.sum(axis=0)
        self.square += np.square(array).sum(axis=0)

    def result(self) -> dict:
        mean = self.total / max(1, self.count)
        variance = np.maximum(self.square / max(1, self.count) - np.square(mean), 0)
        return {"count": self.count, "mean": mean.tolist(), "std": np.sqrt(variance).tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument(
        "--manifest", type=Path, default=REPO / "results/so100_contract_manifest.jsonl"
    )
    parser.add_argument("--samples-per-split", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output", type=Path, default=REPO / "results/so100_contract_audit.json"
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sampled(records: list[dict], count: int, seed: int) -> list[dict]:
    order = list(records)
    random.Random(seed).shuffle(order)
    return order[: min(count, len(order))]


def main() -> None:
    args = parse_args()
    records = load_records(args.manifest)
    if not records:
        raise SystemExit(f"empty manifest: {args.manifest}")
    selected = []
    for offset, split in enumerate(("train", "holdout")):
        selected.extend(
            sampled([row for row in records if row["split"] == split], args.samples_per_split, args.seed + offset)
        )

    metrics: dict[str, list[float]] = defaultdict(list)
    moments = {
        name: Moments()
        for name in (
            "absolute_command",
            "source_state",
            "source_anchor_delta",
            "deployable_step",
            "target_residual",
            "realized_delta",
        )
    }
    failures: list[str] = []
    per_split = defaultdict(int)
    next_better = 0
    for record in selected:
        try:
            table = pd.read_parquet(
                args.data_root / record["data_rel"],
                columns=["action", "observation.state"],
            )
            indices = list(range(record["start"], record["start"] + record["frames"]))
            actions = np.stack(table["action"].iloc[indices].to_numpy()).astype(np.float32)
            states = np.stack(table["observation.state"].iloc[indices].to_numpy()).astype(np.float32)
            contract = build_transition_contract(actions, states)
        except (OSError, KeyError, ValueError, IndexError) as error:
            failures.append(f"{record['id']}: {error}")
            continue
        if len(contract.commands) != 15 or record["excluded_command_index"] != record["start"] + 15:
            failures.append(f"{record['id']}: alignment invariant failed")
            continue
        per_split[record["split"]] += 1
        current_mae = float(np.abs(contract.commands - contract.current_states).mean())
        next_mae = float(np.abs(contract.commands - contract.next_states).mean())
        next_better += int(next_mae < current_mae)
        metrics["action_to_current_mae"].append(current_mae)
        metrics["action_to_next_mae"].append(next_mae)
        metrics["state0_action0_proxy_mae"].append(
            float(np.abs(contract.commands[0] - contract.current_states[0]).mean())
        )
        metrics["target_residual_to_realized_mae"].append(
            float(np.abs(contract.target_residual - contract.realized_delta).mean())
        )
        metrics["deployable_step_to_realized_mae"].append(
            float(np.abs(contract.deployable_step - contract.realized_delta).mean())
        )
        moments["absolute_command"].add(contract.commands)
        moments["source_state"].add(contract.current_states[:1])
        moments["source_anchor_delta"].add(contract.source_anchor_delta)
        moments["deployable_step"].add(contract.deployable_step)
        moments["target_residual"].add(contract.target_residual)
        moments["realized_delta"].add(contract.realized_delta)

    valid = sum(per_split.values())
    train_uploaders = {r["uploader"] for r in records if r["split"] == "train"}
    holdout_uploaders = {r["uploader"] for r in records if r["split"] == "holdout"}
    result = {
        "manifest": str(args.manifest.resolve()),
        "selected_records": len(selected),
        "valid_records": valid,
        "valid_by_split": dict(per_split),
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "uploader_overlap": sorted(train_uploaders & holdout_uploaders),
        "alignment": {
            "source_frame": 0,
            "target_frames": [1, 15],
            "used_action_indices": [0, 14],
            "excluded_action_index": 15,
            "action_to_current_mae": float(np.mean(metrics["action_to_current_mae"])),
            "action_to_next_mae": float(np.mean(metrics["action_to_next_mae"])),
            "next_state_better_fraction": next_better / max(1, valid),
        },
        "proxy_errors": {
            key: float(np.mean(value)) for key, value in metrics.items() if key not in {"action_to_current_mae", "action_to_next_mae"}
        },
        "train_only_feature_statistics": {name: value.result() for name, value in moments.items()},
    }
    passed = (
        valid == len(selected)
        and not failures
        and not result["uploader_overlap"]
        and result["alignment"]["action_to_next_mae"] < result["alignment"]["action_to_current_mae"]
        and result["alignment"]["next_state_better_fraction"] >= 0.9
    )
    result["verdict"] = "PASS_DATA_CONTRACT" if passed else "FAIL_DATA_CONTRACT"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
