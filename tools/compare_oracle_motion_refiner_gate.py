"""Strict promotion gate for the train-only oracle motion-field renderer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def paired(a: dict, b: dict, key: str) -> np.ndarray:
    left = {row["sample_id"]: row for row in a["per_sample"]}
    right = {row["sample_id"]: row for row in b["per_sample"]}
    if left.keys() != right.keys():
        raise ValueError("Oracle score files contain different sample IDs")
    return np.asarray([left[sid][key] - right[sid][key] for sid in sorted(left)])


def ci95(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--zero", required=True)
    parser.add_argument("--batch-roll", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    normal, zero, roll = load(args.normal), load(args.zero), load(args.batch_roll)
    deltas = {}
    for index, key in enumerate(("weighted", "dino", "video", "action")):
        zero_delta = paired(normal, zero, key)
        roll_delta = paired(normal, roll, key)
        deltas[f"normal_minus_zero_{key}"] = float(zero_delta.mean())
        deltas[f"normal_minus_zero_{key}_ci95"] = ci95(zero_delta, 2 * index)
        deltas[f"normal_minus_batch_roll_{key}"] = float(roll_delta.mean())
        deltas[f"normal_minus_batch_roll_{key}_ci95"] = ci95(roll_delta, 2 * index + 1)

    normal_weighted = float(normal["prediction"]["weighted"])
    static_weighted = float(normal["static"]["weighted"])
    strict_control = (
        deltas["normal_minus_zero_weighted_ci95"][1] < 0
        and deltas["normal_minus_batch_roll_weighted_ci95"][1] < 0
    )
    point_control = (
        deltas["normal_minus_zero_weighted"] < 0
        and deltas["normal_minus_batch_roll_weighted"] < 0
    )
    component_sanity = (
        deltas["normal_minus_zero_dino"] < 0
        and deltas["normal_minus_zero_action"] < 0
        and deltas["normal_minus_batch_roll_dino"] < 0
        and deltas["normal_minus_batch_roll_action"] < 0
    )
    beats_static = normal_weighted < static_weighted
    if strict_control and component_sanity and beats_static:
        verdict = "PROMOTE_ACTION_TO_FIELD_STAGE"
    elif point_control and component_sanity and beats_static:
        verdict = "INCONCLUSIVE_REPEAT_WITH_24_HOLDOUTS"
    else:
        verdict = "REJECT_MOTION_FIELD_REFINER"
    report = {
        "scope": "train-only oracle; never submit these predictions",
        "normal_weighted": normal_weighted,
        "static_weighted": static_weighted,
        "normal_minus_static": normal_weighted - static_weighted,
        **deltas,
        "criteria": {
            "weighted_normal_better_than_zero_and_roll_ci95": bool(strict_control),
            "dino_and_action_point_sanity": bool(component_sanity),
            "normal_beats_static": bool(beats_static),
        },
        "verdict": verdict,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
