"""Rank 1k action-representation screens using paired normal-vs-wrong actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def paired(a: dict, b: dict, key: str) -> np.ndarray:
    ar = {row["sample_id"]: row for row in a["per_sample"]}
    br = {row["sample_id"]: row for row in b["per_sample"]}
    if ar.keys() != br.keys():
        raise SystemExit("representation 결과의 sample ID가 다르다")
    return np.asarray([ar[sid][key] - br[sid][key] for sid in sorted(ar)])


def ci95(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(20_000, n))].mean(1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", nargs=3, action="append", required=True,
        metavar=("LABEL", "NORMAL_JSON", "BATCH_ROLL_JSON"),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for seed, (label, normal_path, roll_path) in enumerate(args.variant):
        normal, roll = read(normal_path), read(roll_path)
        action_delta = paired(normal, roll, "action")
        weighted_delta = paired(normal, roll, "weighted")
        action_ci = ci95(action_delta, seed)
        robust = action_ci[1] < 0.0
        point_correct = float(action_delta.mean()) < 0.0 and int((action_delta < 0).sum()) >= 14
        rows.append(
            {
                "variant": label,
                "normal": normal["prediction"],
                "static_weighted": normal["static"]["weighted"],
                "normal_minus_roll_action": float(action_delta.mean()),
                "normal_minus_roll_action_ci95": action_ci,
                "normal_action_wins": int((action_delta < 0).sum()),
                "normal_minus_roll_weighted": float(weighted_delta.mean()),
                "robust_action_correct": robust,
                "point_action_correct": point_correct,
            }
        )

    robust = [row for row in rows if row["robust_action_correct"]]
    point = [row for row in rows if row["point_action_correct"]]
    if robust:
        selected = min(robust, key=lambda row: row["normal"]["weighted"])["variant"]
        verdict = "PROMOTE_TO_2K"
    elif point:
        selected = min(point, key=lambda row: row["normal_minus_roll_action"])["variant"]
        verdict = "INCONCLUSIVE_2K_CONFIRM"
    else:
        selected = None
        verdict = "REJECT_LOW_DIM_FRAME_ADALN"

    result = {"verdict": verdict, "selected": selected, "variants": rows}
    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
