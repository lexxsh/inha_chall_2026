"""Compare a frozen action-conditioned candidate against static/incumbent gates.

This script intentionally consumes train-only holdout score JSON files, never
eval/public-leaderboard feedback. Lower is better for every component.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def paired_delta(
    a: dict, b: dict, key: str, *, allow_b_superset: bool = False
) -> np.ndarray:
    """Return per-sample A-B after checking identical sample IDs."""
    ar = {row["sample_id"]: row for row in a["per_sample"]}
    br = {row["sample_id"]: row for row in b["per_sample"]}
    compatible = ar.keys() == br.keys()
    if allow_b_superset:
        compatible = set(ar).issubset(br)
    if not compatible:
        raise SystemExit("score JSON의 sample ID가 달라 paired 비교를 할 수 없다")
    return np.asarray([ar[sid][key] - br[sid][key] for sid in sorted(ar)], dtype=np.float64)


def bootstrap_ci(values: np.ndarray, seed: int = 0, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--zero", required=True)
    ap.add_argument("--batch-roll", required=True)
    ap.add_argument("--incumbent", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    normal, zero, roll = load(args.normal), load(args.zero), load(args.batch_roll)
    static_w = float(normal["static"]["weighted"])
    normal_w = float(normal["prediction"]["weighted"])

    action_zero = paired_delta(normal, zero, "action")
    action_roll = paired_delta(normal, roll, "action")
    zero_ci = bootstrap_ci(action_zero)
    roll_ci = bootstrap_ci(action_roll, seed=1)

    report = {
        "normal_weighted": normal_w,
        "static_weighted": static_w,
        "normal_minus_static": normal_w - static_w,
        "normal_action_minus_zero": float(action_zero.mean()),
        "normal_action_minus_zero_ci95": list(zero_ci),
        "normal_action_minus_batch_roll": float(action_roll.mean()),
        "normal_action_minus_batch_roll_ci95": list(roll_ci),
    }

    if args.incumbent:
        incumbent = load(args.incumbent)
        # A cheap candidate screen may score only the first N fixed holdout
        # rows while the frozen incumbent JSON contains all 24.  Pair on the
        # candidate IDs, but never permit a candidate sample to be missing.
        delta = paired_delta(normal, incumbent, "weighted", allow_b_superset=True)
        report["normal_minus_incumbent"] = float(delta.mean())
        report["normal_minus_incumbent_ci95"] = list(bootstrap_ci(delta, seed=2))

    # Promotion is deliberately strict. Point improvements with a CI crossing
    # zero are useful but remain INCONCLUSIVE rather than triggering a long run.
    beats_static = normal_w < static_w
    action_correct = zero_ci[1] < 0.0 and roll_ci[1] < 0.0
    beats_incumbent = True
    if args.incumbent:
        beats_incumbent = report["normal_minus_incumbent_ci95"][1] < 0.0

    point_action_correct = action_zero.mean() < 0.0 and action_roll.mean() < 0.0
    point_beats_incumbent = True
    if args.incumbent:
        point_beats_incumbent = report["normal_minus_incumbent"] < 0.0

    if beats_static and action_correct and beats_incumbent:
        verdict = "PROMOTE"
    elif beats_static and point_action_correct and point_beats_incumbent:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECT"
    report["verdict"] = verdict

    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
