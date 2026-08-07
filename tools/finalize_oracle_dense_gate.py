"""Combine oracle dense-flow pixel diagnostics with official feature scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_delta(score: dict, key: str) -> np.ndarray:
    prediction = {row["sample_id"]: row for row in score["per_sample"]}
    static = {row["sample_id"]: row for row in score["static_per_sample"]}
    if prediction.keys() != static.keys():
        raise SystemExit("prediction/static sample IDs do not match")
    return np.asarray(
        [prediction[sample_id][key] - static[sample_id][key] for sample_id in sorted(prediction)],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-json", required=True)
    parser.add_argument("--score-json", required=True)
    parser.add_argument(
        "--incumbent-score-json",
        default=None,
        help="Paired Dream-10k score JSON on the exact same holdout.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-point-improvement", type=float, default=0.01)
    args = parser.parse_args()

    pixel = json.loads(Path(args.pixel_json).read_text())
    score = json.loads(Path(args.score_json).read_text())
    weighted_delta = paired_delta(score, "weighted")
    action_delta = paired_delta(score, "action")
    weighted_ci = bootstrap_ci(weighted_delta, seed=0)
    action_ci = bootstrap_ci(action_delta, seed=1)

    pixel_pass = bool(pixel["pixel_gate"])
    strict_metric_pass = weighted_ci[1] < 0.0 and action_ci[1] < 0.0
    point_metric_pass = (
        weighted_delta.mean() <= -args.min_point_improvement
        and action_delta.mean() <= -args.min_point_improvement
    )
    incumbent_report: dict[str, object] = {}
    strict_incumbent_pass = True
    point_incumbent_pass = True
    if args.incumbent_score_json:
        incumbent = json.loads(Path(args.incumbent_score_json).read_text())
        oracle_rows = {row["sample_id"]: row for row in score["per_sample"]}
        incumbent_rows = {row["sample_id"]: row for row in incumbent["per_sample"]}
        if oracle_rows.keys() != incumbent_rows.keys():
            raise SystemExit("oracle/incumbent sample IDs do not match")
        incumbent_delta = np.asarray(
            [
                oracle_rows[sample_id]["weighted"] - incumbent_rows[sample_id]["weighted"]
                for sample_id in sorted(oracle_rows)
            ],
            dtype=np.float64,
        )
        incumbent_ci = bootstrap_ci(incumbent_delta, seed=2)
        strict_incumbent_pass = incumbent_ci[1] < 0.0
        point_incumbent_pass = incumbent_delta.mean() <= -args.min_point_improvement
        incumbent_report = {
            "incumbent_weighted": float(incumbent["prediction"]["weighted"]),
            "oracle_minus_incumbent_weighted": float(incumbent_delta.mean()),
            "oracle_minus_incumbent_weighted_ci95": incumbent_ci,
            "strict_incumbent_gate": bool(strict_incumbent_pass),
            "point_incumbent_gate": bool(point_incumbent_pass),
        }

    if not pixel_pass:
        verdict = "STOP_DENSE_WARP_PATH"
    elif strict_metric_pass and strict_incumbent_pass:
        verdict = "PASS_ORACLE_PRESERVATION_ONLY"
    elif point_metric_pass and point_incumbent_pass:
        verdict = "INCONCLUSIVE_EXPAND_HOLDOUT"
    else:
        # The train-only proxy ranked Dream-10k below static while the actual
        # leaderboard ranked it far above static (0.28188 vs ~0.517).  A local
        # feature conflict is therefore not sufficient evidence to kill a
        # representation whose direct GT reconstruction gate passed.
        verdict = "HOLD_LOCAL_METRIC_CONFLICT"

    report = {
        "scope_warning": (
            "Future GT optical flow was used. PASS validates only the dense spatial representation; "
            "it does not validate joint-action-to-control prediction or a video generator."
        ),
        "count": int(score["count"]),
        "pixel_gate": pixel_pass,
        "median_full_l1_reduction_fraction": pixel["median_full_l1_reduction_fraction"],
        "median_motion_l1_reduction_fraction": pixel["median_motion_l1_reduction_fraction"],
        "oracle_weighted": float(score["prediction"]["weighted"]),
        "static_weighted": float(score["static"]["weighted"]),
        "oracle_minus_static_weighted": float(weighted_delta.mean()),
        "oracle_minus_static_weighted_ci95": weighted_ci,
        "oracle_action": float(score["prediction"]["action"]),
        "static_action": float(score["static"]["action"]),
        "oracle_minus_static_action": float(action_delta.mean()),
        "oracle_minus_static_action_ci95": action_ci,
        "strict_metric_gate": bool(strict_metric_pass),
        "point_metric_gate": bool(point_metric_pass),
        **incumbent_report,
        "verdict": verdict,
        "next_step": {
            "PASS_ORACLE_PRESERVATION_ONLY": (
                "Run a tiny Dream dense-control overfit gate, then separately learn action-to-control."
            ),
            "INCONCLUSIVE_EXPAND_HOLDOUT": (
                "Regenerate this oracle gate on at least 24 dataset-group holdout clips; do not train yet."
            ),
            "HOLD_LOCAL_METRIC_CONFLICT": (
                "Inspect oracle montages and expand the pixel/counterfactual holdout; local weighted score alone "
                "cannot reject the path because it mis-ranked the public Dream/static result."
            ),
            "STOP_DENSE_WARP_PATH": (
                "Do not build the Dream spatial adapter; keep Dream/static incumbent and revisit representation."
            ),
        }[verdict],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
