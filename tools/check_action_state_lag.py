"""Train parquet만 사용해 action과 observation.state의 시간 정렬을 진단한다.

이 검사는 물리적 command/state 시차를 찾을 뿐, 공식 Action extractor에 가장 유리한
생성 정렬을 직접 결정하지는 않는다. 그 구분은 EMPIRICAL.md 8절을 참고한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "open/data/train"


def mean_of_episode_means(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-lag", type=int, default=3)
    args = parser.parse_args()

    files = sorted(args.root.glob("*/*/data/chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"parquet을 찾지 못했다: {args.root}")
    if args.sample_size <= 0:
        raise SystemExit("--sample-size는 양수여야 한다")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_size, len(files))
    picks = rng.choice(len(files), size=sample_count, replace=False)
    lags = list(range(-args.max_lag, args.max_lag + 1))

    lag_abs_sum = {lag: 0.0 for lag in lags}
    lag_element_count = {lag: 0 for lag in lags}
    best_lag_counts: Counter[int] = Counter()
    episode_metrics: dict[str, list[float]] = {
        "anchor_same": [],
        "anchor_shifted": [],
        "step_same": [],
        "step_shifted": [],
        "command_residual": [],
        "anchor_same_corr": [],
        "anchor_shifted_corr": [],
    }
    skipped = 0

    for pick in picks:
        try:
            table = pd.read_parquet(
                files[int(pick)], columns=["action", "observation.state"]
            )
            action = np.stack(table["action"].to_numpy()).astype(np.float64)
            state = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
        except (OSError, ValueError, KeyError):
            skipped += 1
            continue

        length = min(len(action), len(state))
        action, state = action[:length], state[:length]
        if length < max(4, args.max_lag + 1):
            skipped += 1
            continue

        per_episode_lag = {}
        for lag in lags:
            if lag > 0:
                aligned_action, aligned_state = action[:-lag], state[lag:]
            elif lag < 0:
                aligned_action, aligned_state = action[-lag:], state[:lag]
            else:
                aligned_action, aligned_state = action, state
            absolute_error = np.abs(aligned_action - aligned_state)
            lag_abs_sum[lag] += float(absolute_error.sum())
            lag_element_count[lag] += int(absolute_error.size)
            per_episode_lag[lag] = float(absolute_error.mean())
        best_lag_counts[min(per_episode_lag, key=per_episode_lag.get)] += 1

        action_anchor = action - action[:1]
        state_anchor = state - state[:1]
        shifted_action_anchor = np.concatenate([action[:1], action[:-1]], axis=0) - action[:1]
        episode_metrics["anchor_same"].append(float(np.abs(action_anchor - state_anchor).mean()))
        episode_metrics["anchor_shifted"].append(
            float(np.abs(shifted_action_anchor - state_anchor).mean())
        )
        episode_metrics["anchor_same_corr"].append(
            float(np.corrcoef(action_anchor.ravel(), state_anchor.ravel())[0, 1])
        )
        episode_metrics["anchor_shifted_corr"].append(
            float(np.corrcoef(shifted_action_anchor.ravel(), state_anchor.ravel())[0, 1])
        )

        action_step = np.diff(action, axis=0)
        state_step = np.diff(state, axis=0)
        episode_metrics["step_same"].append(float(np.abs(action_step - state_step).mean()))
        episode_metrics["step_shifted"].append(
            float(np.abs(action_step[:-1] - state_step[1:]).mean())
        )
        episode_metrics["command_residual"].append(
            float(np.abs((action[:-1] - state[:-1]) - state_step).mean())
        )

    valid = sum(best_lag_counts.values())
    print(f"parquet={len(files):,} sampled={sample_count:,} valid={valid:,} skipped={skipped:,}")
    print("\nraw action[t] vs state[t+lag] weighted MAE (degree)")
    for lag in lags:
        value = lag_abs_sum[lag] / lag_element_count[lag]
        print(f"  lag {lag:+d}: {value:.4f}")
    print("\nper-episode best lag counts")
    for lag in lags:
        if best_lag_counts[lag]:
            print(f"  lag {lag:+d}: {best_lag_counts[lag]:,}")

    print("\nchange representation (mean of episode MAE)")
    print(f"  anchor same    : {mean_of_episode_means(episode_metrics['anchor_same']):.4f}")
    print(f"  anchor shifted : {mean_of_episode_means(episode_metrics['anchor_shifted']):.4f}")
    print(f"  step same      : {mean_of_episode_means(episode_metrics['step_same']):.4f}")
    print(f"  step shifted   : {mean_of_episode_means(episode_metrics['step_shifted']):.4f}")
    print(f"  command residual: {mean_of_episode_means(episode_metrics['command_residual']):.4f}")
    print("\nanchor correlation (mean of episode correlations)")
    print(f"  same    : {np.nanmean(episode_metrics['anchor_same_corr']):.4f}")
    print(f"  shifted : {np.nanmean(episode_metrics['anchor_shifted_corr']):.4f}")


if __name__ == "__main__":
    main()
