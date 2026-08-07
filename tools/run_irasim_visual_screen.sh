#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
ROOT="${ROOT:-$REPO/open/baseline/outputs/irasim_faithful_500}"
VALSET="${VALSET:-$REPO/valset_holdout}"
LIMIT="${LIMIT:-8}"
STEPS="${INFERENCE_STEPS:-50}"

run_one() {
  local step="$1"
  local branch="$2"
  local label="irasim_faithful_${step}_${branch}"
  "$PYTHON" "$REPO/train/generate_irasim_videos.py" \
    --checkpoint "$ROOT/step-${step}.pt" \
    --weight-branch "$branch" \
    --challenge-root "$VALSET" \
    --prediction-root "$REPO/diagnostics/$label" \
    --limit "$LIMIT" \
    --num-inference-steps "$STEPS" \
    --action-mode delta_step \
    --overwrite \
    --benchmark-json "$REPO/results/${label}_benchmark.json"
}

# Same samples and seeds: isolate optimizer-step and EMA-lag effects.
run_one 250 model
run_one 500 model
run_one 500 ema
