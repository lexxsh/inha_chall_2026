#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
: "${CHECKPOINT:?Set CHECKPOINT to a faithful SO-100 IRASim checkpoint (not the old 17-frame run)}"
VALSET="${VALSET:-$REPO/valset_holdout}"
LABEL="${LABEL:-irasim_faithful}"
LIMIT="${LIMIT:-24}"
STEPS="${INFERENCE_STEPS:-50}"
BRANCH="${WEIGHT_BRANCH:-model}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/irasim_gate}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"

run_one() {
  local variant="$1"
  local output="$OUT_ROOT/${LABEL}_${variant}"
  "$PYTHON" "$REPO/train/generate_irasim_videos.py" \
    --checkpoint "$CHECKPOINT" --challenge-root "$VALSET" \
    --weight-branch "$BRANCH" \
    --prediction-root "$output" --limit "$LIMIT" \
    --num-inference-steps "$STEPS" --action-mode "${ACTION_MODE:-delta_step}" \
    --action-ablation "$variant" --overwrite \
    --benchmark-json "$RESULT_ROOT/${LABEL}_${variant}_benchmark.json"
  "$PYTHON" "$REPO/tools/score_predictions.py" \
    --valset "$VALSET" --prediction-root "$output" --limit "$LIMIT" \
    --out "$RESULT_ROOT/${LABEL}_${variant}_holdout_scores.json"
}

run_one none
run_one zero
run_one batch-roll

"$PYTHON" "$REPO/tools/compare_generation_gate.py" \
  --normal "$RESULT_ROOT/${LABEL}_none_holdout_scores.json" \
  --zero "$RESULT_ROOT/${LABEL}_zero_holdout_scores.json" \
  --batch-roll "$RESULT_ROOT/${LABEL}_batch-roll_holdout_scores.json" \
  --incumbent "$RESULT_ROOT/step10k_main_eta1_holdout_scores.json" \
  --out "$RESULT_ROOT/${LABEL}_gate.json"
