#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/flow_world_2k/step-2000.safetensors}"
LABEL="${LABEL:-flow_world_2k}"
VALSET="${VALSET:-$REPO/valset_holdout}"
LIMIT="${LIMIT:-24}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/flow_world_gate}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"

run_one() {
  local variant="$1"
  local label="${LABEL}_${variant}"
  "$PYTHON" "$REPO/train/generate_flow_videos.py" \
    --checkpoint "$CHECKPOINT" --challenge-root "$VALSET" \
    --prediction-root "$OUT_ROOT/$label" --limit "$LIMIT" --batch-size "$BATCH_SIZE" \
    --action-ablation "$variant" --overwrite
  "$PYTHON" "$REPO/tools/score_predictions.py" \
    --valset "$VALSET" --prediction-root "$OUT_ROOT/$label" --limit "$LIMIT" \
    --out "$RESULT_ROOT/${label}_holdout_scores.json"
}

run_one none
run_one zero
run_one batch-roll
"$PYTHON" "$REPO/tools/compare_generation_gate.py" \
  --normal "$RESULT_ROOT/${LABEL}_none_holdout_scores.json" \
  --zero "$RESULT_ROOT/${LABEL}_zero_holdout_scores.json" \
  --batch-roll "$RESULT_ROOT/${LABEL}_batch-roll_holdout_scores.json" \
  --incumbent "$REPO/results/wan_action_v2_2k_none_holdout_scores.json" \
  --out "$RESULT_ROOT/${LABEL}_gate.json"
