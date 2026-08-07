#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/wan_action_1k/step-1000.safetensors}"
LABEL="${LABEL:-wan_action_1k}"
VALSET="${VALSET:-$REPO/valset_holdout}"
LIMIT="${LIMIT:-24}"
STEPS="${INFERENCE_STEPS:-20}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/wan_action_gate}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
ACTION_MODE="${ACTION_MODE:-delta}"
ACTION_VERSION="${ACTION_VERSION:-v1}"
PROMPT="${PROMPT-A fixed-camera video of a robot arm manipulating objects.}"

run_one() {
  local variant="$1"
  local label="${LABEL}_${variant}"
  "$PYTHON" "$REPO/train/generate_wan_videos.py" \
    --checkpoint "$CHECKPOINT" --challenge-root "$VALSET" \
    --prediction-root "$OUT_ROOT/$label" --limit "$LIMIT" \
    --num-inference-steps "$STEPS" --cfg-scale 1 --action-ablation "$variant" --overwrite \
    --action-mode "$ACTION_MODE" --action-conditioner-version "$ACTION_VERSION" --prompt "$PROMPT"
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
  --incumbent "$REPO/results/step10k_main_eta1_holdout_scores.json" \
  --out "$RESULT_ROOT/${LABEL}_gate.json"
