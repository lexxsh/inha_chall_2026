#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/wan_oracle_track_250/step-250.safetensors}"
MANIFEST="${MANIFEST:-$REPO/diagnostics/spatial_control_gate/manifest.json}"
VALSET="${VALSET:-$REPO/diagnostics/spatial_control_gate/valset}"
LABEL="${LABEL:-wan_oracle_track_250}"
LIMIT="${LIMIT:-8}"
STEPS="${INFERENCE_STEPS:-20}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/wan_oracle_gate}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"

EXTRA_ARGS=()
if [ "${STRUCTURAL_ZERO:-0}" = "1" ]; then EXTRA_ARGS+=(--structural-zero); fi
if [ "${ADAPTER_ONLY:-0}" = "1" ]; then EXTRA_ARGS+=(--adapter-only); fi

run_one() {
  local variant="$1"
  local output="$OUT_ROOT/${LABEL}_${variant}"
  "$PYTHON" "$REPO/train/generate_wan_oracle_videos.py" \
    --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --valset "$VALSET" \
    --prediction-root "$output" --ablation "$variant" --limit "$LIMIT" \
    "${EXTRA_ARGS[@]}" \
    --num-inference-steps "$STEPS" --cfg-scale 1 --overwrite \
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
  --incumbent "$RESULT_ROOT/${LABEL}_zero_holdout_scores.json" \
  --out "$RESULT_ROOT/${LABEL}_gate.json"
