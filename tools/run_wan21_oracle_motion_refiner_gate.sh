#!/usr/bin/env bash
# Generate normal/zero/cross-field videos and apply the strict train-only oracle gate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/wan21_oracle_motion_refiner_250/step-250.safetensors}"
ORACLE_ROOT="${ORACLE_ROOT:-$REPO/diagnostics/oracle_motion_field_gate}"
LABEL="${LABEL:-wan21_oracle_motion_refiner_250}"
LIMIT="${LIMIT:-8}"
STEPS="${STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GEN_NPROC="${GEN_NPROC:-1}"
PRED_ROOT="${PRED_ROOT:-$REPO/diagnostics/$LABEL}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export USE_TF="${USE_TF:-0}"
export DIFFSYNTH_REDIRECT_COMMON_FILES="false"
export DIFFSYNTH_SKIP_DOWNLOAD="true"

GENERATOR=("$REPO/.venv/bin/python")
if (( GEN_NPROC > 1 )); then
  GENERATOR+=( -m torch.distributed.run --standalone --nproc_per_node "$GEN_NPROC" )
fi
GENERATOR+=("$REPO/train/generate_wan21_oracle_motion_refiner.py")

"${GENERATOR[@]}" \
  --checkpoint "$CHECKPOINT" \
  --oracle-root "$ORACLE_ROOT" \
  --prediction-root "$PRED_ROOT" \
  --limit "$LIMIT" \
  --num-inference-steps "$STEPS" \
  --ablation all \
  --overwrite \
  --benchmark-json "$RESULT_ROOT/${LABEL}_benchmark.json"

for MODE in none zero batch-roll; do
  "$REPO/.venv/bin/python" "$REPO/tools/score_predictions.py" \
    --valset "$ORACLE_ROOT/valset" \
    --prediction-root "$PRED_ROOT/$MODE" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --out "$RESULT_ROOT/${LABEL}_${MODE}_holdout_scores.json"
done

"$REPO/.venv/bin/python" "$REPO/tools/compare_oracle_motion_refiner_gate.py" \
  --normal "$RESULT_ROOT/${LABEL}_none_holdout_scores.json" \
  --zero "$RESULT_ROOT/${LABEL}_zero_holdout_scores.json" \
  --batch-roll "$RESULT_ROOT/${LABEL}_batch-roll_holdout_scores.json" \
  --out "$RESULT_ROOT/${LABEL}_gate.json"
