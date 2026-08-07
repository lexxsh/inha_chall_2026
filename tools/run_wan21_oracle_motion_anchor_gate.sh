#!/usr/bin/env bash
# CPU composite of existing oracle videos, followed by the frozen local feature gate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$REPO/diagnostics/wan21_oracle_motion_refiner_250}"
ORACLE_ROOT="${ORACLE_ROOT:-$REPO/diagnostics/oracle_motion_field_gate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/diagnostics/wan21_oracle_motion_refiner_250_anchored}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
LABEL="${LABEL:-wan21_oracle_motion_refiner_250_anchored}"
LIMIT="${LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"

"$REPO/.venv/bin/python" "$REPO/tools/postprocess_oracle_motion_anchor.py" \
  --prediction-root "$INPUT_ROOT" \
  --oracle-root "$ORACLE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --report "$RESULT_ROOT/${LABEL}_composite.json" \
  --limit "$LIMIT" \
  --overwrite

for MODE in none zero batch-roll; do
  "$REPO/.venv/bin/python" "$REPO/tools/score_predictions.py" \
    --valset "$ORACLE_ROOT/valset" \
    --prediction-root "$OUTPUT_ROOT/$MODE" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --out "$RESULT_ROOT/${LABEL}_${MODE}_holdout_scores.json"
done

"$REPO/.venv/bin/python" "$REPO/tools/compare_oracle_motion_refiner_gate.py" \
  --normal "$RESULT_ROOT/${LABEL}_none_holdout_scores.json" \
  --zero "$RESULT_ROOT/${LABEL}_zero_holdout_scores.json" \
  --batch-roll "$RESULT_ROOT/${LABEL}_batch-roll_holdout_scores.json" \
  --out "$RESULT_ROOT/${LABEL}_gate.json"
