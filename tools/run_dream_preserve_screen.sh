#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
VALSET="${VALSET:-$REPO/valset_holdout}"
INPUT="${INPUT:-$REPO/diagnostics/step_screen/step10k_main_eta1}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/dream_preserve_screen}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
LIMIT="${LIMIT:-24}"

run_variant() {
  local label="$1"
  shift
  local output="$OUT_ROOT/$label"
  "$PYTHON" "$REPO/tools/postprocess_dream_preserve.py" \
    --prediction-root "$INPUT" \
    --challenge-root "$VALSET" \
    --output-root "$output" \
    --limit "$LIMIT" \
    --overwrite \
    "$@"
  "$PYTHON" "$REPO/tools/score_predictions.py" \
    --valset "$VALSET" \
    --prediction-root "$output" \
    --limit "$LIMIT" \
    --out "$RESULT_ROOT/dream_preserve_${label}_holdout_scores.json"
}

# Conservative global attenuation and two motion-localized variants.  The
# latter keep the initial image bit-for-bit outside a feathered edit mask.
run_variant global85 --mode global-blend --alpha 0.85
run_variant mask06 --mode motion-mask --threshold 0.06 --alpha 1.0
run_variant mask04a85 --mode motion-mask --threshold 0.04 --alpha 0.85
