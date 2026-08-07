#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
VALSET="${VALSET:-$REPO/diagnostics/spatial_control_gate/valset}"
OUTPUT="${OUTPUT:-$REPO/diagnostics/oracle_dense_control_gate}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
LIMIT="${LIMIT:-8}"
INCUMBENT_ROOT="${INCUMBENT_ROOT:-$OUTPUT/dream_10k_incumbent}"
INCUMBENT_SCORE="$RESULT_ROOT/oracle_gate_dream_10k_holdout_scores.json"
CHECKPOINT="${DREAM_CHECKPOINT:-$REPO/open/baseline/outputs/full_unet/inha_full_unet/checkpoints/epoch=7-step=10000.ckpt}"

"$PYTHON" "$REPO/tools/oracle_dense_control_gate.py" \
  --valset "$VALSET" \
  --output "$OUTPUT" \
  --result-json "$RESULT_ROOT/oracle_dense_control_pixels.json" \
  --limit "$LIMIT" \
  --overwrite

"$PYTHON" "$REPO/tools/score_predictions.py" \
  --valset "$VALSET" \
  --prediction-root "$OUTPUT/predictions" \
  --limit "$LIMIT" \
  --out "$RESULT_ROOT/oracle_dense_control_holdout_scores.json"

# The public Dream-10k result (0.28188) is the incumbent.  Re-generate it on
# this exact group holdout so the oracle is never promoted merely for beating
# the much weaker static baseline.
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python USE_TF=0 \
"$PYTHON" "$REPO/train/generate_videos.py" \
  --checkpoint "$CHECKPOINT" \
  --challenge-root "$VALSET" \
  --prediction-root "$INCUMBENT_ROOT" \
  --limit "$LIMIT" \
  --batch-size 4 \
  --no-ema \
  --overwrite

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python USE_TF=0 \
"$PYTHON" "$REPO/tools/score_predictions.py" \
  --valset "$VALSET" \
  --prediction-root "$INCUMBENT_ROOT" \
  --limit "$LIMIT" \
  --out "$INCUMBENT_SCORE"

"$PYTHON" "$REPO/tools/finalize_oracle_dense_gate.py" \
  --pixel-json "$RESULT_ROOT/oracle_dense_control_pixels.json" \
  --score-json "$RESULT_ROOT/oracle_dense_control_holdout_scores.json" \
  --incumbent-score-json "$INCUMBENT_SCORE" \
  --out "$RESULT_ROOT/oracle_dense_control_gate.json"
