#!/usr/bin/env bash
# 10k checkpoint가 올바른 action에 유리한지 zero/wrong-action과 비교한다.
# 사용: CUDA_VISIBLE_DEVICES=7 LIMIT=24 BATCH_SIZE=2 bash tools/run_action_gate.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
VALSET="${VALSET:-$REPO/valset_holdout}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/action_gate}"
LIMIT="${LIMIT:-24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CHECKPOINT="$REPO/open/baseline/outputs/full_unet/inha_full_unet/checkpoints/epoch=7-step=10000.ckpt"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0

run_one() {
    local variant="$1"
    local label="step10k_${variant}_eta1"
    local prediction_root="$OUT_ROOT/$label"

    "$PYTHON" "$REPO/train/generate_videos.py" \
        --checkpoint "$CHECKPOINT" \
        --challenge-root "$VALSET" \
        --prediction-root "$prediction_root" \
        --limit "$LIMIT" \
        --batch-size "$BATCH_SIZE" \
        --seed 0 \
        --ddim-steps 50 \
        --ddim-eta 1.0 \
        --guidance-scale 1.0 \
        --action-ablation "$variant" \
        --no-ema

    "$PYTHON" "$REPO/tools/score_predictions.py" \
        --valset "$VALSET" \
        --prediction-root "$prediction_root" \
        --limit "$LIMIT" \
        --out "$REPO/results/${label}_holdout_scores.json"
}

run_one "zero"
run_one "batch-roll"

echo "action gate complete: $OUT_ROOT"
