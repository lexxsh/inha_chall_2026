#!/usr/bin/env bash
# 같은 고정 holdout/seed/sampler에서 2k, 6k, 10k checkpoint만 비교한다.
# 사용: CUDA_VISIBLE_DEVICES=7 LIMIT=24 BATCH_SIZE=2 bash tools/run_step_screen.sh
# 일부만 추가 실행: STEPS="4 8" CUDA_VISIBLE_DEVICES=7 bash tools/run_step_screen.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
VALSET="${VALSET:-$REPO/valset_holdout}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/step_screen}"
LIMIT="${LIMIT:-24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CKPT_ROOT="$REPO/open/baseline/outputs/full_unet/inha_full_unet/checkpoints"
STEPS="${STEPS:-2 6 10}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0

run_one() {
    local label="$1"
    local checkpoint="$2"
    local prediction_root="$OUT_ROOT/$label"

    "$PYTHON" "$REPO/train/generate_videos.py" \
        --checkpoint "$checkpoint" \
        --challenge-root "$VALSET" \
        --prediction-root "$prediction_root" \
        --limit "$LIMIT" \
        --batch-size "$BATCH_SIZE" \
        --seed 0 \
        --ddim-steps 50 \
        --ddim-eta 1.0 \
        --guidance-scale 1.0 \
        --no-ema

    "$PYTHON" "$REPO/tools/score_predictions.py" \
        --valset "$VALSET" \
        --prediction-root "$prediction_root" \
        --limit "$LIMIT" \
        --out "$REPO/results/${label}_holdout_scores.json"
}

for step in $STEPS; do
    case "$step" in
        2) checkpoint="$CKPT_ROOT/epoch=1-step=2000.ckpt" ;;
        4) checkpoint="$CKPT_ROOT/epoch=2-step=4000.ckpt" ;;
        6) checkpoint="$CKPT_ROOT/epoch=4-step=6000.ckpt" ;;
        8) checkpoint="$CKPT_ROOT/epoch=5-step=8000.ckpt" ;;
        10) checkpoint="$CKPT_ROOT/epoch=7-step=10000.ckpt" ;;
        *) echo "지원하지 않는 step: $step (2, 4, 6, 8, 10 중 선택)" >&2; exit 2 ;;
    esac
    run_one "step${step}k_main_eta1" "$checkpoint"
done

echo "step screen complete: $OUT_ROOT"
