#!/usr/bin/env bash
# Generate and score the Stage-1 Frame-Ada candidate on the fixed train-only holdout.
#
#   CUDA_VISIBLE_DEVICES=7 bash tools/run_frame_adaln_gate.sh
#   CHECKPOINT=/path/to/step.ckpt LIMIT=24 BATCH_SIZE=2 bash tools/run_frame_adaln_gate.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
BASE_CONFIG="$REPO/train/configs/inha_full_unet.yaml"
OVERLAY_CONFIG="$REPO/train/configs/inha_frame_adaln.yaml"
VALSET="${VALSET:-$REPO/valset_holdout}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/frame_adaln_gate}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
LIMIT="${LIMIT:-24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/frame_adaln/inha_frame_adaln_action_only/checkpoints/last.ckpt}"
INCUMBENT="${INCUMBENT:-$REPO/results/step10k_main_eta1_holdout_scores.json}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0

if [ ! -f "$CHECKPOINT" ]; then
    echo "Frame-Ada checkpoint가 없다: $CHECKPOINT" >&2
    echo "먼저 CUDA_VISIBLE_DEVICES=<GPU> bash train/run_frame_adaln.sh 를 실행할 것" >&2
    exit 2
fi

run_one() {
    local variant="$1"
    local label="frame_adaln_2k_${variant}"
    local prediction_root="$OUT_ROOT/$label"

    "$PYTHON" "$REPO/train/generate_videos.py" \
        --config "$BASE_CONFIG" "$OVERLAY_CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --challenge-root "$VALSET" \
        --prediction-root "$prediction_root" \
        --limit "$LIMIT" --batch-size "$BATCH_SIZE" \
        --seed 0 --ddim-steps 50 --ddim-eta 1.0 --guidance-scale 1.0 \
        --action-ablation "$variant" --no-ema --overwrite

    "$PYTHON" "$REPO/tools/score_predictions.py" \
        --valset "$VALSET" --prediction-root "$prediction_root" --limit "$LIMIT" \
        --out "$RESULT_ROOT/${label}_holdout_scores.json"
}

run_one none
run_one zero
run_one batch-roll

COMPARE_ARGS=(
    --normal "$RESULT_ROOT/frame_adaln_2k_none_holdout_scores.json"
    --zero "$RESULT_ROOT/frame_adaln_2k_zero_holdout_scores.json"
    --batch-roll "$RESULT_ROOT/frame_adaln_2k_batch-roll_holdout_scores.json"
    --out "$RESULT_ROOT/frame_adaln_2k_gate.json"
)
if [ "$LIMIT" = "24" ] && [ -f "$INCUMBENT" ]; then
    COMPARE_ARGS+=(--incumbent "$INCUMBENT")
fi
"$PYTHON" "$REPO/tools/compare_generation_gate.py" "${COMPARE_ARGS[@]}"

echo "Frame-Ada gate complete: $RESULT_ROOT/frame_adaln_2k_gate.json"
