#!/usr/bin/env bash
# Stage 1b: same Frame-Ada model, 1k steps, action representation/alignment only.
#
# All variants keep seed, effective batch, LR, sampler and holdout fixed.
# Existing delta/shift0 1k checkpoint is reused; three new variants are trained.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/run_action_representation_screen.sh
#   PHASE=train CUDA_VISIBLE_DEVICES=... bash tools/run_action_representation_screen.sh
#   PHASE=gate  CUDA_VISIBLE_DEVICES=7 bash tools/run_action_representation_screen.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PHASE="${PHASE:-all}"
VALSET="${VALSET:-$REPO/valset_holdout}"
LIMIT="${LIMIT:-24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/action_representation_1k}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
MODEL_ROOT="$REPO/open/baseline/outputs/frame_adaln"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0

train_variant() {
    local label="$1" mode="$2" shift="$3" dims="$4"
    local ckpt="$MODEL_ROOT/inha_frame_adaln_${label}/checkpoints/last.ckpt"
    if [ -f "$ckpt" ]; then
        echo "[skip train] $label: $ckpt"
        return
    fi
    CONFIG_OVERLAY="$REPO/train/configs/inha_frame_adaln.yaml" \
      bash "$REPO/train/run_frame_adaln.sh" \
        name="inha_frame_adaln_${label}" group="inha_frame_adaln_representation" \
        data.params.action_mode="$mode" data.params.action_shift="$shift" \
        model.params.unet_config.params.action_dims="$dims" \
        lightning.trainer.max_steps=1000 \
        lightning.callbacks.model_checkpoint.params.every_n_train_steps=1000
}

if [ "$PHASE" = "all" ] || [ "$PHASE" = "train" ]; then
    train_variant "delta_sm1" "delta" -1 6
    train_variant "step_s0" "delta_step" 0 6
    train_variant "anchor_s0" "delta_anchor" 0 12
fi

gate_variant() {
    local label="$1" config="$2" checkpoint="$3"
    if [ ! -f "$checkpoint" ]; then
        echo "checkpoint가 없다: $checkpoint" >&2
        exit 2
    fi
    for ablation in none batch-roll; do
        local run_label="rep1k_${label}_${ablation}"
        local pred="$OUT_ROOT/$run_label"
        "$PYTHON" "$REPO/train/generate_videos.py" \
            --config "$config" --checkpoint "$checkpoint" \
            --challenge-root "$VALSET" --prediction-root "$pred" \
            --limit "$LIMIT" --batch-size "$BATCH_SIZE" \
            --seed 0 --ddim-steps 50 --ddim-eta 1.0 --guidance-scale 1.0 \
            --action-ablation "$ablation" --no-ema --overwrite
        "$PYTHON" "$REPO/tools/score_predictions.py" \
            --valset "$VALSET" --prediction-root "$pred" --limit "$LIMIT" \
            --out "$RESULT_ROOT/${run_label}_holdout_scores.json"
    done
}

if [ "$PHASE" = "all" ] || [ "$PHASE" = "gate" ]; then
    gate_variant \
      "delta_s0" \
      "$MODEL_ROOT/inha_frame_adaln_action_only/configs/model.yaml" \
      "$MODEL_ROOT/inha_frame_adaln_action_only/checkpoints/epoch=0-step=1000.ckpt"
    for label in delta_sm1 step_s0 anchor_s0; do
        gate_variant \
          "$label" \
          "$MODEL_ROOT/inha_frame_adaln_${label}/configs/model.yaml" \
          "$MODEL_ROOT/inha_frame_adaln_${label}/checkpoints/last.ckpt"
    done

    COMPARE_ARGS=()
    for label in delta_s0 delta_sm1 step_s0 anchor_s0; do
        COMPARE_ARGS+=(
          --variant "$label"
          "$RESULT_ROOT/rep1k_${label}_none_holdout_scores.json"
          "$RESULT_ROOT/rep1k_${label}_batch-roll_holdout_scores.json"
        )
    done
    "$PYTHON" "$REPO/tools/compare_action_representations.py" \
      "${COMPARE_ARGS[@]}" --out "$RESULT_ROOT/action_representation_1k_gate.json"
fi
