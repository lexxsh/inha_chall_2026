#!/usr/bin/env bash
# Native BWM-style Wan2.2-5B post-training. GPU phases are launched only by the user.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PHASE="${PHASE:-audit}"
NPROC="${NPROC:-8}"
GEN_GPUS="${GEN_GPUS:-1}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/models/Wan-AI/Wan2.2-TI2V-5B}"
DATA_ROOT="${DATA_ROOT:-$REPO/open/data/train}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-$REPO/valset_holdout}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LIMIT="${LIMIT:-8}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO/open/baseline/outputs/bwm_native_so100_10k}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_PATH/step-${MAX_STEPS}.safetensors}"
PREDICTION_ROOT="${PREDICTION_ROOT:-$REPO/diagnostics/bwm_native_so100_${MAX_STEPS}}"

export PYTHONPATH="$REPO/train:$REPO/third_party/boundless-world-model:$REPO/third_party/DiffSynth-Studio:${PYTHONPATH:-}"
export DIFFSYNTH_REDIRECT_COMMON_FILES=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/inha-matplotlib}"

train_args=(
  --dataset-root "$DATA_ROOT"
  --model-root "$MODEL_ROOT"
  --height "$HEIGHT"
  --width "$WIDTH"
  --output-path "$OUTPUT_PATH"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --dataset-repeat "${DATASET_REPEAT:-2}"
  --dataset-num-workers "${NUM_WORKERS:-2}"
  --action-learning-rate "${ACTION_LR:-5e-5}"
  --dit-learning-rate "${DIT_LR:-1e-5}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
)

fsdp_launch=(
  .venv/bin/accelerate launch
  --num_processes "$NPROC"
  --use_fsdp
  --fsdp_version 1
  --fsdp_sharding_strategy FULL_SHARD
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
  --fsdp_transformer_layer_cls_to_wrap DiTBlock
  --fsdp_state_dict_type FULL_STATE_DICT
  --fsdp_use_orig_params true
  --fsdp_sync_module_states true
  --fsdp_cpu_ram_efficient_loading false
  --fsdp_activation_checkpointing false
)

generate_args=(
  --checkpoint "$CHECKPOINT"
  --model-root "$MODEL_ROOT"
  --challenge-root "$CHALLENGE_ROOT"
  --stats-root "$DATA_ROOT"
  --prediction-root "$PREDICTION_ROOT"
  --limit "$LIMIT"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num-inference-steps "$INFERENCE_STEPS"
  --overwrite
)

case "$PHASE" in
  audit)
    exec .venv/bin/python tools/audit_bwm_native_so100.py \
      --dataset-root "$DATA_ROOT" --model-root "$MODEL_ROOT"
    ;;
  smoke)
    smoke_steps="${SMOKE_STEPS:-5}"
    smoke_output="${OUTPUT_PATH%/}_smoke"
    exec "${fsdp_launch[@]}" train/train_bwm_native_so100.py \
      "${train_args[@]}" --height "${SMOKE_HEIGHT:-320}" --width "${SMOKE_WIDTH:-512}" \
      --output-path "$smoke_output" --max-steps "$smoke_steps" --save-steps "$smoke_steps"
    ;;
  overfit)
    overfit_steps="${OVERFIT_STEPS:-250}"
    overfit_output="${OUTPUT_PATH%/}_overfit"
    exec "${fsdp_launch[@]}" train/train_bwm_native_so100.py \
      "${train_args[@]}" --height "${OVERFIT_HEIGHT:-320}" --width "${OVERFIT_WIDTH:-512}" \
      --output-path "$overfit_output" --max-steps "$overfit_steps" \
      --save-steps "${OVERFIT_SAVE_STEPS:-125}" --single-clip-overfit \
      --action-learning-rate "${OVERFIT_ACTION_LR:-1e-4}" \
      --dit-learning-rate "${OVERFIT_DIT_LR:-2e-5}"
    ;;
  overfit-generate)
    overfit_steps="${OVERFIT_STEPS:-250}"
    overfit_output="${OUTPUT_PATH%/}_overfit"
    exec .venv/bin/python train/generate_bwm_native_overfit.py \
      --checkpoint "${OVERFIT_CHECKPOINT:-$overfit_output/step-${overfit_steps}.safetensors}" \
      --sample "$overfit_output/overfit_sample.pt" \
      --output "${OVERFIT_PREDICTION_ROOT:-$REPO/diagnostics/bwm_native_so100_overfit_${overfit_steps}}" \
      --model-root "$MODEL_ROOT" --num-inference-steps "$INFERENCE_STEPS"
    ;;
  screen)
    screen_steps="${SCREEN_STEPS:-500}"
    screen_output="${OUTPUT_PATH%/}_screen500"
    exec "${fsdp_launch[@]}" train/train_bwm_native_so100.py \
      "${train_args[@]}" --output-path "$screen_output" --max-steps "$screen_steps" \
      --save-steps "${SCREEN_SAVE_STEPS:-250}"
    ;;
  train)
    exec "${fsdp_launch[@]}" train/train_bwm_native_so100.py "${train_args[@]}"
    ;;
  resume)
    if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
      echo "PHASE=resume requires RESUME_CHECKPOINT=/path/to/step-N.safetensors" >&2
      exit 2
    fi
    exec "${fsdp_launch[@]}" train/train_bwm_native_so100.py \
      "${train_args[@]}" --resume-checkpoint "$RESUME_CHECKPOINT"
    ;;
  generate|gate)
    .venv/bin/python tools/audit_bwm_native_checkpoint.py --checkpoint "$CHECKPOINT"
    ablation=none
    [[ "$PHASE" == gate ]] && ablation=all
    full_args=("${generate_args[@]}" --action-ablation "$ablation")
    if [[ "$GEN_GPUS" -gt 1 ]]; then
      exec .venv/bin/torchrun --standalone --nproc_per_node "$GEN_GPUS" \
        train/generate_bwm_native_so100.py "${full_args[@]}"
    else
      exec .venv/bin/python train/generate_bwm_native_so100.py "${full_args[@]}"
    fi
    ;;
  score-gate)
    label="${LABEL:-bwm_native_so100_${MAX_STEPS}}"
    for mode in none zero-motion batch-roll; do
      .venv/bin/python tools/score_predictions.py \
        --valset "$CHALLENGE_ROOT" --prediction-root "$PREDICTION_ROOT/$mode" \
        --limit "$LIMIT" --out "$REPO/results/${label}_${mode}_holdout_scores.json"
    done
    exec .venv/bin/python tools/compare_generation_gate.py \
      --normal "$REPO/results/${label}_none_holdout_scores.json" \
      --zero "$REPO/results/${label}_zero-motion_holdout_scores.json" \
      --batch-roll "$REPO/results/${label}_batch-roll_holdout_scores.json" \
      --out "$REPO/results/${label}_gate.json"
    ;;
  *)
    echo "Unknown PHASE=$PHASE (audit, smoke, overfit, overfit-generate, screen, train, resume, generate, gate, score-gate)" >&2
    exit 2
    ;;
esac
