#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PHASE="${PHASE:-audit}"
NPROC="${NPROC:-8}"
GEN_GPUS="${GEN_GPUS:-1}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-512}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LIMIT="${LIMIT:-8}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO/open/baseline/outputs/bwm_so100_lora_10k}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_PATH/step-${MAX_STEPS}.safetensors}"
PREDICTION_ROOT="${PREDICTION_ROOT:-$REPO/diagnostics/bwm_so100_lora_${MAX_STEPS}}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-$REPO/valset_holdout}"
BWM_CHECKPOINT="${BWM_CHECKPOINT:-$REPO/checkpoints/Boundless-World-Model/step-12000.safetensors}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/models/Wan-AI/Wan2.2-TI2V-5B}"

export PYTHONPATH="$REPO/third_party/boundless-world-model:$REPO/third_party/DiffSynth-Studio:${PYTHONPATH:-}"
export DIFFSYNTH_REDIRECT_COMMON_FILES=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/inha-matplotlib}"

common_train_args=(
  --dataset-root "$REPO/open/data/train"
  --model-root "$MODEL_ROOT"
  --bwm-checkpoint "$BWM_CHECKPOINT"
  --height "$HEIGHT"
  --width "$WIDTH"
  --output-path "$OUTPUT_PATH"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --lora-rank 32
  --action-learning-rate "${ACTION_LR:-1e-4}"
  --lora-learning-rate "${LORA_LR:-5e-5}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --dataset-num-workers "${NUM_WORKERS:-4}"
)

case "$PHASE" in
  audit)
    .venv/bin/python tools/audit_bwm_so100_contract.py \
      --dataset-root "$REPO/open/data/train" \
      --model-root "$MODEL_ROOT" \
      --bwm-checkpoint "$BWM_CHECKPOINT"
    ;;
  smoke)
    smoke_output="${OUTPUT_PATH%/}_smoke"
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_bwm_so100_lora.py \
      "${common_train_args[@]}" \
      --output-path "$smoke_output" \
      --max-steps "${SMOKE_STEPS:-5}" \
      --save-steps "${SMOKE_STEPS:-5}"
    ;;
  train)
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_bwm_so100_lora.py "${common_train_args[@]}"
    ;;
  resume)
    if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
      echo "PHASE=resume requires RESUME_CHECKPOINT=/path/to/step-N.safetensors" >&2
      exit 2
    fi
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_bwm_so100_lora.py \
      "${common_train_args[@]}" \
      --resume-checkpoint "$RESUME_CHECKPOINT"
    ;;
  generate|gate)
    ablation=none
    if [[ "$PHASE" == gate ]]; then
      ablation=all
    fi
    generation_args=(
      --checkpoint "$CHECKPOINT"
      --model-root "$MODEL_ROOT"
      --bwm-checkpoint "$BWM_CHECKPOINT"
      --challenge-root "$CHALLENGE_ROOT"
      --stats-root "$REPO/open/data/train"
      --prediction-root "$PREDICTION_ROOT"
      --limit "$LIMIT"
      --height "$HEIGHT"
      --width "$WIDTH"
      --num-inference-steps "$INFERENCE_STEPS"
      --action-ablation "$ablation"
      --overwrite
    )
    if [[ "$GEN_GPUS" -gt 1 ]]; then
      .venv/bin/torchrun --standalone --nproc_per_node "$GEN_GPUS" \
        train/generate_bwm_so100_videos.py "${generation_args[@]}"
    else
      .venv/bin/python train/generate_bwm_so100_videos.py "${generation_args[@]}"
    fi
    ;;
  *)
    echo "Unknown PHASE=$PHASE (use audit, smoke, train, resume, generate, or gate)" >&2
    exit 2
    ;;
esac
