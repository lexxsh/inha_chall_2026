#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PHASE="${PHASE:-audit}"
NPROC="${NPROC:-8}"
GEN_GPUS="${GEN_GPUS:-1}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-512}"
MAX_STEPS="${MAX_STEPS:-12000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LIMIT="${LIMIT:-8}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO/open/baseline/outputs/bwm_so100_eef14_12k}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_PATH/step-${MAX_STEPS}.safetensors}"
PREDICTION_ROOT="${PREDICTION_ROOT:-$REPO/diagnostics/bwm_so100_eef14_${MAX_STEPS}}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-$REPO/valset_holdout}"
STATS_PATH="${STATS_PATH:-$REPO/train/so100_eef_statistics.json}"
BWM_CHECKPOINT="${BWM_CHECKPOINT:-$REPO/checkpoints/Boundless-World-Model/step-12000.safetensors}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/models/Wan-AI/Wan2.2-TI2V-5B}"

export PYTHONPATH="$REPO/train:$REPO/third_party/boundless-world-model:$REPO/third_party/DiffSynth-Studio:${PYTHONPATH:-}"
export DIFFSYNTH_REDIRECT_COMMON_FILES=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/inha-matplotlib}"

train_args=(
  --dataset-root "$REPO/open/data/train"
  --stats-path "$STATS_PATH"
  --model-root "$MODEL_ROOT"
  --bwm-checkpoint "$BWM_CHECKPOINT"
  --height "$HEIGHT"
  --width "$WIDTH"
  --output-path "$OUTPUT_PATH"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --lora-rank "${LORA_RANK:-32}"
  --action-learning-rate "${ACTION_LR:-2e-5}"
  --lora-learning-rate "${LORA_LR:-2e-5}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --dataset-num-workers "${NUM_WORKERS:-4}"
)

generate_args=(
  --model-root "$MODEL_ROOT"
  --bwm-checkpoint "$BWM_CHECKPOINT"
  --stats-path "$STATS_PATH"
  --challenge-root "$CHALLENGE_ROOT"
  --prediction-root "$PREDICTION_ROOT"
  --limit "$LIMIT"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num-inference-steps "$INFERENCE_STEPS"
  --overwrite
)

case "$PHASE" in
  stats)
    .venv/bin/python tools/compute_so100_eef_stats.py --output "$STATS_PATH"
    ;;
  audit)
    .venv/bin/python tools/audit_bwm_so100_eef_contract.py \
      --stats-path "$STATS_PATH" --challenge-root "$CHALLENGE_ROOT" \
      --bwm-checkpoint "$BWM_CHECKPOINT"
    ;;
  zeroshot)
    .venv/bin/python train/generate_bwm_so100_eef_videos.py \
      "${generate_args[@]}" --action-ablation all
    ;;
  smoke)
    smoke_output="${OUTPUT_PATH%/}_smoke"
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_bwm_so100_eef_lora.py "${train_args[@]}" \
      --output-path "$smoke_output" --max-steps "${SMOKE_STEPS:-5}" \
      --save-steps "${SMOKE_STEPS:-5}"
    ;;
  train)
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_bwm_so100_eef_lora.py "${train_args[@]}"
    ;;
  resume)
    if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
      echo "PHASE=resume requires RESUME_CHECKPOINT=/path/to/step-N.safetensors" >&2
      exit 2
    fi
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_bwm_so100_eef_lora.py "${train_args[@]}" \
      --resume-checkpoint "$RESUME_CHECKPOINT"
    ;;
  generate|gate)
    ablation=none
    [[ "$PHASE" == gate ]] && ablation=all
    full_generate_args=("${generate_args[@]}" --checkpoint "$CHECKPOINT" --action-ablation "$ablation")
    if [[ "$GEN_GPUS" -gt 1 ]]; then
      .venv/bin/torchrun --standalone --nproc_per_node "$GEN_GPUS" \
        train/generate_bwm_so100_eef_videos.py "${full_generate_args[@]}"
    else
      .venv/bin/python train/generate_bwm_so100_eef_videos.py "${full_generate_args[@]}"
    fi
    ;;
  *)
    echo "Unknown PHASE=$PHASE (stats, audit, zeroshot, smoke, train, resume, generate, gate)" >&2
    exit 2
    ;;
esac
