#!/usr/bin/env bash
# Faithful IRASim Frame-Ada adaptation. Override MAX_STEPS/OUT for a smoke run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PRETRAINED="${PRETRAINED:-$REPO/models/IRASim/0300000.pt}"
OUT="${OUT:-$REPO/open/baseline/outputs/irasim_action_500}"
NPROC="${NPROC:-8}"

LAUNCH_ARGS=(--num_processes "$NPROC")
if [ "$NPROC" -gt 1 ]; then LAUNCH_ARGS+=(--multi_gpu); fi

export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_TF=1

"$PYTHON" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
  "$REPO/train/train_irasim_action.py" \
  --pretrained "$PRETRAINED" \
  --vae "$REPO/models/IRASim/sdxl-base" \
  --dataset-root "$REPO/open/data/train" \
  --output "$OUT" \
  --max-steps "${MAX_STEPS:-500}" \
  --save-every "${SAVE_EVERY:-250}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${ACCUM:-2}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --dataset-repeat "${REPEAT:-100}" \
  --holdout-count "${HOLDOUT_COUNT:-6}" \
  --action-mode "${ACTION_MODE:-absolute}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --ema-decay "${EMA_DECAY:-0.9999}" \
  --mixed-precision "${MIXED_PRECISION:-no}"
