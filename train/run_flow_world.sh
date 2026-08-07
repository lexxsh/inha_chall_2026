#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PHASE="${PHASE:-smoke}"
NPROC="${NPROC:-1}"
OUT="${OUT:-$REPO/open/baseline/outputs/flow_world_2k}"

if [ "$PHASE" = "smoke" ]; then
  STEPS="${MAX_STEPS:-10}"
  SAVE_STEPS="$STEPS"
elif [ "$PHASE" = "train" ]; then
  STEPS="${MAX_STEPS:-2000}"
  SAVE_STEPS="${SAVE_STEPS:-500}"
else
  echo "Unknown PHASE=$PHASE (use smoke or train)" >&2
  exit 2
fi

LAUNCH_ARGS=(--num_processes "$NPROC")
if [ "$NPROC" -gt 1 ]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

"$PYTHON" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
  "$REPO/train/train_flow_world.py" \
  --data-root "$REPO/open/data/train" \
  --output "$OUT" \
  --height "${TRAIN_HEIGHT:-160}" --width "${TRAIN_WIDTH:-256}" \
  --batch-size "${BATCH_SIZE:-4}" --workers "${WORKERS:-4}" \
  --max-steps "$STEPS" --save-steps "$SAVE_STEPS" \
  --learning-rate "${LR:-2e-4}" --warmup-steps "${WARMUP_STEPS:-200}" \
  --ranking-margin "${RANKING_MARGIN:-0.01}"
