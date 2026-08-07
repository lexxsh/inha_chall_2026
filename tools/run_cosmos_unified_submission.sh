#!/usr/bin/env bash
# Generate all 216 evaluation videos with the unified Cosmos checkpoint on
# multiple GPUs, verify completeness, and build the official feature CSV.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$REPO/inha_worldmodel_scratch_training/cosmos/train"
COSMOS_REPO="${COSMOS_REPO:-$REPO/third_party/cosmos-predict2.5}"
COSMOS_PYTHON="${COSMOS_PYTHON:-$COSMOS_REPO/.venv/bin/python}"
FEATURE_PYTHON="${FEATURE_PYTHON:-$REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_DIR/runs/cosmos_unified_action_v2/latest.pt}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-$REPO/open/data/eval}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-$REPO/submissions/cosmos_unified_action_v2_13067}"
VIDEO_ROOT="$SUBMISSION_ROOT/videos"
OUTPUT_CSV="$SUBMISSION_ROOT/submission_features.csv"
NPROC="${NPROC:-8}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
mkdir -p "$VIDEO_ROOT"

export COSMOS_REPO
export COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-$REPO/inha_worldmodel_scratch_training/cosmos/checkpoints}"
export SO100_EVAL_ROOT="$CHALLENGE_ROOT"
export RUN_CKPT="$CHECKPOINT"
export OUTDIR="$VIDEO_ROOT"
export ATOK=1
export ATOK_GATE_SCALE="${ATOK_GATE_SCALE:-1.0}"
export LORA_SCALE="${LORA_SCALE:-1.0}"
export STEPS="${STEPS:-30}"
export SHIFT="${SHIFT:-5.0}"
export SEED="${SEED:-7}"
export GUIDANCE="${GUIDANCE:-0}"
export ACFG="${ACFG:-0}"
export AUTO_G="${AUTO_G:-0}"
export NAVG="${NAVG:-1}"
export SUBSET=0
export RESUME="${RESUME:-0}"
export OVERWRITE="${OVERWRITE:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$COSMOS_PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  "$TRAIN_DIR/generate_eval.py"

EXPECTED="$(find "$CHALLENGE_ROOT/images" -maxdepth 1 -type f -name 'sample_*.png' | wc -l)"
GENERATED="$(find "$VIDEO_ROOT" -maxdepth 1 -type f -name 'sample_*.mp4' | wc -l)"
if [[ "$GENERATED" -ne "$EXPECTED" ]]; then
  echo "Generated video count mismatch: expected=$EXPECTED generated=$GENERATED" >&2
  exit 2
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python USE_TF=0 \
"$FEATURE_PYTHON" "$REPO/open/submission_kit/make_submission_csv.py" \
  --prediction-root "$VIDEO_ROOT" \
  --challenge-root "$CHALLENGE_ROOT" \
  --output-csv "$OUTPUT_CSV" \
  --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
  --action-extractor-ckpt "$REPO/open/submission_kit/checkpoints/action_extractor.ckpt" \
  --feature-batch-size "${FEATURE_BATCH_SIZE:-4}"

echo "Submission ready: $OUTPUT_CSV"
