#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

NPROC="${NPROC:-4}"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/fomm_action_kp_hard101/step-500.safetensors}"
RENDERER_CHECKPOINT="${RENDERER_CHECKPOINT:-$REPO/open/baseline/outputs/fomm_oracle_hard101_single/step-1000.safetensors}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-$REPO/submissions/fomm_action_kp_hard101_500}"
VIDEO_ROOT="$SUBMISSION_ROOT/videos"
OUTPUT_CSV="$SUBMISSION_ROOT/submission_features.csv"

mkdir -p "$VIDEO_ROOT"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  train/generate_fomm_action_videos.py \
  --checkpoint "$CHECKPOINT" \
  --renderer-checkpoint "$RENDERER_CHECKPOINT" \
  --challenge-root open/data/eval \
  --prediction-root "$VIDEO_ROOT" \
  --limit 0 \
  --overwrite

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python USE_TF=0 \
.venv/bin/python open/submission_kit/make_submission_csv.py \
  --prediction-root "$VIDEO_ROOT" \
  --challenge-root open/data/eval \
  --output-csv "$OUTPUT_CSV" \
  --action-stats-path open/data/train/so100_action_statistics.json \
  --action-extractor-ckpt open/submission_kit/checkpoints/action_extractor.ckpt \
  --feature-batch-size 4

echo "submission -> $OUTPUT_CSV"
