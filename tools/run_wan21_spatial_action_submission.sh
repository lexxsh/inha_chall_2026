#!/usr/bin/env bash
# Generate all 216 competition samples with the valid (non-oracle) Wan2.1
# spatial-action checkpoint, then convert them to the official feature CSV.
#
# Example (8 H100s):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash tools/run_wan21_spatial_action_submission.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/wan21_spatial_action_250/step-250.safetensors}"
BASE_MODEL="${BASE_MODEL:-$REPO/checkpoints/Wan2.1-I2V-14B-480P}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-$REPO/open/data/eval}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-$REPO/submissions/wan21_spatial_action_250}"
PREDICTION_ROOT="$SUBMISSION_ROOT/videos"
OUTPUT_CSV="$SUBMISSION_ROOT/submission_features.csv"
STEPS="${STEPS:-50}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
else
  NPROC=1
fi

mkdir -p "$PREDICTION_ROOT"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_FLAX="${USE_FLAX:-0}"
export DIFFSYNTH_REDIRECT_COMMON_FILES="false"
export DIFFSYNTH_SKIP_DOWNLOAD="true"

GENERATOR=("$PYTHON")
if (( NPROC > 1 )); then
  GENERATOR+=( -m torch.distributed.run --standalone --nproc_per_node "$NPROC" )
fi
GENERATOR+=("$REPO/train/generate_wan21_spatial_action.py")

GENERATE_ARGS=(
  --checkpoint "$CHECKPOINT"
  --base-model-path "$BASE_MODEL"
  --challenge-root "$CHALLENGE_ROOT"
  --prediction-root "$PREDICTION_ROOT"
  --limit 0
  --height 480
  --width 640
  --num-inference-steps "$STEPS"
  --action-ablation none
  --benchmark-json "$SUBMISSION_ROOT/generation_benchmark.json"
)
if [[ "$OVERWRITE" == "1" ]]; then
  GENERATE_ARGS+=(--overwrite)
fi

"${GENERATOR[@]}" "${GENERATE_ARGS[@]}"

EXPECTED="$(find "$CHALLENGE_ROOT/images" -maxdepth 1 -type f -name 'sample_*.png' | wc -l)"
GENERATED="$(find "$PREDICTION_ROOT" -maxdepth 1 -type f -name 'sample_*.mp4' | wc -l)"
if [[ "$GENERATED" -ne "$EXPECTED" ]]; then
  echo "Generated video count mismatch: expected=$EXPECTED generated=$GENERATED" >&2
  exit 2
fi

"$PYTHON" "$REPO/open/submission_kit/make_submission_csv.py" \
  --prediction-root "$PREDICTION_ROOT" \
  --challenge-root "$CHALLENGE_ROOT" \
  --output-csv "$OUTPUT_CSV" \
  --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
  --action-extractor-ckpt "$REPO/open/submission_kit/checkpoints/action_extractor.ckpt" \
  --feature-batch-size "$FEATURE_BATCH_SIZE"

echo "Submission ready: $OUTPUT_CSV"
