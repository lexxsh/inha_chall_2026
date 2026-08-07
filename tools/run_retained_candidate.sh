#!/usr/bin/env bash
# Stable entry point for the only two retained submitted candidates.
# This wrapper does not train or mutate checkpoints.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="${CANDIDATE:-}"
PHASE="${PHASE:-verify}"

usage() {
  cat <<'EOF'
Usage:
  CANDIDATE=cosmos|wan PHASE=verify|audit|submission|csv bash tools/run_retained_candidate.sh

verify      Check the exact retained checkpoint, videos, CSV, and hashes (CPU only).
audit       Run the candidate's structural/data audit (CPU only).
submission Generate missing videos and rebuild the official submission CSV (uses GPU).
csv         Rebuild only the CSV from the existing 216 videos (uses feature extractor GPU).
EOF
}

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 1; }
}

case "$CANDIDATE" in
  cosmos)
    NAME="Cosmos Unified Action v2 @ 13,067"
    CHECKPOINT="$REPO/inha_worldmodel_scratch_training/cosmos/train/runs/cosmos_unified_action_v2/latest.pt"
    CHECKPOINT_SHA256="bcd5f4e656cfa17834d4553ee4a864a9b0a9a1632e6bba855a290b8ce17fd12b"
    SUBMISSION_ROOT="$REPO/submissions/cosmos_unified_action_v2_13067"
    CSV_SHA256="7090a49e903f17bd2d57d1d3f0a59653ba9f4f5ef4de8178f63b42e8b63472e2"
    PUBLIC_SCORE="0.22"
    ACTION_MAE="0.3224969483"
    ;;
  wan)
    NAME="Wan2.1 Spatial Action @ 250"
    CHECKPOINT="$REPO/open/baseline/outputs/wan21_spatial_action_250/step-250.safetensors"
    CHECKPOINT_SHA256="21d4ef9a8ca9759de8d0795f355fac1ab8fff5614517134e199808df44c97976"
    SUBMISSION_ROOT="$REPO/submissions/wan21_spatial_action_250"
    CSV_SHA256="09af6415ed9d9ba75ea66635d045ef8072604987a0e7d6222cd0ae125d7b6d48"
    PUBLIC_SCORE="0.25"
    ACTION_MAE="0.4857981627"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

VIDEO_ROOT="$SUBMISSION_ROOT/videos"
OUTPUT_CSV="$SUBMISSION_ROOT/submission_features.csv"

case "$PHASE" in
  verify)
    require_file "$CHECKPOINT"
    require_dir "$VIDEO_ROOT"
    require_file "$OUTPUT_CSV"
    video_count="$(find "$VIDEO_ROOT" -maxdepth 1 -type f -name 'sample_*.mp4' | wc -l)"
    csv_lines="$(wc -l < "$OUTPUT_CSV")"
    [[ "$video_count" -eq 216 ]] || { echo "ERROR: expected 216 videos, found $video_count" >&2; exit 1; }
    [[ "$csv_lines" -eq 649 ]] || { echo "ERROR: expected 649 CSV lines, found $csv_lines" >&2; exit 1; }
    echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status || {
      echo "ERROR: retained checkpoint hash mismatch: $CHECKPOINT" >&2
      exit 1
    }
    echo "$CSV_SHA256  $OUTPUT_CSV" | sha256sum --check --status || {
      echo "ERROR: retained CSV hash mismatch: $OUTPUT_CSV" >&2
      exit 1
    }
    printf '%s\n' \
      "candidate=$NAME" \
      "checkpoint=$CHECKPOINT" \
      "videos=$video_count" \
      "csv=$OUTPUT_CSV" \
      "public_score=$PUBLIC_SCORE" \
      "mean_action_mae=$ACTION_MAE" \
      "verdict=PASS_RETAINED_ARTIFACTS"
    ;;
  audit)
    if [[ "$CANDIDATE" == "cosmos" ]]; then
      exec env PHASE=audit bash \
        "$REPO/inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh"
    else
      exec env PHASE=audit bash "$REPO/train/run_wan21_spatial_action.sh"
    fi
    ;;
  submission)
    require_file "$CHECKPOINT"
    if [[ "$CANDIDATE" == "cosmos" ]]; then
      exec env CHECKPOINT="$CHECKPOINT" SUBMISSION_ROOT="$SUBMISSION_ROOT" RESUME="${RESUME:-1}" \
        bash "$REPO/tools/run_cosmos_unified_submission.sh"
    else
      exec env CHECKPOINT="$CHECKPOINT" SUBMISSION_ROOT="$SUBMISSION_ROOT" \
        bash "$REPO/tools/run_wan21_spatial_action_submission.sh"
    fi
    ;;
  csv)
    require_dir "$VIDEO_ROOT"
    exec env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python USE_TF=0 \
      "$REPO/.venv/bin/python" "$REPO/open/submission_kit/make_submission_csv.py" \
      --prediction-root "$VIDEO_ROOT" \
      --challenge-root "$REPO/open/data/eval" \
      --output-csv "$OUTPUT_CSV" \
      --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
      --action-extractor-ckpt "$REPO/open/submission_kit/checkpoints/action_extractor.ckpt" \
      --feature-batch-size "${FEATURE_BATCH_SIZE:-4}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
