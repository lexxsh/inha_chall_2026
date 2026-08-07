#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="${PHASE:-smoke}"
METHODS="${METHODS:-multiframe,singleframe,mask}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cpu}"
RESPONSE="${RESPONSE:-1.0}"

case "$PHASE" in
  smoke)
    LIMIT="${LIMIT:-1}"
    MULTIFRAME_MAXITER="${MULTIFRAME_MAXITER:-4}"
    SINGLEFRAME_MAXITER="${SINGLEFRAME_MAXITER:-4}"
    MASK_MAXITER="${MASK_MAXITER:-12}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-diagnostics/oscar_so100_skeletons_smoke}"
    RESULT_JSON="${RESULT_JSON:-results/oscar_so100_skeleton_smoke.json}"
    ;;
  gate)
    LIMIT="${LIMIT:-8}"
    MULTIFRAME_MAXITER="${MULTIFRAME_MAXITER:-60}"
    SINGLEFRAME_MAXITER="${SINGLEFRAME_MAXITER:-35}"
    MASK_MAXITER="${MASK_MAXITER:-180}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-diagnostics/oscar_so100_skeletons_gate}"
    RESULT_JSON="${RESULT_JSON:-results/oscar_so100_skeleton_gate.json}"
    ;;
  *)
    echo "PHASE must be smoke or gate, got: $PHASE" >&2
    exit 2
    ;;
esac

HF_HUB_OFFLINE=1 .venv/bin/python tools/prepare_oscar_so100_skeletons.py \
  --methods "$METHODS" \
  --limit "$LIMIT" \
  --multiframe-maxiter "$MULTIFRAME_MAXITER" \
  --singleframe-maxiter "$SINGLEFRAME_MAXITER" \
  --mask-maxiter "$MASK_MAXITER" \
  --detector-device "$DETECTOR_DEVICE" \
  --response "$RESPONSE" \
  --output-root "$OUTPUT_ROOT" \
  --result-json "$RESULT_JSON"
