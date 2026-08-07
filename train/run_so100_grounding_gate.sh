#!/usr/bin/env bash
# Data-contract and action-grounding gate. No GPU is selected automatically.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PHASE="${PHASE:-audit}"
MANIFEST="${MANIFEST:-$REPO/results/so100_contract_manifest.jsonl}"
FEATURES="${FEATURES:-$REPO/results/so100_grounding_features.pt}"
OUT="${OUT:-$REPO/open/baseline/outputs/so100_grounding_probe_v2}"
CONFIRM_OUT="${CONFIRM_OUT:-$REPO/open/baseline/outputs/so100_grounding_probe_v2_confirm}"

case "$PHASE" in
  manifest)
    "$PYTHON" "$REPO/tools/build_so100_contract_manifest.py" \
      --output "$MANIFEST" \
      --windows-per-episode "${WINDOWS_PER_EPISODE:-2}" \
      --holdout-uploaders "${HOLDOUT_UPLOADERS:-8}"
    ;;
  audit)
    "$PYTHON" "$REPO/tools/audit_so100_contract.py" \
      --manifest "$MANIFEST" \
      --samples-per-split "${SAMPLES_PER_SPLIT:-1200}" \
      --output "${AUDIT_OUTPUT:-$REPO/results/so100_contract_audit.json}"
    ;;
  cache)
    "$PYTHON" "$REPO/tools/cache_so100_grounding_features.py" \
      --manifest "$MANIFEST" \
      --output "$FEATURES" \
      --train-records "${TRAIN_RECORDS:-4096}" \
      --holdout-records "${HOLDOUT_RECORDS:-1024}" \
      --feature-batch-size "${FEATURE_BATCH_SIZE:-4}" \
      --device "${DEVICE:-cpu}"
    ;;
  probe)
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" "$PYTHON" \
      "$REPO/train/train_so100_grounding_probe.py" \
      --features "$FEATURES" \
      --output "$OUT" \
      --max-steps "${MAX_STEPS:-2000}" \
      --batch-size "${BATCH_SIZE:-128}" \
      --eval-every "${EVAL_EVERY:-250}" \
      --device "${DEVICE:-cpu}"
    ;;
  confirm)
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" "$PYTHON" \
      "$REPO/train/train_so100_grounding_probe.py" \
      --features "$FEATURES" \
      --output "$CONFIRM_OUT" \
      --max-steps "${MAX_STEPS:-250}" \
      --batch-size "${BATCH_SIZE:-128}" \
      --eval-every "${EVAL_EVERY:-25}" \
      --split-seed "${SPLIT_SEED:-20260805}" \
      --confirmatory \
      --device "${DEVICE:-cpu}"
    ;;
  *)
    echo "PHASE must be one of: manifest, audit, cache, probe, confirm" >&2
    exit 2
    ;;
esac
