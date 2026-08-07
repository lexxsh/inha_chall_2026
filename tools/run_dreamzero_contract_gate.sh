#!/usr/bin/env bash
# Staged, train-only DreamZero-SO101 input-contract gate. This script never
# selects a GPU by itself; the caller must set CUDA_VISIBLE_DEVICES for GPU phases.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${PHASE:-audit}"
VIEW_MODE="${VIEW_MODE:-front-only}"
OUTPUT="${OUTPUT:-$REPO/diagnostics/dreamzero_so101_contract_gate}"
VALSET="$REPO/diagnostics/oracle_motion_field_gate/valset"
DZ_PY="$REPO/third_party/dreamzero/.venv/bin/python"
SCORE_PY="$REPO/.venv/bin/python"
GENERATOR="$REPO/train/generate_dreamzero_contract_gate.py"
SAMPLES=(--sample-id holdout_0000 --sample-id holdout_0003)

case "$PHASE" in
  audit)
    CUDA_VISIBLE_DEVICES="" "$DZ_PY" "$GENERATOR" \
      "${SAMPLES[@]}" \
      --prediction-root "$OUTPUT" \
      --dry-run
    ;;
  view)
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "PHASE=view requires an explicit CUDA_VISIBLE_DEVICES (for example 0)." >&2
      exit 2
    fi
    "$DZ_PY" "$GENERATOR" \
      "${SAMPLES[@]}" \
      --prediction-root "$OUTPUT" \
      --view-modes front-only replicate-three \
      --variants normal
    ;;
  counterfactual)
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "PHASE=counterfactual requires an explicit CUDA_VISIBLE_DEVICES." >&2
      exit 2
    fi
    "$DZ_PY" "$GENERATOR" \
      "${SAMPLES[@]}" \
      --prediction-root "$OUTPUT" \
      --view-modes "$VIEW_MODE" \
      --variants normal zero batch-roll
    ;;
  score)
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "PHASE=score requires an explicit CUDA_VISIBLE_DEVICES." >&2
      exit 2
    fi
    for variant in normal zero batch-roll; do
      "$SCORE_PY" "$REPO/tools/score_predictions.py" \
        --valset "$VALSET" \
        --prediction-root "$OUTPUT/$VIEW_MODE/$variant" \
        "${SAMPLES[@]}" \
        --batch-size 2 \
        --out "$REPO/results/dreamzero_contract_${VIEW_MODE}_${variant}_scores.json"
    done
    "$SCORE_PY" "$REPO/tools/compare_generation_gate.py" \
      --normal "$REPO/results/dreamzero_contract_${VIEW_MODE}_normal_scores.json" \
      --zero "$REPO/results/dreamzero_contract_${VIEW_MODE}_zero_scores.json" \
      --batch-roll "$REPO/results/dreamzero_contract_${VIEW_MODE}_batch-roll_scores.json" \
      --incumbent "$REPO/results/oracle_gate_dream_10k_holdout_scores.json" \
      --out "$REPO/results/dreamzero_contract_${VIEW_MODE}_gate.json"
    ;;
  *)
    echo "Unknown PHASE=$PHASE (expected audit, view, counterfactual, or score)" >&2
    exit 2
    ;;
esac
