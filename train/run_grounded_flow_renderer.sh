#!/usr/bin/env bash
# Frozen grounding tokens -> source-preserving flow renderer. GPU selection is explicit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PHASE="${PHASE:-train}"
GROUNDING="${GROUNDING:-$REPO/open/baseline/outputs/so100_grounding_probe_v2_confirm/best.pt}"
OUTPUT="${OUTPUT:-$REPO/open/baseline/outputs/grounded_flow_renderer_v2_500}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT/step-500.safetensors}"
VALSET="${VALSET:-$REPO/valset_holdout}"
DIAGNOSTICS="${DIAGNOSTICS:-$REPO/diagnostics/grounded_flow_renderer_v2_500}"
LIMIT="${LIMIT:-8}"

case "$PHASE" in
  train)
    "$REPO/.venv/bin/accelerate" launch \
      --num_processes "${NUM_PROCESSES:-8}" \
      "$REPO/train/train_grounded_flow_renderer.py" \
      --grounding-checkpoint "$GROUNDING" \
      --output "$OUTPUT" \
      --max-steps "${MAX_STEPS:-500}" \
      --save-steps "${SAVE_STEPS:-250}" \
      --batch-size "${BATCH_SIZE:-4}" \
      --workers "${WORKERS:-2}" \
      --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.25}" \
      --flow-supervision-weight "${FLOW_SUPERVISION_WEIGHT:-5.0}" \
      --flow-motion-boost "${FLOW_MOTION_BOOST:-20.0}" \
      --mask-supervision-weight "${MASK_SUPERVISION_WEIGHT:-0.5}"
    ;;
  generate)
    "$PYTHON" "$REPO/train/generate_grounded_flow_videos.py" \
      --checkpoint "$CHECKPOINT" \
      --grounding-checkpoint "$GROUNDING" \
      --challenge-root "$VALSET" \
      --prediction-root "$DIAGNOSTICS/none" \
      --limit "$LIMIT" --batch-size "${GEN_BATCH_SIZE:-4}" --overwrite
    ;;
  gate)
    for variant in none zero batch-roll; do
      "$PYTHON" "$REPO/train/generate_grounded_flow_videos.py" \
        --checkpoint "$CHECKPOINT" \
        --grounding-checkpoint "$GROUNDING" \
        --challenge-root "$VALSET" \
        --prediction-root "$DIAGNOSTICS/$variant" \
        --limit "$LIMIT" --batch-size "${GEN_BATCH_SIZE:-4}" \
        --action-ablation "$variant" --overwrite
      "$PYTHON" "$REPO/tools/score_predictions.py" \
        --valset "$VALSET" --prediction-root "$DIAGNOSTICS/$variant" \
        --limit "$LIMIT" \
        --out "$REPO/results/grounded_flow_renderer_v2_500_${variant}_holdout_scores.json"
    done
    "$PYTHON" "$REPO/tools/compare_generation_gate.py" \
      --normal "$REPO/results/grounded_flow_renderer_v2_500_none_holdout_scores.json" \
      --zero "$REPO/results/grounded_flow_renderer_v2_500_zero_holdout_scores.json" \
      --batch-roll "$REPO/results/grounded_flow_renderer_v2_500_batch-roll_holdout_scores.json" \
      --incumbent "$REPO/results/step10k_main_eta1_holdout_scores.json" \
      --out "$REPO/results/grounded_flow_renderer_v2_500_gate.json"
    ;;
  *)
    echo "PHASE must be one of: train, generate, gate" >&2
    exit 2
    ;;
esac
