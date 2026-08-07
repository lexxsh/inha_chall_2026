#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${PHASE:-audit}"
EDGE_ENV="${EDGE_ENV:-$REPO/.venv-cosmos3-edge}"
EDGE_PYTHON="$EDGE_ENV/bin/python"
MODEL="${MODEL:-nvidia/Cosmos3-Edge}"
VALSET="${VALSET:-$REPO/valset_holdout}"
if [[ -z "${LIMIT+x}" ]]; then
  if [[ "$PHASE" == "smoke" ]]; then
    LIMIT=1
  else
    LIMIT=8
  fi
fi
STEPS="${INFERENCE_STEPS:-30}"
RESOLUTION="${RESOLUTION_TIER:-480}"
LABEL="${LABEL:-cosmos3_edge_so100_zeroshot}"
PRED_ROOT="${PRED_ROOT:-$REPO/diagnostics/$LABEL}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"

score_gate() {
  for MODE in none zero-motion batch-roll; do
    "$REPO/.venv/bin/python" "$REPO/tools/score_predictions.py" \
      --valset "$VALSET" --prediction-root "$PRED_ROOT/$MODE" --limit "$LIMIT" \
      --out "$RESULT_ROOT/${LABEL}_${MODE}_holdout_scores.json"
  done
  COMPARE=(
    "$REPO/.venv/bin/python" "$REPO/tools/compare_generation_gate.py"
    --normal "$RESULT_ROOT/${LABEL}_none_holdout_scores.json"
    --zero "$RESULT_ROOT/${LABEL}_zero-motion_holdout_scores.json"
    --batch-roll "$RESULT_ROOT/${LABEL}_batch-roll_holdout_scores.json"
    --out "$RESULT_ROOT/${LABEL}_gate.json"
  )
  INCUMBENT="${INCUMBENT:-$REPO/results/step10k_main_eta1_holdout_scores.json}"
  [[ ! -f "$INCUMBENT" ]] || COMPARE+=(--incumbent "$INCUMBENT")
  "${COMPARE[@]}"
}

case "$PHASE" in
  setup)
    uv venv "$EDGE_ENV" --python 3.13 --seed --managed-python
    uv pip install --python "$EDGE_PYTHON" --torch-backend=cu128 \
      "diffusers @ git+https://github.com/huggingface/diffusers.git" \
      accelerate av huggingface_hub imageio imageio-ffmpeg pillow scipy torch torchvision transformers
    "$EDGE_PYTHON" - <<'PY'
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "pipeline": Cosmos3OmniPipeline.__name__, "action": CosmosActionCondition.__name__})
PY
    ;;
  audit)
    "$REPO/.venv/bin/python" "$REPO/tools/audit_cosmos3_so100_action.py" --limit "$LIMIT"
    ;;
  download)
    if [[ ! -x "$EDGE_ENV/bin/hf" ]]; then
      echo "Run PHASE=setup first: missing $EDGE_ENV/bin/hf" >&2
      exit 1
    fi
    "$EDGE_ENV/bin/hf" download nvidia/Cosmos3-Edge \
      --local-dir "$REPO/checkpoints/Cosmos3-Edge"
    ;;
  smoke)
    "$EDGE_PYTHON" "$REPO/train/generate_cosmos3_edge_so100.py" \
      --model "$MODEL" --challenge-root "$VALSET" --prediction-root "$PRED_ROOT" \
      --limit "$LIMIT" --num-inference-steps "$STEPS" --resolution-tier "$RESOLUTION" \
      --action-ablation smoke --overwrite \
      --benchmark-json "$RESULT_ROOT/${LABEL}_benchmark.json"
    ;;
  gate)
    "$EDGE_PYTHON" "$REPO/train/generate_cosmos3_edge_so100.py" \
      --model "$MODEL" --challenge-root "$VALSET" --prediction-root "$PRED_ROOT" \
      --limit "$LIMIT" --num-inference-steps "$STEPS" --resolution-tier "$RESOLUTION" \
      --action-ablation all --overwrite \
      --benchmark-json "$RESULT_ROOT/${LABEL}_benchmark.json"
    score_gate
    ;;
  score)
    score_gate
    ;;
  generate)
    "$EDGE_PYTHON" "$REPO/train/generate_cosmos3_edge_so100.py" \
      --model "$MODEL" --challenge-root "${CHALLENGE_ROOT:-$REPO/open/data/eval}" \
      --prediction-root "${SUBMISSION_VIDEO_ROOT:-$REPO/submissions/cosmos3_edge_so100/videos}" \
      --limit 0 --num-inference-steps "$STEPS" --resolution-tier "$RESOLUTION" \
      --action-ablation none \
      --benchmark-json "$RESULT_ROOT/${LABEL}_eval_benchmark.json"
    ;;
  submission)
    SUBMISSION_ROOT="${SUBMISSION_ROOT:-$REPO/submissions/cosmos3_edge_so100}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$REPO/.venv/bin/python" "$REPO/open/submission_kit/make_submission_csv.py" \
      --prediction-root "$SUBMISSION_ROOT/videos" \
      --challenge-root "$REPO/open/data/eval" \
      --output-csv "$SUBMISSION_ROOT/submission_features.csv" \
      --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
      --action-extractor-ckpt "$REPO/open/submission_kit/checkpoints/action_extractor.ckpt" \
      --feature-batch-size "${FEATURE_BATCH_SIZE:-4}"
    ;;
  *)
    echo "Unknown PHASE=$PHASE (setup|audit|download|smoke|gate|score|generate|submission)" >&2
    exit 2
    ;;
esac
