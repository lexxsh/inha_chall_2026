#!/usr/bin/env bash
# Released Ctrl-World architecture, adapted only at the SO-100 data boundary.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PHASE="${PHASE:-audit}"
NPROC="${NPROC:-8}"
GEN_NPROC="${GEN_NPROC:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LIMIT="${LIMIT:-8}"
INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LABEL="${LABEL:-ctrl_world_so100_10k}"

DATA_ROOT="${DATA_ROOT:-$REPO/open/data/train}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/cache/ctrl_world_so100_svd}"
STATS_PATH="${STATS_PATH:-$REPO/results/ctrl_world_so100_pose_stats.json}"
SVD_MODEL_PATH="${SVD_MODEL_PATH:-$REPO/checkpoints/stable-video-diffusion-img2vid}"
CTRL_CHECKPOINT="${CTRL_CHECKPOINT:-$REPO/checkpoints/Ctrl-World/checkpoint-10000.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO/open/baseline/outputs/$LABEL}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_PATH/step-${MAX_STEPS}.safetensors}"
PREDICTION_ROOT="${PREDICTION_ROOT:-$REPO/diagnostics/$LABEL}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-$REPO/valset_holdout}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/results}"
INCUMBENT="${INCUMBENT:-$REPO/results/step10k_main_eta1_holdout_scores.json}"

export PYTHONPATH="$REPO:$REPO/third_party/Ctrl-World:$REPO/open/baseline/challenge_kit/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

prepare_common=(
  --data-root "$DATA_ROOT"
  --cache-root "$CACHE_ROOT"
  --stats-path "$STATS_PATH"
  --svd-model-path "$SVD_MODEL_PATH"
)

train_common=(
  --data-root "$DATA_ROOT"
  --cache-root "$CACHE_ROOT"
  --stats-path "$STATS_PATH"
  --svd-model-path "$SVD_MODEL_PATH"
  --output-path "$OUTPUT_PATH"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --num-workers "${NUM_WORKERS:-4}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
)

case "$PHASE" in
  assets)
    mkdir -p "$SVD_MODEL_PATH" "$(dirname "$CTRL_CHECKPOINT")"
    .venv/bin/hf download stabilityai/stable-video-diffusion-img2vid \
      --local-dir "$SVD_MODEL_PATH" \
      --include 'model_index.json' 'feature_extractor/*' 'image_encoder/*' \
        'scheduler/*' 'unet/*' 'vae/*'
    .venv/bin/hf download yjguo/Ctrl-World checkpoint-10000.pt \
      --local-dir "$(dirname "$CTRL_CHECKPOINT")"
    ;;
  audit)
    .venv/bin/python tools/prepare_ctrl_world_so100.py \
      --phase audit "${prepare_common[@]}"
    ;;
  stats)
    .venv/bin/python tools/prepare_ctrl_world_so100.py \
      --phase stats "${prepare_common[@]}"
    ;;
  cache)
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      tools/prepare_ctrl_world_so100.py \
      --phase cache "${prepare_common[@]}" \
      --encode-batch-size "${ENCODE_BATCH_SIZE:-64}"
    ;;
  smoke)
    smoke_output="${OUTPUT_PATH%/}_smoke"
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_ctrl_world_so100.py \
      "${train_common[@]}" \
      --ctrl-world-checkpoint "$CTRL_CHECKPOINT" \
      --output-path "$smoke_output" \
      --max-steps "${SMOKE_STEPS:-5}" \
      --save-steps "${SMOKE_STEPS:-5}"
    ;;
  train)
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_ctrl_world_so100.py \
      "${train_common[@]}" \
      --ctrl-world-checkpoint "$CTRL_CHECKPOINT"
    ;;
  resume)
    if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
      echo "PHASE=resume requires RESUME_CHECKPOINT=/path/to/step-N.safetensors" >&2
      exit 2
    fi
    .venv/bin/accelerate launch --multi_gpu --num_processes "$NPROC" \
      train/train_ctrl_world_so100.py \
      "${train_common[@]}" \
      --resume-checkpoint "$RESUME_CHECKPOINT"
    ;;
  generate|gate)
    action_ablation=none
    if [[ "$PHASE" == gate ]]; then
      action_ablation=all
    fi
    generator=(.venv/bin/python)
    if (( GEN_NPROC > 1 )); then
      generator+=( -m torch.distributed.run --standalone --nproc_per_node "$GEN_NPROC" )
    fi
    generator+=(train/generate_ctrl_world_so100.py)
    "${generator[@]}" \
      --checkpoint "$CHECKPOINT" \
      --svd-model-path "$SVD_MODEL_PATH" \
      --stats-path "$STATS_PATH" \
      --challenge-root "$CHALLENGE_ROOT" \
      --prediction-root "$PREDICTION_ROOT" \
      --limit "$LIMIT" \
      --num-inference-steps "$INFERENCE_STEPS" \
      --action-ablation "$action_ablation" \
      --overwrite

    if [[ "$PHASE" == gate ]]; then
      for mode in none zero-motion batch-roll; do
        .venv/bin/python tools/score_predictions.py \
          --valset "$CHALLENGE_ROOT" \
          --prediction-root "$PREDICTION_ROOT/$mode" \
          --limit "$LIMIT" \
          --batch-size "${SCORE_BATCH_SIZE:-4}" \
          --out "$RESULT_ROOT/${LABEL}_${mode}_holdout_scores.json"
      done
      compare=(
        .venv/bin/python tools/compare_generation_gate.py
        --normal "$RESULT_ROOT/${LABEL}_none_holdout_scores.json"
        --zero "$RESULT_ROOT/${LABEL}_zero-motion_holdout_scores.json"
        --batch-roll "$RESULT_ROOT/${LABEL}_batch-roll_holdout_scores.json"
        --out "$RESULT_ROOT/${LABEL}_gate.json"
      )
      if [[ -f "$INCUMBENT" ]]; then
        compare+=(--incumbent "$INCUMBENT")
      fi
      "${compare[@]}"
    fi
    ;;
  *)
    echo "Unknown PHASE=$PHASE (assets, audit, stats, cache, smoke, train, resume, generate, gate)" >&2
    exit 2
    ;;
esac
