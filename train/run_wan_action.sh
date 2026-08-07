#!/usr/bin/env bash
# PHASE=smoke: vanilla 2-step feasibility; PHASE=train: LoRA+action gate training.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PHASE="${PHASE:-smoke}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
OUT="${OUT:-$REPO/open/baseline/outputs/wan_action_1k}"
MAX_STEPS="${MAX_STEPS:-1000}"
NPROC="${NPROC:-1}"
ACTION_MODE="${ACTION_MODE:-delta}"
ACTION_VERSION="${ACTION_VERSION:-v1}"
PROMPT="${PROMPT-A fixed-camera video of a robot arm manipulating objects.}"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO/models}"
export DIFFSYNTH_REDIRECT_COMMON_FILES=false
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HOME="${HF_HOME:-$REPO/models/.hf_cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"

if [ "$PHASE" = "download" ]; then
  "$PYTHON" "$REPO/train/download_wan_weights.py" \
    --model-root "$DIFFSYNTH_MODEL_BASE_PATH" --retries "${DOWNLOAD_RETRIES:-20}"
elif [ "$PHASE" = "smoke" ]; then
  "$PYTHON" "$REPO/train/generate_wan_videos.py" \
    --vanilla --limit "${LIMIT:-1}" --num-inference-steps "${INFERENCE_STEPS:-2}" \
    --challenge-root "${VALSET:-$REPO/valset_holdout}" \
    --prediction-root "$REPO/diagnostics/wan_smoke" --prompt "$PROMPT" --overwrite \
    --benchmark-json "$REPO/results/wan_smoke_benchmark.json"
elif [ "$PHASE" = "train" ]; then
  LAUNCH_ARGS=(--num_processes "$NPROC")
  if [ "$NPROC" -gt 1 ]; then
    LAUNCH_ARGS+=(--multi_gpu)
  fi
  "$PYTHON" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
    "$REPO/train/train_wan_action.py" \
    --dataset_base_path "$REPO/open/data/train" \
    --height 320 --width 512 --num_frames 17 --dataset_repeat 100 \
    --model_id_with_origin_paths "$MODEL_ID:diffusion_pytorch_model*.safetensors,$MODEL_ID:models_t5_umt5-xxl-enc-bf16.pth,$MODEL_ID:Wan2.2_VAE.pth" \
    --learning_rate "${LR:-1e-4}" --weight_decay 0.01 --max_steps "$MAX_STEPS" \
    --action_learning_rate "${ACTION_LR:-${LR:-1e-4}}" \
    --lora_learning_rate "${LORA_LR:-${LR:-1e-4}}" \
    --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
    --use_gradient_checkpointing --gradient_accumulation_steps "${ACCUM:-1}" \
    --save_steps "${SAVE_STEPS:-250}" --output_path "$OUT" \
    --remove_prefix_in_ckpt "pipe.dit." --action_mode "$ACTION_MODE" --action_shift 0 \
    --action_conditioner_version "$ACTION_VERSION" --prompt "$PROMPT"
else
  echo "Unknown PHASE=$PHASE (use download, smoke, or train)" >&2
  exit 2
fi
