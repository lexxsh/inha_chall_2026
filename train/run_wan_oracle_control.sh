#!/usr/bin/env bash
# Train-only oracle gate. This is not the final action-conditioned model.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
OUT="${OUT:-$REPO/open/baseline/outputs/wan_oracle_track_250}"
MANIFEST="${MANIFEST:-$REPO/diagnostics/spatial_control_gate/manifest.json}"
NPROC="${NPROC:-4}"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO/models}"
export DIFFSYNTH_REDIRECT_COMMON_FILES=false
export HF_HOME="${HF_HOME:-$REPO/models/.hf_cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"

LAUNCH_ARGS=(--num_processes "$NPROC")
if [ "$NPROC" -gt 1 ]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

EXTRA_ARGS=()
if [ "${STRUCTURAL_ZERO:-0}" = "1" ]; then EXTRA_ARGS+=(--structural_zero); fi
if [ "${ADAPTER_ONLY:-0}" = "1" ]; then EXTRA_ARGS+=(--adapter_only); fi

"$PYTHON" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
  "$REPO/train/train_wan_oracle_control.py" \
  --manifest "$MANIFEST" --dataset_base_path "$REPO/open/data/train" \
  --height 320 --width 512 --num_frames 17 --dataset_repeat "${REPEAT:-100}" \
  --model_id_with_origin_paths "$MODEL_ID:diffusion_pytorch_model*.safetensors,$MODEL_ID:models_t5_umt5-xxl-enc-bf16.pth,$MODEL_ID:Wan2.2_VAE.pth" \
  --learning_rate "${LR:-2e-5}" --track_learning_rate "${TRACK_LR:-1e-4}" \
  --lora_learning_rate "${LORA_LR:-2e-5}" --weight_decay 0.01 \
  --control_rank_margin "${RANK_MARGIN:-0.005}" --control_rank_weight "${RANK_WEIGHT:-0.1}" \
  --wrong_control_mode "${WRONG_CONTROL_MODE:-reverse}" "${EXTRA_ARGS[@]}" \
  --max_steps "${MAX_STEPS:-250}" --save_steps "${SAVE_STEPS:-250}" \
  --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --use_gradient_checkpointing --gradient_accumulation_steps "${ACCUM:-1}" \
  --output_path "$OUT" --remove_prefix_in_ckpt "pipe.dit."
