#!/usr/bin/env bash
# H100 DDP launcher for source-conditioned SO100 control on Wan2.1-I2V-14B.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${PHASE:-smoke}"
NPROC="${NPROC:-8}"
BASE_MODEL="${BASE_MODEL:-$REPO/checkpoints/Wan2.1-I2V-14B-480P}"
DATA_ROOT="${DATA_ROOT:-$REPO/open/data/train}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-432}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-}"
OUT="${OUT:-}"

case "$PHASE" in
  audit)
    exec "$REPO/.venv/bin/python" "$REPO/tools/audit_wan21_spatial_action.py" \
      --dataset-root "$DATA_ROOT" --base-model-path "$BASE_MODEL"
    ;;
  smoke)
    MAX_STEPS="${MAX_STEPS:-5}"
    SAVE_STEPS="${SAVE_STEPS:-5}"
    OUT="${OUT:-$REPO/open/baseline/outputs/wan21_spatial_action_smoke}"
    ;;
  gate250)
    MAX_STEPS="${MAX_STEPS:-250}"
    SAVE_STEPS="${SAVE_STEPS:-125}"
    OUT="${OUT:-$REPO/open/baseline/outputs/wan21_spatial_action_250}"
    ;;
  train500)
    MAX_STEPS="${MAX_STEPS:-500}"
    SAVE_STEPS="${SAVE_STEPS:-250}"
    OUT="${OUT:-$REPO/open/baseline/outputs/wan21_spatial_action_500}"
    ;;
  *)
    echo "PHASE must be one of: audit, smoke, gate250, train500" >&2
    exit 2
    ;;
esac

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export USE_TF="${USE_TF:-0}"
export TOKENIZERS_PARALLELISM="false"
export DIFFSYNTH_REDIRECT_COMMON_FILES="false"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[Wan21 spatial] phase=$PHASE gpus=$NPROC steps=$MAX_STEPS size=${WIDTH}x${HEIGHT} out=$OUT"
exec "$REPO/.venv/bin/python" -m accelerate.commands.launch \
  --num_processes "$NPROC" \
  --multi_gpu \
  "$REPO/train/train_wan21_spatial_action.py" \
  --base_model_path "$BASE_MODEL" \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_repeat 100 \
  --dataset_num_workers 2 \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames 17 \
  --max_steps "$MAX_STEPS" \
  --output_path "$OUT" \
  --save_steps "$SAVE_STEPS" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 16 \
  --control_hidden_dim 192 \
  --injection_layers 0,10,20,30 \
  --action_mode hybrid \
  --control_learning_rate 1e-4 \
  --lora_learning_rate 1e-5 \
  --control_rank_margin 0.002 \
  --control_rank_weight 0.5 \
  --motion_loss_weight 2.0 \
  --static_probability 0.15 \
  --gradient_accumulation_steps 1 \
  --use_gradient_checkpointing
