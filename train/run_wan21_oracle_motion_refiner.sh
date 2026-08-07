#!/usr/bin/env bash
# Train-only oracle dense-motion refiner. Controls use future RGB and are not submissible.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${PHASE:-audit}"
NPROC="${NPROC:-8}"
BASE_MODEL="${BASE_MODEL:-$REPO/checkpoints/Wan2.1-I2V-14B-480P}"
DATA_ROOT="${DATA_ROOT:-$REPO/open/data/train}"
ORACLE_ROOT="${ORACLE_ROOT:-$REPO/diagnostics/oracle_motion_field_gate}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-432}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-}"
OUT="${OUT:-}"

case "$PHASE" in
  prepare)
    exec "$REPO/.venv/bin/python" "$REPO/tools/prepare_oracle_motion_field_gate.py" \
      --data-root "$DATA_ROOT" \
      --output "$ORACLE_ROOT" \
      --height "$HEIGHT" --width "$WIDTH" \
      "${@:1}"
    ;;
  audit)
    exec "$REPO/.venv/bin/python" "$REPO/tools/audit_wan21_oracle_motion_refiner.py" \
      --oracle-root "$ORACLE_ROOT"
    ;;
  smoke)
    MAX_STEPS="${MAX_STEPS:-5}"
    SAVE_STEPS="${SAVE_STEPS:-5}"
    OUT="${OUT:-$REPO/open/baseline/outputs/wan21_oracle_motion_refiner_smoke}"
    ;;
  gate250)
    MAX_STEPS="${MAX_STEPS:-250}"
    SAVE_STEPS="${SAVE_STEPS:-125}"
    OUT="${OUT:-$REPO/open/baseline/outputs/wan21_oracle_motion_refiner_250}"
    ;;
  *)
    echo "PHASE must be one of: prepare, audit, smoke, gate250" >&2
    exit 2
    ;;
esac

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export USE_TF="${USE_TF:-0}"
export TOKENIZERS_PARALLELISM="false"
export DIFFSYNTH_REDIRECT_COMMON_FILES="false"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[Wan oracle refiner] phase=$PHASE gpus=$NPROC steps=$MAX_STEPS size=${WIDTH}x${HEIGHT} out=$OUT"
exec "$REPO/.venv/bin/python" -m accelerate.commands.launch \
  --num_processes "$NPROC" \
  --multi_gpu \
  "$REPO/train/train_wan21_oracle_motion_refiner.py" \
  --base_model_path "$BASE_MODEL" \
  --oracle_manifest "$ORACLE_ROOT/manifest.json" \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_repeat 100 \
  --dataset_num_workers 1 \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames 17 \
  --max_steps "$MAX_STEPS" \
  --output_path "$OUT" \
  --save_steps "$SAVE_STEPS" \
  --control_hidden_dim 192 \
  --injection_layers 0,10,20,30 \
  --control_learning_rate 1e-4 \
  --control_rank_margin 0.002 \
  --control_rank_weight 0.5 \
  --motion_loss_weight 2.0 \
  --gradient_accumulation_steps 1 \
  --use_gradient_checkpointing
