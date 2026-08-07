#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK="$REPO/third_party/cosmos-framework"
FRAMEWORK_ENV="${FRAMEWORK_ENV:-$FRAMEWORK/.venv}"
PYTHON="$FRAMEWORK_ENV/bin/python"
TORCHRUN="$FRAMEWORK_ENV/bin/torchrun"
PHASE="${PHASE:-audit}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ "$PHASE" == "smoke" ]]; then
  MAX_STEPS="${MAX_STEPS:-5}"
else
  MAX_STEPS="${MAX_STEPS:-500}"
fi
SAVE_STEPS="${SAVE_STEPS:-250}"
RUN_NAME="${RUN_NAME:-cosmos3_edge_so100_native}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
EXPORT_TAG="${EXPORT_TAG:-}"

EDGE_HF="${EDGE_HF:-$REPO/checkpoints/Cosmos3-Edge}"
EDGE_DCP="${EDGE_DCP:-$REPO/checkpoints/Cosmos3-Edge-DCP}"
EDGE_MODEL_CONFIG="${EDGE_MODEL_CONFIG:-$FRAMEWORK/cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$REPO/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
SO100_TRAIN_ROOT="${SO100_TRAIN_ROOT:-$REPO/open/data/train}"
SO100_INDEX_PATH="${SO100_INDEX_PATH:-$REPO/inha_worldmodel_scratch_training/cosmos/train/episode_index.parquet}"
SO100_HOLDOUT_MANIFEST="${SO100_HOLDOUT_MANIFEST:-$REPO/valset_holdout/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/open/baseline/outputs}"
RUN_DIR="$OUTPUT_ROOT/cosmos3_action_fd/action_sft/$RUN_NAME"
TOML="$FRAMEWORK/examples/toml/sft_config/action_fd_so100_edge.toml"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 1; }
}

require_framework_env() {
  if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import loguru, torch, hydra, cosmos_framework' >/dev/null 2>&1; then
    echo "ERROR: Cosmos Framework training environment is incomplete: $FRAMEWORK_ENV" >&2
    echo "Run this first: PHASE=setup bash train/run_cosmos3_edge_so100_native.sh" >&2
    exit 1
  fi
}

export COSMOS3_EDGE_HF_PATH="$EDGE_HF"
export BASE_CHECKPOINT_PATH="$EDGE_DCP"
export WAN_VAE_PATH SO100_TRAIN_ROOT SO100_INDEX_PATH SO100_HOLDOUT_MANIFEST
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$REPO/train:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=""

case "$PHASE" in
  setup)
    cd "$FRAMEWORK"
    uv sync --all-extras --group=cu128-train
    "$PYTHON" -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda})'
    ;;
  audit)
    "$REPO/.venv/bin/python" "$REPO/tools/audit_cosmos3_edge_so100_training.py" --samples "${LIMIT:-8}"
    ;;
  convert-base)
    require_framework_env
    require_dir "$EDGE_HF"
    require_file "$EDGE_HF/model.safetensors.index.json"
    require_file "$EDGE_MODEL_CONFIG"
    mkdir -p "$EDGE_DCP"
    cd "$FRAMEWORK"
    CUDA_VISIBLE_DEVICES= "$PYTHON" -m cosmos_framework.scripts.convert_model_to_dcp \
      --checkpoint-path "$EDGE_HF" \
      --config-file "$EDGE_MODEL_CONFIG" \
      -o "$EDGE_DCP"
    ;;
  dryrun)
    require_framework_env
    require_dir "$EDGE_DCP"
    require_file "$WAN_VAE_PATH"
    cd "$FRAMEWORK"
    "$PYTHON" -m cosmos_framework.scripts.train --sft-toml "$TOML" --dryrun
    ;;
  smoke|train)
    require_framework_env
    require_dir "$EDGE_DCP"
    require_file "$WAN_VAE_PATH"
    require_file "$SO100_INDEX_PATH"
    require_file "$SO100_HOLDOUT_MANIFEST"
    if [[ "$PHASE" == "smoke" && "$RUN_NAME" == "cosmos3_edge_so100_native" ]]; then
      RUN_NAME="cosmos3_edge_so100_smoke"
    fi
    cd "$FRAMEWORK"
    "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-50137}" \
      -m cosmos_framework.scripts.train --sft-toml "$TOML" -- \
      "job.name=$RUN_NAME" \
      "model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE" \
      "trainer.straggler_detection.enabled=false" \
      "trainer.max_iter=$MAX_STEPS" \
      "checkpoint.save_iter=$SAVE_STEPS"
    ;;
  export)
    require_framework_env
    require_dir "$RUN_DIR/checkpoints"
    if [[ -n "$CHECKPOINT_NAME" ]]; then
      CHECKPOINT="$RUN_DIR/checkpoints/$CHECKPOINT_NAME"
    else
      require_file "$RUN_DIR/checkpoints/latest_checkpoint.txt"
      LATEST="$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpoint.txt")"
      CHECKPOINT="$RUN_DIR/checkpoints/$LATEST"
    fi
    require_dir "$CHECKPOINT"
    MODEL_DIR="$RUN_DIR/model${EXPORT_TAG:+_$EXPORT_TAG}"
    DIFFUSERS_DIR="$RUN_DIR/diffusers${EXPORT_TAG:+_$EXPORT_TAG}"
    cd "$FRAMEWORK"
    "$PYTHON" -m cosmos_framework.scripts.export_model \
      --checkpoint-path "$CHECKPOINT" \
      --config-file cosmos_framework/configs/base/config.py \
      --experiment action_fd_so100_edge \
      --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
      --vit-checkpoint-path "$EDGE_HF" \
      -o "$MODEL_DIR"
    "$PYTHON" -m cosmos_framework.scripts.convert_model_to_diffusers \
      --checkpoint-path "$MODEL_DIR" \
      --edge-reasoner-path "$EDGE_HF" \
      -o "$DIFFUSERS_DIR"
    ;;
  gate)
    DIFFUSERS_DIR="$RUN_DIR/diffusers${EXPORT_TAG:+_$EXPORT_TAG}"
    require_dir "$DIFFUSERS_DIR"
    MODEL="$DIFFUSERS_DIR" \
    LABEL="${LABEL:-cosmos3_edge_so100_native_${EXPORT_TAG:-$MAX_STEPS}}" \
    LIMIT="${LIMIT:-8}" \
    INFERENCE_STEPS="${INFERENCE_STEPS:-30}" \
    PHASE=gate bash "$REPO/train/run_cosmos3_edge_so100.sh"
    ;;
  generate)
    DIFFUSERS_DIR="$RUN_DIR/diffusers${EXPORT_TAG:+_$EXPORT_TAG}"
    require_dir "$DIFFUSERS_DIR"
    MODEL="$DIFFUSERS_DIR" \
    LABEL="${LABEL:-cosmos3_edge_so100_native_${EXPORT_TAG:-$MAX_STEPS}}" \
    INFERENCE_STEPS="${INFERENCE_STEPS:-30}" \
    SUBMISSION_VIDEO_ROOT="${SUBMISSION_VIDEO_ROOT:-$REPO/submissions/cosmos3_edge_so100_native/videos}" \
    PHASE=generate bash "$REPO/train/run_cosmos3_edge_so100.sh"
    ;;
  submission)
    SUBMISSION_ROOT="${SUBMISSION_ROOT:-$REPO/submissions/cosmos3_edge_so100_native}" \
    PHASE=submission bash "$REPO/train/run_cosmos3_edge_so100.sh"
    ;;
  *)
    echo "Unknown PHASE=$PHASE (setup|audit|convert-base|dryrun|smoke|train|export|gate|generate|submission)" >&2
    exit 2
    ;;
esac
