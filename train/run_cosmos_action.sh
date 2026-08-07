#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSMOS_REPO="${COSMOS_REPO:-$REPO/third_party/cosmos-predict2.5}"
PHASE="${PHASE:-dryrun}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-12341}"

export PYTHONPATH="$REPO/train:$REPO:$COSMOS_REPO${PYTHONPATH:+:$PYTHONPATH}"
export INHA_DATA_ROOT="${INHA_DATA_ROOT:-$REPO/open/data/train}"
export HF_HOME="${HF_HOME:-$REPO/models/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/models/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO/models/uv-python}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$REPO/open/baseline/outputs/cosmos_action}"
export COSMOS_HEIGHT="${COSMOS_HEIGHT:-256}"
export COSMOS_WIDTH="${COSMOS_WIDTH:-320}"
export COSMOS_ACTION_MODE="${COSMOS_ACTION_MODE:-delta}"
export COSMOS_ACTION_MODE_V2="${COSMOS_ACTION_MODE_V2:-hybrid_step}"
export COSMOS_ACTION_SHIFT="${COSMOS_ACTION_SHIFT:-0}"
export TOKENIZERS_PARALLELISM=false

# Reuse an existing CLI login without copying the token into the project.
HF_TOKEN_FILE="/nfsdata/home/lexxsh/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -r "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

if [[ ! -d "$COSMOS_REPO" ]]; then
  echo "Cosmos source not found: $COSMOS_REPO" >&2
  exit 2
fi

case "$PHASE" in
  setup)
    cd "$COSMOS_REPO"
    uv python install 3.10
    uv sync --python 3.10 --extra=cu128
    ;;
  download)
    cd "$COSMOS_REPO"
    uvx "hf>=1.3.5" download nvidia/Cosmos-Predict2.5-2B \
      --repo-type model --revision main \
      robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
      tokenizer.pth
    ;;
  data)
    "$REPO/.venv/bin/python" "$REPO/train/check_cosmos_so100_data.py" \
      --data-root "$INHA_DATA_ROOT" \
      --action-mode "$COSMOS_ACTION_MODE"
    ;;
  dryrun)
    cd "$COSMOS_REPO"
    .venv/bin/torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
      -m scripts.train --dryrun \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_500 \
      '~dataloader_train.dataloaders' \
      trainer.max_iter=2 \
      job.wandb_mode=disabled
    ;;
  smoke)
    cd "$COSMOS_REPO"
    COSMOS_SMOKE=1 .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_500 \
      '~dataloader_train.dataloaders' \
      trainer.max_iter=2 \
      checkpoint.save_iter=2 \
      job.wandb_mode=disabled
    ;;
  train)
    cd "$COSMOS_REPO"
    .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_500 \
      '~dataloader_train.dataloaders' \
      job.wandb_mode=disabled
    ;;
  probe)
    cd "$COSMOS_REPO"
    .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_probe \
      '~dataloader_train.dataloaders' \
      job.wandb_mode=disabled
    ;;
  adapter)
    cd "$COSMOS_REPO"
    COSMOS_ACTION_LAYOUT=so1006 COSMOS_NUM_FRAMES=17 \
      .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_joint_adapter_500 \
      '~dataloader_train.dataloaders' \
      job.wandb_mode=disabled
    ;;
  adapter_v2_smoke)
    cd "$COSMOS_REPO"
    COSMOS_ACTION_LAYOUT=so1006 COSMOS_NUM_FRAMES=17 \
      .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_joint_adapter_v2_250 \
      '~dataloader_train.dataloaders' \
      trainer.max_iter=5 \
      checkpoint.save_iter=5 \
      model.config.fsdp_shard_size="$GPUS" \
      trainer.callbacks.every_n_sample_reg.every_n=999999999 \
      model.action_log_every=1 \
      job.name=cosmos_predict2p5_so100_joint_adapter_v2_smoke \
      job.wandb_mode=disabled
    ;;
  adapter_v2)
    cd "$COSMOS_REPO"
    COSMOS_ACTION_LAYOUT=so1006 COSMOS_NUM_FRAMES=17 \
      .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_joint_adapter_v2_250 \
      '~dataloader_train.dataloaders' \
      model.config.fsdp_shard_size="$GPUS" \
      job.wandb_mode=disabled
    ;;
  adapter_v2_diagnose)
    "$COSMOS_REPO/.venv/bin/python" "$REPO/train/diagnose_cosmos_adapter.py" \
      --checkpoint "$IMAGINAIRE_OUTPUT_ROOT/inha_cosmos/so100_joint_adapter_v2/cosmos_predict2p5_so100_joint_adapter_v2_17f/checkpoints/iter_000000250" \
      --data-root "$INHA_DATA_ROOT" \
      --action-mode "$COSMOS_ACTION_MODE_V2" \
      --num-samples 8 \
      --output "$REPO/results/cosmos_adapter_v2_250_gate.json"
    ;;
  adapter_video)
    cd "$COSMOS_REPO"
    COSMOS_ACTION_LAYOUT=so1006 COSMOS_NUM_FRAMES=17 \
      .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_adapter_video \
      '~dataloader_train.dataloaders' \
      job.wandb_mode=disabled
    ;;
  adapter_v2_video)
    cd "$COSMOS_REPO"
    COSMOS_ACTION_LAYOUT=so1006 COSMOS_NUM_FRAMES=17 \
      .venv/bin/torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
      -m scripts.train \
      --config=cosmos_so100/config.py -- \
      experiment=inha_cosmos_so100_adapter_v2_video \
      '~dataloader_train.dataloaders' \
      trainer.max_iter="${COSMOS_PROBE_ITERS:-1}" \
      model.config.fsdp_shard_size="$GPUS" \
      model.config.net.joint_adapter_output_scale="${COSMOS_ADAPTER_OUTPUT_SCALE:-0.5}" \
      job.wandb_mode=disabled
    ;;
  *)
    echo "Unknown PHASE=$PHASE (expected setup|download|data|dryrun|smoke|train|probe|adapter|adapter_v2_smoke|adapter_v2|adapter_v2_diagnose|adapter_video|adapter_v2_video)" >&2
    exit 2
    ;;
esac
