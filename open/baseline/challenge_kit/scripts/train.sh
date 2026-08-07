#!/bin/bash

# Initialize variables
CONFIG=""
SCRIPT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --script) SCRIPT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if the config path and script path were provided
if [ -z "$CONFIG" ] || [ -z "$SCRIPT" ]; then
    echo "Usage: $0 --config config/path --script script/path"
    exit 1
fi

export HOST_GPU_NUM=1
export PYTHONPATH="$(pwd)/src:$(pwd)/libs/dynamicrafter:${PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0

if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
    :
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

if [ -f "/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
fi

## run
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12353 --node_rank=0 \
$SCRIPT \
--base $CONFIG \
--train \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1
