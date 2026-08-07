#!/usr/bin/env bash
# 액션 조건부 DynamiCrafter 파인튜닝 실행기.
#
#   bash train/run_train.sh                      # 기본 학습 (temporal 정책)
#   bash train/run_train.sh --trainable all      # 전체 파인튜닝
#   bash train/run_train.sh --bench-steps 8      # 처리량/메모리만 측정
#
# GPU는 CUDA_VISIBLE_DEVICES로 지정한다. 예: CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIT="$REPO/open/baseline/challenge_kit"
PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-$REPO/train/configs/inha_full_unet.yaml}"
CONFIG_OVERLAY="${CONFIG_OVERLAY:-}"

BASE_ARGS=(--base "$CONFIG")
if [ -n "$CONFIG_OVERLAY" ]; then
    BASE_ARGS+=("$CONFIG_OVERLAY")
fi

# CUDA_VISIBLE_DEVICES에서 GPU 수를 센다.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
else
    NGPU=1
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-12355}"

cd "$KIT"

if [ "$NGPU" -gt 1 ]; then
    # 기본 단일-GPU 설정은 local batch 4 × accumulation 2 = effective batch 8이다.
    # DDP에서 accumulation만 줄이면 4/8 GPU의 effective batch가 16/32로 바뀌므로
    # local batch도 함께 줄여 비교 실험의 총 샘플 수를 고정한다.
    TARGET_EFFECTIVE_BATCH="${TARGET_EFFECTIVE_BATCH:-8}"
    if [ $(( TARGET_EFFECTIVE_BATCH % NGPU )) -ne 0 ]; then
        echo "TARGET_EFFECTIVE_BATCH=$TARGET_EFFECTIVE_BATCH 를 GPU $NGPU 장에 정확히 나눌 수 없다" >&2
        exit 2
    fi
    LOCAL_BATCH=$(( TARGET_EFFECTIVE_BATCH / NGPU ))
    ACCUM=1
    echo "[run] GPU ${NGPU}장 DDP, local_batch=${LOCAL_BATCH}, accumulate=${ACCUM}, effective_batch=${TARGET_EFFECTIVE_BATCH}"
    exec "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NGPU" --nnodes=1 \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        "$REPO/train/train.py" "${BASE_ARGS[@]}" --train --devices "$NGPU" \
        data.params.batch_size="$LOCAL_BATCH" \
        lightning.trainer.accumulate_grad_batches="$ACCUM" "$@"
else
    # 단일 GPU면 lvdm의 get_env_vars가 요구하는 값만 채워 준다.
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    export RANK="${RANK:-0}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    exec "$PYTHON" "$REPO/train/train.py" "${BASE_ARGS[@]}" --train "$@"
fi
