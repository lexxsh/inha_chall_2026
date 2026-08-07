#!/usr/bin/env bash
# Stage 1 challenger: pretrained DynamiCrafter + frame-wise AdaLN-Zero.
#
# 기본은 action conditioner만 2k step 학습한다. 이 gate를 통과한 뒤에만
# TRAINABLE=temporal로 시간 블록까지 여는 중간 실험을 실행한다.
#
#   CUDA_VISIBLE_DEVICES=7 bash train/run_frame_adaln.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINABLE=action_only bash train/run_frame_adaln.sh
#   CUDA_VISIBLE_DEVICES=7 bash train/run_frame_adaln.sh --bench-steps 8
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$REPO/train/configs/inha_full_unet.yaml}"
export CONFIG_OVERLAY="${CONFIG_OVERLAY:-$REPO/train/configs/inha_frame_adaln.yaml}"
TRAINABLE="${TRAINABLE:-action_only}"

exec bash "$REPO/train/run_train.sh" --trainable "$TRAINABLE" "$@"
