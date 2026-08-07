# Conservative Action Flow — 실행 메모

이 후보는 영상 전체를 생성하는 대신 첫 프레임을 action-conditioned flow로 이동시킨다. 현재 계획은
절대적인 최종안이 아니며 train-only holdout gate가 반박하면 폐기한다.

## 구성

- 모델: 약 4.64M parameters, ImageNet-pretrained ResNet18 일부 + 2-layer action Transformer + flow decoder
- 입력: 320x512 첫 이미지, hybrid action `(16,18)`
- 출력: 입력 frame 1개 + flow/mask/residual 미래 frame 15개
- 학습 화면: 160x256; normalized flow라 추론은 320x512
- loss: motion-weighted Charbonnier + temporal delta + flow TV + sparse mask/residual + correct-vs-roll ranking
- 공식 action extractor: 학습에 사용하지 않음

## 실행 순서

```bash
# 1-GPU 20-step RAFT-supervision engineering smoke
CUDA_VISIBLE_DEVICES=0 PHASE=smoke NPROC=1 MAX_STEPS=20 WARMUP_STEPS=5 BATCH_SIZE=1 \
OUT=open/baseline/outputs/flow_world_smoke_v3 \
bash train/run_flow_world.sh

# smoke checkpoint로 1개 생성/shape/속도 확인
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/generate_flow_videos.py \
  --checkpoint open/baseline/outputs/flow_world_smoke_v3/step-20.safetensors \
  --limit 1 --batch-size 1 --overwrite \
  --prediction-root diagnostics/flow_world_smoke_v3 \
  --benchmark-json results/flow_world_smoke_v3_benchmark.json

# engineering pass 뒤 8-GPU 2k screen
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PHASE=train NPROC=8 \
MAX_STEPS=2000 SAVE_STEPS=500 BATCH_SIZE=4 \
OUT=open/baseline/outputs/flow_world_2k \
bash train/run_flow_world.sh

# 24 holdout normal/zero/batch-roll gate
CUDA_VISIBLE_DEVICES=0 bash tools/run_flow_world_gate.sh
```

## 중단 기준

- frame 0이 입력과 다르거나 background가 움직이면 구현 실패.
- mask가 계속 0에 붙고 normal/zero가 같으면 static shortcut.
- mask가 화면 전체로 커지며 흐려지면 sparse/motion 설계 실패.
- 2k에서 normal Action이 zero와 batch-roll보다 낮지 않으면 이 방법을 연장하지 않는다.
- static weighted를 이기고 action point estimate가 둘 다 음수일 때만 더 긴 학습을 검토한다.

## 실패 기록: unsupervised flow smoke v2

50-step direct-reconstruction smoke는 mask가 화면 약 20%로 퍼지고 normalized flow 평균 `0.056`, 최대
`0.32`까지 커졌다. 마지막 프레임의 Laplacian sharpness가 첫 프레임의 69%로 감소했고 로봇이 이동하기보다
배경 픽셀로 지워졌다. 따라서 direct photometric loss만으로 flow를 찾는 버전은 폐기했다. 현재 v3는
제공 train frame pair에 frozen torchvision RAFT-small을 적용해 1/8/15 horizon의 backward pseudo-flow를
감독하고, GT static 영역의 predicted flow를 별도로 0으로 누른다.

v3 20-step smoke 결과: `0.203 s/step`, peak `1.10 GiB`(batch 1), 생성 `0.56 s/sample`.
마지막/첫 프레임 픽셀 차이는 `0.25/255`, Laplacian sharpness 보존은 93.5%로 v2의 전역 blur가 사라졌다.
mask는 평균 약 0.105로 아직 균일하고 영상은 거의 정지지만, normal-zero 출력 차이 `5.84e-4`가 전체
normal motion `1.15e-3`의 절반 정도라 action 경로는 열려 있다. 판정은 **ENGINEERING PASS / QUALITY
INCONCLUSIVE**이며 8-GPU 500-step screen으로 이동한다.

## v3 250-step 시각 판정 — REJECT

500-step 실행은 250 checkpoint만 남기고 종료됐으나, 8개 생성으로 renderer 표현력을 먼저 감사했다.
추론은 batch 4에서 `0.19 s/sample`, peak `3.12 GiB`로 매우 빠르다. 그러나 품질은 승격 불가다.

- `val_000000/000001/000006`: 관절 운동이 아니라 robot texture가 늘어나거나 흰 배경으로 지워지는 ghost/소멸.
- `val_000003/000005/000007`: 사실상 static.
- sample별 마지막-첫 프레임 MAE는 `0.15~7.65/255`로 움직임 크기가 불안정하다.
- sharpness ratio는 최저 약 `0.83`; background는 비교적 보존하지만 움직이는 물체의 새 표면과 articulation을
  backward warp 하나로 표현하지 못한다.

판정: **REJECT / DO NOT EXTEND**. 이 결과는 단순 step 부족보다 renderer의 구조적 한계다. flow-only RGB
renderer를 500/2k로 연장하거나 loss를 추가하지 않는다. 이후 spatial flow는 RGB를 직접 warp하는 출력기가
아니라 pretrained video generator의 보조 control로만 재검토할 수 있다. 다음 주 후보는 video prior를
유지하면서 action identity를 직접 읽는 action-token cross-attention + correct-vs-roll objective다.
