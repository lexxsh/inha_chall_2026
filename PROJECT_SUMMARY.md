# INHA World Model Challenge — 프로젝트 총정리

최종 갱신: 2026-08-07

이 문서는 대회 정의부터 현재 코드와 제출 결과, 실패 원인, 보존할 후보와 다음 실험까지 한 번에 보는
진입점이다. 세부 근거는 `ANALYSIS.md`, `EMPIRICAL.md`, `METHOD.md`, `RESEARCH_SOTA.md`에 남아 있다.

> **계획의 지위:** 아래 우선순위는 절대적인 최종 계획이 아니다. 논문과 리더보드의 모델 순위를 이 대회에
> 그대로 옮기지 않으며, train-only holdout과 실제 제출 결과가 반박하면 구조, 학습 범위, 순서를 바꾼다.
> eval 입력과 이미 만든 제출물의 분석은 포렌식으로만 보존하고 학습 데이터 선택이나 checkpoint 선택에
> 되먹이지 않는다.

## 1. 현재 결론

- 현재 공개점수 최고는 **Cosmos Unified Action v2 13,067-step: 0.22**다.
- 현재 시각 품질 기준은 **Wan2.1-I2V-14B Spatial Action 250-step: 0.25**다.
- Cosmos는 action을 더 잘 따르지만 흐리고, Wan 14B는 형태가 선명하지만 action grounding이 약하다.
- 새 BWM-style Wan2.2-5B 2K는 공식 CSV까지 만들었으나 robot morphology가 무너져 현재 제출 권장 후보가 아니다.
- BWM 5B 실패는 BWM 구조 자체의 반증이 아니다. 공개 BWM checkpoint와 학습 레시피 없이 vanilla Wan에서
  구조만 재구축했고, one-image/6D SO100 계약도 공개 BWM의 9-history/14D EEF 계약과 다르다.
- 다음 핵심 문제는 새 생성기를 계속 만드는 것이 아니라 **Wan 14B급 source fidelity와 Cosmos급 action
  fidelity를 같은 모델에서 동시에 얻는 것**이다.

## 2. 대회 개요

### 2.1 입출력

| 항목 | 내용 |
|---|---|
| 입력 sample 수 | eval 216개 |
| 시작 관측 | RGB PNG 1장, 640×480 |
| 행동 조건 | `(16,6)` NumPy action trajectory |
| action 의미 | SO100 절대 목표 관절각 6D: shoulder pan/lift, elbow, wrist flex/roll, gripper |
| 출력 | sample당 MP4 1개, 정확히 16 frames, 6 fps |
| 제출 파일 | 공식 `make_submission_csv.py`가 영상에서 추출한 648-row CSV |

학습 데이터는 LeRobot v2.1 형식의 커뮤니티 SO100 데이터셋 128개, 56 uploader, 11,132 episodes,
약 1,025,666 frames다. 원본 30 fps 영상을 stride 5로 줄여 대부분 6 fps로 통일했다. 영상은 주로 외부
고정 camera 640×480이고 parquet에는 `action`, `observation.state`가 함께 있다.

train-only 검사에서 `action[t]`는 동시 state보다 `state[t+1]`에 더 가까웠다. 1,195개 episode의 최적 lag는
대부분 `+1`이었고 raw MAE도 lag 0의 2.5951도에서 lag +1의 1.7038도로 낮아졌다. 따라서 action은 현재
관절 상태가 아니라 다음 관측을 향한 absolute target command로 해석하는 근거가 강하다. 다만 공식 action
extractor의 최적 frame alignment까지 자동으로 확정되는 것은 아니다.

### 2.2 평가식

```text
Score = 0.3 × DINO Component
      + 0.3 × Video Feature Component
      + 0.4 × Action Component
```

모든 component와 최종 score는 **낮을수록 좋다**.

| component | 공식 평가 내용 | 실무상 의미 |
|---|---|---|
| DINO | DINOv2 frame feature와 GT의 cosine distance | 로봇 형태, 물체, 배경, frame별 appearance 보존 |
| Video Feature | R3D-18 video feature와 GT의 cosine distance | 전체 motion과 시간적 영상 패턴 |
| Action | 생성 영상에서 회귀한 16×6 action과 제공 action의 normalized MAE | 주어진 trajectory를 실제 영상이 반영하는지 |

공식 제출킷은 영상을 320×512 letterbox canvas로 변환한다. 640×480 영상은 내용 영역 427×320과 좌우 검은
띠가 된다. 제출 CSV를 수동 편집하거나 비공식 extractor로 대체하면 안 된다. Public/Private split은 30%/70%다.

Action 가중치가 0.4로 가장 높지만 appearance를 버려도 된다는 뜻은 아니다. DINO와 Video가 각각 `ΔD`,
`ΔV`만큼 악화되면 Action MAE가 최소 `0.75 × (ΔD + ΔV)`보다 더 개선돼야 가중합상 이득이다.

### 2.3 규정과 실행 제약

- eval image/action을 모델 학습, pseudo-label, 데이터 선택, hyperparameter 또는 checkpoint 선택에 사용하지 않는다.
- 모델 선택은 train에서 완전히 제외한 dataset-group holdout으로 한다.
- 공개 weight와 허용 license를 사용하는 로컬 모델만 후보로 둔다. 외부 API inference는 사용하지 않는다.
- 최종 216개 inference와 공식 제출킷 처리는 제한 시간 안에 재현돼야 한다.
- 개발 환경은 주로 H100 80GB 8장과 NFS이며, RTX PRO 6000 Blackwell에서는 별도 sm_120 호환 PyTorch가 필요하다.

## 3. 실제 제출 및 artifact 현황

> 공개점수와 eval CSV 통계는 이미 만들어진 제출물을 설명하기 위한 사후 기록이다. 다음 모델의 학습 결정은
> 독립 train-only gate를 먼저 사용한다.

| 후보 | 공개점수 | eval CSV 평균 Action MAE | 시각 관찰 | 현재 지위 |
|---|---:|---:|---|---|
| Dream/DynamiCrafter 10K | 0.28188 | 0.38322 | motion은 있으나 후반 blur와 관절 뒤틀림 | 최초 제출 기준선 |
| Cosmos Unified Action v2 13,067 | **0.22** | **0.32250** | 다소 흐리지만 action 반영이 가장 좋음 | 점수 incumbent |
| Wan2.1 Spatial Action 250 | 0.25 | 0.48580 | 로봇·배경·질감 완성도가 가장 좋음 | 시각 reference |
| BWM-style Wan2.2-5B 2K | 미제출 | 0.45717 | 로봇 소실·재등장·관절/morphology 붕괴 | 제출 보류 |
| FOMM hard101 500 | 미확정 | 0.62121 | trajectory는 보이나 renderer가 뭉개짐 | 종료 |
| Wan2.2 xattn 6K | 미확정 | 0.34633 | CSV는 존재하나 retained 승격 근거 부족 | 보존만 함 |

공개점수 최고 두 후보의 핵심 artifact는 `RETAINED_CANDIDATES.md`에 hash와 함께 고정했다.

### 점수 incumbent: Cosmos Unified v2

- checkpoint: `inha_worldmodel_scratch_training/cosmos/train/runs/cosmos_unified_action_v2/latest.pt`
- 영상: `submissions/cosmos_unified_action_v2_13067/videos`
- CSV: `submissions/cosmos_unified_action_v2_13067/submission_features.csv`
- 구조: 공개 robot action-conditioned Cosmos 2B, action AdaLN + 16 action tokens + rank-32 LoRA
- 학습량: 13,067 optimizer steps, 104,536 sample exposures
- 장점: 현재 가장 낮은 Action MAE와 공개 score
- 단점: Wan 14B보다 blur와 morphology 손실이 큼

### 시각 reference: Wan2.1-I2V-14B Spatial 250

- checkpoint: `open/baseline/outputs/wan21_spatial_action_250/step-250.safetensors`
- 영상: `submissions/wan21_spatial_action_250/videos`
- CSV: `submissions/wan21_spatial_action_250/submission_features.csv`
- 구조: 전용 Wan2.1-I2V-14B base + 2.69M source/action adapter + 76.68M rank-16 LoRA
- 학습량: 약 2,000 clip exposures, base clip 한 pass의 약 19%
- 장점: source robot identity, 배경, 질감과 영상 완성도
- 단점: correct-vs-wrong loss 분리가 0 부근이고 Action MAE가 높음

## 4. 실험에서 얻은 핵심 교훈

| 계열 | 결과 | 남은 교훈 |
|---|---|---|
| DynamiCrafter 10K | 공개 0.28188 | 단순 장기학습으로 관절 뒤틀림이 해결되지 않음 |
| Frame-AdaLN/AdaLN action | condition tensor는 연결됐지만 semantic action gate 실패 | action sensitivity와 action correctness는 다름 |
| Wan2.2 action v1/v2 | 1K/2K에서 normal-zero-roll CI 분리 실패 | 더 많은 step만으로 약한 conditioning을 구제하지 못함 |
| Cosmos adapters | adapter output은 nonzero지만 영상이 고정되거나 흐림 | 구조 PASS는 생성 PASS가 아님 |
| Cosmos Unified v2 | 0.22 | pretrained robot-action prior와 충분한 exposure가 실제 점수에 기여 |
| FOMM/keypoint/dense warp | 단일 clip oracle은 좋고 multi-clip renderer는 실패 | pose 이동보다 unseen appearance rendering이 병목 |
| IRASim 500 | 영상 prior는 있으나 correct action 분리 실패 | robot video pretraining만으로 action grounding은 생기지 않음 |
| Wan oracle track/refiner | GT motion control을 줘도 static보다 나쁜 경우가 많음 | 좋은 control도 generator와 결합이 틀리면 실패 |
| HMA/MaskGIT | 고정 또는 다른 scene 생성 | tokenizer/domain이 다르면 action 학습 이전에 reconstruction이 실패 |
| DreamZero SO101 LoRA | source 보존은 있으나 고정/blur/noise | SO101 3-view 30fps 계약을 SO100 1-view 6fps에 바로 옮길 수 없음 |
| Ctrl-World/SVD | 생성 품질과 action gate 모두 부족 | 저해상도 latent와 다른 action semantics의 domain gap |
| Cosmos3-Edge zero-shot | 움직임은 확인 | native action FD로의 SO100 mapping과 post-training이 필요 |
| BWM-style Wan2.2-5B 2K | morphology 붕괴 | 공개 architecture와 공개 성능 checkpoint/recipe는 구분해야 함 |

반복 실패의 공통 패턴은 다음과 같다.

1. condition을 바꾸면 출력이 달라지는 것은 올바른 action을 생성한다는 증거가 아니다.
2. pixel/flow reconstruction loss는 action을 무시하고 평균 motion과 appearance shortcut으로 내려갈 수 있다.
3. 생성 backbone이 source identity를 보존하지 못하면 perfect track, flow, skeleton도 좋은 영상이 되지 않는다.
4. 반대로 아름다운 일반 I2V 영상도 numeric joint trajectory와 맞지 않으면 Action Component에서 진다.
5. local holdout score와 공개 score의 순위가 항상 일치하지 않으므로 작은 gate는 탈락 장치이지 점수 예측기가 아니다.

## 5. Cosmos v2와 Wan 14B가 서로 다른 이유

Cosmos v2가 더 낮은 score를 얻은 주된 차이는 action이다. eval CSV에서 Cosmos의 평균 Action MAE는 0.32250,
Wan은 0.48580으로 `0.16330` 차이가 난다. Action 가중치 0.4를 고려하면 appearance 우위만으로 이를 상쇄하기
어렵다.

Wan 250이 undertrained인 것은 사실이다. 약 2,000 exposure와 Cosmos의 104,536 exposure는 약 52배 차이다.
그러나 Wan loss에서 reconstruction은 내려가는 동안 correct-minus-wrong denoising error는 계속 0 주변이었다.
따라서 같은 구조를 오래 학습하면 action이 자동으로 해결된다고 확정할 수 없다.

구조적으로 Cosmos는 이미 robot action-conditioned인 backbone의 28개 block에서 action을 사용한다. Wan은
일반 I2V prior에 새 spatial residual을 40개 block 중 4곳만 넣었다. Wan LoRA가 영상 domain을 맞추는 동안
action을 쓰지 않는 shortcut이 남을 수 있다.

## 6. BWM-style Wan2.2-5B 2K 분석

### 6.1 현재 구현

- base: vanilla `Wan2.2-TI2V-5B`
- action: SO100 normalized absolute 6D, 16 future commands 앞에 history proxy 1개를 붙인 17 tokens
- temporal grouping: `[a0×4] / [a0..a3] / [a4..a7] / [a8..a11] / [a12..a15]`
- injection: BWM `encode_ti2v2` action context + timestep AdaLN
- trainable: fresh action encoder 약 47.5M + DiT 약 4.41B; VAE/text path frozen
- loss: plain flow-matching SFT
- run: 8-GPU FSDP, 320×512, global batch 8, DiT LR 1e-5, action LR 5e-5

정상 checkpoint는 20.36GB인 `step-1000`, `step-2000`이다. `step-3000`, `step-4000`은 약 462MB로
불완전하므로 사용하지 않는다. 2K exposure는 16,000개로 repeat 포함 loader의 약 0.752 epoch다.

2K에서 216개 영상과 공식 CSV를 만들었고 Action MAE는 0.45717이었다. 최초 320×512/30-step 결과뿐 아니라
480×640/50-step 재검사에서도 robot의 소실, 재등장, 관절 변형이 남았다. 따라서 upscaling과 sampling step은
blur를 악화시키지만 핵심 morphology 붕괴의 단독 원인은 아니다.

### 6.2 왜 14B보다 완성도가 낮았나

| 항목 | Wan2.1-I2V-14B | BWM-style Wan2.2-TI2V-5B |
|---|---|---|
| image conditioning | `has_image_input=True`, VAE condition + CLIP image embedding | `has_image_input=False`, first-frame latent prefix fusion |
| text | prompt + anti-deformation negative prompt, CFG 5 | text off, CFG 1 |
| prior 보존 | 14B base 고정, adapter/LoRA만 학습 | 4.41B DiT 전체 업데이트 |
| 초기 world-model state | 강한 일반 I2V 480P prior | vanilla TI2V + fresh action path |
| 현재 병목 | action grounding | source morphology와 action을 함께 학습 중 |

14B의 우위는 parameter 수만의 효과가 아니다. 전용 I2V image encoder, CLIP semantic anchor, text guidance,
base prior 보존이 함께 작용했다.

### 6.3 공개 BWM과 다른 부분

공개 BWM은 vanilla Wan에 구조만 붙인 모델이 아니라 별도 `step-12000.safetensors`를 로드한다. 공개 inference
profile은 14D dual-arm EEF/state, 9 history frames, 57-frame rollout, 672×896이다. 현재 대회 경로는 SO100
6D joint command, history image 1장, 17-frame rollout이다. 공개 저장소의 실제 training recipe도 아직
`Coming soon`이다.

따라서 현재 모델은 **BWM-style architecture initialized from vanilla Wan2.2-5B**이며 공개 BWM 자체를
fine-tune한 것으로 부르지 않는다.

## 7. 현재 코드 사용 경계

### 계속 보존하고 사용할 코드

- Cosmos incumbent: `inha_worldmodel_scratch_training/cosmos/train/`
- Wan 14B visual reference: `train/train_wan21_spatial_action.py`,
  `train/generate_wan21_spatial_action.py`, `train/run_wan21_spatial_action.sh`
- 공식 artifact 재검증/재제출: `tools/run_retained_candidate.sh`
- BWM 진단 코드: `train/bwm_native_so100.py`, `train/train_bwm_native_so100.py`,
  `train/generate_bwm_native_so100.py`, `train/run_bwm_native_so100.sh`
- 고정 train-only holdout와 scorer: `valset_holdout/`, `tools/score_predictions.py`,
  `tools/compare_generation_gate.py`

### 기본 경로에서 제외할 코드/실험

- 실패 gate를 통과하지 못한 FOMM, HMA, IRASim, oracle track, Ctrl-World를 장기 연장하지 않는다.
- BWM `step-3000`, `step-4000`을 로드하거나 resume하지 않는다.
- normal/zero/batch-roll이 분리되지 않은 adapter를 step 수만 늘리지 않는다.
- eval sample을 보고 특정 train dataset을 고르거나 loss를 조정하지 않는다.

## 8. 조건부 다음 실험 순서

이 순서는 후속 gate가 반박하면 바뀐다.

### P0. 기존 두 후보 보존

Cosmos v2 0.22와 Wan 14B 0.25의 checkpoint, 영상, CSV를 절대 덮어쓰지 않는다. 새 run은 별도 directory와
manifest를 사용한다.

### P1. Cosmos3-Edge native Action FD 3K 확인

현재 다음 artifact가 존재한다.

- DCP: `open/baseline/outputs/cosmos3_action_fd/action_sft/cosmos3_edge_so100_native_10k/checkpoints/iter_000003000`
- diffusers export: `open/baseline/outputs/cosmos3_action_fd/action_sft/cosmos3_edge_so100_native_10k/model_3000`

생성, train-only normal/zero/batch-roll gate와 공식 CSV가 아직 최종 판정되지 않았다. 형태 보존과 action 분리를
동시에 통과할 때만 10K까지 계속한다.

### P2. Wan2.1-I2V-14B action grounding 강화

source fidelity를 새로 학습하지 않고 기존 14B base를 유지한다. 250-step checkpoint를 맹목적으로 연장하기보다
action branch가 correct-vs-wrong에 실제 gradient와 분리를 만들도록 loss/conditioning 비중을 먼저 확인한다.
1K~2K screen은 train-only holdout에서 다음을 모두 만족할 때만 승격한다.

1. correct가 zero와 batch-roll보다 action component에서 일관되게 낫다.
2. robot morphology와 background가 250 visual reference보다 악화되지 않는다.
3. weighted score가 static과 기존 Wan holdout을 이긴다.

### P3. 실제 공개 BWM checkpoint 전이

5B를 다시 사용할 경우 vanilla에서 처음부터 full-SFT하는 경로보다 공개 BWM `step-12000`의 DiT/action prior를
초기값으로 쓴다. SO100 6D를 BWM 14D space에 연결하는 bridge를 먼저 학습하고, 9-history/dual-arm domain gap을
별도 gate한다. 이 경로도 성공 보장은 없다.

## 9. 승격 및 중단 기준

새 모델은 다음을 모두 보고 판단한다.

1. checkpoint가 완전하고 독립 audit에서 모든 expected tensor를 포함한다.
2. 동일 train-only holdout과 seed에서 normal, zero-motion, batch-roll을 생성한다.
3. correct condition이 counterfactual보다 paired GT에 가까우며 CI가 지속적으로 올바른 방향이다.
4. static보다 움직인다는 사실이 아니라 weighted score가 실제로 낮다.
5. 일부 sample만 좋고 나머지가 morphology collapse이면 평균 하나로 승격하지 않는다.
6. DINO/Video 개선과 Action 손해를 공식 0.3/0.3/0.4 가중치로 함께 본다.
7. inference 216개, 공식 CSV 생성, license와 재현 경로를 최종 제출 전에 확인한다.

학습 loss 하락, adapter nonzero, action permutation에 따른 출력 변화, 큰 모델이라는 사실은 어느 것도 단독
승격 근거가 아니다.

## 10. 문서 지도

| 문서 | 역할 |
|---|---|
| `PROJECT_SUMMARY.md` | 현재 상태와 전체 의사결정의 단일 진입점 |
| `RETAINED_CANDIDATES.md` | 보존 후보의 checkpoint/hash/재실행 명령 |
| `ANALYSIS.md` | 데이터, 평가식, 제출 사후 비교와 BWM/14B 원인 분석 |
| `EMPIRICAL.md` | 제출킷 기반 metric 실측, action lag, 출력 기하 |
| `METHOD.md` | 구현과 실험을 시간순으로 기록한 상세 방법론 |
| `EXPERIMENT_LOG.md` | 반복 gate와 당시 판정의 원본 로그 |
| `RESEARCH.md` | 초기 논문·유사대회 조사 |
| `RESEARCH_SOTA.md` | 최신 모델, license, architecture 조사 |
