# INHA World Model Challenge — 데이터 분석 (2026-07-20)

> 현재 대회/실험/제출/후속 계획의 단일 총정리는 `PROJECT_SUMMARY.md`를 먼저 본다. 이 문서는 데이터와
> 평가 및 제출 사후 분석의 상세 근거를 보존한다.

## 과제 정의
- 입력: 시작 이미지 1장 (640×480 PNG) + 액션 시퀀스 (16, 6) `.npy`
- 출력: 16프레임, 6fps, mp4 영상 216개 (`submission_kit/input_videos/sample_XXXXXX.mp4`)
- 제출: `make_submission_csv.py`가 영상에서 feature를 추출한 CSV (648행 = 216샘플 × 3컴포넌트)

## 평가 (submission_kit 코드 확정)
| 컴포넌트 | 모델 | 상세 |
|---|---|---|
| DINO | timm `vit_small_patch14_dinov2.lvd142m` | 프레임별 CLS 토큰, (16, 384). 518×518 letterbox 입력 |
| Video Feature | torchvision `r3d_18` (Kinetics400) | 16×112×112 trilinear 리사이즈, (512,) |
| Action | 주최측 고정 ckpt (3D CNN + BiGRU, 55MB) | 생성 영상에서 액션 회귀 → 정규화 GT와 MAE **스칼라** (1,1) |
- DINO/Video는 리더보드에서 GT feature와 cosine distance. Action은 제출킷이 **로컬에서 MAE를 직접 계산**해 CSV에 기록 (GT 액션 = data/eval/actions).
- 공식 최종 점수는 **`0.3 × DINO + 0.3 × Video Feature + 0.4 × Action`**이며 낮을수록 좋다.
  Public/Private 분할은 30%/70%다. 따라서 세 원시 지표의 숫자를 같은 비중처럼 읽으면 안 된다.
- 전처리: 영상 → 320×512 aspect-preserving letterbox (pad, 검은 배경). 640×480 입력이면 좌우 패딩.
- 영상은 정확히 16프레임이어야 함 (`expected_frames` 검사).
- 공식 근거: https://dacon.io/en/competitions/official/236736/overview/evaluation

## 학습 데이터
- HuggingFace **LeRobot v2.1** 커뮤니티 SO-100 데이터셋 128개 재패키징 (56 uploader)
- 총 11,132 에피소드 / 1,025,666 프레임 (~47.5시간) / 16프레임 윈도우 약 859,000개
- 원본 30fps → stride 5 다운샘플 → **6fps** 통일 (`original_fps: 30, downsample_stride: 5`)
- 카메라 1개(`observation.images.image`), 480×640이 대부분
- 이상치: dragon-95/so100_sorting(10fps), pranavsaroha 3개+mikechambers 1개+carrot_5(720p/1080p), triton7777(`s_left` 카메라 키)
- parquet 컬럼: action, observation.state, timestamp, frame_index, episode_index, index, task_index
- 액션 = **절대 목표 관절각(도)**, 6차원 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper 0~48)
  - `|a_t − s_{t+1}| < |a_t − s_t|` 확인 → 목표 위치 semantics
- 에피소드 길이: 중앙값 80, p5=42, p95=165; len≥16이 99.7%
- 정규화 통계 제공: `data/train/so100_action_statistics.json` (mean/std, count=974,661)

## 포렌식 기록(설계 근거 사용 금지): eval/train 외형 차이
- eval 216개 = 사실상 **2개 물리 환경**:
  1. 주황/흰색 SO-100, 어두운 회색 책상, 흰 트레이 2개, 빨강·파랑 큐브, 이모지 스티커 (~170샘플)
  2. 검은 SO-100, 나무 책상, 체커 바닥, 투명 박스, 소품(지우개·핑크 물체·회색 링·흰 박스 등) (~46샘플)
- train 128개 데이터셋 대표 프레임 몽타주와 대조 → **일치하는 장면 없음** (썸네일 NN 매칭 MSE<0.01 = 0/216)
- 표본 내 최근접 장면을 못 찾았다는 관찰일 뿐, 전수 불일치나 특정 일반화 전략의 우위를 뜻하지 않는다.
- eval 액션 분포도 train 전체 통계와 어긋남 (z-mean: shoulder_lift −0.63, wrist_roll +1.50)

> **규정상 격리:** 위 내용은 이미 제공된 평가 입력을 설명한 포렌식 관찰일 뿐이다. eval 외형·액션 통계,
> 최근접 train 데이터셋, `valset_evallike`를 학습 데이터 선택, 홀드아웃 구성, 동결 정책, 하이퍼파라미터,
> 체크포인트 선택에 사용하지 않는다. 실제 설계 판단은 eval과 무관하게 만든 train-only 고정 분할에서 한다.
> 규정은 eval을 “어떠한 형태로도 모델 학습에 활용”하는 것과 모델 학습에서 eval 정보를 쓰는
> data leakage를 금지한다. 경계가 애매한 활용은 주최측에 서면 확인한다.

## 베이스라인
- DynamiCrafter(lvdm) 코드베이스, `Doubiiu/DynamiCrafter_512` backbone에서 VAE·OpenCLIP·Resampler만 로드
- UNet은 **11M 초소형 신규 학습** (model_channels=32), action_conditioned=True(action_dims=6), v-param, zero-SNR, 16프레임, latent 40×64
- conditioning: hybrid (concat 시작프레임 latent + cross-attn 이미지/텍스트 + 액션)
- 추론: DDIM 50 steps, cfg 1.0, guidance_rescale 0.7, ddim_eta 1.0, fps 6 → 320×512 padded mp4 저장
- 학습 설정: batch 1 × grad_accum 2, max 100k steps, max_time 48h

## 제약
- 학습: RTX PRO 6000 96GB 1장, 4일 이내 / 추론: 216개, 1시간 이내
- 사전학습모델: 공개 가중치 + 상업/비상업 허용 라이선스 (CC BY-NC 가능 → SVD 등 연구용 라이선스 검토 여지)
- 외부 데이터 금지, eval 학습 활용 금지, submission_kit 수정 금지
- eval 기반 pseudo-label뿐 아니라 eval 정보가 학습·모델 선택에 스며드는 것도 금지. 공식 규칙:
  https://dacon.io/competitions/official/236736/overview/rules
- 환경: uv 사용 (사용자 지시)

## 실행 환경 (개발 서버)
- H100 80GB × 8 (공유), NFS, Python 3.10.6, torch 2.10.0 시스템 설치

## 제출 사후 분석: Cosmos Unified v2와 Wan2.1 Spatial Action (2026-08-07)

> **포렌식 격리:** 이 절의 공개 점수와 eval Action Component는 이미 만든 제출물의 차이를 설명하기 위한
> 사후 기록이다. eval 표본별 결과를 train 데이터 선택, loss 가중치, checkpoint 선택에 되먹임하지 않는다.
> 다음 모델의 채택과 중단은 eval과 독립적인 train-only group holdout에서 먼저 결정한다.

비교 대상은 다음 두 제출이다.

- `submissions/cosmos_unified_action_v2_13067`: Cosmos-Predict2.5-2B action-conditioned backbone,
  기존 action AdaLN + 16개 frame-level action token + rank-32 LoRA, 13,067 optimizer step
- `submissions/wan21_spatial_action_250`: Wan2.1-I2V-14B, source-conditioned spatial action residual
  4개 block 주입 + rank-16 LoRA, 250 optimizer step

육안상 Wan2.1은 로봇과 배경의 형태, 질감, 프레임 완성도가 Cosmos v2보다 낫다. 그러나 공개 점수는
Cosmos v2 `0.22`, Wan2.1 `0.25`로 Cosmos가 더 낮았다. 이 역전은 Action Component에서 직접 설명된다.
`make_submission_csv.py`는 생성 영상에서 고정 action extractor가 회귀한 16×6 trajectory와 제공된 eval
action의 정규화 MAE를 계산해 CSV에 스칼라로 기록한다. 두 CSV의 216개 값을 다시 집계한 결과는 다음과 같다.

| 제출 | 평균 Action MAE | 중앙값 | p25 | p75 | 공개 점수 |
|---|---:|---:|---:|---:|---:|
| Cosmos Unified v2 13,067 | **0.32250** | **0.27244** | 0.18081 | 0.43295 | **0.22** |
| Wan2.1 Spatial Action 250 | 0.48580 | 0.45911 | 0.30762 | 0.64825 | 0.25 |

Wan의 Action MAE가 전체 216개 평균에서 `+0.16330` 높다. 공식 가중치가 Action 0.4이므로, 평가 subset의
분포가 같다고 단순 가정하면 이 차이만으로 약 `+0.065`의 가중 손해가 날 수 있다. Public은 전체 216개의
일부이므로 이 값을 공개 점수의 정확한 component 분해로 읽을 수는 없지만, 좋은 시각 품질만으로 최종 순위가
정해지지 않은 이유는 분명하다. Wan은 그럴듯한 로봇 동작을 만들었어도 주어진 관절 trajectory의 방향, 크기,
위상을 충분히 재현하지 못했다.

### 250 step 부족 가설의 검증

Wan은 8 GPU에서 rank당 batch 1이므로 250 optimizer step 동안 약 2,000 clip exposure를 보았다. base clip이
10,697개이므로 한 epoch의 약 19%다. 반면 Cosmos Unified v2는 effective batch 8로 13,067 step, 즉
104,536 exposure를 사용해 중복 없는 window prefix를 거의 한 번 통과했다. 노출량이 약 52배 차이나므로
Wan 250 step이 부족하다는 판단 자체는 맞다.

하지만 `step 부족`만이 원인이라는 증거는 없다. Wan 학습 로그의 구간 평균은 다음과 같다.

| step 구간 | reconstruction | ranking | correct − wrong denoising error |
|---|---:|---:|---:|
| 1–50 | 0.10626 | 0.001979 | `+2.21e-5` |
| 51–125 | 0.09819 | 0.001952 | `-1.14e-5` |
| 126–200 | 0.09230 | 0.001981 | `+3.53e-5` |
| 201–250 | 0.09276 | 0.001978 | `+3.97e-6` |

reconstruction은 좋아졌지만 correct action과 reversed action의 오차 차이는 처음부터 끝까지 0 주변이고,
margin ranking도 약 0.002에서 줄지 않았다. 즉 이 구간에서 학습된 주된 기능은 SO100 영상 외형/평균 동역학이며,
action 식별성은 개선 추세가 없다. 같은 objective를 10k까지 단순 연장하면 action보다 appearance shortcut을 더
학습할 수도 있다.

구조적 차이도 크다.

- Cosmos v2는 robot action-conditioned 사전학습 backbone의 기존 action AdaLN을 유지하고, 16개 action token을
  28개 DiT block의 별도 attention softmax에 넣었다.
- Wan2.1은 일반 I2V prior에 2.69M spatial adapter를 새로 붙이고 40개 block 중 4곳에 같은 source-aligned
  residual을 넣었다. 약 76.68M LoRA는 영상 domain을 빠르게 맞출 수 있지만 action을 사용하지 않아도
  reconstruction loss를 줄일 수 있다.
- Wan의 hybrid 18D action은 absolute/anchor delta/step delta를 모두 제공하지만, 숫자 trajectory를 영상의
  특정 관절 위치에 대응시키는 사전학습은 없다. source VAE feature가 위치 단서를 주더라도 robot kinematics와
  camera geometry를 데이터에서 새로 학습해야 한다.
- 정답-vs-counterfactual 항은 존재하지만 전체 reconstruction 규모에 비해 작고, wrong action이 실제 출력에
  차이를 만들지 않는 상태에서 250 step이 종료됐다.

따라서 현재 결론은 **“Wan 250은 명백히 undertrained이지만, 현 구조를 오래 돌리면 해결된다고 확정할 수
없다”**다. Wan 연장 실험은 Action MAE 하락, correct-vs-wrong 음의 분리, normal-vs-zero/batch-roll gate,
형태 보존을 동시에 확인해야 한다. 이 중 action 분리가 계속 0이면 step을 더 주는 대신 conditioning 경로를
바꿔야 한다.

## BWM-style Wan2.2-5B 2k 시각 붕괴 분석 (2026-08-07)

> **포렌식 격리:** eval 216개와 그중 8개 480p/50-step 재생성의 육안 비교는 이미 만든 모델의 실패를
> 설명하기 위한 기록이다. eval sample별 외형을 학습 데이터, loss, hyperparameter 또는 checkpoint 선택에
> 되먹이지 않는다. 구조 선택은 코드/공개 checkpoint 계약과 train-only group holdout으로 다시 검증한다.

정상 checkpoint `open/baseline/outputs/bwm_native_so100_5k/step-2000.safetensors`는 20.36GB이며 30개 DiT
block과 6D action encoder를 모두 포함한다. 따라서 이번 출력은 과거의 441~462MB 불완전 checkpoint를 잘못
읽은 결과가 아니다. 반면 같은 폴더의 `step-3000`, `step-4000`은 약 462MB로 불완전하므로 사용하지 않는다.

최초 320x512/30-step 영상은 전체적으로 부드럽고 robot morphology가 불안정했다. 같은 2k checkpoint를
480x640/50-step으로 다시 생성해 업스케일과 sampling budget을 교정했지만 소실, 재등장, 관절 변형이 남았다.
그러므로 낮은 생성 해상도와 30-step은 blur를 악화시키는 보조 요인이지만 핵심 붕괴의 충분한 설명은 아니다.

코드 기준으로 14B와 5B의 source condition은 다음처럼 다르다.

- Wan2.1-I2V-14B는 `has_image_input=True`이며 VAE image condition과 CLIP image embedding을 함께 사용한다.
- Wan2.2-TI2V-5B는 `has_image_input=False`, `require_clip_embedding=False`,
  `fuse_vae_embedding_in_latents=True`로 clean history latent를 temporal prefix에 직접 융합한다.
- 14B 후보는 prompt/negative prompt와 CFG 5를 사용하고 base를 보존한 채 2.69M adapter와 76.68M LoRA만
  학습했다. 5B 후보는 text 없이 CFG 1로 동작하며 fresh 47.5M action encoder와 4.41B DiT를 함께 갱신했다.

따라서 “14B가 더 크기 때문에 선명하다”만으로는 설명이 부족하다. 전용 I2V source anchoring, semantic image/text
condition, pretrained prior 보존이 동시에 14B에 유리했다. 반대로 현재 5B는 공개 BWM robot checkpoint를 쓰지
않아 SO100 appearance, dynamics, 6D action grounding을 한 번에 배워야 했다.

공개 BWM의 높은 영상 품질과 현재 결과를 직접 비교할 수도 없다. 공개 inference는 BWM `step-12000`, 14D
dual-arm EEF/state, 9 history frames, 57-frame rollout을 사용하지만 현재 모델은 vanilla Wan2.2-5B, 6D SO100
joint command, 1 history image, 17-frame rollout이다. 공개 저장소의 실제 training recipe도 아직 공개되지 않았다.
즉 재사용한 것은 action encoder와 DiT 주입 **구조**이며, 성능을 만든 pretrained world-model state와 학습 조건은
재현되지 않았다.

2k의 총 exposure는 global batch 8 기준 16,000개로 repeat 포함 loader의 약 0.752 epoch다. 따라서 undertraining은
실재하는 가능성이지만, 더 많은 step이 morphology와 action을 함께 개선할지는 아직 증명되지 않았다. 현재 5B를
계속 연장하려면 train-only holdout에서 다음 세 조건이 먼저 필요하다.

1. normal 영상의 robot identity와 sharpness가 checkpoint가 진행될수록 악화되지 않는다.
2. correct action이 zero-motion과 batch-roll보다 paired GT에 일관되게 가깝다.
3. 시각 손해를 포함한 weighted score가 static과 기존 후보를 이긴다.

현 증거에서는 source fidelity를 이미 보인 Wan2.1-I2V-14B에서 action grounding만 강화하거나, 공개 BWM
step-12000을 실제 초기값으로 사용해 SO100 6D bridge를 학습하는 편이 현재 vanilla-5B full-SFT의 무조건 연장보다
우선이다. 이 우선순위 역시 후속 train-only gate가 반박하면 변경한다.
