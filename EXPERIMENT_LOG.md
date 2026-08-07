# INHA World Model Challenge — 반복 실험 로그

> 이 파일은 계획을 고정하는 명세가 아니라 **실험 결과로 계속 갱신하는 의사결정 기록**이다.
> 논문·모델 크기·한 번의 좋은 수치보다 동일 조건의 train-only 반증을 우선한다.
> eval 입력에서 얻은 통계·유사도·육안 관찰은 이 로그의 모델 선택 근거로 넣지 않는다.

## 1. 현재 안전 gate

| ID | gate | 상태 | 증거 / 다음 조건 |
|---|---|---|---|
| G00 | partial EMA 최소 재현 | **PASS** | `tools/check_partial_ema_roundtrip.py`: frozen main 보존, partial EMA 적용, restore 성공 |
| G01 | 실제 학습 checkpoint EMA round-trip | PENDING | main/EMA 각각 생성, 학습 당시 frozen 파라미터 hash 불변, action EMA key 완전성 확인 |
| G02 | 모델 선택용 고정 holdout | **PASS** | `valset_holdout/`: 실제 제외 6 datasets × 4 clips, manifest SHA256 `ccf8b19f...fcab` |
| G03 | action 표현별 scale | PENDING | absolute/anchor/step에 train-fold 통계 적용 후 conditioner 진폭 기록 |
| G04 | full-model 추론 제한 | PENDING | 목표 장비 상당 환경에서 I/O 포함 216개 < 60분 |
| G05 | checkpoint 저장 예산 | PENDING | 1개 실제 크기와 84시간 예상 총량 측정, 보존 상한 설정 |

`G01~G04` 전 결과는 탐색용일 뿐 최종 baseline으로 승격하지 않는다. EMA가 의심되면 먼저
`--no-ema`로 main weights를 사용하고, EMA 결과는 별도 행으로 기록한다.

## 2. 최소 baseline 순서

모든 생성 점수는 **동일한 train-only 고정 manifest, seed, 출력 기하, 공식 extractor**로 잰다.

| ID | 목적 | 변경점 | 상태 |
|---|---|---|---|
| B00 | 하한/무동작 기준 | 첫 입력 프레임 16회 반복 | READY |
| B01 | 제공 모델 기준 | 제공 11M checkpoint, 제공 설정 | READY |
| B10-M | incumbent 학습 sanity | 10k full-UNet 학습, main weights(`--no-ema`) | **VALID SCREEN / NOT PROMOTED** |
| B10-E | EMA 구현 검증 | B10과 동일 checkpoint, 재구성 EMA | G01 뒤 실행 |
| A10 | action 표현 | absolute vs anchor delta vs step delta | G02·G03 뒤 실행 |
| A11 | 시간 정렬 | same-index vs causal shift | A10 상위 표현에서 실행 |
| A12 | 시작 자세 보존 | anchor delta vs `delta_anchor` | A10/A11 결과가 모호할 때 실행 |
| A20 | 학습 정책 | action_only vs temporal vs all/LoRA | action 표현·정렬 고정 뒤 실행 |
| A30 | conditioner | additive vs token/AdaLN/spatial auxiliary | incumbent가 action gate에 실패할 때 실행 |

초기 screen은 seed 1개로 할 수 있지만 작은 차이로 후보를 제거하지 않는다. 상위 2개는 같은 예산으로
2개 이상 추가 seed 또는 2~3개 group fold에서 순위가 유지되는지 확인한다. `action_only`에서 action 반응이
없다는 이유만으로 표현을 폐기하지 않는다. conditioner 용량 부족과 표현 실패를 구분할 수 없기 때문이다.

### B10-M 학습 완료 — 2026-07-20

- 판정: **TRAINING HEALTH PASS / MODEL QUALITY PENDING**
- checkpoint: `open/baseline/outputs/full_unet/inha_full_unet/checkpoints/epoch=7-step=10000.ckpt`
- checkpoint 내부 확인: `global_step=10000`, `epoch=7`, tensor key 2,366개, main action key 5개,
  partial EMA key 849개(그중 action 5개)
- 설정: temporal 정책 기본값, anchor delta, shift 0, LR 1e-5, warmup 500 optimizer step
- 실행: 8×H100 DDP, GPU당 batch 1, accumulation 1, 유효 배치 8, 10,000 optimizer step
- wall time: 모델 초기화 포함 약 2시간 26분
- 저장: 2k/4k/6k/8k/10k 및 `last.ckpt`, 파일당 8,054,083,101 bytes, 총 약 46GB
- 로그: `open/baseline/outputs/logs/inha_full_unet/version_3/metrics.csv`

rank 0에 기록된 epoch train loss는 0.07110 → 0.06658 → 0.06741 → 0.06789 → 0.07012 →
0.06844 → 0.06867이었고, 마지막 partial epoch는 0.06566이었다. NaN/OOM/수치 발산은 없었다.
그러나 DDP epoch log에 `sync_dist=True`가 없어 이 값은 전체 8-rank 평균이 아니며, 현재 val도 무작위
8 batch뿐이다. 따라서 이 loss들로 4k/10k 우열, 과적합, action following을 판정하지 않는다.

종료 시 `destroy_process_group()` 경고가 있었지만 `max_steps=10000`으로 정상 종료했고 10k와
`last.ckpt`가 모두 `torch.load(..., mmap=True)`로 열렸다. 학습 결과를 무효화할 증거는 아니다.

중요한 연장 제약: `save_weights_only: true`라 checkpoint 최상위에는 `state_dict`, loop metadata만 있고
optimizer/scheduler state가 없다. 그러므로 현재 10k에서 30k로의 **정확한 resume는 아직 불가능**하다.
생성 평가 후 연장이 결정되면 action head를 다시 zero-init하지 않는 weight-init 경로를 만들고 optimizer를
새로 시작할지, full-state checkpoint를 저장하도록 바꿔 다시 학습할지 명시적으로 선택한다.

다음 판정 순서는 잠정적이며 결과에 따라 교체할 수 있다.

1. G02의 완전히 분리된 고정 train-only holdout을 만든다.
2. 동일 manifest/seed에서 4k와 10k **main weights**를 먼저 생성한다.
3. 정지 영상·제공 backbone과 공식 세 component를 비교하고 정상 action 대 permutation 반응을 잰다.
4. 10k가 4k보다 낫고 action gate를 통과할 때만 actual-checkpoint EMA와 30k 연장을 검토한다.

### DIAG-20260721-01 — eval 8개 붕괴 관찰(모델 선택에는 사용하지 않음)

- checkpoint: 10k main weights, seed 0, DDIM 50, eta 1, CFG 1, batch 2
- 관찰: 8개 모두 첫 프레임과 배경은 보존됐다. 전역 노이즈 발산이 아니라 로봇/그리퍼가 중후반에
  형태를 잃거나 다른 구조로 바뀌는 local temporal morphology collapse다.
- 심한 예: `sample_000003`은 frame 5~6부터 로봇 형태가 흐려지고 frame 12 이후 다른 구조로 굳는다.
  `sample_000000`도 frame 11 이후 오른쪽에 분리된 팔/그리퍼처럼 보이는 구조가 생긴다.
- 8개 입력 대비 생성 frame 0 pixel MAE는 2.75~3.27/255라 첫 프레임 pin/VAE 경로는 정상이다.
- 첫 8개 중 심한 샘플의 anchor-delta 조건 크기는 outlier가 아니었다. 단순 action 크기 하나로 설명되지 않는다.
- 같은 eval 샘플의 제공 11M 출력은 frame 1~4부터 장면이 patch 형태로 전역 붕괴했다. 10k full model이
  명백히 더 안정적이므로 backbone 선택이 완전히 실패한 것은 아니다.

이 관찰은 eval 입력을 사용했으므로 학습 step/방법 선택 근거로 되먹이지 않는다. 아래의 고정 train-only
holdout에서 같은 현상을 재현해 판정한다.

### G02 고정 holdout — 2026-07-21

- 경로: `valset_holdout/`
- 구성: 학습에서 실제로 제외된 6 datasets, dataset당 4 clips, 총 24개
- 각 sample: 입력 PNG, raw action `(16,6)`, GT video `(16,H,W,3)`, episode/start/camera manifest 고정
- manifest SHA256: `ccf8b19f0193ebab10dad1182effc711155bca7a42de2270dd41d6a5f5e7fcab`
- 생성: `tools/build_valset.py --training-group-holdout 6`
- checkpoint step 1차 screen: `CUDA_VISIBLE_DEVICES=<GPU> bash tools/run_step_screen.sh`

1차 판정은 다른 축을 고정하고 2k/6k/10k main, eta 1만 비교한다.

- 점수와 붕괴율이 2k → 6k → 10k로 일관되게 개선: **undertraining 가설 유지**, 20~30k 연장 후보
- 2k가 가장 낫거나 6k 이후 악화: **prior 훼손/동결 정책 문제**, 더 오래 학습하지 않음
- 세 checkpoint가 비슷하고 action permutation 반응이 약함: **conditioner/표현 문제**, additive 대신 token/AdaLN 등 검토
- step 추세가 애매할 때만 10k eta 0, reconstructed EMA, zero/batch-roll action을 2차 screen한다.

### STEP-SCREEN-20260721-01 — 고정 holdout 24개, main/eta 1

낮을수록 좋다. 모든 행은 동일한 manifest, seed 0, DDIM 50, CFG 1, batch 2다.

| checkpoint | DINO | Video | Action | weighted | static 대비 |
|---|---:|---:|---:|---:|---:|
| static | **0.077689** | 0.071936 | **1.276147** | **0.555346** | 0 |
| 2k | 0.186359 | 0.073070 | 1.285989 | 0.592224 | +0.036878 |
| 6k | 0.208427 | 0.067677 | 1.284082 | 0.596464 | +0.041118 |
| 10k | 0.188623 | **0.053763** | 1.281031 | **0.585128** | +0.029782 |

10k−2k paired bootstrap(24 samples)의 weighted 평균 차이는 -0.00710이지만 95% CI
`[-0.02026, +0.00468]`로 0을 포함했다. 반면 Video는 24개 중 21개에서 좋아졌고 평균 -0.01931,
95% CI `[-0.02816, -0.01149]`였다. Action은 평균 -0.00496, 12/24 승리,
95% CI `[-0.03006, +0.01774]`로 개선 증거가 없다.

dataset별 weighted는 10k가 6개 중 4개에서 최선, 6k와 2k가 각각 1개에서 최선이었다. 따라서 10k가
6k보다 낫다는 증거는 있지만 2k→10k의 단조 개선이나 현 설정의 30k 연장을 정당화하지는 못한다.
세 checkpoint 모두 정지 기준선을 넘지 못했다.

별도 eval 6개 자기참조 DINO 진단은 late drift가 4k 0.2516, 6k 0.2969, **8k 0.1845**,
10k 0.3149로 비단조였다. 정상-vs-reversed action 0.1985, 정상-vs-shuffled 0.1694라 conditioner에
대한 민감도는 강했다. 그러나 다른 action에 영상이 달라지는 것은 올바른 방향의 action following 증거가 아니다.
GT 대표 샘플에서는 정답이 거의 정지/평행 이동인데 모델이 큰 팔 이동, 회전, 형태 변형을 만드는 경우가 있었다.

현재 판정:

- **순수 undertraining 가설은 지지되지 않는다.** 30k 연장은 보류한다.
- backbone은 제공 11M보다 장면 보존이 훨씬 좋아 완전 폐기하지 않는다.
- 현 병목은 action을 무시하는 collapse가 아니라, action에 강하게 반응하면서도 잘못된 motion/morphology를
  만드는 **action correctness + temporal identity 문제**에 가깝다.
- 8k를 같은 GT holdout에서 추가 채점하고, 10k normal/zero/batch-roll의 Action 점수를 비교한 뒤
  현 additive conditioner를 유지할지 결정한다.

### ARCH-DECISION-20260721-01 — incumbent 이후의 잠정 순서

이 결정은 절대적인 계획이 아니다. 새 holdout 결과, 단일 RTX PRO 6000 처리량, 구현 난이도에 따라
후보 순서를 바꾸거나 폐기한다.

1. **현 DynamiCrafter를 그대로 장기 연장하지 않는다.** 장면 prior는 보존하되, 프레임별 action
   AdaLN-Zero/Frame-Ada 또는 action token을 짧은 동일예산 실험으로 비교한다.
2. 백본 교체 1순위 challenger는 **Wan2.2-TI2V-5B + 프레임별 action AdaLN-Zero + LoRA**다.
   1X 유사대회 근거는 강하지만, 원 레시피의 multi-node B200 비용과 이 대회의 1시간 추론 제한이 달라
   1-step VRAM/처리량과 216개 환산 시간을 먼저 통과해야 한다.
3. Wan이 메모리·시간 gate를 실패하면 **Cosmos-Predict2.5-2B action-conditioned**를 우선 비교한다.
   공식 post-training/distillation 경로가 장점이지만 7D EE delta를 6D joint target에 맞추는 것은 별도 실험이다.
4. train-only 영상에서 URDF/카메라 투영 오차가 충분히 작을 때만 **OSCAR-2B skeleton control**을 올린다.
   이 gate를 못 넘으면 pixel-aligned 계열은 optical-flow/point-track auxiliary 정도로 축소한다.
5. **latent autoregressive**는 독립 challenger로 유지한다. iVideoGPT식 video/action token interleave는
   action 정렬·추론 속도가 장점이지만 16프레임 rollout drift를 같은 DINO/Video/Action으로 먼저 잰다.
6. IRASim Frame-Ada는 더 가벼운 직접 구조 및 conditioner 참고선이다. LTX, Ctrl-World,
   EnerVerse, VideoVLA/Motus/UVA/FRAPPE는 위 후보가 실패하거나 고유 장점이 필요한 경우의 후순위다.

제출 판단도 모델 승격과 분리한다. 10k는 train-only weighted `0.585128`로 static `0.555346`보다
나빠 최종 후보 근거는 없지만, 고정된 checkpoint의 파이프라인/분포 확인용 1회 제출은 의미가 있다.
정지 제출을 기준점으로 먼저 보존하고, 10k 제출 결과를 다음 학습의 eval-derived 튜닝 신호로 사용하지 않는다.
현재 8개 생성 파일의 steady-state 간격은 batch 2당 약 14.4초라 H100 단순 환산은 216개 약 26분이지만,
RTX PRO 6000 96GB의 1시간 제한 통과를 증명하지는 않는다.

### A30-FRAME-ADALN-IMPL-20260721 — Stage 1 2k 학습 완료, 생성 gate 대기

- 상태: **VALID TRAIN / GATE PENDING**
- 유일한 의도적 구조 변경: pretrained timestep/additive 경로와 분리된 ResBlock별
  `action_modulation(1280 → 2C)`로 frame-wise normalization scale/shift 적용
- 초기화: action modulation projection 22개 zero-init. step 0에서 pretrained 출력 보존
- 학습 정책: 첫 screen은 `action_only`, 2k optimizer step, effective batch 8, LR 2e-5
- 전체 모델: 1.4901B. generic backbone 1.4368B checkpoint load 100.0%, shape mismatch 0
- 신규 action embed/modulation: 약 51.2M
- 작은 CPU 검사: zero-init bitwise 동일, modulation gradient 발생, 학습 후 action별 출력 차이 통과
- 기존 additive config: generic backbone 1.4368B 100.0%, shape mismatch 0으로 재검증
- 실행기 보정: 1/2/4/8 GPU에서 effective batch 8이 유지되도록 DDP local batch를 자동 조정
- 실행: H100 8장, GPU당 batch 1, accumulation 1, effective batch 8
- checkpoint: 1k/2k/last, 각 6,174,207,747 bytes. `global_step=2000`, `epoch=1`
- 시간: config 생성 10:17:17 → last 저장 10:43:34로 초기화·검증 포함 약 26분 17초.
  1k→2k checkpoint 간격은 약 12분 44초(optimizer step당 약 0.76초, 8-GPU wall time)
- checkpoint action 키: main 49, partial EMA 49, non-finite 0
- logged epoch train loss: 0.07254 → 0.06852. epoch 0 val loss 0.09293
  (고정 생성 점수가 아니므로 승격 근거로 사용하지 않음)
- GPU benchmark stdout은 파일에 보존되지 않아 peak VRAM 수치는 아직 로그에 없다.
- `diagnostics/frame_adaln_gate/`와 `results/frame_adaln_2k_*`가 없어 생성 gate는 아직 실행되지 않은 상태다.

실행:

```bash
CUDA_VISIBLE_DEVICES=<GPU> bash train/run_frame_adaln.sh --bench-steps 8
CUDA_VISIBLE_DEVICES=<GPU> bash train/run_frame_adaln.sh
CUDA_VISIBLE_DEVICES=<GPU> bash tools/run_frame_adaln_gate.sh
```

### A33-SPATIAL-ORACLE-PREGATE-20260803

- 상태: **PASS PRE-GATE / 250-STEP WAN ORACLE GATE REJECT**
- 장기 학습 승인: **아니오**
- 입력 표현: RAFT cycle-consistent point tracks → occupancy/dx/dy control video
- split: 121 train datasets / 6 group-holdout datasets; fixed clips 64/8
- 시각 감사에서 발견·수정: 정지 clip에 padding-border track을 채우던 fallback 제거
- 학습 품질 통과: 50/64 train clips
- holdout: tracks≥8 7/8, cycle 0.8477, coverage 0.5916, action-motion rho 0.3505
- 결과: `results/spatial_control_gate.json`
- 첫 4-GPU launch engineering failure: paired batch에 timestep `(2,)`를 넘겨 Wan separated-timestep의
  `(4,160) * timestep` broadcast가 실패했다. 동일 timestep 원칙대로 `(1,)`을 유지하고 token modulation이
  paired batch에 broadcast되게 수정했으며, 정확한 shape 회귀검사와 paired gradient 검사를 통과했다.
- 다음 중단 조건: oracle normal이 zero/batch-roll과 구별되지 않거나, 로봇 밖 새 형상/팔을 생성하면
  action→track 구현 없이 spatial Wan 경로를 중단한다.

250-step 실행 및 holdout gate:

- optimizer step `250`, world size `4`, wall time `187.48 s` (`0.7499 s/step`), rank0 peak `22.61 GiB`
- train clips `50`, track-adapter LR `1e-4`, LoRA LR `2e-5`
- checkpoint: `open/baseline/outputs/wan_oracle_track_250/step-250.safetensors`

| variant | weighted | normal 대비 |
|---|---:|---:|
| oracle normal | 0.618634 | 0 |
| zero track | 0.617440 | -0.001194 |
| batch-roll track | 0.616648 | -0.001986 |
| static | **0.586220** | **-0.032414** |

- normal−zero Action `+0.003389`, CI `[-0.001379,+0.009987]`: 정답 oracle 우위 없음
- normal−batch-roll Action `+0.002498`, CI `[-0.001592,+0.008056]`: 다른 clip의 track보다도 우위 없음
- normal−static weighted `+0.032414`: 생성이 정지 기준선보다 크게 나쁨
- normal−zero weighted `+0.001194`, CI `[-0.001619,+0.004198]`: 생성 전체도 구별되지 않음
- 최종 gate: `REJECT`

체크포인트 내부 진단은 이 REJECT가 단순한 250-step 부족이 아님을 보인다. holdout에서 adapter 출력
abs mean은 normal `0.015332`, zero `0.015235`였고, 실제 `f(control)-f(0)`은 `0.0001068`, 즉 전체 출력의
`0.70%`뿐이었다. batch-roll에 따른 변화도 `1.16%`였다. 입력이 0이어도 Conv3D bias를 통해 거의 같은
시공간 보정이 발생했으며, 영상 normal/zero/batch-roll도 육안상 거의 동일했다. adapter가 oracle trajectory를
번역한 것이 아니라 또 하나의 dataset/domain 보정을 학습한 것이다.

따라서 현 checkpoint를 1k로 연장하거나 action→track predictor를 구현하지 않는다. 단, 이 결과만으로
point-track 표현 자체를 기각하지는 않는다. 현재 adapter가 Cosmos 실패 뒤 세운 `f(0)=0` 구조 불변식을
지키지 않았고, 학습 wrong condition도 cross-clip roll이 아니라 같은 track의 시간 역순이었다. 재시험한다면
`g(c)=raw(c)-raw(0)`의 exact-zero adapter, frozen Wan/LoRA, cross-clip wrong-track ranking만 적용한 **마지막
250-step oracle v2**로 제한한다. 이 수정 후에도 정답 track 우위가 없으면 spatial Wan 경로를 종료한다.

v2 구현 상태: **UNIT PASS / GPU TRAIN PENDING**. `structural_zero`, adapter-only, cross-dataset wrong-track을
각각 코드와 전용 실행 스크립트에 고정했다. exact-zero, nonzero gradient, 50개 cross-dataset pairing,
Python compile 및 shell syntax 검사를 통과했다. 실행은 `train/run_wan_oracle_v2.sh`, 평가는
`tools/run_wan_oracle_v2_gate.sh`를 사용한다.

v2 250-step 결과:

- 상태: **VALID / REJECT / SPATIAL WAN TERMINATED**
- 4x H100, `185.46 s`, `0.7418 s/step`, rank0 peak `22.15 GiB`
- checkpoint는 adapter-only 1.7 MiB; zero 입력 adapter output은 exact `0.0`
- normal adapter output abs mean `0.010556`: v1과 달리 조건 신호 자체는 실제로 발생

| variant | DINO | Video | Action | weighted |
|---|---:|---:|---:|---:|
| oracle normal | 0.282524 | 0.137346 | 1.311725 | 0.650651 |
| zero track | 0.405136 | 0.178027 | **1.280819** | 0.687276 |
| batch-roll track | **0.250600** | **0.127665** | 1.321830 | **0.642211** |
| static | 0.117142 | 0.083038 | 1.315416 | **0.586220** |

- normal−static weighted `+0.064431`: oracle 생성이 정지 기준선보다 크게 악화
- normal−zero Action `+0.030906`, CI `[-0.002745,+0.080086]`: 정답 track이 zero보다 평균상 더 나쁨
- normal−batch-roll Action `-0.010105`, CI `[-0.052758,+0.024683]`: point 우위만 있고 불확실
- normal−zero weighted `-0.036625`, CI `[-0.075055,-0.005015]`: adapter가 vanilla Wan 환각을 억제하지만,
  이는 trajectory correctness가 아니라 domain regularization 효과
- 육안 감사: zero에서 로봇 장면과 무관한 고채도 기둥/박스/새 물체가 생성됨. normal과 batch-roll은 이를
  일부 억제하지만 glove/robot의 정답 운동을 재현하지 못함

exact-zero로 v1의 bias shortcut을 제거해도 correctness가 통과하지 않았으므로 사전에 정한 중단 조건을
충족한다. Wan point-track adapter, action→track predictor, Cosmos/Wan 추가 장기학습은 진행하지 않는다.
다음 후보는 생성 prior가 아니라 source frame을 보존하는 oracle dense-flow warp → retrieval motion-transfer의
두 단계 gate로 제한한다.

후속 문헌 재대조에 따른 정정: 위 문장의 범위는 **백업 후보**로 축소한다. 1X Sampling 우승과 IRASim은
모두 생성모델이며, 현재 실험은 이 계열을 충분히 반증하지 못했다. 특히 공식 IRASim Frame-Ada checkpoint와
SO-100 adapter 코드는 준비됐지만 GPU 실행 결과가 전혀 없다. 따라서 다음 주력 gate는 faithful IRASim
500-step이고, dense-flow/retrieval은 생성모델을 대체하는 결론이 아니라 static 보존형 진단·백업이다.

gate는 normal/static/incumbent weighted와 normal-vs-zero/batch-roll Action의 paired bootstrap을 기록한다.
통과 전에는 temporal 장기 학습, DINO auxiliary, Wan/Cosmos/latent-AR 구현으로 자동 승격하지 않는다.

### A34-IRASIM-OFFICIAL-SMOKE-20260803

- 상태: **PUBLIC WEIGHT LOAD PASS / GENERATION SMOKE PASS / TRAIN SMOKE PENDING**
- 공식 checkpoint: RT-1 `frame_ada/0300000.pt`; archive 뒤쪽은 truncated지만 필요한 첫 PyTorch zip은
  중앙 디렉터리 1,481 entry와 `torch.load`를 통과함
- 모델: IRASim-XL/2, `679,394,320` parameters
- compatibility: 의도한 `temp_embed`, `embed_state.fc1.weight` 두 shape만 교체; unexpected `0`, finite
- 11 GiB 공개 checkpoint는 `mmap=True`로 로드해 optimizer state를 DDP rank마다 materialize하지 않음
- BF16 수정: timestep sinusoidal embedding을 MLP weight dtype으로 변환; BF16 unit test 통과
- PNDM 제약: PRK warm-up 때문에 inference step 최소 `4`
- zero-step 1-sample smoke: `1.2513 s`, output 16 frames, 6fps, 640x480
- 4-step 영상은 심한 노이즈이며 품질 판정에 사용하지 않음; 목적은 VAE/PNDM/기하 실행 검증
- wrapper: `train/run_irasim_action.sh`, `tools/run_irasim_gate.sh`

500-step 실행 결과 (4x H100):

- 상태: **TRAIN PASS / 50-STEP GATE PENDING**
- optimizer step `500`, wall time `107.08 s`, `0.2142 s/step`
- rank0 peak allocated `22.10 GiB`, world size `4`
- dataset `121`, fixed clip index `10,347`, action mode anchor delta
- `step-{250,500}.pt`, 각 2.6 GiB, 297 tensors, non-finite `0`
- step loss는 diffusion timestep 난이도에 따라 `3.9e-5~0.669`로 변동; NaN/OOM 없음
- 종료 시 `destroy_process_group` 경고는 학습 산출물과 무관한 cleanup warning

500-step 24-holdout gate:

- 상태: **VALID / REJECT AT 500 / CHECKPOINT TREND PENDING**
- normal: DINO `0.477168`, Video `0.162884`, Action `1.238549`, weighted `0.687435`
- static weighted `0.555346`; normal−static `+0.132089`
- normal−zero Action `-0.006219`, CI `[-0.018916,+0.005828]`
- normal−batch-roll Action `-0.003446`, CI `[-0.029804,+0.023517]`
- normal−10k incumbent weighted `+0.102307`, CI `[+0.056656,+0.149590]`
- 영상 감사: 세 action 조건 모두 첫 프레임 뒤 RT-1식 다른 팔/물체를 생성하며 SO-100 외형을 보존하지 못함
- 50-step inference `1.89 s/sample`, peak `1.69 GiB`; 216개/1시간 제한은 충분히 통과
- 500×4 = 2,000 clip exposure로 10,347 고유 clip 대비 약 0.19 epoch. correctness point 두 개가 모두
  처음으로 올바른 방향이므로, 2k 연장 전 이미 저장된 step-250 normal을 같은 설정으로 평가해 외형 추세를 본다.
  250→500 DINO/weighted가 개선되지 않으면 종료하고, 명확히 개선될 때만 2k를 허용한다.

step-250 normal 추세 비교:

| step | DINO | Video | Action | weighted |
|---:|---:|---:|---:|---:|
| 250 | 0.517987 | 0.196924 | **1.192829** | 0.691605 |
| 500 | **0.477168** | **0.162884** | 1.238549 | **0.687435** |

- 250→500 DINO `-0.040819` (7.9% 개선), Video `-0.034040` (17.3% 개선): SO-100 외형 적응은 진행 중
- Action `+0.045720` 악화, weighted는 `-0.004170`만 개선: action 학습은 아직 불충분하거나 trade-off
- 500-step은 0.19 epoch뿐이고 2k도 약 0.77 epoch이므로 **2k screen만 승인**. 2k에서도 static gap이
  크게 줄지 않거나 correctness가 발전하지 않으면 5k/10k로 연장하지 않는다.

### A30-FRAME-ADALN-GATE-20260721 — 2k action-only

- 상태: **VALID / REJECT**
- checkpoint: `frame_adaln/inha_frame_adaln_action_only/.../epoch=1-step=2000.ckpt`
- holdout: 고정 train-only 24개, seed 0, DDIM 50, eta 1, main weights

| variant | DINO | Video | Action | weighted | static 대비 |
|---|---:|---:|---:|---:|---:|
| static | **0.077689** | 0.071936 | 1.276147 | **0.555346** | 0 |
| Frame-Ada normal | 0.215488 | 0.085253 | 1.270586 | 0.598457 | +0.043110 |
| Frame-Ada zero | 0.188581 | 0.078579 | 1.271119 | 0.588596 | +0.033249 |
| Frame-Ada batch-roll | 0.223140 | 0.084194 | **1.262323** | 0.597130 | +0.041783 |
| additive 10k incumbent | 0.188623 | **0.053763** | 1.281031 | 0.585128 | +0.029782 |

paired 결과:

- normal−static weighted `+0.04311`, 3/24 승리, 95% CI `[+0.02479,+0.06291]`: 명확히 악화
- normal−10k: DINO `+0.02686` CI `[+0.00338,+0.05129]`, Video `+0.03149`
  CI `[+0.02131,+0.04178]`; Action `-0.01044`는 CI가 0을 포함
- normal−zero: DINO `+0.02691` CI `[+0.00941,+0.04619]`; 올바른 action이 외형을 더 훼손
- normal−batch-roll Action `+0.00826`, 11/24 승리, CI `[-0.02227,+0.03812]`:
  wrong action이 평균상 더 좋고 correctness 증거가 없음
- dataset별 weighted도 6개 모두 static보다 나빴고, 5/6에서 additive 10k보다 나빴다.

판정:

1. Frame-Ada action-only를 5k/10k로 연장하지 않는다.
2. 주입 세기는 충분하지만 action 의미·시간 정렬과 대응하지 않는다. 백본 크기만 바꾼 Wan에서도 반복될 수 있다.
3. 다음으로 가장 싼 반증은 같은 Frame-Ada 1k에서 `delta shift=-1`, `delta_step shift=0`,
   `delta_anchor shift=0`의 normal-vs-batch-roll Action을 비교하는 표현 screen이다.
4. 어느 표현도 올바른 action 우위를 보이지 않으면 low-dimensional conditioner 계열을 중단하고
   Wan/Cosmos의 token conditioner 또는 latent-AR, 이후 spatial control로 이동한다.

## 3. 공통 측정표

낮을수록 좋은 공식 component와 가중합을 함께 쓴다.

| experiment | fold | seed | DINO | Video | Action | `0.3D+0.3V+0.4A` | 정지 대비 | 시간/샘플 | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TODO | | | | | | | | | |

다음 진단은 공식 점수를 대신하지 않지만 failure 원인을 구분하는 데 사용한다.

| experiment | action permutation 반응 | counterfactual 방향 상관 | 첫 전이 discontinuity | NaN/붕괴 frame | 판정 |
|---|---:|---:|---:|---:|---|
| TODO | | | | | |

- **action permutation 반응**: 같은 시작 이미지에서 정상/순열 action의 생성 feature가 거의 같으면 action collapse다.
- **counterfactual 방향 상관**: 두 action의 목표 변화 방향과 extractor가 읽은 변화 방향의 상관을 본다.
  단순히 영상 차이가 크다는 것만으로는 올바른 action following이 아니다.
- **첫 전이 discontinuity**: 고정된 frame 0과 생성 frame 1 사이의 feature/pixel jump를 별도 기록한다.
- seed 하나의 아주 작은 우위는 `INCONCLUSIVE`로 두고, 총점뿐 아니라 세 component의 trade-off를 남긴다.

## 4. 실험 1건 기록 템플릿

아래 블록을 복사해 결과가 나올 때마다 뒤에 붙인다.

```markdown
### EXP-YYYYMMDD-NN — 짧은 이름

- 상태: PLANNED / RUNNING / VALID / INVALID / INCONCLUSIVE
- 가설: 무엇이 왜 좋아질 것으로 예상되는가
- 반증 조건: 어떤 결과면 이 가설을 버릴 것인가
- 비교 기준: baseline experiment ID
- 유일한 의도적 변경점:
- code/config hash:
- checkpoint path/hash/size:
- data manifest path/hash, train overlap:
- action mode / shift / scale source:
- trainable policy / parameters:
- optimizer steps / effective batch / LR / seed:
- GPU / wall time / peak VRAM:
- inference main 또는 EMA / sampler / steps / CFG / eta:
- 216개 환산 추론 시간:

결과:

| fold | seed | DINO | Video | Action | weighted | static 대비 |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

- action permutation / counterfactual:
- 첫 전이 / 붕괴 시점 / 대표 failure:
- 예상과 달랐던 점:
- 판정: PROMOTE / REJECT / INCONCLUSIVE / INVALID
- 다음으로 가장 싼 반증 실험:
```

## 5. 판정 규칙

1. manifest overlap, weight load, action 변환, EMA, 출력 16프레임·6fps·letterbox 중 하나라도 틀리면
   점수가 좋아도 `INVALID`다.
2. 승격은 공식 가중 train-only 점수 개선, 올바른 action counterfactual 반응, 추론 1시간 제한을 모두 요구한다.
3. DINO/Video와 Action이 trade-off면 `0.3/0.3/0.4`로 판단하되 fold/seed 변동보다 작은 차이는 보류한다.
4. diffusion val loss만 좋아진 후보는 승격하지 않는다.
5. Public/Private 제출 결과는 학습 로그와 분리한다. 제출 전 이미 동결한 후보의 확인 기록으로만 남기고,
   eval 입력 통계나 최근접 데이터 탐색을 다음 학습 결정에 되먹이지 않는다.
6. 결과가 가설을 반박하면 incumbent를 보호하지 않는다. 백본·action 표현·conditioner·동결 정책을 모두 교체 가능하다.

## EXP-20260721-03 — Wan2.2-TI2V-5B temporal action AdaLN-Zero

- 상태: PLANNED / UNIT-VALID
- 가설: video-native TI2V prior가 DynamiCrafter의 후반 붕괴를 줄이고, future latent에 직접 넣은 action AdaLN이
  action following을 만든다.
- 반증 조건: smoke OOM/17-frame 불일치, 216개 환산 60분 초과, 또는 1k에서 normal action이
  zero/batch-roll보다 Action metric상 유리하지 않음
- 비교 기준: static, additive 10k, EXP-20260721-02 Frame-Ada 2k
- 유일한 의도적 변경점: Wan2.2-TI2V-5B frozen backbone + LoRA r32 + shared temporal action AdaLN-Zero
- upstream: DiffSynth-Studio `fb337fbb90945ff829de69dbd44ded618f73e889`
- temporal alignment: 17 RGB = clean latent 1 + future latent 4; action 16 = ordered 4x4 pooling
- 출력: decode 17, 제출/점수는 frame 0..15
- action mode / shift / scale source: delta / 0 / `train/delta_action_stats.json`
- trainable policy: `q,k,v,o,ffn.0,ffn.2` LoRA rank 32 + action conditioner
- 예정 optimizer steps / effective batch / LR: 1-step feasibility 후 1k / GPU 수 / 1e-4
- 예정 inference: 20 steps, CFG 1, 320x512, 17 decode frames
- 실행 파일: `train/run_wan_action.sh`, `tools/run_wan_action_gate.sh`

구현 검증:

- 16 action -> 4 future latent -> spatial token expansion shape 검증 통과
- clean latent action modulation exact zero 통과
- zero-init에서 tiny Wan의 action 없음/있음 출력 bitwise 동일 (`max diff=0`)
- backward에서 zero-init output projection gradient nonzero (`sum(abs(grad))=6.2565`)
- 실제 train dataset sample: 17 PIL frames `(512,320)`, actions `(16,6)`, train dataset 121개/10,697 clips
- Python compile 및 shell syntax 통과
- 실제 GPU smoke: **통과** (2026-07-21, H100, `INFERENCE_STEPS=2`, 1 sample)
  - model/T5/VAE load 정상, 누락된 image encoder 등은 TI2V-5B에서 불필요한 optional module
  - VAE encode/decode 및 17 frame 생성 정상, 제출용 frame 0..15의 16 frame MP4 저장 정상
  - 생성 구간 `1.3657 s/sample`, peak allocated VRAM `23.0929 GiB`
  - 고정 holdout 24개 단순 환산 `32.78 s`; 이는 2-step이며 모델 로드 시간과 공식 eval 216개 환산은 별도 측정 필요
  - 결과: `results/wan_smoke_benchmark.json`, 영상: `diagnostics/wan_smoke/val_000000.mp4`

판정: **SMOKE PASS / INCONCLUSIVE**. 다음은 1 training-step의 forward/backward, peak VRAM,
step time, LoRA+action checkpoint 키를 확인하며 이를 통과하기 전에는 1k를 시작하지 않는다.

1k 실행 결과 (2026-07-21, 8x H100):

- optimizer step `1000`, world size `8`, accumulation `1`
- wall time `943.61 s` (`15m 43.6s`), `0.9436 s/step`
- rank0 peak allocated VRAM `22.6726 GiB`
- checkpoint `step-{250,500,750,1000}.safetensors`, 각 약 `174 MiB`
- step-1000 감사: 총 610 tensors = action conditioner 10 + LoRA 600(A 300/B 300)
- non-finite tensor `0`; zero-init output projection은 abs mean `5.81e-4`, max `6.74e-3`로 학습됨
- checkpoint path: `open/baseline/outputs/wan_action_1k/step-1000.safetensors`

판정: **TRAIN PASS / 생성 품질 INCONCLUSIVE**. 24개 normal/zero/batch-roll 공식 component gate로 이동한다.

1k holdout gate 결과:

| variant | DINO | Video | Action | weighted |
|---|---:|---:|---:|---:|
| static | 0.077689 | 0.071936 | 1.276147 | 0.555346 |
| Wan normal | 0.189824 | 0.053631 | 1.271536 | 0.581651 |
| Wan zero action | 0.157179 | 0.057546 | 1.271585 | 0.573052 |
| Wan batch-roll | 0.191448 | 0.054582 | 1.267498 | 0.580808 |
| additive 10k incumbent | 0.188623 | 0.053763 | 1.281031 | 0.585128 |

- normal−static weighted `+0.026304`: static보다 나쁨
- normal−incumbent weighted `-0.003478`, CI `[-0.017897,+0.011466]`: 평균은 조금 좋지만 불확실
- normal−zero Action `-0.000049`, CI `[-0.016661,+0.015011]`: 올바른 action 이점 없음
- normal−batch-roll Action `+0.004038`, CI `[-0.012584,+0.018569]`: 평균상 wrong action이 더 좋음
- normal−zero DINO `+0.032645`, CI `[+0.014331,+0.052296]`, normal 승리 4/24:
  action 신호가 외형을 유의하게 훼손
- normal−zero Video `-0.003915`, CI `[-0.008078,-0.000194]`:
  action이 시간적 motion prior에는 영향을 주지만 올바른 robot action으로 연결되지는 않음
- normal−zero weighted `+0.008599`, CI `[+0.001341,+0.016332]`, normal 승리 5/24
- 여섯 holdout dataset 모두 normal weighted가 static보다 나쁨

최종 판정: **REJECT**. 현재 Wan AdaLN 설정을 3k/5k로 단순 연장하지 않는다. 학습량 부족이라면
action 반응 자체가 약해야 하지만, 실제로는 DINO/Video가 유의하게 변하면서 Action correctness만 없다.
다음 Wan 후보는 (1) 4-frame 평균 pooling을 learned stride-4 temporal encoder로 바꿔 그룹 내부 순서를
보존하고, (2) correct-vs-wrong action denoising ranking을 보조목적으로 추가하는 경우에만 1k 재시험한다.
이 변경도 correctness gate를 통과하지 못하면 AdaLN 계열을 끝내고 action-token cross-attention 또는
latent-AR/spatial control로 이동한다.

체크포인트 추세 감사 (동일 24 holdout, 20 inference steps):

| step | normal weighted | normal−static | Action: normal−zero | Action: normal−batch-roll |
|---:|---:|---:|---:|---:|
| 250 | 0.583663 | +0.028316 | +0.011419 | +0.033951 `[+0.003948,+0.072028]` |
| 500 | 0.587972 | +0.032626 | -0.007281 | +0.004257 |
| 1000 | 0.581651 | +0.026304 | -0.000049 | +0.004038 |

250-step에서는 batch-roll이 정상 action보다 Action metric상 유의하게 좋았다. 500-step에서 초기의 잘못된
민감도는 줄었지만 두 correctness CI가 모두 0을 포함했고, 1000-step에서도 정상 action 우위로 발전하지
않았다. weighted 역시 세 checkpoint 모두 static보다 `+0.026~+0.033` 나빴다. 이는 점진적인 action
학습 곡선보다 **초기 잘못된 민감도가 사라진 뒤 action을 무시하는 수렴**에 가깝다. 따라서 현 설정의
5k 연장은 하지 않으며, 위 conditioner/objective 변경 뒤 새 1k gate를 수행한다.

## EXP-20260721-04 — Wan Action v2 hybrid-state temporal encoder

- 상태: IMPLEMENTED / UNIT-VALID / GPU-UNTESTED
- 수정된 해석: v1 1k는 world size 8에서 약 8,000 clip 노출로 고유 train clip 10,697개보다 적다.
  따라서 correctness 추세 부재만으로 학습량 부족을 완전히 기각한 이전 결론은 강했다.
- 유지점: Wan2.2-TI2V-5B, timestep AdaLN-Zero, LoRA rank 32, 17-frame 학습/16-frame 저장
- 변경점:
  - action `delta` 6D -> `hybrid` 18D = normalized absolute + anchor delta + step velocity
  - value `x` -> `[x, sin(pi*x), cos(pi*x)]` 및 시간 sin/cos 위치 특징
  - Conv1d + adaptive average -> Conv1d(k3) + learned Conv1d(k4,s4)
  - 설명 prompt -> 공식 1X 설정에 가까운 empty prompt
  - 흐림/사전학습 prior 손상을 줄이기 위해 action conditioner LR `1e-4`, LoRA LR `1e-5`로 분리
- 보조 ranking loss는 이번에는 넣지 않는다. AdaLN 방식 자체의 재검증을 위해 representation/encoder/exposure만
  바꾸며, correctness가 다시 실패할 때 별도 실험으로 분리한다.
- 예정 학습: 8x H100, 5k optimizer steps, effective batch 8, 약 40,000 clip exposure(약 3.7 epoch),
  예상 wall time 약 79분. 500-step마다 저장하고 1k/2.5k/5k gate를 비교한다.
- unit 검증: 실제 sample `(17 frames, 16x18 actions)`, finite; `16 -> 4` temporal shape;
  clean modulation exact zero; zero-init 출력 projection gradient nonzero; Python/bash syntax 통과.

판정 규칙: 1k/2.5k에서 normal−zero 및 normal−batch-roll Action이 음수 방향으로 발전하지 않으면 5k를
조기 중단한다. 5k에서도 두 CI 상단이 0 아래가 아니면 hybrid AdaLN을 REJECT하고 action-token
cross-attention 또는 latent-AR로 이동한다. 이 계획도 절대적이지 않으며 실제 gate 결과가 우선한다.

실행 감사 (2026-07-21):

- optimizer step `2000`, world size `8`, accumulation `1`; 정상 완료.
- wall time `1817.02 s` (`30m 17.0s`), `0.9085 s/step`; rank0 peak allocated VRAM `22.6820 GiB`.
- action conditioner LR `1e-4`, LoRA LR `1e-5`가 benchmark에 기록되어 의도한 분리 학습을 확인했다.
- `step-{500,1000,1500,2000}.safetensors`, 각 약 176 MiB.
- step-2000은 총 612 tensors = v2 action conditioner 12 + LoRA 600, non-finite `0`.
- zero-init output projection은 weight abs mean `9.46e-4`, max `1.43e-2`로 실제 학습됨.
- 2000 step x world size 8 = 약 16,000 clip 노출, 고유 train clip 10,697개 대비 약 1.5 epoch.

판정: **TRAIN PASS / GATE PENDING**. step-2000의 24-sample normal/zero/batch-roll gate로 이동한다.

2k holdout gate 결과:

| variant | DINO | Video | Action | weighted |
|---|---:|---:|---:|---:|
| static | 0.077689 | 0.071936 | 1.276147 | 0.555346 |
| v2 normal | 0.149889 | 0.062060 | 1.271424 | 0.572154 |
| v2 zero | 0.121194 | 0.061519 | 1.279283 | 0.566527 |
| v2 batch-roll | 0.146069 | 0.067062 | 1.270647 | 0.572198 |

- normal−static weighted `+0.016808`: static보다 나쁨.
- normal−incumbent weighted `-0.012974`, CI `[-0.025532,-0.000645]`: additive 10k보다 유의하게 개선.
- v1 대비 normal weighted `0.581651 -> 0.572154`, DINO `0.189824 -> 0.149889`: prior 보존/외형은 개선.
- normal−zero Action `-0.007859`, CI `[-0.025866,+0.007384]`: action 존재/크기 효과의 약한 신호지만 불확실.
- normal−batch-roll Action `+0.000777`, CI `[-0.012986,+0.012814]`: action identity/trajectory 구별 없음.

최종 판정: **REJECT FOR EXTENSION**. v2는 현재 learned video 후보 중 개선됐지만 static을 이기지 못하고
가장 중요한 wrong-action correctness를 통과하지 못했다. 같은 AdaLN을 5k로 연장하지 않는다. 다음 실험은
Wan의 visual prior를 가급적 고정한 action-token cross-attention과 correct-vs-rolled paired objective로
조건 주입 및 학습 목적을 함께 바꾼다. 이것도 짧은 gate에서 실패하면 Wan을 중단하고 spatial control 또는
latent-AR로 이동한다.

## 6. 다음 결과를 전달할 때 필요한 최소 정보

사용자가 다음 네 가지를 이 파일에 채우면 후속 분석에서 원인과 다음 실험을 바로 분리할 수 있다.

1. 실행한 experiment ID와 정확한 명령/config
2. checkpoint step, main/EMA 구분, 학습·추론 wall time와 VRAM
3. 고정 holdout의 DINO/Video/Action/weighted 및 정지 기준선
4. 정상 action 대 permutation/counterfactual 결과와 대표 실패 영상의 sample ID

## EXP-20260804-01 — Dream 10k 실제 제출과 평가 기준 교정

- 제출 모델: DynamiCrafter/Dream full-UNet additive action, step 10,000, no-EMA, eval 216 videos
- 실제 leaderboard score: **0.28188** (낮을수록 좋음)
- static 공개 기준 약 `0.517` 대비 절대 `-0.23512`, 상대 약 45.5% 감소
- 목표: `0.1 이하`; 현재 점수에서 추가로 약 64.5% 감소 필요
- 판정: **Dream 10k를 incumbent로 유지**. 실패작으로 폐기하지 않는다.

중요한 교정: train-only local proxy는 같은 Dream을 `0.585128`, static을 `0.555346`으로 계산해 실제
leaderboard와 우열의 부호가 반대였다. local weighted는 실제 점수 추정이나 단독 REJECT 기준으로 사용하지
않고, pixel reconstruction·action counterfactual·실제 제출과 함께 본다.

다음 실행은 미래 GT flow를 쓰는 8-sample oracle dense preservation gate다. 이는 제출 후보가 아니라
VAP/MVA 계열 dense visual control을 구현할 가치가 있는지 확인하는 낙관적 진단이다.

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_oracle_dense_gate.sh
```

### Oracle dense gate 결과

- 판정: `STOP_DENSE_WARP_PATH`
- pixel: full L1 감소 중앙값 `0.00472`, motion L1 감소 중앙값 `0.07368`, cycle confidence `0.88302`
- local weighted: oracle `0.601028`, static `0.586220`, Dream-10k `0.642947`
- oracle−Dream weighted CI95 `[-0.060029,-0.022551]`: local Dream보다는 유의하게 좋음
- oracle−static weighted CI95 `[-0.001458,+0.034297]`: static 우위 없음
- oracle−static Action `-0.006703`, CI95 `[-0.028637,+0.009622]`: 불확실

대표 실패는 `holdout_0002`, `holdout_0003`, `holdout_0004`, `holdout_0006`이다. 첫 프레임에 없는 robot
surface와 disocclusion을 source-only warp가 만들 수 없어 팔 중복·반투명 구멍이 생긴다. pure warp와
retrieval-flow renderer는 종료하지만, rendered future robot을 조건으로 새 pixel을 채우는 VAP/MVA까지
반증된 것은 아니다.

다음 후보는 (1) 실제 제출 근거가 있는 Dream step-8000, (2) Dream motion 영역 밖을 첫 프레임으로 고정하는
후처리 screen이다. action→visual-prompt 정확도를 먼저 증명하지 않은 VAP/MVA ControlNet 장기 학습은 보류한다.

### Dream 보존 후처리 및 step-8k screen

- global blend 0.85: weighted `0.581822`, raw 10k 대비 `-0.003306`, CI95 `[-0.009846,+0.003363]`
- mask 0.06: DINO `-0.018186`로 유의 개선, Video `+0.006170`로 유의 악화
- mask 0.04 + blend 0.85: weighted `0.583380`, 개선 불확실
- raw 8k: DINO `0.153839`, Video `0.063459`, Action `1.294889`, weighted `0.583145`
- 8k−10k: DINO `-0.034784` 유의 개선, Video `+0.009696` 유의 악화, weighted `-0.001983` 불확실

판정: 후처리는 제출 보류. 다음 leaderboard probe는 raw Dream 8k로 정한다. 10k 실제 점수 `0.28188`은
incumbent로 유지하고 8k가 실제로 개선되는지 확인한 뒤 다음 학습 방향을 결정한다.

## EXP-20260804-02 — SO100 rendered prompt / VAP 정렬 가능성 검증

- 목적: MVA/VAP를 GPU로 학습하기 전에 action으로 만든 미래 robot render가 실제 영상에 정렬 가능한지 확인
- GPU 사용: 없음(CPU gate만 수행)
- geometry: ManiSkill corrected SO100 URDF/mesh
- action 변환: `q(t)=q0+scale*(a(t)-a(0))`; camera와 `q0`는 train-only RAFT track으로 fit
- Gate A: normal이 reverse `7/7`, batch-roll `6/7`에서 reprojection 우위였으나 zero는 `4/7`에 불과
- Gate A 실패 원인: sparse moving track 일부에는 맞지만 전체 rendered skeleton과 robot silhouette 불일치
- Gate B: 공식 MatchAnything ELoFTR와 foreground-boundary/target-edge 검증 사용
- Gate B 결과: strict alignment `2/7`, pass rate `28.6%`
- 최종 판정: **STOP_BEFORE_MVA / REJECT_MATCHANYTHING_ALIGNMENT**

관련 산출물:

- `results/so100_track_alignment_gate.json`
- `results/matchanything_alignment_gate.json`
- `diagnostics/so100_alignment/`
- `diagnostics/matchanything_alignment/`
- `train/so100_renderer.py`
- `tools/render_so100_prompt.py`
- `tools/fit_so100_track_alignment.py`
- `tools/matchanything_alignment_gate.py`

해석: action delta는 시간 방향/trajectory 신호를 포함하지만, 데이터에 per-camera extrinsic과 절대 joint
calibration이 없어 visual prompt의 절대 위치를 안정적으로 복원하지 못한다. 이는 MVA/VAP 논문의 calibrated
setting과 현재 challenge 사이의 실제 차이다. strict alignment 80%와 deployable inference 조건을 달성하기
전에는 MVA GPU 학습을 재개하지 않는다. Dream 10k `0.28188`을 incumbent로 유지하고 raw 8k 실제 제출
probe를 다음 비교점으로 사용한다.

## EXP-20260804-03 — IRASim faithful rebuild, pre-GPU audit

기존 `irasim_action_500` 포팅을 공식 RT-1 Frame-Ada 코드와 다시 대조한 결과, 품질 판정을 흐리는 세 구현
차이를 발견했다.

1. 공식 checkpoint는 `16 frames + 15 actions`인데 기존 포팅은 17프레임을 생성한 뒤 source를 버리고
   미래 16프레임을 저장했다. challenge 입력 PNG는 GT frame 0과 pixel-exact 동일하므로 올바른 출력은
   `source frame 0 + generated frames 1..15`다.
2. 공식 7D action MLP를 6D로 직접 바꿔 `embed_state.fc1.weight` 공개 가중치를 버렸다.
3. 공식 학습/평가는 decay 0.9999 EMA를 사용하지만 기존 포팅은 main weight만 저장·추론했다.

수정본은 공식 16-frame RT-1 backbone과 7D action MLP를 그대로 유지하고, 별도의 zero-initialized
`Linear(6,7,bias=False)`만 SO-100 전용 새 파라미터로 둔다. 기본 action 표현은 공식
`accumulate_action=False`에 맞춘 `delta_step`이며, 16개 command를 먼저 변환한 뒤 frame 1..15에 해당하는
15개를 사용한다. 전처리도 공식과 같이 256×320 direct resize로 맞췄다. full model constant LR `1e-4`,
gradient accumulation 2, clip norm 0.1, action dropout 0.1, PNDM 50-step, EMA 0.9999를 보존한다.

CPU audit 결과:

- 공개 EMA checkpoint tensor `297/297` shape-exact load
- temporal embedding `(1,16,1152)`, public action input `(4608,7)` 보존
- 새 adapter `(7,6)`, init abs max `0.0`, 첫 backward grad abs sum `0.106300354`
- train sample `(16,3,256,320)` + action `(15,6)`
- eval raw action `(16,6)` -> model action `(15,6)`
- input PNG == GT frame 0: exact `true`
- official direct-resize numerical equality: exact `true`
- 결과: `results/irasim_faithful_audit.json`, verdict **PASS**

따라서 기존 500-step 결과는 IRASim 방법 전체의 최종 반증으로 사용하지 않는다. 그렇다고 새 구현의 품질이
확정된 것도 아니다. GPU 순서는 VAE-only reconstruction gate → public RT-1 zero-action native-prior 50-step
8-sample → faithful SO-100 short training 순이며, 앞 단계 실패 시 다음 단계로 넘어가지 않는다.

추가 감사에서 기존 dataset wrapper의 `pad=False`가 direct resize가 아니라 center crop을 선택한다는 것을
발견했다. 이를 `pad=None` direct-resize 모드로 분리하고, training dataset·challenge inference·VAE audit가
동일한 antialiased interpolation helper를 호출하도록 수정했다. 최초 VAE 결과(PSNR median 33.91,
gradient-energy ratio 0.930)는 decoder ceiling이 충분하다는 강한 증거지만 수정 전 PIL/direct 경로의
결과이므로, 수정 후 공식 posterior-sample gate를 최종 기록으로 한 번 재실행한다.

수정 후 최종 VAE gate (`limit=8`)도 **PASS_VAE**였다. 공식 posterior `sample()` 기준 median PSNR은
`33.9813 dB`, median MAE는 `2.1193/255`, median gradient-energy ratio는 `0.9247`이다. posterior
`mode()` median도 `33.9807 dB`로 동일했다. 따라서 기존 IRASim 출력의 blur/collapse를 SDXL VAE
codec 탓으로 설명할 수 없으며, 다음 진단 범위는 공개 RT-1 denoising prior와 SO-100 transfer다.

Public RT-1 native-prior 50-step 8-sample 영상은 약 절반이 source에 거의 정지했고, 나머지 절반은 몇
프레임 안에 robot/background 형상을 잃었다. 자연스러운 motion과 외형 보존을 함께 만족한 샘플은 거의
없었다. 이는 VAE가 아니라 RT-1→SO-100 embodiment/domain mismatch이며, zero-shot gate 자체는 FAIL이다.
다만 faithful 500-step transfer 비용이 작으므로 방법 전체를 zero-shot만으로 폐기하지 않고 short screen을
실행했다.

4-GPU faithful training은 500 optimizer step을 완료했다. wall time `480.46 s`, `0.9609 s/step`, rank0
peak `29.72 GiB`였고 model/EMA 모두 298 tensor를 저장했다. 500-step adapter abs mean은 main
`0.0038754`, EMA `0.0000948`이다. 공식 decay 0.9999 EMA는 짧은 transfer에서 main 변화의 약 2.4%만
반영하므로 main/EMA를 분리하지 않으면 under-training으로 오판할 수 있다. 250-main, 500-main,
500-EMA를 동일 seed로 먼저 육안 비교하고 승자 branch에서만 counterfactual gate를 수행한다.

500-step **main** branch를 50-step PNDM으로 8개 생성해 직접 확인한 결과, 8개 모두 frame 1 이후
robot silhouette가 작은 patch/조각 형태로 분해되고 배경까지 함께 재합성됐다. 자연스러운 motion이나
source appearance 보존으로 볼 수 있는 샘플은 없었다. 이는 EMA lag가 만든 가짜 실패가 아니며, main은
public zero-shot의 “일부 정지”보다 오히려 더 불안정했다. 공식 full-model LR `1e-4`를 짧은 SO-100
transfer에 적용하면서 RT-1 prior를 빠르게 훼손한 양상과 일치하지만, 더 낮은 LR가 회복을 보장하지는 않는다.

결론: **IRASim faithful transfer REJECT / TERMINATED.** VAE, checkpoint shape, horizon, direct resize,
source-frame 정합은 모두 통과했으므로 구현 경계 문제가 아니라 RT-1→SO-100 visual-domain transfer와
짧은 full fine-tuning의 실패로 판정한다. 250/EMA, zero/batch-roll, 2k 연장은 실행하지 않는다. 다음 후보는
HMA의 공개 tokenizer/reconstruction 및 native dynamics prior를 학습 전에 먼저 검증한다.

## EXP-20260804-04 — architecture pivot decision (no GPU)

누적 결과를 방법 단위가 아니라 failure mode 단위로 재검토했다. Dream은 실제 제출 `0.28188`로 현재
incumbent지만 robot morphology가 불안정하다. Wan/Cosmos/oracle control은 action sensitivity 또는 외형
보존을 함께 만족하지 못했고, faithful IRASim은 VAE PSNR 33.98 dB에도 denoising prior 단계에서 붕괴했다.
따라서 codec 품질이나 단순 학습 step 부족만으로 공통 현상을 설명할 수 없다. full-frame repaint와
cross-embodiment prior mismatch가 더 직접적인 공통 원인이다.

결정은 HMA 장기학습 직행이 아니라 `source-anchored motion renderer`의 단계적 검증이다. HMA discrete는
tokenizer reconstruction/crop gate를 위한 challenger로 유지한다. 새 주력 후보는 FOMM/TPS 계열의 공식
occlusion-aware source-feature warping을 fidelity 있게 이식하고, driving motion을 action-conditioned
keypoint/part predictor로 바꾼다. dense warp 밖의 disoccluded 영역만 residual decoder가 생성하며 정지
배경은 source-copy 경로를 보존한다.

이 시점에는 GPU 실행이나 성능 주장을 하지 않았다. 다음 구현의 첫 성공 기준은 full training loss가 아니라
GT-derived motion을 사용한 renderer oracle 복원이다. oracle 실패 시 즉시 중단하고, 통과할 때만 single-clip
action overfit으로 넘어간다. 이 순서로 renderer 실패와 action mapping 실패를 다시 섞지 않는다.

공식 MIT FOMM 코드를 `third_party/first-order-model`에 고정해 핵심 모듈을 CPU에서 구조 감사했다. challenge
해상도 160×256 입력에 KPDetector와 OcclusionAwareGenerator forward가 정상 동작했고, 출력은 dense component
mask `(B, K+1, 40, 64)`, occlusion map `(B,1,40,64)`, deformed source 및 prediction `(B,3,160,256)`로
확인됐다. 아직 이 결과는 품질 PASS가 아니라 upstream 구조·현재 PyTorch 호환성만 확인한 것이다.

SO-100 oracle adapter를 `train/source_anchored_fomm.py`, `train/train_fomm_oracle.py`,
`tools/eval_fomm_oracle.py`로 구현했다. target frame은 target keypoint를 얻는 detector에만 들어가고 generator는
source RGB, source keypoint, target keypoint만 받는다. 공식 dense-motion component 0을 identity/background로
사용하고 나머지 component 및 occlusion으로 edit mask를 만들어 최종 RGB에 명시적 source-copy 경로를
추가했다. identity pair도 같은 DDP forward에 함께 넣어 buffer version 문제를 피한다.

공식 base capacity는 59,790,759 params다. 5-block KP hourglass와 0.25 pre-scale의 skip 정합을 보장하려면
입력이 128의 배수여야 하므로 기본 해상도를 crop 없는 256×384 letterbox로 정했다. 작은 CPU smoke는
tiny 1,966,423 params, 160×256에서 1 optimizer step, checkpoint reload, 16-frame mp4/metric 저장까지
PASS했다. 1-step metric의 `REJECT_RENDERER_ORACLE`은 예상된 값이며 품질 증거로 사용하지 않는다. 실제 gate는
base overfit 후 median PSNR >=25 dB, static 대비 median +3 dB, background MAE <=0.02,
gradient-energy ratio 0.70..1.35를 모두 요구한다.

Single RTX PRO 6000에서 dataset index 0 (`00ri/so100_battery`, start 0)을 1000-step overfit한 결과
median PSNR `38.2188 dB`, static 대비 `+8.9286 dB`, background MAE `0.00908`, gradient-energy ratio
`0.9367`, mean edit fraction `0.2049`로 수치 gate는 **PASS_RENDERER_ORACLE**였다. prediction/target/static
contact sheet를 직접 확인하니 blur나 배경 재합성은 없었지만, robot이 화면 위에 일부만 보이고 16 frame의
움직임도 작았다. 따라서 이는 renderer 구현/선명도 PASS이지 큰 robot motion의 충분한 증거로 과대해석하지
않는다.

32-episode renderer 2000-step 후 unseen 8-episode oracle는 median PSNR `22.0702 dB`, static 대비
`+3.1409 dB`, background MAE `0.00196`, global gradient ratio `0.7027`로
**REJECT_MULTICLIP_RENDERER_ORACLE**였다. 네 저장 clip을 직접 비교하면 background가 metric을 높인 반면,
움직이는 robot은 검은 평균 형상으로 심하게 뭉개졌다. 따라서 single-clip PASS/action-overfit을 일반화로
해석하지 않으며 multi-clip action predictor로 넘어가지 않는다.

원인은 official FOMM architecture를 사용했지만 학습 loss에서 공식 multi-scale VGG perceptual 항을 생략하고
pixel/gradient 중심으로 최적화해 pose 다양성에서 decoder 평균화가 발생한 것으로 좁혔다. cached ImageNet
VGG19를 사용해 GT-motion mask 안의 robot/object feature에만 4-scale perceptual loss를 적용하는 500-step
bounded fine-tune을 추가했다. 평가는 background가 가리는 기존 global edge ratio 외에 motion-region gradient
ratio를 필수 gate로 추가한다. 이 마지막 수정도 실패하면 FOMM 경로를 종료한다.

`tools/rank_fomm_oracle_clips.py`로 127개 dataset의 deterministic clip 0을 CPU 전수 스캔하고 static PSNR,
pixel change, action travel을 기록했다. 상위 후보를 직접 확인한 뒤 dataset index 101
(`roboticshack/team9-pick_cube_place_static_plate`, start 28)을 hard oracle로 선정했다. 이 clip은 고정 camera,
화면 중앙의 큰 robot, t=0..15의 큰 관절 이동과 cube interaction을 함께 포함한다. 이 hard oracle이 같은
품질 gate를 통과하기 전에는 action-to-keypoint predictor를 구현하지 않는다.

Hard oracle 1000-step 결과는 median PSNR `26.5117 dB`, static 대비 median `+9.9925 dB`, background MAE
`0.00167`, gradient-energy ratio `0.7012`, mean edit fraction `0.3282`로 **PASS_RENDERER_ORACLE**였다.
프레임별/육안 비교에서 robot, gripper, cube의 위치와 큰 궤적은 target을 따랐고 background는 안정적이었지만,
t=3..11 robot 경계의 gradient ratio는 약 `0.66..0.71`로 내려가 국소 texture/윤곽이 뭉개졌다. 이 단계는
target frame에서 얻은 GT-motion keypoint를 사용하므로 action conditioning 성공으로 해석하지 않는다.

기존 checkpoint의 official pixel-space `deformed` 결과를 occlusion visibility로 직접 섞는 무학습 sharpen
ablation도 수행했다. aggregate gradient ratio는 `0.7012→0.8433`으로 올랐지만 median PSNR은
`26.5117→18.2247`, static gain은 `+9.9925→+2.4389`로 붕괴해 **REJECT**했다. sharp warp의 작은 위치
오차가 이중 윤곽을 만들기 때문이다. 이 합성 변경은 원복했다. blur 개선은 direct warp가 아니라 official
FOMM의 perceptual/multi-scale 학습 loss 또는 이후 TPS/multi-resolution occlusion으로 다룬다.

Frozen hard renderer에 2.48M causal action-to-keypoint predictor를 500 step overfit했다. normal median PSNR
`25.9259 dB`, static 대비 `+9.0282 dB`, zero 대비 `+7.9356 dB`, reverse 대비 `+8.9103 dB`, background
MAE `0.00176`으로 **PASS_ACTION_KP_OVERFIT**였다. contact sheet에서도 normal만 target의 robot/cube 궤적을
재현하고 first-action-hold는 거의 정지했으며 reverse/roll4는 잘못된 위치로 이동했다. action→motion 연결이
한 clip에서 실제 동작한다는 증거지만, predictor가 동일 trajectory와 명시적 counterfactual ranking을 외울 수
있으므로 일반화 증거는 아니다.

다음 저비용 gate로 같은 고정 camera/domain의 40 episode 중 32개로 renderer를 이어 학습하고 나머지 8개를
GT-keypoint oracle holdout으로 평가하는 multi-clip 경로를 추가했다. single-clip renderer가 1000 step에
41.6초였으므로 정보 대비 비용이 작다. held renderer가 통과하기 전에는 full 127-domain이나 제출 생성을 하지
않는다.

## EXP-20260804-05 — DreamZero-SO101 asset/config audit (no GPU)

`Wan-AI/Wan2.1-I2V-14B-480P`의 7 DiT shard, T5, CLIP, Wan2.1 VAE 총 77GB와
`Vizuara/dreamzero-so101-lora` 217MB를 로컬에서 size/key audit했다. LoRA의 40-block PEFT key와
action/state encoder/decoder key는 현재 DreamZero `CausalWanModel`과 일치했다. Python 3.11 전용 venv에
PyTorch 2.8.0+cu128, transformers 4.51.3, PEFT 0.5.0, DeepSpeed 0.16.7을 설치했고 CPU import 및 tokenizer
smoke를 통과했다. 이 단계에서 GPU inference는 실행하지 않았다.

LoRA 저장소에 빠진 metadata 때문에 임의 mean/std를 사용하지 않고 raw checkpoint 저장소를 전수 조회했다.
`lora-100k/checkpoint-20000/experiment_cfg/{conf.yaml,metadata.json}`을 찾아 내려받았고 SO-101 3-view
layout, q01/q99 normalization, relative action, action horizon 24, fps 30, embodiment mapping을 확인했다.
Challenge 첫 action을 SO-101 absolute state로 직접 정규화하는 audit에서는 첫 8개 sample의 dimension
saturation fraction이 0.167--0.833이었다. 따라서 native prior gate 기본 state는 metadata center이고,
`action0-q99`는 진단 옵션으로만 남겼다.

native smoke generator는 두 causal chunk를 이어 5 latent/17 RGB frame을 만들고 front view 16 frame과 원래
multi-view composite 16 frame을 함께 저장한다. future challenge action은 아직 사용하지 않으며 JSON에도 이를
명시한다. 또한 upstream loader가 F32 DiT shard 7개를 RAM에 전부 누적하던 부분을 shard별 즉시 load/release로
바꿔, 동일 weight를 유지하면서 약 66GiB의 불필요한 host-memory peak를 제거했다. 다음 실행은 H100 단일 GPU
`limit=1`, 4 full DiT steps이며 결과가 식별 가능한 robot motion을 보일 때만 given-action inpainting으로 간다.

## EXP-20260805-01 — Wan2.1 14B native prior PASS, spatial-action implementation

DreamZero-SO101 native/action stress 모두 challenge single-view에서 실질적으로 정지했고 frame 1 이후 blur와
경계 노이즈가 누적됐다. 이를 Wan2.1 자체의 실패와 분리하기 위해 동일 base 14B를 LoRA/action/state 없이
`val_000009`에 50-step 생성했다. 결과는 robot이 charger 방향으로 명확히 이동하고 source morphology도
식별 가능했다. runtime 47.16초, peak 43.57 GiB, last sharpness ratio 0.687이었다. scene-wide zoom과 edit
fraction 0.925는 남았으므로 submission 후보 PASS가 아니라 **native motion/appearance prior PASS**로만 기록한다.
결과는 `diagnostics/wan21_vanilla_motion/vanilla_benchmark.json`에 있다.

이 증거에 따라 Wan2.1-I2V-14B를 frozen visual prior로 두고 source-conditioned spatial action branch와 rank-16
LoRA를 추가했다. 이전 Wan2.2 5B의 global AdaLN과 sparse track branch가 액션을 무시했던 점을 반영해,
16-step hybrid action을 4개 미래 latent에 정렬하고 첫 프레임 VAE feature에 FiLM한 spatial residual을 40개
block 중 4개 depth에 주입한다. exact-zero output projection, 15% static/constant-action augmentation,
correct-vs-reverse paired noise/timestep ranking, motion-weighted reconstruction을 함께 사용한다. 정렬 실패한
URDF visual prompt는 사용하지 않았다.

GPU를 사용하지 않은 `PHASE=audit` 결과는 **PASS_STRUCTURE**였다.

- exact-zero init: true
- output projection gradient L1: `2.12186`
- warmup 후 action/source gradient L1: `8.30e-8` / `1.87e-6`
- reverse-action residual sensitivity: `2.12e-5`
- paired loss: finite, action gradient L1 `0.59334`
- dataset: 121 groups, 10,697 clips, 17 frames, 16x18 action, static exact repeat
- local Wan components/tokenizer: complete

아직 GPU smoke/training/generation 품질은 실행하지 않았다. 다음 순서는 8-H100 5-step smoke → 250-step →
8-sample normal/constant/batch-roll gate다. normal action sensitivity, static 우위, 육안 morphology가 함께 보일
때만 500-step으로 연장한다. 이 후보 역시 절대적인 계획이 아니며, 250-step 실패 시 Wan이라는 이유만으로
추가 step을 투입하지 않는다.
