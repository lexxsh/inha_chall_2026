# 최신 SoTA 추가 조사 (2026-07 기준)

`RESEARCH.md`(1차 조사)의 갱신본. 2025년 말~2026년에 나온 것들 위주로 확인했다.
**라이선스는 대회 규정(공개 가중치 + 상업/비상업 허용, API 전용 불가) 통과 여부가 핵심이므로 우선 확인했다.**

> **주의:** 이 문서는 정답 레시피가 아니라 현재 증거로 만든 후보 지도다. 논문 결과는 서로 다른 데이터·평가·
> 계산량에서 나온 것이며 이 대회의 DINO/Video/Action 가중 점수를 보장하지 않는다. 아래 권고는 자체 ablation과
> 시간 제한을 통과하면 채택하고, 반례가 나오면 폐기한다. 특히 “delta”, “spatial control”, “큰 백본”도 단일 정답이 아니다.

## 1. ★ Wan 2.5/2.6/2.7 서비스는 있어도 공개 로컬 가중치가 없다

| 모델 | 공개 시점 | 가중치 | 라이선스 | 대회 사용 |
|---|---|---|---|---|
| **Wan2.2-TI2V-5B** | 2025.08 | HF 공개 | **Apache 2.0** | 가능, challenger 후보 |
| Wan2.2-I2V-A14B | 2025.08 | HF 공개 | Apache 2.0 | 가능 (14B, 무거움) |
| Wan-Dancer-14B | 2026.07 | HF 공개 | Apache 2.0 | I2V이나 댄스/인체 특화로 보임, 부적합 가능성 |
| Wan 2.5 | 2025 | Alibaba Cloud API, 공개 로컬 weight 미발견 | — | **API 사용 불가** |
| Wan 2.6 | 2025 | Alibaba Cloud API, 공개 로컬 weight 미발견 | — | **API 사용 불가** |
| Wan 2.7 | 2026 | Alibaba Cloud API, 공개 로컬 weight 미발견 | — | **API 사용 불가** |
| **LTX-2** | 2026.01 | 공개(코드+가중치+학습코드) | 자체 라이선스: 학술 무료, 상업은 ARR $10M 미만 무료 | 가능하나 라이선스 문구 확인 권장 |
| **Cosmos-Predict2.5 (2B)** | 2025~2026 | HF 공개 | NVIDIA Open Model License (상업 이용·파생 허용) | 가능 |
| **Cosmos3-Nano (16B)** | 2026.05 | HF·코드 공개 | OpenMDW-1.1, 상업/비상업 사용 가능 표기 | 최신이나 계산·action 이식 gate 필요 |
| HunyuanVideo, Open-Sora, CogVideoX 등 | ~2025 | 공개 | 각기 다름 | 개별 확인 필요 |

**핵심**: Alibaba Cloud 공식 문서에는 Wan 2.5·2.6·2.7 서비스가 실제로 존재한다. 그러나 이 대회는
원격 API를 금지하고, 2026-07-20 현재 공식 `Wan-AI`/`Wan-Video`에서 로컬 실행 가능한 2.5~2.7
가중치를 찾지 못했다. 따라서 **사용 가능한 최신 공개 계열은 Wan2.2**다. 이것은 “Wan2.2가 최선”이라는
성능 결론이 아니라 자격 조건을 통과하는 후보라는 뜻이다.

> **구분할 것:** “제품/API가 존재한다”와 “다운로드 가능한 가중치가 공개됐다”는 다른 주장이다.
> 비공식 블로그의 Apache-2.0/open-weight 주장은 공식 HF/GitHub 파일과 license를 직접 확인하기 전에는 채택하지 않는다.

### 1-A. 1X 우승법은 conditioner 참고이지 이 대회의 예산 레시피가 아니다

Team Revontuli의 1X Sampling 1위가 Wan2.2-TI2V-5B + state AdaLN-Zero + LoRA r32인 것은 원문에서
확인했다. 그러나 그대로 옮길 수 없는 차이가 크다.

- 학습은 23k step, **4개 노드의 B200(184GB), 유효 배치 1024**였고 이 대회는 96GB 단일 GPU다.
- 1X는 과거 5프레임과 robot angle/velocity를 조건으로 미래를 예측했지만 우리는 이미지 1장+6D action이다.
- 1X Sampling은 마지막 프레임 PSNR이 핵심이고 추론 제한이 없었다. 20-sample 픽셀 평균은 PSNR에는
  유리했지만 LPIPS/FID를 악화시켰고, 이 대회의 DINO/Video/Action 및 1시간 제한과 맞지 않는다.

따라서 이 경로의 첫 질문은 “논문에서 우승했는가”가 아니라 **단일 RTX PRO 6000에서 LoRA 1 step의
메모리/시간, 216개 단일-sample 추론, train-only 공식 가중 점수**다. 통과 전에는 4일 내 가능하다고 쓰지 않는다.

- 원문: https://arxiv.org/html/2510.07092v1

## 2. ★ Cosmos-Predict2.5 — 최신은 아니지만 2B 액션 컨디셔닝 레시피가 있다

NVIDIA가 **action-conditioned video prediction post-training 공식 문서와 distillation 가이드(2026.02)** 를 제공한다.
- 모델: Cosmos-Predict2-2B-Video2World (2B)
- 액션 포맷: 7차원 (gripper 좌표계 x,y,z,roll,pitch,yaw 변위 + gripper 개폐 이진값)
- 주입 방식: `conditioner.py`에 `action_conditioned_video_conditioner` 커스텀 컨디셔너 추가
- 데이터: Bridge 형식(mp4 + 프레임별 state/action JSON)
- Cosmos Policy(RoboCasa, Libero 사후학습 모델)도 함께 공개

**우리 과제와의 차이**: Cosmos는 **엔드이펙터 변위(delta)** 를 조건으로 쓰는데, 우리 데이터는 **관절각 절대값**이다.
내 실측(EMPIRICAL.md 4-3)에서도 데이터셋별 offset 차이가 관찰됐다. 다만 EE-local delta를 쓰는 Cosmos와
joint-space 목표각을 쓰는 이 대회는 action 의미가 다르다. 이는 delta 계열을 **비교할 근거**이지,
현재 anchor delta를 채택해야 한다는 증거는 아니다.
- 문서: https://docs.nvidia.com/cosmos/latest/predict2.5/post-training/video2world_action-conditioned.html
- 코드: https://github.com/nvidia-cosmos/cosmos-predict2.5

### 2-A. Cosmos 3 — 구조적으로 가장 직접적이지만 현재는 고위험 challenger

NVIDIA는 2026-05-31 공개한 **Cosmos 3**를 최신 계열로 명시하고 Predict2.5 저장소는 제한적 유지보수로
전환했다. Cosmos3-Nano는 16B omnimodal MoT 모델이며 `image + action → video` forward dynamics를
공식 지원한다. 문제 형태만 보면 이 대회와 매우 가깝다.

그러나 곧바로 주력으로 바꾸기에는 다음 gate가 남는다.

- 공개 forward-dynamics 예시는 AgiBot 29D, 공식 지원 single-arm domain은 10D 계열이다. 이 대회의
  6D SO-100 joint target과 calibration/시간축을 맞추려면 자체 post-training이 필요하다.
- action post-training recipe는 공식 저장소에서 아직 `Coming Soon`으로 표시된 부분이 있고,
  vLLM-Omni action 경로도 main/review 상태가 섞여 있다. 재현 surface를 먼저 고정해야 한다.
- Nano도 16B BF16이며 공식 serving 권장은 H200, H100은 테스트 대상이다. 96GB 단일 GPU LoRA와
  216개/1시간이 가능한지는 공개 수치로 보장되지 않는다.
- 공식 생성 설정은 10fps 이상이고 action 예시는 조건 프레임+action chunk 구조다. 6fps·16 output frame의
  정확한 horizon을 다시 정의해야 한다.

따라서 **최신성은 Cosmos 3, 현재 실험 용이성은 Predict2.5-2B**로 분리한다. Cosmos 3는 1샘플 로컬
forward-dynamics, VRAM/시간, 6D adapter 최소학습이 모두 통과할 때만 2B challenger보다 앞에 둔다.

- 공식 코드/모델표: https://github.com/NVIDIA/Cosmos , https://huggingface.co/nvidia/Cosmos3-Nano
- 기술 보고서: https://research.nvidia.com/labs/cosmos-lab/cosmos3/

## 3. ★ LeRobot v0.6.0 (2026) — 월드모델이 프레임워크에 들어왔다

| 모델 | 성격 | 특징 |
|---|---|---|
| **FastWAM** | ~5B 비디오 생성 전문가 + 소형 액션 전문가 결합 | `lerobot/fastwam_base`에서 파인튜닝 가능. 추론 시 "dreaming"을 건너뛰고 액션만 디노이즈 |
| **LingBot-VA** | autoregressive 비디오-액션 모델, 청크 단위 예측 | 추론에 24~32GB GPU 1장. 실제 관측을 되먹여 drift 억제 |
| **VLA-JEPA** | Qwen3-VL-2B 기반, latent 공간 미래 예측 | 월드모델이 학습만 감독하고 추론 시엔 사라짐 (추론 비용 0) |

주의: 이들은 **정책(policy) 모델**이라 우리 과제(비디오 생성 자체가 산출물)와 목적이 다르다.
다만 FastWAM의 비디오 전문가 부분과 LingBot-VA의 drift 억제 기법(실관측 되먹임)은 참고 가치가 있다.
라이선스는 블로그에 명시돼 있지 않으므로 사용 전 개별 확인 필요.
- https://huggingface.co/blog/lerobot-release-v060

## 4. ★★ SO-100 캘리브레이션 관례 — 자체 관찰과 정합하는 외부 근거

내가 측정한 "데이터셋마다 관절각 오프셋이 다르다"(EMPIRICAL.md 4-3)는 커뮤니티에서 알려진 문제였다:
- URDF의 0 자세와 **LeRobot의 0 자세 관례가 다르다**
- **구 캘리브레이션 방식**은 팔을 수평으로 완전히 편 자세를 각 관절의 가상 0으로 잡는다 (`so101_old_calib.xml`) — 신 방식과 다름
- LeRobot은 각 관절의 최대/최소를 서보에 저장하는 방식으로 캘리브레이션하는데, **phosphobot 등 다른 소프트웨어는 첫 캘리브레이션 자세를 0으로 두고 min/max를 갱신하지 않는다**
- LeRobot v0.6.0이 "calibration correction이 적용된 체크포인트"를 배포한다는 것 자체가 이 문제의 존재를 증명한다

**현재 가설**: 커뮤니티 SO-100 데이터를 합칠 때 calibration offset을 처리하지 않은 절대 관절각은 위험하다.
상대 표현과 시작 이미지/anchor 조합이 유력하지만, absolute·anchor delta·step delta 비교 전에는 확정하지 않는다.
- https://huggingface.co/docs/lerobot/so101
- https://www.waveshare.com/wiki/SO-ARM100/101_Robotic_Arm_Calibration_and_Remote_Control

## 4-B. ★★ DreamZero-SO101 — 관련은 있지만 방향이 다른 공개 WAM

| 항목 | 내용 | 검증 |
|---|---|---|
| 어댑터 | `Vizuara/dreamzero-so101-lora`, **Apache 2.0** | HF에서 직접 확인 |
| 입·출력 | 시작 이미지+언어 → 미래 6-DoF action 24개와 RGB 33프레임 **공동 예측** | 모델 카드 |
| 베이스/방식 | `Wan-AI/Wan2.1-I2V-14B-480P` + causal Wan/action transformer, joint flow matching | 모델 카드 |
| action | relative joint positions 6-DoF(action 32차원 pad, state 64차원 pad) | 모델 카드 |
| LoRA | rank 4, attention·FFN 대상 + action head, trainable 약 5천만, 파일 207MB | 모델 카드 |
| 데이터 | `whosricky/so101-megamix-v1` (**Apache 2.0**, SO-101, 400 에피소드 / 8 태스크 / 3 카메라 / 179,166 프레임 / 30fps / LeRobot v3.0) | HF에서 직접 확인 |
| 학습 비용 | 72,000 step, **2×H100으로 약 127시간** | 모델 카드 |
| 추론(자기 보고) | 320×176, 33프레임, 4 Euler step, H100 약 600ms | 우리 파이프라인 end-to-end 수치가 아님 |

**중요한 차이와 시사점**
1. 이 모델은 **주어진 action으로 미래 영상을 생성하지 않는다.** 이미지+언어에서 action과 영상을 함께
   예측하므로 이 대회의 conditioner로 바로 이식할 수 없다. 참고할 부분은 relative joint 표현과 joint
   video/action auxiliary head이며, 현재 additive conditioner의 우위를 반박하는 직접 비교도 아니다.
2. **학습 예산 초과 주의**: 2×H100 × 127h = 약 254 GPU-hour인데 대회 제한은 1 GPU × 4일 = 96 GPU-hour다.
   원 schedule은 재현할 수 없고, 더 작은 백본/짧은 schedule로 바꾸면 별도 방법이 된다.
3. 공개 카드는 학습 loss를 보고하지만 이 대회와 같은 action-conditioned video 품질이나 OOD 점수를 주지 않는다.
   600ms 자기 보고도 4-step·저해상도 core inference라 216개 제출 파이프라인의 시간 보장은 아니다.

- 모델 카드: https://huggingface.co/Vizuara/dreamzero-so101-lora
- 베이스 모델: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P

## 4-C. ★★ delta 액션의 문헌적 근거 (직접 검증함)

**Demystifying Action Space Design for Robotic Manipulation Policies** (arXiv 2602.23408)
- 저자: Yuchun Feng, Jinliang Zheng, Zhihao Wang, Dongxiu Liu, Jianxiong Li, Jiangmiao Pang, Tai Wang, Xianyuan Zhan
- **500개 이상 학습 모델, 13,000회 이상 실제 롤아웃**으로 absolute vs delta, joint-space vs task-space를 비교한 대규모 실증 연구
- 초록에서 확인된 결론: **"delta action을 예측하도록 설계하면 성능이 일관되게 향상된다"**

이것은 내 실측(EMPIRICAL.md 4-3)과 **독립적으로 같은 방향**을 가리킨다. 또한 VLA-REPLICA
(arXiv 2605.20774) 3.2절은 SO-101 조립체별 서보 초기 offset이 각 팔을 고유 action space에 묶는 문제와,
offset/min/max calibration을 통한 universal joint-degree action 변환을 직접 설명한다.

다만 정책 출력에서 delta가 좋다는 결과나 서로 다른 실물 팔의 calibration 문제는 이 영상 생성 과제의
anchor delta 우위를 증명하지 않는다. 제공 데이터가 이미 degree로 보정된 정도도 dataset마다 확인되지 않았다.

- Action-space 연구: https://arxiv.org/abs/2602.23408
- VLA-REPLICA 3.2절: https://arxiv.org/html/2605.20774

## 4-D. ★★ MiraBench — 시각 지표와 액션 지표는 따로 논다

**MiraBench** (arXiv 2605.29360) 원문을 직접 확인했다. 12개 모델 구성, 906개 영상,
16,704개 구조화된 사람 판단을 사용해 얻은 핵심 결론은 다음과 같다.
1. **시각 충실도는 액션 충실도의 나쁜 대리지표다**
2. 모델 스케일을 키워도 action following이 안정적으로 좋아지지는 않는다
3. optimism bias — 실패해야 할 액션에도 성공하는 장면을 그린다

우리 평가가 DINO/r3d_18(시각)과 IDM MAE(액션)로 분리돼 있고 내 실측에서도
같은 장면 기준 움직임 오답의 대가가 DINO +0.264 vs Action +0.026으로 스케일이 전혀 달랐다.
→ 최소한 두 목표를 **별도 metric과 failure gate로 측정**해야 한다. 분리 loss, 가중 multi-objective loss,
checkpoint 선택 중 어느 방식이 좋은지는 자체 ablation으로 정한다.
- 원문: https://arxiv.org/html/2605.29360

## 4-E. 기타 참고 (에이전트 보고, 코드 공개 확인된 것 위주)

| 이름 | arXiv | 핵심 | 공개 |
|---|---|---|---|
| Nano World Models | 2605.23993 | diffusion forcing 미니멀 구현, **액션 컨디셔닝 방식을 통일 인터페이스로 비교** | 코드·체크포인트 공개 |
| ABot-PhysWorld | 2603.23376 | 14B DiT, parallel context block으로 spatial action injection | github.com/amap-cvlab/ABot-PhysWorld |
| Cosmos Policy | 2601.16163 | 액션을 diffusion **latent frame**으로 인코딩, 아키텍처 변경 없이 post-training | 공개 예정 |
| EVA | 2603.17808 | IDM을 RL post-training **reward**로 사용 | — |
| LDA-1B | 2602.12215 (RSS 2026) | latent action, 30k+ 시간 | 코드 공개 |
| MolmoAct2 | 2605.02881 | 공개 LeRobot 1,222 데이터셋에서 SO100/101 38,059 에피소드 큐레이션 | 데이터 참고용 |

액션을 **픽셀 공간 신호로 렌더해 주입**하는 계열(MTV-World 2511.12882의 trajectory video,
BridgeV2W 2602.03793의 URDF embodiment mask + ControlNet, IOI 2606.23296의 FK 정사영,
FlowWAM 2607.13017의 optical flow)이 2026년의 뚜렷한 흐름이다.
저차원 벡터를 AdaLN으로 전역 주입하면 픽셀과 정렬되지 않는다는 문제의식이다.
다만 이들 대부분은 **코드/가중치 공개 여부가 확인되지 않았다.**

## 4-F. ★★★ 픽셀 정렬 액션 조건 — OSCAR·iMaC·FlowWAM·JOPAT

최신 연구에서 가장 일관된 비판은 “6차원 숫자를 전역 feature에 더하는 것만으로는 어느 픽셀이 어떻게
움직여야 하는지 알려주기 어렵다”는 점이다. 다만 방법마다 이 대회로 옮기는 비용이 크게 다르다.

| 방법 | 액션 표현 | 공개 상태 | 이 대회에서의 판단 |
|---|---|---|---|
| **OSCAR-2B** (2606.04463) | URDF FK로 만든 2D skeleton video를 별도 latent stream으로 주입 | 코드 Apache-2.0, HF weight Apache-2.0 | **조건부 강한 후보**. 카메라/URDF gate가 먼저 |
| **iMaC** (2606.09813) | motion image + RGB-D/contact image | 논문 확인 | 3-view·depth·camera calibration 의존도가 높아 현재는 고위험 |
| **FlowWAM** (2607.13017) | RGB와 optical-flow video의 dual stream | 논문·project 확인 | 숫자 action→target flow adapter가 필요; 보조 실험 후보 |
| **JOPAT** (2605.23856) | RGB latent + 2D point tracks/visibility 공동 예측 | 논문 확인 | train 영상에서 track label을 만들 수 있어 auxiliary loss 후보 |

**OSCAR**는 Cosmos-Predict2.5-2B를 바탕으로 관절 상태를 URDF forward kinematics와 카메라 내·외부
파라미터로 투영해 skeleton control video를 만든다. 공개 모델은 약 4GB이고 ≥24GB VRAM을 권장하며,
공개 구현은 81프레임·5 sampling step을 Blackwell GPU에서 warmup 후 약 1분으로 보고한다.
정밀한 action following에는 매력적이지만, 로컬 제공 데이터에서 URDF/카메라 calibration 파일을 찾지 못했다.
논문 자체도 calibration이 없는 세션에는 MoGe-v2/CtRNet-X 추정과 수동 overlay 검수를 사용했다.
따라서 **train-only 영상에서 skeleton overlay reprojection error가 충분히 낮아지는지**를 먼저 검사하고,
실패하면 OSCAR 이식에 장기 예산을 쓰지 않는다.

- OSCAR 원문: https://arxiv.org/html/2606.04463
- 코드/가중치: https://github.com/wuzy2115/oscar-public , https://huggingface.co/zywu2115/OSCAR-2B
- iMaC: https://arxiv.org/html/2606.09813
- FlowWAM: https://arxiv.org/abs/2607.13017
- JOPAT: https://arxiv.org/abs/2605.23856

**낮은 위험의 중간안**은 전체 백본 교체가 아니라 현재 모델에 motion auxiliary를 붙이는 것이다.
제공 train 영상에서만 optical flow/point track을 추출해 미래 motion head를 함께 학습하거나,
action permutation에 대한 contrastive sensitivity loss를 둔다. 이것은 eval-derived label 없이 구현 가능하지만,
공식 세 component 개선은 보장되지 않으므로 소규모 실험을 먼저 한다.

## 4-G. 더 볼 만한 벤치마크와 대회

| 벤치/대회 | 무엇을 재는가 | 우리에게 가져올 것 |
|---|---|---|
| **MiraBench** | physics, action following, failure/optimism bias | action perturbation·failure-preservation test |
| **WorldArena / 2.0 / CVPR 2026 Challenge** | 지각 품질과 data engine/policy evaluation/planning의 간극 | DINO가 좋아도 기능적으로 틀릴 수 있다는 진단 틀 |
| **RoboWM-Bench** | 생성 영상을 IDM/retargeting해 simulator에서 실행 | spatial/contact/deformation failure taxonomy |
| **AGIBOT World Challenge 2026 + EWMBench** | image quality, scene consistency, trajectory adherence | 공개 evaluator와 실패 trajectory 중심 검증 |
| **Nano World Models** | objective·scale·action injection·sampling의 통제 비교 | 작은 모델로 ablation 설계를 먼저 검증 |

AGIBOT 대회는 10개 task, 30,000개 이상 실제 trajectory에 missed grasp/collision 같은 imperfect action도
포함하고 공개 baseline/EWMBench를 제공한다. 데이터 자체는 이 대회에서 외부 데이터라 사용할 수 없지만,
평가 코드와 실패 케이스 분류는 참고할 가치가 있다. WorldArena와 RoboWM-Bench도 제출 점수를 대신하는
벤치는 아니며, train-only 진단 항목을 설계하는 데만 쓴다.

- WorldArena: https://arxiv.org/abs/2602.08971 , https://world-arena.ai/ (공식 페이지에서 CVPR 2026 Challenge 연결)
- RoboWM-Bench: https://arxiv.org/html/2604.19092
- AGIBOT World Challenge: https://www.agibot.com/article/231/detail/45.html
- EWMBench: https://github.com/AgibotTech/EWMBench
- Nano World Models: https://arxiv.org/abs/2605.23993

## 5. 액션 조건 비디오 생성 최신 연구 (코드/가중치 공개된 것)

| 이름 | 시기 | 액션 주입 방식 | 공개 |
|---|---|---|---|
| VideoVLA | NeurIPS 2025 | 비디오-액션 결합 디노이징, cross-attention 결합 | 코드 + HF 가중치 |
| Motus | 2025 | multi-stream diffusion, dynamics/action 스트림 hidden-state 결합 | 코드 + HF 체크포인트 |
| UVA | RSS 2025 | 비디오·액션 공유 표현 동시 생성 | 코드 |
| FRAPPE | 2026 | 미래 표현 정렬 + cross-attention 액션 조건 | HF 컬렉션 |

정리 목록: https://github.com/OpenMOSS/Awesome-WAM , https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy
서베이: World Model for Robot Learning: A Comprehensive Survey (https://arxiv.org/pdf/2605.00080)

## 6. few-step distillation — full-model 시간 측정 전에는 후순위다

2026년 들어 Causal Forcing(2602.02214), Causal Forcing++(2605.15141), Causal-rCM(2606.25473) 등
CausVid·Self-Forcing 계열 AR distillation이 크게 발전했다.

11M 베이스라인은 216샘플에 14분 47초였지만 이 값은 1.44B/2B/5B 모델로 외삽할 수 없다.
먼저 후보별 batch inference를 재고 1시간을 넘을 때만 step 축소나 distillation을 검토한다.
시간이 남는다면 품질을 위해 sampling budget을 쓰되, “베이스라인이 빨랐으니 full model도 빠르다”는 가정은 하지 않는다.

## 7. 갱신된 권고

1차 조사의 3단계 전략을 실측 결과와 최신 지형에 맞춰 수정한다.

**[incumbent / 이식 비용 최소] DynamiCrafter 풀 UNet + 상대 액션 조건**
- 제공된 코드베이스를 그대로 쓰되 11M UNet을 backbone(약 1.44B) 파인튜닝으로 교체하고 상대 액션을 비교한다.
- 근거: 제공 구조·데이터로더·기하를 재사용하면서 공개 사전학습 prior를 보존할 수 있다.
  eval에서 관찰한 기존 출력 붕괴는 이 선택의 근거로 쓰지 않고, train-only holdout에서 incumbent 자체를 검증한다.
  코드 이식 리스크는 낮지만 action 정렬·검증·1.44B 추론시간이 미해결이라 아직 “안전”하다고 부를 수 없다.

**[challenger] Wan2.2-TI2V-5B + 프레임별 상대 액션 조건 + LoRA**
- Apache 2.0과 강한 비디오 prior가 장점이다. 일반화 우위는 eval 관찰이 아니라 train-only group holdout에서 판정한다.
- 원 우승 학습은 multi-node B200이므로 이 대회 예산에서의 학습 가능성은 미확정이다(1-A).
- 1X 우승 레시피의 컨디셔닝 구조를 비교하되 **픽셀 앙상블 평균은 기본값에서 제외**한다
  (섭동 probe에서는 역효과였고 실제 생성 샘플 재검증은 남음, EMPIRICAL.md 3).
- 출력 기하(427×320 내용 + 검은 띠)를 반드시 맞출 것(EMPIRICAL.md 6).

**[challenger] Cosmos-Predict2.5-2B + 공식 action-conditioned 레시피**
- 액션 컨디셔닝 공식 문서·distillation 가이드가 있어 출발점은 명확하지만, joint action 이식 리스크는 남는다.
- 7차원 EE delta 레시피를 6차원 joint action에 맞춰 고쳐야 한다. 표현·시간 정렬·conditioner 인터페이스가
  모두 달라 “차원만 바꾸는” 작업으로 보지는 않는다.
- NVIDIA Open Model License의 상업 이용·파생 허용 문언상 자격 후보이나, 정확한 weight 카드와 제출 시점을 보존한다.

**[watchlist challenger] Cosmos3-Nano forward dynamics**
- 현재 공개 계열 중 과제 입·출력 구조에는 가장 가깝지만 16B, 6D/domain 불일치, 6fps horizon,
  미완성 post-training surface 때문에 “최신 = 우선”으로 두지 않는다(2-A).
- 1샘플 local forward-dynamics와 target 장비 환산 시간이 통과하고 6D adapter를 짧은 예산에 학습할 수 있을
  때만 본 비교표에 올린다.

**[참고 설계] DreamZero-SO101의 joint action/video auxiliary**
- 방향이 반대이므로 conditioner를 복사하지 않는다. 대신 제공 train 데이터만으로 미래 latent에서 action을
  예측하는 보조 head나 joint flow-matching 아이디어를 작은 백본에서 비교할 수 있다.
- 보고된 원 레시피(약 254 GPU-h)는 대회 한도 96 GPU-h를 초과한다. auxiliary가 공식 세 component를
  개선하는지 짧은 실험에서 보이지 않으면 장기 이식하지 않는다.

**공통으로 우선 검증할 설계 가설**
- 상대 action은 유력하지만 **anchor delta(`a_t-a_0`)와 step delta(`a_t-a_{t-1}`)를 구분**한다.
  정책 논문과 Cosmos는 현재 구현의 anchor delta를 직접 검증하지 않는다. absolute/anchor/step,
  same-index/causal-shift, anchor 포함 여부를 train-only 고정 manifest에서 비교한다(EMPIRICAL.md 8).
- 시각 목표와 액션 목표를 별도 metric/failure gate로 본다(4-D MiraBench). loss 구성은 비교 후 정한다.
- 정지 영상은 train probe의 강한 기준선일 뿐 “확신 없으면 덜 움직인다”는 보편 규칙은 아니다(EMPIRICAL.md 1).
- 블러/스무딩/픽셀 평균은 현 probe의 기본값에서 제외하되 실제 모델 샘플로 반증 가능하게 둔다(EMPIRICAL.md 2, 3).
- 추론 예산은 베이스라인 기준 4배 여유지만 **14B급 백본을 쓰면 사라진다**. 백본 선정 시 216샘플 1시간을 먼저 계산할 것.
- **기본적으로 피할 것**: ID lookup형 per-dataset embedding. 새 ID에서 값이 정의되지 않는다.
  다만 이미지에서 연속 calibration latent를 추론하는 방식은 다른 가설이므로 별도 비교 가능하다.

## 출처
- Wan 공식 조직(직접 확인): https://huggingface.co/Wan-AI , https://github.com/orgs/Wan-Video/repositories
- Wan2.2 저장소: https://github.com/Wan-Video/Wan2.2 , https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Wan 2.5/2.6/2.7 API 서비스: https://www.alibabacloud.com/help/en/model-studio/use-video-generation/
- DreamZero-SO101 모델 카드: https://huggingface.co/Vizuara/dreamzero-so101-lora
- LTX-2 오픈 웨이트: https://www.globenewswire.com/news-release/2026/01/06/3213304/0/en/Lightricks-Open-Sources-LTX-2-the-First-Production-Ready-Audio-and-Video-Generation-Model-With-Truly-Open-Weights.html , https://ltx.io/model/open-source
- Cosmos-Predict2.5: https://github.com/nvidia-cosmos/cosmos-predict2.5 , https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
- Cosmos 3: https://github.com/NVIDIA/Cosmos , https://huggingface.co/nvidia/Cosmos3-Nano
- LeRobot v0.6.0: https://huggingface.co/blog/lerobot-release-v060
- SO-101 캘리브레이션: https://huggingface.co/docs/lerobot/so101
- Awesome-WAM: https://github.com/OpenMOSS/Awesome-WAM
- Causal Forcing++: https://arxiv.org/html/2605.15141v1

## 미확인 사항
- LTX-2 라이선스가 대회 규정 "상업적 또는 비상업적 이용이 허용된 라이선스"에 해당하는지 주최측 확인 권장.
- LeRobot v0.6.0 월드모델(FastWAM 등) 가중치 라이선스 미확인.
- 5절 논문 4건(VideoVLA, Motus, UVA, FRAPPE)은 Awesome-WAM 목록 기반이며 각 저장소를 직접 열어보지 않았다.
- 4-C의 구체 수치(33.7→55.0 등)는 **원문 표에서 직접 확인 전**이므로 인용하지 않는다.
- PAIWorld·Qwen-RobotWorld·MTV-World·BridgeV2W·IOI·DexAC-WM의 코드/가중치 공개 여부 미확인.
- FlowWAM 논문·project는 확인했지만 숫자 action만으로 world mode를 실행하는 공개 adapter/weight는 미확인.
- WorldArena 리더보드 원본(worldarena.ai) 접속 실패 — 관련 순위는 논문 자기 보고다.
- 흔히 인용되는 "additive < concat < cross-attn < AdaLN (FVD 87.31/67.89/62.56/56.47)" 비교는
  **원 논문을 특정하지 못했고 다른 비교에서는 순서가 정반대**다. 인용하지 말 것.
  액션 주입 방식에는 단일 정답이 없으므로 자체 ablation이 필요하며,
  Nano World Models(2605.23993)가 통일 비교 인터페이스를 코드로 제공한다.

## 직접 검증한 항목 (신뢰도 높음)
`Vizuara/dreamzero-so101-lora`(Apache 2.0, image+language→joint action/video, 72k step 자기 보고) 실재 ·
`Wan-AI/Wan2.1-I2V-14B-480P` Apache 2.0 · `whosricky/so101-megamix-v1`(400ep) 실재 ·
arXiv 2602.23408 실재 및 delta 우위 결론 ·
Wan 2.5/2.6/2.7 공식 API 서비스 존재 및 로컬 공개 weight 미발견 · Cosmos-Predict2.5 액션 조건 공식 문서 존재.
Cosmos 3가 최신 계열이고 Nano 16B/OpenMDW-1.1/forward-dynamics를 공개한 사실 ·
MiraBench 원문 결론 · OSCAR 코드/OSCAR-2B Apache-2.0 weight · iMaC/FlowWAM/JOPAT 원문 ·
WorldArena/RoboWM-Bench/AGIBOT World Challenge 및 EWMBench 공개 자료.

## 조사 방법 메모
검색 결과에는 제품 API와 공개 weight를 혼동하거나 라이선스를 임의로 붙인 페이지가 섞여 있다.
**모델 채택 판단은 공식 API 문서, HuggingFace 조직, 공식 GitHub와 실제 license 파일을 서로 구분해 확인**해야 한다.

## 8. 실행 우선순위 — 모델명이 아니라 증거 gate로 결정

이 절이 7절의 모델 순위를 보완한다. 현재는 어느 한 방법도 최종안이 아니다.

| 순서 | 질문 | 가장 싼 판정 | 실패하면 |
|---:|---|---|---|
| 0 | 공식 규정상 eval 정보가 학습 선택에 섞였는가? | 결정 로그에서 eval-derived 근거 제거 | train-only 기준으로 재선정 |
| 1 | action/frame 정렬과 표현은 무엇이 맞는가? | 짧은 action-only/temporal ablation | causal shift·step/anchor 구조 수정 |
| 2 | 모델이 action을 실제로 쓰는가? | permutation/counterfactual sensitivity | token/spatial condition으로 교체 |
| 3 | incumbent가 정지 영상보다 공식 가중 train-only 점수가 좋은가? | 고정 생성 manifest + 3 component | DynamiCrafter 장기 학습 중단 |
| 4 | skeleton을 정확히 투영할 수 있는가? | train-only reprojection overlay | OSCAR/iMaC 경로 중단 |
| 5 | 추론 216개가 1시간 안인가? | 목표 장비와 동일 설정 dry-run | steps/백본/batch 변경 |

권장 자원 배분은 “84시간 한 번”이 아니라 **짧은 반증 → 중간 학습 → 승자만 장기 학습**이다.
현재 incumbent, Wan, Cosmos/OSCAR 중 무엇이 더 좋을지는 이 gate 결과가 정하며, 논문 연도나 파라미터 수가 정하지 않는다.
