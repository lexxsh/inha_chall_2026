# INHA World Model Challenge — 관련 연구·유사대회 리서치 (2026-07-20)

ANALYSIS.md(데이터 분석)와 짝을 이루는 문서. 리서치 에이전트 2개(논문/데이터 포맷) 결과를 교차 검증해 통합.

> **문서 상태:** 이 파일은 1차 조사 스냅샷이며 절대적인 계획이나 최종 모델 순위가 아니다.
> 이후 실증·구현 감사에서 액션 시간 정렬, 정확한 점수 가중치, eval leakage 경계, 추론 예산 등의 빈틈이
> 발견됐다. 최신 후보와 반증 실험은 `RESEARCH_SOTA.md` 8절과 `METHOD.md` 10절을 우선해서 본다.

## 1. 과제와 관련성이 높은 선행 연구 후보

| 모델 | 컨디셔닝 방식 | 가중치 | 라이선스 | 규모 | 96GB·4일 FT |
|---|---|---|---|---|---|
| **IRASim** (ByteDance, ICCV'25) | 프레임별 action → linear → AdaLN scale/shift (Frame-Ada) | GitHub+ckpt | **Apache 2.0** | DiT 수억급 | ◎ 과제와 1:1 |
| **Wan2.2 TI2V-5B** + 1X 우승 구조 | state → sinusoidal → MLP(256) → 1D conv(VAE 시간압축 정합) → AdaLN-Zero, LoRA r32 | HF | **Apache 2.0** | 5B | △ 원 레시피는 multi-node B200, 단일 96GB 미측정 |
| **Cosmos-Predict2-2B Action-Conditioned** (NVIDIA) | 7D EE action conditioner, video2world | HF | NVIDIA Open Model | 2B | ○ |
| **Ctrl-World** (Stanford) | frame-level action + memory retrieval | HF | SVD 연구용(비상용) | ~1.5B | ○ (라이선스 재확인 필요) |
| **iVideoGPT** (THU) | AR 토큰 (obs+action) | GitHub | **CC BY 4.0** | 수억급 | ◎ 가볍고 추론 빠름 |
| **EnerVerse-AC** (AgiBot) | multi-level action + ray-map | GitHub | 확인 필요 | DiT | ○ |
| DynamiCrafter (베이스라인 backbone) | 이미지 dual-stream (CLIP cross-attn + latent concat) | HF | 코드 Apache-2.0, weight 카드 non-commercial | 1.1B | ◎ |

- 유사 대회: **1X World Model Challenge** (Sampling track 우승: Team Revontuli — Wan2.2-TI2V-5B + 프레임별 state AdaLN-Zero + LoRA r32, https://arxiv.org/html/2510.07092v1). WorldModelBench(CVPR'25)도 참고.
- LTX-Video(Apache 2.0, 초고속 추론), CogVideoX-I2V(+ControlNet 주입)도 후보이나 action 브랜치 직접 구현 필요.

## 2. 메트릭별 최적화 인사이트 (코드 확정 사실과 결합)

### DINO Component (DINOv2 ViT-S/14 프레임별 CLS, cosine dist)
- drift는 causal AR rollout에서 발생 → 16프레임 single-shot 생성(베이스라인 구조)이 유리. 시작 프레임 강앵커.
- **1X 우승 트릭: 복수 샘플 예측 앙상블(픽셀 평균)** — Gaussian blur보다 우수, 움직임 영역만 선택적으로 부드러워짐.
- "DINO as Foundation for Video World Models"(arXiv 2507.19468): 생성 프레임의 DINO feature를 GT에 정렬하는 보조 loss로 메트릭을 직접 최적화 가능.
- appearance 일반화는 eval 관찰이 아니라 train-only 데이터셋 홀드아웃에서 평가한다. 시작 프레임 충실도와
  사전학습 prior 보존은 우선 검증할 가설이다.

### Video Feature Component (r3d_18 Kinetics400, 112×112, 512-d) ※ 코드로 확정
- 저해상도(112×112)로 뭉개지므로 대략적 움직임 패턴·시간 일관성이 중요. 16프레임·320×512 letterbox 기하를 GT와 정확히 일치시킬 것.

### Action Component (고정 IDM의 MAE, 제출킷이 로컬 계산)
- 프레임별 action conditioning(AdaLN 계열)이 IDM 재추출 정확도에 직결. trajectory 전체를 한 벡터로 뭉치면 불리.
- **IDM-reward 사후학습** (EVA arXiv 2603.17808, RLIR arXiv 2509.23958): 자체 IDM으로 생성 영상에서 action을 되뽑아 MAE를 loss/reward로 파인튜닝 — 대회 메트릭③과 동일 신호.
  - 주의: 대회 IDM(action_extractor.ckpt) 자체를 loss에 쓰는 것은 규정상 "변경"은 아니지만(추론만), 재현 검증 관점에서 안전한지 규정 재확인 필요. 자체 IDM 학습이 안전.
- 시간축 스무딩과 앙상블은 초기 probe에서 손해였지만 보편 법칙은 아니다. 실제 모델 샘플의 세 component로 재검증한다.

## 3. 데이터 포맷 검증 (data-research 에이전트 ↔ 로컬 확인)
- 대회 데이터는 LeRobot v2.1 커뮤니티 데이터셋 재패키징 확정. 예: ZGGZZG/so100_drop0 (Apache 2.0, 원본 30fps·2캠(left/up)·AV1) → 대회판은 6fps·단일 `observation.images.image`·h264로 리인코딩 + 카메라 키 리네이밍.
- 원본 HF 데이터셋 사용은 "외부 데이터" 규정 위반 소지 → 제공본만 사용.
- action = 절대 목표 관절각(deg), 5관절 + gripper(0~48). 로컬 분포 확인 완료(ANALYSIS.md).

## 4. 1차 후보 전략 (현재 순위가 아님)

**[초기 후보] IRASim 파인튜닝 또는 베이스라인 스케일업**
- IRASim Frame-Ada가 관절 action 시퀀스를 받는 구조다. 라이선스·4일 처리량·출력 기하는 실제 실행 전 재확인한다.
- 또는 베이스라인 UNet을 11M → DynamiCrafter 풀 UNet(약 1.44B)으로 교체하고 action conditioner를 추가한다.
  코드 재사용 비용은 낮지만 additive/AdaLN/token 중 어느 주입이 좋은지는 미확정이다.

**[확장 후보] Wan2.2-TI2V-5B + 프레임별 action AdaLN-Zero + LoRA (1X 우승 레시피 이식)**
- Apache 2.0과 비디오 prior는 장점이지만 이 데이터의 일반화 우위는 미검증이다.
- LoRA rank·윈도우 수·해상도는 4일 처리량을 재고 정한다.
- 추론 방식과 앙상블은 공식 가중 train-only 점수 및 1시간 제한으로 선택한다.

**[공격적] 메인 라인 + DINO feature loss + IDM-reward 사후학습**
- 세 메트릭을 각각 대리 loss로 직접 겨냥. 상한 최고, 밸런싱 리스크 있음. 중간 전략 안정화 후 얹기.

## 5. 남은 확인 사항
- SVD 계열(Ctrl-World) 라이선스가 대회 "비상업 허용" 문구에 부합하는지 — 사용할 경우에만 확인.
- 추론 1시간 제한: 5B 모델 + 216샘플 → step 수 예산 계산 필요 (50 step diffusion이면 빠듯할 수 있음, distillation/적은 step 검토).
- 대회 action_extractor를 학습 loss에 활용하는 것의 규정 적합성.

## 주요 링크
- IRASim: https://github.com/bytedance/IRASim
- Wan2.2: https://github.com/Wan-Video/Wan2.2 · 1X 우승 리포트: https://arxiv.org/html/2510.07092v1
- Cosmos-Predict2: https://github.com/nvidia-cosmos/cosmos-predict2
- iVideoGPT: https://github.com/thuml/iVideoGPT
- EnerVerse-AC: https://github.com/AgibotTech/EnerVerse-AC
- DINO world model: https://arxiv.org/html/2507.19468v1 · EVA: https://arxiv.org/pdf/2603.17808 · RLIR: https://arxiv.org/html/2509.23958v1
- LeRobot: https://github.com/huggingface/lerobot · ZGGZZG/so100_drop0: https://huggingface.co/datasets/ZGGZZG/so100_drop0

## 6. 구현 가능성 재감사, 2026-07-21

후보 이름만 비교하던 단계에서 공식 구현·checkpoint 단위로 다시 확인했다. IRASim RT-1 Frame-Ada는
`16 frame`, `256x320`, `7D action`, IRASim-XL/2, SDXL VAE, 300k 학습이며 공개 checkpoint와
Apache-2.0 코드가 모두 있다. 시작 이미지와 action trajectory로 로봇 영상을 만드는 목적도 대회와 가장
직접적으로 일치한다. 반면 Cosmos 공식 action recipe는 7D EE displacement이고, iVideoGPT의 공개
고해상도 checkpoint는 action-free이며 공개 action-conditioned downstream checkpoint는 64x64라서
현재 DINO 중심 점수에는 불리하다. 따라서 다음 검증 대상은 공식 IRASim의 최소 이식으로 정했다.

다만 IRASim의 action 의미는 RT-1 상대 Cartesian command이고 우리 입력은 SO-100 절대 관절 target이라
checkpoint를 무수정 추론하는 것은 타당하지 않다. 영상 backbone은 전이하되 첫 action projection은
재학습해야 한다. 이 차이 때문에 IRASim이 논문 SOTA라는 사실만으로 이 대회에서의 우위는 보장되지 않으며,
500-step train-only action perturbation gate를 통과하지 못하면 채택하지 않는다. 구체적인 변경·탈락 기준은
`METHOD.md` 13절에 기록한다.
