# INHA World Model Challenge 2026

SO100 로봇의 첫 관측 이미지와 16-step, 6D action trajectory로 16-frame 영상을 생성하는 대회의 연구·실험 저장소입니다.

현재 전체 결론과 대회 정의는 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)에서 시작하면 됩니다. 이 저장소는 성공한 두 후보뿐 아니라 실패한 실험과 그 원인도 보존합니다. 아래 계획은 절대적인 최종 계획이 아니며, 독립적인 train-only holdout과 실제 제출 결과가 반박하면 변경합니다.

## 현재 핵심 결과

| 후보 | 공개 점수 | 현재 역할 |
|---|---:|---|
| Cosmos Unified Action v2 13,067-step | **0.22** | action/점수 incumbent |
| Wan2.1-I2V-14B Spatial Action 250-step | **0.25** | 시각 품질 reference |
| Dream/DynamiCrafter 10K | 0.28188 | 최초 제출 baseline |

점수는 낮을수록 좋으며, 공식 산식은 `0.3 × DINO + 0.3 × Video Feature + 0.4 × Action`입니다.

## 문서 안내

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): 대회 개요, 데이터, 평가, 모든 주요 실험, 현재 결론
- [RETAINED_CANDIDATES.md](RETAINED_CANDIDATES.md): 보존한 두 후보의 checkpoint/hash/재실행 경로
- [METHOD.md](METHOD.md): 방법론과 구현 변화
- [ANALYSIS.md](ANALYSIS.md): 실패 원인과 결과 분석
- [EMPIRICAL.md](EMPIRICAL.md): 실증 점검과 gate
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): 시간순 실험 기록
- [RESEARCH.md](RESEARCH.md), [RESEARCH_SOTA.md](RESEARCH_SOTA.md): 관련 연구와 모델 조사
- [FLOW_METHOD.md](FLOW_METHOD.md): flow/warp 계열 방법 기록
- [inha_worldmodel_scratch_training/docs](inha_worldmodel_scratch_training/docs): Cosmos scratch-training 세부 문서
- [THIRD_PARTY.md](THIRD_PARTY.md): 외부 저장소 버전과 로컬 변경 적용 방법
- [ARTIFACTS.md](ARTIFACTS.md): 공개 저장소에서 제외한 데이터·가중치·영상 목록

## 저장소 구성

```text
train/                              모델 학습·추론 코드
tools/                              audit, gate, scoring, 제출 실행기
results/                            소형 JSON/CSV 실험 지표
open/baseline/challenge_kit/        공식 제출킷 관련 코드
inha_worldmodel_scratch_training/   Cosmos 계열 코드와 문서
third_party_overlays/               pin된 외부 저장소에 적용할 수정 파일
```

## 재현 전 주의사항

대용량 모델, competition data, latent cache, checkpoint, 생성 영상, 제출 CSV는 Git에 포함하지 않았습니다. 원래 경로와 SHA-256은 문서에 남겨 두었으며, 필요한 공개 base model과 third-party repository는 각 라이선스와 접근 조건에 따라 별도로 받아야 합니다.

실험 스크립트 상당수는 원 연구 환경의 절대 경로와 H100 80GB 다중 GPU 구성을 전제로 합니다. 실행 전 경로, checkpoint, dataset root, CUDA 환경을 확인해야 합니다. RTX PRO 6000 Blackwell은 `sm_120`을 지원하는 PyTorch 빌드가 별도로 필요합니다.

공식 제출 파일은 반드시 competition `make_submission_csv.py`로 생성해야 합니다. eval 입력을 학습, 데이터 선택, checkpoint 선택에 사용하지 않습니다.
