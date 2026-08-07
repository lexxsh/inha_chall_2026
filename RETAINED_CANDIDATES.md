# Retained candidates (2026-08-07)

대회 정의와 전체 실험 회고, BWM-style 5B와 Wan 14B 비교, 조건부 다음 계획은 `PROJECT_SUMMARY.md`에
통합했다. 이 파일은 실제로 보존할 checkpoint와 재실행 경로만 좁게 관리한다.

현재 실제 제출 결과와 재사용 가치가 확인된 두 경로만 남긴 실행 인덱스다. 다른 실험 파일은 결과의
출처와 실패 회고를 보존하기 위해 삭제하지 않지만, 새 작업의 기본 진입점으로 사용하지 않는다.

이 선택 역시 절대적인 최종 결론은 아니다. 이후 Cosmos3-Edge 또는 Wan2.2-5B가 독립적인 train-only gate와
공식 제출에서 둘을 이기면 retained 목록을 갱신한다. 현재 문서의 목적은 과거 실패 코드를 실수로 다시 실행하거나
서로 다른 checkpoint/영상/CSV를 섞는 일을 막는 것이다.

## 1. Cosmos Unified Action v2 — 점수 우선 incumbent

| 항목 | 값 |
|---|---|
| 공개 점수 | **0.22** (낮을수록 좋음) |
| eval CSV 평균 Action MAE | **0.3224969483** |
| optimizer step / exposure | 13,067 / 104,536 |
| checkpoint | `inha_worldmodel_scratch_training/cosmos/train/runs/cosmos_unified_action_v2/latest.pt` |
| checkpoint SHA-256 | `bcd5f4e656cfa17834d4553ee4a864a9b0a9a1632e6bba855a290b8ce17fd12b` |
| 제출 영상 | `submissions/cosmos_unified_action_v2_13067/videos` |
| 제출 CSV | `submissions/cosmos_unified_action_v2_13067/submission_features.csv` |
| CSV SHA-256 | `7090a49e903f17bd2d57d1d3f0a59653ba9f4f5ef4de8178f63b42e8b63472e2` |

`latest.pt`와 `step_013067.pt`는 직렬화 해시는 다르지만 step/exposure metadata와 605개 state tensor가
원소 단위로 동일함을 확인했다. 실제 제출 manifest가 가리킨 `latest.pt`를 retained 기준으로 사용한다.

핵심 코드는 다음뿐이다.

- 학습/재개 orchestration: `inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh`
- 모델: `inha_worldmodel_scratch_training/cosmos/train/action_token_dit.py`
- 손실: `inha_worldmodel_scratch_training/cosmos/train/unified_losses.py`
- 학습 loop: `inha_worldmodel_scratch_training/cosmos/train/train_lora.py`
- 생성: `inha_worldmodel_scratch_training/cosmos/train/generate_eval.py`, `validate.py`
- 제출: `tools/run_cosmos_unified_submission.sh`

장점은 두 retained 후보 중 action 충실도와 공개 점수가 더 낫다는 것이다. 단점은 Wan보다 blur가 많고 로봇
morphology/세부 질감이 약하다는 것이다.

## 2. Wan2.1 Spatial Action 250 — 시각 품질 reference

| 항목 | 값 |
|---|---|
| 공개 점수 | **0.25** |
| eval CSV 평균 Action MAE | 0.4857981627 |
| optimizer step / exposure | 250 / 약 2,000 |
| checkpoint | `open/baseline/outputs/wan21_spatial_action_250/step-250.safetensors` |
| checkpoint SHA-256 | `21d4ef9a8ca9759de8d0795f355fac1ab8fff5614517134e199808df44c97976` |
| 제출 영상 | `submissions/wan21_spatial_action_250/videos` |
| 제출 CSV | `submissions/wan21_spatial_action_250/submission_features.csv` |
| CSV SHA-256 | `09af6415ed9d9ba75ea66635d045ef8072604987a0e7d6222cd0ae125d7b6d48` |

핵심 코드는 다음뿐이다.

- 학습: `train/train_wan21_spatial_action.py`
- 데이터/action 표현: `train/wan_spatial_action_dataset.py`, `train/data_module.py`
- Wan action residual 구현: `third_party/DiffSynth-Studio/diffsynth/models/wan_video_dit.py`
- pipeline 주입: `third_party/DiffSynth-Studio/diffsynth/pipelines/wan_video.py`
- 생성: `train/generate_wan21_spatial_action.py`
- 학습 실행: `train/run_wan21_spatial_action.sh`
- train-only gate: `tools/run_wan21_spatial_action_gate.sh`
- 제출: `tools/run_wan21_spatial_action_submission.sh`

장점은 로봇·배경 보존과 영상 완성도가 Cosmos v2보다 좋다는 것이다. 단점은 action MAE가 높고 학습 로그의
correct-vs-wrong 분리가 0 주변이라는 점이다. 이 250-step checkpoint는 시각 reference와 제출 가능한 후보로
보존하되, 같은 loss로 무조건 장기 연장할 기본 모델로 간주하지 않는다.

## 3. 통합 실행기

CPU에서 artifact가 정확한지 확인한다.

```bash
CANDIDATE=cosmos PHASE=verify bash tools/run_retained_candidate.sh
CANDIDATE=wan PHASE=verify bash tools/run_retained_candidate.sh
```

구조/data audit을 다시 실행한다.

```bash
CANDIDATE=cosmos PHASE=audit bash tools/run_retained_candidate.sh
CANDIDATE=wan PHASE=audit bash tools/run_retained_candidate.sh
```

기존 216개 영상에서 공식 CSV만 다시 만든다.

```bash
CUDA_VISIBLE_DEVICES=0 CANDIDATE=cosmos PHASE=csv bash tools/run_retained_candidate.sh
CUDA_VISIBLE_DEVICES=0 CANDIDATE=wan PHASE=csv bash tools/run_retained_candidate.sh
```

누락 영상을 생성하고 CSV까지 완성한다. Cosmos는 기본 `RESUME=1`, Wan은 기존 파일을 자동으로 건너뛴다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  CANDIDATE=cosmos PHASE=submission bash tools/run_retained_candidate.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  CANDIDATE=wan PHASE=submission bash tools/run_retained_candidate.sh
```

`PHASE=verify`는 GPU를 사용하지 않는다. `submission`과 `csv`는 사용자가 명시적으로 실행할 때만 GPU를 쓴다.
통합 실행기는 checkpoint를 학습하거나 덮어쓰지 않는다.

## 4. 다음 개발과의 경계

- `Cosmos Unified v2 13,067`: 현재 점수 incumbent이자 action reference
- `Wan2.1 Spatial Action 250`: 현재 시각 품질 reference
- `Cosmos3-Edge native`: 두 장점을 native action bridge에서 결합하려는 별도 challenger
- `Wan2.2-TI2V-5B action-token`: Cosmos3-Edge가 실패할 때만 구현할 fallback

새 challenger는 두 retained artifact를 덮어쓰지 않고 별도 run/submission directory를 사용한다. 승격 조건은
train-only correct/zero/batch-roll gate, morphology 검사, 공식 CSV 완결성, 실제 제출 점수다.
