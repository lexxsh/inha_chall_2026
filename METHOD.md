# 학습 파이프라인 — 무엇을 만들었고 왜 그렇게 했나 (2026-07-20)

> 2026-08-07 기준 대회 개요, 제출 결과, 보존 후보와 현재 우선순위는 `PROJECT_SUMMARY.md`를 먼저 본다.
> 이 파일은 구현과 실험이 누적된 상세 방법론/변경 기록이다.

`ANALYSIS.md`(데이터) · `EMPIRICAL.md`(메트릭 실측) · `RESEARCH_SOTA.md`(선행연구)에서 나온 현재 가설을
실제 학습 코드로 옮긴 결과물이다. 이 문서 하나만 읽으면 파이프라인 전체를 이해할 수 있게 썼다.

> **계획의 지위:** 아래 구성은 절대적인 최종안이 아니라 2026-07-20 시점의 **incumbent(현재 비교 기준)** 다.
> 아직 장기 학습·공식 가중 점수·RTX PRO 6000 추론을 통과하지 않았고, 액션 시간 정렬에도 반례가 발견됐다.
> 새 실험이 틀렸음을 보이면 작은 수정이 아니라 백본·조건 표현·학습 정책까지 교체할 수 있다.

**현재 실행 판정: HOLD.** partial-EMA의 실제 checkpoint 검증, action/frame 정렬, 고정 metric validation,
full-model 추론시간의 네 P0를 해결하기 전에는 84시간 본 학습을 시작하지 않는다.

---

## 1. 한 줄 요약

**사전학습된 DynamiCrafter I2V UNet(1.44B)을 불러와, 프레임별 anchor-delta 액션을 timestep embedding에 더하고,
공간 레이어는 얼린 채 시간·액션 경로만 파인튜닝한다.**

베이스라인은 같은 코드베이스에서 11M UNet을 **처음부터** 학습한다. 현재 incumbent는 공개된 1.44B
사전학습 가중치를 같은 자리에 로드해 prior를 보존하는 비교 후보이며, 우위는 train-only holdout에서 검증한다.

---

## 2. 만든 파일

| 파일 | 역할 |
|---|---|
| `train/configs/inha_full_unet.yaml` | 학습 설정. UNet 하이퍼파라미터를 backbone.ckpt에서 역추출해 맞췄다 |
| `train/data_module.py` | absolute/anchor/step/anchor+absolute 변환, 시간 shift + 데이터셋 단위 홀드아웃 |
| `train/train.py` | 학습 스크립트. 사전학습 로드 검증 / zero-init / 학습대상 선택 / 벤치마크 |
| `train/verify_checkpoint_load.py` | 사전학습 UNet 로드율 검증 (독립 실행용) |
| `train/compute_delta_stats.py` | delta 액션 정규화 통계 계산 |
| `train/delta_action_stats.json` | 위 산출물 (delta std) |
| `train/generate_videos.py` | eval 영상 생성. **학습과 동일한 delta 변환을 적용**하고 추론 시간을 잰다 |
| `train/run_train.sh` | 환경변수 세팅 포함 실행기 |
| `tools/check_action_state_lag.py` | train parquet에서 action/state 시간 정렬 진단 |
| `tools/check_partial_ema_roundtrip.py` | partial EMA를 full inference EMA로 복원하는 최소 재현 검사 |
| `EXPERIMENT_LOG.md` | baseline 결과와 다음 반증 실험을 누적하는 템플릿 |

`open/baseline/challenge_kit`(대회 제공 코드)는 **한 줄도 고치지 않았다.** 전부 감싸거나 상속했다.
`open/submission_kit`은 규정상 수정 금지이며 역시 손대지 않았다.

---

## 3. 핵심 설계 결정 4가지

### 3-1. 사전학습 UNet을 통째로 불러온다

베이스라인 설정은 `only_reload_modules`에서 UNet을 **일부러 빼고** VAE·CLIP·image projector만 불러왔다.
우리는 `"model.diffusion_model"`을 목록에 추가한다.

```yaml
only_reload_modules:
  ["first_stage_model", "cond_stage_model", "embedder", "image_proj_model", "model.diffusion_model"]
```

문제는 `load_checkpoints()`가 `load_state_dict(strict=False)`로 부른다는 점이다.
설정이 어긋나면 **예외 없이 조용히 랜덤 초기화 상태로 남는다.** 4일을 태우고 나서야 알게 된다.

그래서 UNet 하이퍼파라미터를 체크포인트에서 직접 역추출했다. 특히 두 가지가 베이스라인과 달랐다.

| 항목 | 베이스라인 | 체크포인트 실제값 | 근거 |
|---|---|---|---|
| `model_channels` | 32 | **320** | `input_blocks.0.0.weight` = (320, 8, 3, 3) |
| `use_scale_shift_norm` | True | **False** | `emb_layers.1.weight` = (320, 1280). True였다면 (640, 1280) |

`channel_mult [1,2,4,4]`, `attention_resolutions [4,2,1]`, `context_dim 1024`, `temporal_conv True`,
`image_cross_attention True`(`to_k_ip`/`to_v_ip` 존재), `addition_attention True`(`init_attn` 존재)도
같은 방식으로 확정했다.

**검증**: `train/verify_checkpoint_load.py`가 생성 모델과 체크포인트를 키 단위로 대조한다.
```
로드 성공     : 키 1512개 / 파라미터 1436.8M (99.9%)
형상 불일치   : 0개
체크포인트 없음: 5개  (action_embed 4 + null_action_emb 1 — 새로 학습할 모듈)
모델에 없음   : 4개  (fps_embedding — fs_condition=false라 미사용)
```
학습 스크립트도 시작할 때 같은 검사를 하고 **95% 미만이면 중단**한다.

### 3-2. 현재 기본 가설은 절대 관절각보다 anchor delta가 낫다는 것이다

상대 표현이 유리할 가능성에는 독립적인 근거 셋이 있지만, 현재의 정확한 변환까지 확정된 것은 아니다.

1. **자체 실측** — 대회 제공 action extractor로 train 128개 데이터셋을 전수 측정하니
   원본 MAE는 0.65~1.99로 3배 넘게 흩어지는데, **클립별 오프셋만 빼면 0.16~0.66으로 수렴**했다
   (EMPIRICAL.md 4-3). 상대 움직임은 데이터셋을 넘어 일관되고 절대 오프셋만 제각각이라는 뜻이다.
2. **문헌** — arXiv 2602.23408 *Demystifying Action Space Design*(500+ 모델, 13,000+ 실제 롤아웃)이
   delta action이 absolute를 일관되게 능가한다고 결론냈다. SO-101은 3D 프린팅 부품과 서보 초기
   오프셋 편차 때문에 팔마다 액션 공간이 다르다는 서술(VLA-REPLICA)도 있다.
3. **관례** — NVIDIA Cosmos의 action-conditioned 레시피가 EE delta를 쓴다.
   LeRobot 문서에도 복수의 캘리브레이션 관례가 공존한다(URDF 0자세 ≠ LeRobot 0자세, 구/신 방식 차이).

중요한 구분: 위 문헌의 delta는 주로 `a_t-a_{t-1}`인 per-step 또는 local-frame EE displacement다.
현재 구현의 `a_t-a_0`은 클립 첫 action에 고정한 **anchor displacement**다. 문헌은 “상대 표현을 시험하라”는
근거이지 anchor delta가 absolute/step delta보다 낫다는 직접 증거가 아니다.

구현은 `ActionRepresentationWrapper`가 담당한다. 전역 mean/std 정규화 뒤
`(a_t − a_0) / delta_std`를 조건으로 준다. 첫 프레임은 정의상 정확히 0이다.

절대 자세 정보는 **시작 이미지가 이미 담고 있으므로** 버려지지 않는다.
`delta_std`는 `compute_delta_stats.py`로 학습 데이터에서 구했다: `[0.78, 0.78, 0.75, 0.74, 0.33, 1.20]`.

현재 코드는 `absolute` / `delta`(anchor) / `delta_step` / `delta_anchor`(anchor delta + `a_0`, 12차원)와
`action_shift`를 지원한다. 기본은 `delta`, shift 0이다. 단, `delta_step`도 현재 anchor-delta용
`delta_std`를 공유하므로 공정한 비교 전에 **표현별 scale 통계**를 따로 계산해야 한다. 같은 sampler에서
step std는 `[0.1365, 0.1252, 0.1351, 0.1490, 0.0598, 0.3496]`로, anchor std의 약 16~29%뿐이었다.
현재 scale을 그대로 쓰면 `delta_step` 조건 진폭이 체계적으로 작아져 불공정하다.
`delta_anchor`를 쓸 때는 UNet의 `action_dims`도 12로 함께 바꿔야 한다.

### 3-3. incumbent additive와 challenger Frame-Ada를 분리한다

UNet의 액션 주입은 이렇게 생겼다.

```python
act_emb = self.action_embed(act)      # (B, T, 1280)
act_emb = rearrange(act_emb, "b t c -> (b t) c")
emb = time_emb + act_emb              # add_act_time_emb=True 일 때
```

`emb`는 각 ResBlock에 프레임별로 들어가지만 **AdaLN은 아니다.** incumbent 설정은
`use_scale_shift_norm: False`이므로 `emb_layers(emb)`를 feature에 더하는 additive bias/FiLM 계열 조건화다.
따라서 이 구현을
“AdaLN-Zero”라고 부르지 않고 **zero-initialized additive action embedding**이라고 부른다.

여기서 `add_act_time_emb: True`가 중요하다. `False`면 `time_embed`의 출력 차원이 절반으로 바뀌어
**사전학습 가중치를 못 불러온다.** `True`여야 형상이 보존된다.

그리고 `action_embed`의 마지막 선형층과 `null_action_emb`을 **0으로 초기화**한다.
그러면 학습 시작 시점에 `act_emb = 0`이므로 `emb = time_emb`, 즉 **모델이 사전학습 DynamiCrafter와
정확히 동일하게 동작**한다. 액션 조건은 0에서 출발해 점진적으로 학습된다.
베이스라인이 초반부터 붕괴한 전철을 밟지 않기 위한 장치다.

새 `action_injection: frame_adaln` challenger는 pretrained `time_embed`와 `emb_layers`의 형상을
바꾸지 않는다. 대신 각 ResBlock에 별도 `action_modulation: 1280 → 2C`를 두고 normalization의
scale/shift를 프레임별로 조절한다. 이 projection만 0-init하므로 step 0 출력은 pretrained 모델과
bitwise 동일하다. 이 설계에서는 backbone 호환성을 위해 `use_scale_shift_norm: False`를 그대로 두며,
action 전용 normalization modulation만 별도 적용한다.

전체 checkpoint 감사 결과 frame_adaln UNet은 1.4901B이고, generic backbone 1.4368B는 100% 일치했다.
새 action embed/modulation은 약 51.2M이다. 작은 CPU 검사는 zero-init 출력 동일성, 첫 backward의
modulation gradient, 학습 후 서로 다른 action에 대한 출력 차이를 모두 통과했다.

### 3-4. 공간 레이어를 얼리고 시간·액션 경로만 학습한다

동결의 정당화는 eval 관찰이 아니라 일반적인 사전학습 prior 보존과 **train-only 데이터셋 홀드아웃**에서
판정한다. 공간 레이어를 건드렸을 때 고정 홀드아웃의 DINO/Video가 나빠지는지가 핵심이다.
`temporal`은 기본 후보일 뿐이며 `action_only`, `all`, LoRA와 같은 예산으로 비교하기 전에는 최종 선택이 아니다.

`--trainable` 정책 셋:

| 정책 | 학습 대상 | 파라미터 |
|---|---|---|
| `all` | UNet 전체 | 1438.5M (100%) |
| **`temporal`** (기본) | TemporalTransformer + TemporalConvBlock + ResBlock의 `emb_layers` + `time_embed` + `init_attn` + 액션 임베딩 | **574.8M (40.0%)** |
| `action_only` | 액션 임베딩/Frame-Ada modulation | additive 약 1.7M / Frame-Ada 약 51.2M |

`emb_layers`는 액션이 더해진 timestep embedding을 feature로 투영하는 지점이라 현재 정책에 포함했다.
하지만 이를 얼려도 고정 projection을 통해 `action_embed`로 gradient가 흐르므로 액션 통로가 사라지는 것은 아니다.
따라서 `emb_layers`까지 학습하는 이득은 `action_only`와의 동예산 비교로 입증해야 한다.

`get_param_list()`가 `requires_grad`를 거르지 않으므로 인스턴스 단에서 필터를 씌웠다.

---

## 4. 검증 분할을 데이터셋 단위로 나눈 이유

기본 데이터모듈은 클립을 무작위로 train/val로 나눈다. 그러면 **같은 장면이 양쪽에 들어가** 일반화가
과대평가된다. 이는 eval을 보지 않아도 성립하는 group leakage 문제다.

그래서 `holdout_datasets: 6`으로 **데이터셋 6개를 통째로 빼서** 검증에 쓴다.
현재 홀드아웃: DorayakiLin/so100_pick_charger_on_tissue, ganker5/so100_dataline_0328,
pranavsaroha/so100_carrot_5, samsam0510/glove_reorientation_1, sihyun77/suho_3_17_1, tkc79/so100_lego_box_2.

또 `dragon-95/so100_sorting`은 혼자 10fps라 제외했다(나머지와 eval은 전부 6fps).

결과: 학습 121개 데이터셋 / 10,698 에피소드, 검증 6개 / 339 에피소드.
매 `__getitem__`마다 시작 지점을 무작위로 뽑으므로 에폭마다 다른 16프레임 윈도우를 본다
(전체 가능 윈도우는 약 86만 개).

---

## 5. 측정된 예산

GPU 1장(H100 80GB) 기준 실측:

| microbatch | microbatch당 | 최대 메모리 | 96시간 환산 샘플 |
|---|---:|---:|---:|
| 1 | 0.70s | 28.0GB | 약 494k |
| **4** (후보) | **2.22s** | **57.6GB** | 약 623k |

대회 재현 환경은 RTX PRO 6000 96GB 1장이므로 57.6GB는 여유 있게 들어간다.
기본 설정은 `batch_size 4` × `accumulate_grad_batches 2` = 유효 배치 8,
`max_steps 68000`, `max_time "03:12:00:00"`(84시간, 한도 96시간 안쪽)이다.

단, 위 2.22초는 `run_benchmark()`가 **한 microbatch마다 optimizer update까지 한 시간**이다.
실제 Lightning의 `max_steps`는 gradient accumulation 뒤의 optimizer step을 세므로, 단순 환산하면
84시간에 약 68k optimizer step(약 545k sample)이고 validation/checkpoint/I/O와 GPU 차이를 넣으면 더 적다.
따라서 과거의 `max_steps=130000`은 시간 제한 안에 도달하지 못하며 `max_time`이 먼저 종료할 설정이었다.
RTX PRO 6000에서 accumulation까지 포함한 wall-clock benchmark 후 step 목표를 다시 정한다.

학습률은 사전학습 파인튜닝이므로 베이스라인 1e-4에서 **1e-5로 낮췄고**, warmup 500 스텝을 뒀다.

2026-07-20의 첫 10k screen은 8×H100에서 GPU당 batch 1, accumulation 1로 유효 배치 8을 유지했고,
초기화 포함 약 2시간 26분에 끝났다. 이는 연구용 병렬 실행의 wall time이며 단일 RTX PRO 6000
재현 시간을 직접 측정한 값은 아니다. NaN/OOM은 없었지만 loss만으로 생성 품질은 판정할 수 없다.

현재의 **잠정적 step ladder**는 다음과 같다. 절대적인 계획이 아니며 생성 metric, action 반응,
처리량에 따라 더 짧게 중단하거나 다른 방법으로 교체할 수 있다.

- 2k: checkpoint/EMA/생성 경로를 확인하는 engineering checkpoint
- **10k: 첫 유효 baseline**(약 7.5 dataset epoch, 유효 샘플 80k)
- 30k: 10k가 정지·제공 기준과 action gate를 통과하고 개선 기울기가 남을 때만 연장
- 55~60k: 30k에서도 공식 가중 점수가 유의하게 좋아질 때만 full-budget 후보
- 68k: H100 단순 환산상 84시간과 거의 맞닿은 상한이라 validation/I/O와 장비 차이를 고려하면 목표값으로
  맹목적으로 사용하지 않는다.

추론 예산은 **11M 베이스라인에 대해서만** 확인했다(216샘플 14분 47초). 이 값을 1.44B 모델에
외삽할 수 없다. DynamiCrafter 공식 표도 512 모델 50-step을 A100에서 샘플당 약 20초로 보고하므로,
학습 전이라도 full backbone + batch 4의 216개 dry-run 시간을 먼저 재야 한다.

---

## 6. 실행 방법

```bash
# 사전학습 로드 확인 (설정을 바꿨다면 반드시)
.venv/bin/python train/verify_checkpoint_load.py

# 처리량·메모리 측정
CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh --bench-steps 8

# 학습
CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh                   # temporal (기본)
CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh --trainable all   # 전체 파인튜닝

# Stage 1: Frame-Ada 2k action-only screen
CUDA_VISIBLE_DEVICES=7 bash train/run_frame_adaln.sh --bench-steps 8
CUDA_VISIBLE_DEVICES=7 bash train/run_frame_adaln.sh

# 2k checkpoint 생성/정상-vs-zero-vs-batch-roll/공식 3-component gate
CUDA_VISIBLE_DEVICES=7 bash tools/run_frame_adaln_gate.sh

# 액션 표현 바꿔 실험
CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh data.params.action_mode=absolute
CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh data.params.action_mode=delta_step
CUDA_VISIBLE_DEVICES=7 bash train/run_train.sh data.params.action_shift=-1
```

Frame-Ada gate는 train-only 24개에서 normal weighted가 static보다 낮고, normal Action이 zero와
batch-roll보다 paired bootstrap 95% CI 기준으로 낮으며, incumbent보다도 유의하게 좋을 때만 `PROMOTE`한다.
그 외에는 장기 step을 늘리지 않고 Wan/Cosmos/latent-AR challenger로 이동한다.

첫 2k Frame-Ada는 이 gate를 실패했다. 다만 conditioner 교체와 action 의미/정렬 실패를 분리하기 위해
마지막 저비용 screen을 다음처럼 실행한다.

```bash
# 8-GPU 학습 3종 + 1k checkpoint 4종의 normal/batch-roll 생성·채점
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/run_action_representation_screen.sh

# 학습과 gate를 나눠 실행할 수도 있다.
PHASE=train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/run_action_representation_screen.sh
PHASE=gate CUDA_VISIBLE_DEVICES=7 bash tools/run_action_representation_screen.sh
```

비교 대상은 `delta/shift0`, `delta/shift-1`, `delta_step/shift0`, `delta_anchor/shift0`다.
어느 후보도 normal Action이 batch-roll보다 우세하지 않으면 low-dimensional Frame-Ada를 종료한다.

기본 체크포인트 간격은 5k optimizer step이다. 첫 10k screen만 비교 곡선을 보기 위해 2k 간격으로 저장했고,
실측 파일당 8,054,083,101 bytes, 6개(2k~10k와 `last`) 합계 약 46GB였다.

현재 `save_weights_only: true`라 optimizer/scheduler state가 저장되지 않는다. 추론 checkpoint로는 정상이나
10k에서 30k로 정확히 resume할 수 없다. 연장 실험 전에는 full-state `last`를 별도로 저장하거나,
10k weight-init을 명시적으로 지원하되 학습된 action head를 다시 zero-init하지 않도록 해야 한다.

---

## 7. 추론 — 학습과 액션 표현을 맞춘다

베이스라인 `generate_baseline_videos.py`는 절대 액션을 그대로 넣는다. 학습을 delta로 했으므로
그대로 쓰면 **예외 없이 조용히 엉뚱한 영상**이 나온다. `train/generate_videos.py`가 같은 변환을 적용한다.

과거에는 학습/추론에 변환 함수가 중복돼 있었지만, 현재는 둘 다
`data_module.transform_actions()`를 호출하므로 새 mode/shift도 한 구현을 공유한다. 기본 변환은 대조해 확인했다.
```
학습/추론 변환 최대 차이: 0.0
첫 프레임이 0인가: True
absolute 모드 항등: True
delta_anchor 앞 6차원 = delta 일치: True
```

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python train/generate_videos.py \
  --checkpoint open/baseline/outputs/full_unet/.../last.ckpt \
  --prediction-root generated_videos
```
생성 중 경과·예상 시간을 찍고, 1시간을 넘기면 경고한다.

---

## 8. 아직 안 한 것 / 다음 단계

1. **본 학습 실행** — 아직 벤치마크(8스텝)만 돌렸다. 로드와 단일-step 실행만 예비 검증됐으며,
   학습 수렴·생성 품질·재현 시간은 검증되지 않았다.
2. **정지 영상 베이스라인 제출** — `results/submission_static.csv`가 준비돼 있다.
   합산식은 공식 공개됐으므로 Public 기준점 확인에만 쓴다.
3. **본 학습보다 먼저** action same-index/causal-shift × absolute/anchor/step/delta_anchor 단기 ablation.
4. 고정 manifest에서 DINO/Video/Action과 action permutation sensitivity를 측정해 후보를 고른다.
5. full model의 RTX PRO 6000 상당 추론 시간, checkpoint 크기, 실제 accumulation 처리량을 실측한다.
6. 그 뒤 `all` vs `temporal` vs `action_only`와 공간 정렬 조건(OSCAR/track/flow)을 비교한다.

---

## 9. 지키고 있는 제약

- `open/submission_kit` 무수정 (규정)
- 학습: 단일 GPU 4일 이내 → 84시간으로 설정
- 추론: eval 216개 1시간 이내 → 학습 후 실측 예정
- 외부 데이터 미사용. eval 기반 분석 산출물은 학습·검증·모델 선택에서 격리
- 사전학습 모델: DynamiCrafter. **코드는 Apache-2.0이지만 HF weight 카드의 직접 사용은
  personal/research/non-commercial로 제한**된다. 대회는 비상업 허용 모델도 인정하므로 문언상 가능하나,
  제출물에 해당 모델 카드와 버전을 보존하고 필요하면 주최측 확인
- 환경: uv (`.venv`)

## 10. 구현 감사에서 추가로 확인한 빈틈과 중단 조건

| 우선순위 | 확인된 빈틈 | 통과 조건 |
|---|---|---|
| **P0** | `action[t]`가 1,195개 train 에피소드 중 1,168개에서 `state[t+1]`에 가장 가까움 | 짧은 동일예산 생성 ablation에서 정렬 선택 |
| **P0** | val은 매 `__getitem__`마다 clip 시작점이 바뀌고 8 batch의 diffusion loss만 봄 | 동일 holdout의 고정 manifest·seed로 세 공식 component 측정 |
| **P0** | 1.44B 추론시간 미측정 | 216개가 목표 장비 1시간 이내, I/O 포함 |
| **P0** | eval 유사도 산출물이 과거 설계 근거에 섞여 있었음 | train-only 근거로 결정 로그 재작성 |
| **P0** | partial EMA를 그대로 full EMA에 로드하면 동결 shadow가 랜덤이 됨; 재구성 완화와 최소 재현은 통과 | 실제 학습 checkpoint의 main/EMA round-trip 통과 |
| **P0** | `delta_step`이 anchor용 `delta_std`를 재사용 | mode별 train-only scale로 공정 비교 |
| P1 | 현재 주입은 AdaLN이 아니라 additive embedding | additive/AdaLN/token 또는 spatial control의 동예산 비교 |
| P1 | `--guidance-scale != 1`이면 uncond dict에서 action과 cross-attn image가 함께 빠짐 | 이를 action-only CFG로 부르지 말고 분리 구현 후 비교 |
| P1 | 학습 loss는 첫 프레임까지 평균하지만 추론은 첫 프레임을 입력으로 고정 | all-frame과 future-only loss를 동일예산 생성 점수로 비교 |
| P1 | `save_top_k=-1`, 1k 간격 | 크기 측정 후 보존 상한 설정 |
| P1 | inference가 `strict=False`; main/EMA action 누락 검사는 추가됐지만 나머지 누락은 허용 | 허용 목록 밖 missing/unexpected key면 즉시 실패 |

### 10-1. partial EMA 추론 버그와 현재 완화 상태

`LitEma`는 생성 시점에 `requires_grad=True`인 파라미터만 shadow buffer로 등록한다. 학습 코드는
`temporal` 동결을 적용한 **뒤** EMA를 다시 만들므로 checkpoint에는 약 40% trainable 경로의 EMA만 있다.
그런데 추론은 모든 파라미터가 trainable인 상태로 full EMA를 먼저 만들고 checkpoint를 `strict=False`로 얹는다.
이때 checkpoint에 없는 frozen-layer EMA buffer는 모델 본체의 사전학습값이 아니라 **초기 랜덤값**으로 남고,
`ema_scope()`가 이를 본체에 복사할 수 있다. 작은 동일 구조 재현에서도 누락 buffer가 생기고 frozen layer가
EMA 진입 직후 바뀌는 것을 확인했다.

안전한 수정 원칙은 다음 중 하나다.

1. 우선 EMA를 쓰지 않고 checkpoint의 main weights로 생성한다.
2. EMA를 쓴다면 main checkpoint를 먼저 로드한 뒤 그 값으로 full inference EMA를 초기화하고,
   checkpoint에 실제 존재하는 partial EMA buffer만 다시 overlay한다.
3. `ema_scope()` 전후로 학습에서 frozen이었던 모든 파라미터가 bitwise 동일한지 검사하고,
   main action key와 EMA action key를 각각 필수 검사한다.

현재 `generate_videos.py`에는 두 경로가 구현돼 있다. `--no-ema`는 checkpoint main weights만 쓰고,
기본 EMA 경로는 main checkpoint를 먼저 로드한 뒤 full EMA를 재생성하고 저장된 partial buffer만 overlay한다.
main action key와 EMA action key도 각각 없으면 중단한다.

`tools/check_partial_ema_roundtrip.py`의 작은 동일 구조 검사에서는 frozen main weights bitwise 보존,
trainable partial EMA 적용, scope 종료 후 main 복원을 모두 통과했다. 다만 이는 실제 1.44B checkpoint의 키 구조와
동결 목록까지 검증한 것은 아니다. 첫 짧은 checkpoint가 생기면 다음을 통과하기 전 EMA 생성 점수를 채택하지 않는다.

1. 같은 입력/seed를 main(`--no-ema`)과 재구성 EMA로 각각 생성하고 NaN·즉시 붕괴가 없는지 확인한다.
2. 학습 당시 frozen 파라미터 이름 목록을 저장하고, EMA scope 전/안/후 hash가 각각 main/동일/main인지 검사한다.
3. overlay 수는 `decay`·`num_updates` 같은 metadata와 parameter shadow를 분리해 기록한다.
4. 허용 목록 밖의 main/EMA missing·unexpected key는 `strict=False` 로그만 남기지 말고 실패시킨다.

실제 checkpoint 검증 전 생성 품질 평가는 main weights 결과만 유효한 탐색값으로 본다.

### 10-2. 현재 `valset/`은 모델 선택용 holdout이 아니다

`valset/manifest.json`은 48개 train 데이터셋에서 probe를 만들었고, 데이터모듈이 통째로 제외한 6개
holdout과 독립적으로 생성됐다. 예를 들어 manifest의 여러 데이터셋은 현재 학습 121개 목록에도 들어간다.
따라서 이 probe로 얻은 EMPIRICAL 수치는 metric 성질을 파악하는 데는 유용하지만, 학습한 checkpoint의
일반화 성능이나 모델 순위를 재는 데 쓰면 안 된다.

모델 선택용 생성 검증셋은 다음 조건으로 새로 만든다.

- 데이터모듈과 **동일한 6개 holdout dataset만** 사용하고 episode/start/camera를 manifest에 고정한다.
- `--eval-like`와 `results/eval_source_match.json`은 절대 사용하지 않는다.
- dataset마다 여러 episode를 균형 있게 뽑고, generation seed도 고정한다.
- 가능하면 holdout seed를 바꾼 2~3개 group fold에서 순위가 유지되는지 확인한다.
- 엄밀한 fold 비교에서는 action/representation normalization 통계도 해당 fold의 train 부분만으로 계산한다.

액션 조건이 실제로 사용되는지도 별도 확인한다. 같은 첫 이미지에서 action 순서를 뒤섞거나 부호/크기를
바꿨을 때 영상과 extractor 출력이 거의 같다면, 낮은 diffusion loss와 좋은 외형에도 불구하고 action collapse다.
MiraBench가 보고한 optimism bias와 같은 유형이므로 **정상 action만 생성해 보는 검증으로는 발견되지 않는다.**

### 10-3. 첫 프레임 loss와 실제 생성 구간이 다르다

현재 `LatentVisualDiffusion.p_losses()`는 오차를 `[C, T, H, W]` 전체에 평균하므로 입력과 같은 첫 프레임도
loss의 1/16을 차지한다. 반면 추론의 `cond_mask[:, :, 0] = 1`은 매 denoising 과정에서 첫 latent를 입력
이미지로 고정한다. 따라서 첫 프레임 재구성 loss는 제출 시 직접 생성되는 15개 미래 프레임의 품질·동작과
경쟁하며, `action[0]`의 의미도 출력 프레임 수와 한 스텝 어긋날 가능성이 있다.

그렇다고 곧바로 첫 프레임 loss를 제거하는 것도 확정 해법은 아니다. DynamiCrafter 사전학습 목적과 달라지고
경계 프레임의 시간 일관성이 나빠질 수 있다. 다음 세 조건을 고정한 짧은 ablation으로만 결정한다.

- all-frame loss(현행) 대 future-only loss(`t >= 1`), 필요하면 작은 first-frame consistency 항을 함께 비교한다.
- 둘 다 같은 action 정렬·정규화, optimizer step, seed, sampler 설정을 사용한다.
- 전체 diffusion loss가 아니라 15개 생성 프레임의 공식 DINO/Video/Action과 첫 전이의 discontinuity로 판정한다.

최종 의사결정은 다음처럼 한다.

1. P0를 통과하지 못하면 84시간 학습을 시작하지 않는다.
2. incumbent보다 공식 가중 train-only 점수가 좋아지고, action perturbation에 올바른 방향으로 반응하며,
   추론 제한을 통과한 후보만 승격한다.
3. DynamiCrafter가 이 기준을 못 넘으면 Wan/Cosmos/OSCAR 계열 또는 point-track/flow 보조 조건으로 바꾼다.
4. 어떤 문헌도 이 데이터·extractor·계산 제한에서의 우위를 보장하지 않으므로, 이 계획 자체를 보존할 이유는 없다.

## 11. Wan2.2-TI2V-5B 전환안 — 잠정 계획, 2026-07-21

Frame-Ada 2k가 static과 additive 10k보다 외형에서 나쁘고 wrong action보다 올바른 action에 유리하다는
증거도 만들지 못했으므로, 같은 백본의 표현 screen을 더 돌리는 대신 video-native 사전학습 백본을 시험한다.
이 선택은 **절대적인 최종 계획이 아니다.** 아래 smoke/gate 중 하나라도 실패하면 Wan을 고집하지 않고
latent-AR, action token cross-attention, spatial track/flow control 또는 더 작은 video backbone으로 바꾼다.

현재 구현은 공식 DiffSynth-Studio `fb337fbb90945ff829de69dbd44ded618f73e889`를 기준으로 한다.
Wan2.2-TI2V-5B의 17 RGB frame은 clean first latent 1개와 future latent 4개가 된다. 16개 action을
순서를 유지한 4개 그룹으로 pooling해 future latent 네 개의 timestep AdaLN에 더하고, clean latent에는
항상 0을 넣는다. 마지막 projection은 zero-init이므로 step 0의 함수는 vanilla Wan과 bitwise 동일하다.
DiT 본체는 고정하고 attention/FFN LoRA rank 32와 action conditioner만 학습한다.

진행 조건은 다음 순서로 고정한다.

1. vanilla 320x512, 17 frame, 2 denoise-step 1개 생성이 성공하고 출력이 정확히 17 frame인지 확인한다.
2. 1 training-step에서 OOM/NaN이 없고, peak VRAM과 step time으로 1k 비용을 산출한다.
3. 1k 학습 뒤 고정 holdout 24개의 normal/zero/batch-roll action을 20-step, CFG 1로 생성한다.
4. normal이 static 및 additive 10k보다 weighted가 좋아야 하며, normal−wrong Action bootstrap CI 상단이
   0보다 작아야 3k/5k로 연장한다.
5. 216개 환산 생성 시간이 60분을 넘으면 품질과 무관하게 현재 sampler 설정은 제출 후보에서 제외한다.

17번째 decode frame은 VAE의 `4n+1` 제약을 만족시키는 문맥으로만 사용하고 제출에는 frame 0..15를 저장한다.
action 0..15는 네 future latent 모두에 들어가므로 마지막 action이 단순 폐기되는 구조는 아니다. 다만
비인과 VAE 때문에 마지막 문맥 frame이 앞 16 frame에 미치는 효과는 smoke/gate의 별도 위험요인으로 남긴다.

구현·실행 파일:

- `third_party/DiffSynth-Studio/diffsynth/models/wan_video_dit.py`: temporal action AdaLN-Zero
- `third_party/DiffSynth-Studio/diffsynth/pipelines/wan_video.py`: 학습/추론 action 전달
- `train/wan_action_dataset.py`: 17 frame/16 action 및 train-only split
- `train/train_wan_action.py`, `train/run_wan_action.sh`: max-step LoRA 학습과 smoke
- `train/generate_wan_videos.py`, `tools/run_wan_action_gate.sh`: 16-frame 생성과 gate

현재 작업 환경에서는 NVIDIA driver가 노출되지 않아 실제 5B smoke와 VRAM/속도 측정은 미실행이다.
따라서 이 절은 아직 `PLANNED/UNIT-VALID`, 모델 선택 결론은 `INCONCLUSIVE`다.

### 11-1. 1k 결과로 수정된 결론

실제 H100 smoke와 8-GPU 1k는 계산상 통과했지만 생성 gate는 `REJECT`였다. normal은 static보다
weighted `+0.02630` 나빴고, additive 10k보다 `-0.00348` 좋았으나 CI가 0을 포함했다. 특히
normal action은 zero action보다 DINO를 `+0.03265` 유의하게 악화시키고 Video를 `-0.00392`
개선했지만 Action 차이는 `-0.00005`에 불과했다. batch-roll action도 normal보다 Action이 평균상 좋았다.

따라서 “Wan이면 자동으로 해결된다”는 가설과 “단순히 더 학습하면 된다”는 해석은 모두 기각한다.
현재 구현의 4-frame pooling은 latent group 안 action 순서를 상당 부분 잃고, diffusion MSE만으로는
correct action과 wrong action을 명시적으로 구분시키지 않는다. 다음 1k를 허용하려면 두 변경을 함께 둔다.

1. learned kernel-4/stride-4 temporal encoder로 네 action의 순서를 보존한다.
2. 동일 영상에서 correct action의 denoising error가 wrong/reversed action보다 margin만큼 작도록 하는
   bounded ranking loss를 추가한다.

이는 아직 유력 후보일 뿐 절대 계획이 아니다. 두 번째 Wan 1k에서도 normal−wrong Action CI 상단이
0 아래로 내려가지 않으면 AdaLN 경로는 더 연장하지 않는다.

## 12. 과제 역설계 전환 — Conservative Action Flow, 2026-07-21

Wan v2는 외형과 weighted를 개선했지만 batch-roll action을 구별하지 못했다. 이 결과는 더 큰 생성모델이
필요하다는 뜻보다, 현재 과제에서 영상 생성의 자유도가 지나치게 크다는 반례로 해석한다. 입력은 시작 이미지
한 장뿐이고 카메라는 대부분 고정이며, DINO 0.3은 흐림을 크게 벌점하고 static baseline도 강하다. 따라서
다음 후보는 새 픽셀을 전부 생성하지 않고 **첫 이미지의 선명한 픽셀을 기본값으로 복사**한다.

`ConservativeFlowWorldModel`은 hybrid action 16개를 causal Transformer로 인코딩하고, ResNet18 image
feature와 결합해 미래 15개 frame의 backward flow, motion mask, 작은 bounded residual을 예측한다.
출력 frame 0은 입력과 정확히 같고, 미래는 다음 식으로 만든다.

`I_t = (1-M_t) I_0 + M_t (warp(I_0, F_t) + R_t)`

이 구조의 과제별 장점은 다음과 같다.

1. 고정 배경과 로봇 texture를 재생성하지 않아 DINO와 선명도를 보호한다.
2. action이 flow를 직접 결정하므로 generic video motion보다 action identity를 배우기 쉽다.
3. GT motion 영역에 reconstruction 가중치를 줘 static shortcut을 막는다.
   또한 `|I_t-I_0|`로 만든 soft motion target으로 mask를 직접 감독해 zero-mask 고착을 막는다.
4. 동일 시작 이미지에서 correct action reconstruction error가 batch-roll보다 margin만큼 작도록 paired
   ranking loss를 넣어 이전 gate의 정확한 실패를 학습 목적에 반영한다.
5. 약 4.64M parameters라 4일/1시간 제한보다 훨씬 작고, 실패 비용도 낮다.

첫 direct-reconstruction 50-step smoke에서는 flow가 전역으로 퍼져 로봇을 배경 픽셀로 지우고 선명도가
약 31% 감소했다. 이는 허용 가능한 초기 품질이 아니라 ill-posed photometric flow 학습의 실패다. 따라서
이 버전은 폐기하고, 공개 torchvision RAFT-small을 frozen teacher로 사용해 제공 train 영상에서만
미래→첫 프레임 backward pseudo-flow를 만든다. 1/8/15 horizon flow supervision과 GT static 영역의
zero-flow penalty를 추가한 v3만 다음 smoke 대상으로 둔다. RAFT 가중치는 공개 사전학습 모델 사용 범위에
해당하지만 최종 코드 제출 시 모델 라이선스와 다운로드 재현 경로를 함께 기록한다.

공식 `submission_kit/checkpoints/action_extractor.ckpt`는 학습 loss에 사용하지 않는다. 규칙은 checkpoint의
재학습·변경을 명시적으로 금지하고 있으나 frozen reward 사용 허용 여부는 명확하지 않으므로, 서면 확인 전에는
평가 모델을 학습 루프에서 완전히 격리한다. paired video reconstruction만 제공 train data로 학습한다.

위 선택도 절대 계획이 아니다. flow-only screen의 한계는 disocclusion과 새로 드러나는 물체 면을 만들기
어렵다는 점이다. screen에서 DINO는 유지하지만 Video/Action이 움직이지 않으면 spatial flow를 끝내고
action-token latent-AR 또는 공개 action-conditioned Cosmos 계열로 이동한다.

## 13. 공식 IRASim Frame-Ada 이식 — 잠정 계획, 2026-07-21

Wan과 직접 flow의 실패 뒤에는 새 주입 구조를 더 만드는 대신, 과제 정의가 거의 같은 공개 방법을 먼저
재현하지 않았다는 절차상 빈틈이 있었다. 공식 [ByteDance IRASim](https://github.com/bytedance/IRASim)
commit `c72b6dade6fcd65971e0aa8ab49ea39b15108c90`와 공개 RT-1 Frame-Ada 300k checkpoint를 기준으로
다음 부분은 그대로 보존한다.

- SDXL VAE latent에서 동작하는 IRASim-XL/2 679M DiT
- action 한 개를 미래 프레임 한 개에 대응시키는 Frame-Ada 공간 블록
- alternating spatial/temporal transformer와 future-only diffusion loss
- 256x320 학습 기하, PNDM 50-step, CFG 1

필요한 변경은 세 가지로 제한한다.

1. RT-1 상대 EE+gripper 7D 대신 SO-100 action 6D를 받도록 첫 action projection만 교체한다.
2. 공식 horizon `source 1 + future 15`를 대회 `source 1 + future 16`으로 늘린다. `temp_embed`는
   학습 파라미터가 아니라 고정 sinusoidal이므로 17 길이로 다시 계산한다.
3. 제공 train 데이터만 읽는 SO-100 어댑터를 연결하고, 기존 전수 측정에서 확인된 calibration offset을
   줄이기 위해 기본 표현은 train/eval이 동일한 anchor delta로 둔다.

공개 checkpoint 로더는 shape가 달라야 하는 `temp_embed`, `embed_state.fc1.weight` 두 키 외의 누락을
오류로 처리한다. 공식 코드의 dropout mask indexed in-place write는 최근 PyTorch autograd에서 version
오류를 만들 수 있어 같은 연산의 `torch.where`로만 바꿨다. 영상 backbone이나 conditioning 위치를
추가 설계하지 않았다.

이 선택 역시 절대적인 계획이 아니다. 먼저 8-GPU 500-step screen을 실행하고 250/500 checkpoint에서
고정 train-only holdout의 normal/zero/batch-roll을 비교한다. normal action이 wrong action보다 유의하게
좋지 않거나 static보다 weighted가 나쁘면 2k로 늘리지 않고 IRASim도 탈락시킨다. 통과할 때만 2k로
연장하며, 이후에도 216개 50-step 추론이 1시간을 넘으면 step 축소를 별도 품질 gate로 판단한다.

구현 파일:

- `third_party/IRASim/models/irasim.py`: action width와 autograd-safe mask의 최소 호환 패치
- `train/irasim_so100.py`: SO-100 데이터/horizon/checkpoint 어댑터
- `train/train_irasim_action.py`: 공식 backbone 전체 저율 fine-tune + action MLP 고율 fine-tune
- `train/generate_irasim_videos.py`: source+16 생성 후 대회용 미래 16프레임 저장

## 14. Cosmos adapter 실패 분석과 식별가능 v2 — 2026-08-03

Cosmos-Predict2.5 공개 action-conditioned prior 자체는 zero-action probe에서 선명한 영상을 만들었지만,
SO-100 adapter 500-step 영상은 normal과 zero가 사실상 같은 운동량을 보였다. 8개 holdout의 guidance 0에서
프레임간 변화량은 normal `0.00940`, zero `0.00953`이고 first-last 변화량도 각각 `0.06206`, `0.06513`이었다.
따라서 보이는 작은 움직임은 액션 효과가 아니라 Cosmos의 무조건부 motion prior로 판정한다.

체크포인트를 직접 열어 원인을 확인했다. 기존 903-parameter adapter는 `f(0)`의 평균 절댓값이
`0.16543`이었고, 실제 액션 출력과 `f(0)`의 차이는 평균 `0.01613`뿐이었다. 즉 adapter가 관절 운동을
번역하기보다 bias를 통해 SO-100 데이터셋용 상수 domain token을 학습했다. 일반 rectified-flow MSE에는
정답 액션과 틀린 액션을 비교하는 항이 없으므로 이 shortcut을 막지 못한다.

입력 의미도 맞지 않았다. 공개 Cosmos Bridge checkpoint는 매 프레임 사이의 gripper-frame EE 변위
6D에 절대 gripper 명령 1D를 붙이고 motion 좌표를 20배 스케일해 학습했다. 반면 v1은 첫 명령 대비
누적 관절 변위를 timestep별 MLP에 넣었다. 이 MLP는 과거 timestep을 보지 않으므로 누적 변위를
step 변위로 바꾸는 것조차 구조적으로 불가능하다.

v2는 세 문제를 함께 고친다.

1. 두 Linear의 bias를 없애 `f(0)=0`을 수치적 희망이 아닌 구조적 불변식으로 만든다.
2. 입력을 12D `[정규화 절대 관절 목표, 정규화 step 관절 변위]`로 만든다. 전자는 관절 구성과 절대
   gripper 상태, 후자는 Cosmos가 기대하는 국소 motion semantics를 제공한다.
3. 같은 GT latent, noise, timestep에 correct action과 다른 data-parallel rank의 action을 각각 넣고
   `L_correct + 0.01 <= L_wrong`을 요구하는 margin ranking loss를 원래 RF loss에 더한다.
   GPU당 batch가 1이므로 local batch-roll이 원본과 동일해지는 문제는 8개 rank의 action을 gather한 뒤
   rank 간 roll하는 방식으로 피한다. 이렇게 하면 단순 시간 역순의 분포 이상을 shortcut으로 쓰지도 못한다.

v2는 공개 checkpoint에서 새로 시작하며 v1 checkpoint를 resume하지 않는다. 5-step smoke에서
`correct-wrong` gap과 rank loss가 실제로 변하지 않으면 250-step을 시작하지 않는다. 250-step 뒤에도
adapter-only gate와 동일 seed normal/zero/reverse 영상 gate를 모두 통과해야 다음 학습을 허용한다.
이 역시 절대적인 계획이 아니다. v2가 action counterfactual을 구분하지 못하면 작은 adapter의 표현력
또는 joint-to-EE 비식별성이 병목이라는 증거로 보고, Cosmos backbone LoRA나 URDF/FK 기반 조건으로
바꾸거나 Cosmos 후보를 종료한다.

---

## 15. Spatial-control 전환: oracle point-track → Wan gate (2026-08-03)

> **계획의 지위:** 이것도 최종안이 아니다. 아래 oracle gate가 통과해야만 action→track을 만들고,
> action→track gate까지 통과해야만 제출 후보를 학습한다. 논문과 사전학습 백본의 명성만으로 장기 학습하지 않는다.

### 실패에서 고정한 결론

- DynamiCrafter/Frame-Ada, Wan action-AdaLN, Cosmos joint adapter는 모두 저차원 action을 전역 feature에
  주입했지만 올바른 픽셀 운동을 만들지 못했다.
- Cosmos adapter는 구조적으로 action에 반응했으나 scale을 올리자 새 팔/형상이 생겼다. 이는 단순한
  신호 부족보다 pretrained action semantics와 SO-100 joint target의 불일치에 가깝다.
- 따라서 다음 후보는 `action vector → video`를 한 번에 학습하지 않고
  `action → pixel-aligned motion → pretrained renderer`로 분해한다.

### URDF skeleton 대신 point-track을 첫 gate로 택한 이유

공식 SO-100 URDF가 있어도 제공 데이터는 uploader별 servo calibration과 camera extrinsic이 없다.
잘못 투영한 skeleton은 정확해 보이는 거짓 조건이 될 수 있다. 먼저 제공 train 영상만으로 RAFT
forward/backward cycle-consistent point track을 만들고, Wan이 **정답 track을 받았을 때조차** 따르지
못하면 action predictor를 만들기 전에 이 경로를 중단한다. FK skeleton은 추후 track에 대한 reprojection
오차를 실제로 통과한 경우에만 추가한다.

### 구현과 1차 결과

- `tools/prepare_spatial_control_gate.py`: 고정 64 train + dataset-group holdout 8 clip, RAFT point track,
  occupancy/dx/dy 3채널 control, 오버레이와 challenge-format GT 생성
- `SpatialTrackAdapter`: control을 Wan token grid로 보간하는 작은 Conv3D adapter. 마지막 projection zero-init
- `train/train_wan_oracle_control.py`: adapter + temporal/spatial LoRA, 동일 noise/timestep의 correct-vs-future-reversed
  control ranking loss
- 품질 필터: train에서 tracks≥8, cycle≥0.6, motion coverage≥0.2인 50/64 clip만 사용
- 잘못된 fallback 제거: 정지 clip은 padding 경계 점을 강제로 채우지 않고 0-track으로 유지

고정 holdout 8개의 pre-gate:

| 지표 | 값 |
|---|---:|
| tracks≥8 비율 | 0.875 (7/8) |
| median cycle-valid | 0.8477 |
| median motion coverage | 0.5916 |
| median action–motion Spearman | 0.3505 |
| state lag 0/1 비율 | 0.875 |
| 판정 | `PASS_ORACLE_TRACKS_AND_TEMPORAL_ALIGNMENT` |

이 PASS는 oracle condition을 만들 수 있다는 뜻뿐이다. action으로 condition을 예측할 수 있다거나 Wan이
따른다는 뜻이 아니다. 다음 250-step oracle-control gate의 normal이 zero와 batch-roll보다 유의하게 좋고,
새 형상을 만들지 않을 때만 action→track 단계로 이동한다.

```bash
# 이미 생성·검증된 artifact를 다시 만들 때만 실행
.venv/bin/python tools/prepare_spatial_control_gate.py \
  --split both --train-clips 64 --holdout-clips 8

# 4-GPU, 250-step oracle controllability gate
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash train/run_wan_oracle_control.sh

# train-only holdout: normal / zero / batch-roll 생성 및 공식 3-component 비교
CUDA_VISIBLE_DEVICES=0 bash tools/run_wan_oracle_gate.sh
```

### 250-step oracle gate 결과와 수정된 판단

gate는 `REJECT`였다. normal weighted `0.618634`는 static `0.586220`보다 `+0.032414` 나빴고,
normal−zero Action은 `+0.003389` (95% CI `[-0.001379,+0.009987]`), normal−batch-roll Action은
`+0.002498` (95% CI `[-0.001592,+0.008056]`)이었다. 즉 정답 영상에서 직접 추출한 trajectory조차
zero나 다른 clip의 trajectory보다 유리하지 않았다.

다만 이것을 곧바로 “point-track 방법의 반증”으로 해석하면 안 된다. checkpoint에서 normal adapter 출력은
abs mean `0.015332`, zero 입력 출력은 `0.015235`였고 순수 control 기여는 전체의 `0.70%`뿐이었다.
zero 입력에서도 Conv3D bias가 거의 같은 보정을 만들었으므로, 이번 gate는 **조건 표현보다 먼저 현재
adapter의 식별가능성 실패**를 드러냈다. 이는 앞서 Cosmos에서 관측한 constant domain-token shortcut과
같은 종류이며, output projection의 zero initialization은 시작점만 보존할 뿐 학습 뒤 `f(0)=0`을 보장하지 않는다.

현재 checkpoint의 장기 연장과 action→track 학습은 중단한다. 선택 가능한 마지막 저비용 반증 실험은
다음 세 조건을 동시에 고정한 250-step oracle v2뿐이다.

1. `g(c)=raw(c)-raw(0)`로 학습 뒤에도 exact `g(0)=0`을 보장한다.
2. LoRA를 끄고 Wan을 완전히 고정해 unconditional denoising shortcut을 제거한다.
3. 같은 clip의 시간 역순 대신 다른 clip의 track을 wrong condition으로 사용한다.

이 v2가 normal−zero와 normal−cross-clip 두 Action CI에서 모두 0 아래로 가지 못하면 spatial Wan을
종료한다. 반대로 통과할 때만 action→track predictor로 이동한다. 이 역시 절대적인 계획이 아니며,
현재 증거에 비해 가장 싸고 직접적인 반증 순서라는 의미다.

v2 구현 완료 및 회귀검사:

- `SpatialTrackAdapter(structural_zero=True)`: zero 입력 출력 bitwise exact `0.0`
- nonzero 입력 output 및 input-convolution gradient nonzero
- 50/50 train clip에 서로 다른 dataset의 quality-passed wrong track 연결
- Wan/LoRA는 동결되고 checkpoint에는 adapter trainable parameter만 저장
- ranking weight는 v1의 `0.1`에서 oracle 식별 gate용 `1.0`으로 높임

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash train/run_wan_oracle_v2.sh
CUDA_VISIBLE_DEVICES=0 bash tools/run_wan_oracle_v2_gate.sh
```

### v2 최종 결과: spatial Wan 종료

v2도 `REJECT`였다. exact-zero로 adapter 무시 문제는 제거되어 normal output abs mean이 `0.010556`, zero는
정확히 `0.0`이 됐다. 그러나 normal weighted `0.650651`은 static `0.586220`보다 `+0.064431` 나빴다.
normal−zero Action은 `+0.030906`, normal−batch-roll Action은 `-0.010105`이지만 두 CI 모두 0을 포함했다.
batch-roll weighted `0.642211`은 normal보다 오히려 좋았다.

영상에서 zero Wan은 원래 장면에 없던 고채도 기둥·박스·물체를 만들었다. normal adapter는 이 환각을
줄여 zero보다 weighted가 유의하게 좋아졌지만, 정답 trajectory 때문이 아니라 장면을 덜 바꾸는
regularizer처럼 작동했다. 따라서 이 결과는 “신호가 약하다”가 아니라 **Wan의 patch-token additive
control이 이 데이터의 pixel motion renderer로 식별되지 않는다**는 반증으로 본다.

사전에 정한 규칙대로 spatial Wan과 action→track 학습을 종료한다. 다음은 다른 대형 video prior를 다시
붙이지 않는다. 먼저 GT에서 얻은 dense target→source flow로 첫 프레임을 warp했을 때 static을 이기는지
oracle gate를 수행한다. 통과할 때만 train clip의 action/appearance nearest-neighbor motion field를 query
첫 프레임에 보수적으로 전이하는 retrieval-flow 후보를 만든다. 이 경로는 source pixel을 유지하므로 지금까지
반복된 새 형상·배경 붕괴를 구조적으로 피한다. oracle warp도 static을 못 이기면 learned generation을 더
시도하지 않고 static/기존 incumbent를 제출 후보로 고정한다.

## 16. 방향 수정: 생성모델은 유지하고, 자체 adapter만 종료 (2026-08-03)

앞 절의 “다른 대형 video prior를 다시 붙이지 않는다”는 결론은 유사 대회와의 비교 범위를 너무 넓게
잡았으므로 철회한다. 종료된 것은 Wan patch-additive track adapter와 joint-space Cosmos adapter이지,
**action-conditioned generative world model 전체가 아니다.** flow warp는 renderer의 상한을 재는 진단 및
안전 기준선으로만 남기고 주력 방법으로 자동 승격하지 않는다.

근거는 다음과 같다.

- 1X Sampling 우승은 Wan2.2-TI2V-5B를 video/state-conditioned flow model로 바꾼 생성모델이다. 다만
  과거 5개 영상 프레임과 77-step 25D state를 받고, 23k step·effective batch 1024·32 B200으로 학습했다.
  우리 Wan v2의 2k×batch 8은 약 16k clip exposure로 우승 레시피의 약 1/1,472이며 입력도 단일 이미지라
  직접적인 재현이나 반증이 아니다.
- IRASim은 시작 관측과 action trajectory에서 16-frame robot video를 만드는 679M diffusion transformer이며,
  action-frame 정렬을 각 block의 Frame-Ada로 넣는다. 과제 형태가 현재 후보 중 가장 가깝다.
- 현재 workspace에는 공식 RT-1 Frame-Ada 300k checkpoint archive(23 GiB), SDXL VAE, SO-100 adapter 및
  train/generate 코드가 있지만 실제 checkpoint 추출·GPU smoke·500-step gate 결과는 없다. 이를 건너뛰고
  Cosmos/Wan 자체 adapter를 반복한 것이 절차상 빈틈이었다.

수정된 순서는 다음과 같다. 이 순서도 절대적이지 않으며 각 gate 결과가 우선한다.

1. 공식 IRASim RT-1 checkpoint에서 `frame_ada/0300000.pt`만 추출하고 weight compatibility를 감사한다.
2. 공개 weight zero-shot 1-sample generation으로 VAE/PNDM/17-frame 기하와 216개 환산 시간을 잰다.
3. 8-GPU 500-step SO-100 screen을 실행해 250/500의 normal/zero/reverse를 비교한다.
4. point-level action correctness가 생기면 2k까지 연장하고, 없으면 IRASim을 종료한다.
5. flow/retrieval은 IRASim과 독립된 static 이상의 보수적 백업 후보로만 평가한다.

참고: [1X 우승 기술보고서](https://arxiv.org/abs/2510.07092),
[IRASim 공식 코드](https://github.com/bytedance/IRASim),
[IRASim 프로젝트](https://gen-irasim.github.io/).

## 17. 실제 제출 기준 재설정과 masked spatial-control oracle gate (2026-08-04)

> **계획의 지위:** 아래 방향은 `0.1 이하`를 목표로 하는 현재 최선의 반증 순서이지 확정된 해법이 아니다.
> 특히 oracle flow는 미래 GT를 사용하는 train-only 진단이며 제출 방법이 아니다.

Dream 10k 제출의 실제 점수는 **0.28188**이었다. static 공개점수 약 `0.517` 대비 약 `0.235` 절대 개선,
약 45.5% 상대 감소이므로 Dream을 실패작으로 폐기하지 않고 공식 incumbent로 고정한다. 목표 `0.1`까지는
현재 점수에서 약 64.5%를 추가로 줄여야 하므로, 새 후보는 단순히 움직이는 영상을 만드는 수준이 아니라
Dream의 외형 보존과 video prior를 유지하면서 action identity와 pixel motion을 더 정확히 연결해야 한다.

이 결과는 로컬 판정도 수정한다. 같은 Dream 10k를 train-only holdout의 임시 cosine/MAE 식으로 계산하면
weighted `0.585128`, static `0.555346`으로 Dream이 `+0.029782` 나쁘게 나온다. 반면 실제 leaderboard에서는
Dream이 약 `-0.235` 좋았다. 따라서 local weighted는 동일 조건의 디버깅 보조값일 뿐 hard reject나 실제
점수 추정치가 아니다. 이후에는 다음 네 증거를 분리한다.

1. 직접 GT reconstruction/pixel motion gate
2. correct action 대 zero/batch-roll counterfactual gate
3. dataset-group holdout의 local feature 비교
4. 제한된 실제 leaderboard 제출

### VAP/MVA 재감사

[VAP](https://arxiv.org/abs/2508.13104)는 raw action을 그대로 AdaLN에 넣지 않는다. 알려진 카메라 파라미터와
로봇 state로 2D skeleton/mesh/depth를 렌더링하고, 3D trajectory encoder와 첫 14 DiT block의 ControlNet,
main-DiT LoRA, gripper-region loss 증폭을 사용한다. 논문의 mesh/depth가 skeleton보다 더 좋다는 결과도
희소점보다 외형을 포함한 dense control이 현재 과제에 더 적합하다는 근거다.

[Masked Visual Actions](https://arxiv.org/abs/2607.19343)의
[공식 코드](https://github.com/HadiZayer/masked-visual-actions)는 `PAI/Wan2.2-Fun-A14B-Control`에 LoRA를
학습하며 입력은 **렌더된 URDF 로봇 control video + 첫 실제 프레임 + prompt**다. 이전 Wan oracle 실험의
16개 sparse track을 작은 additive adapter로 넣은 방식과 동일하지 않다. 다만 우리 eval에는 camera extrinsic,
정확히 정합된 URDF render, servo offset이 없고 MVA의 rendering tool도 아직 공개되지 않았으므로 원 구현을
그대로 복제하는 것은 불가능하다.

### 구현한 낙관적 관문

`tools/oracle_dense_control_gate.py`는 dataset-group train holdout의 미래 GT에서 RAFT target→source flow를
구하고, 첫 프레임 픽셀을 inverse warp한다. cycle confidence가 낮은 가림/노출 영역과 작은 background flow는
첫 프레임을 그대로 복사해 새 형상과 배경 붕괴를 막는다. 동시에 warped RGB와 motion mask를 4채널
masked-control artifact로 저장한다. 이 control은 미래 GT flow를 썼으므로 action predictor 입력이나 제출물이
아니며, “정확한 pixel motion을 알고 있어도 reference preservation이 무의미한가”만 검사한다.

`tools/finalize_oracle_dense_gate.py`는 pixel L1, motion-region L1, cycle confidence와 공식 extractor 기반
local score를 결합하고 동일 holdout의 Dream 10k와 paired 비교한다. 단, 위 leaderboard/local 부호 불일치
때문에 pixel gate가 통과한 후보를 local weighted 하나만으로 종료하지 않고 `HOLD_LOCAL_METRIC_CONFLICT`로
보낸다. strict PASS도 dense representation만 승격하며 action→control이나 generator 성공을 뜻하지 않는다.

1-sample CPU smoke에서 output은 16×320×512 MP4, control은 `(4,16,160,256)`, backward flow는
`(15,2,160,256)`으로 확인했다. 해당 샘플의 full L1 감소는 14.0%, motion-region L1 감소는 23.9%,
cycle confidence는 0.931이었다. 시각적으로 로봇 외형과 배경은 유지되지만 팔 내부 blur/ghost가 있어 전체
8-sample 결과 전에 성공으로 판정하지 않는다.

```bash
# GPU 0 하나가 비었을 때: 8-sample oracle + local score + 같은 holdout Dream 10k + 최종 gate
CUDA_VISIBLE_DEVICES=0 bash tools/run_oracle_dense_gate.sh
```

판정 뒤 순서는 `oracle preservation → Dream dense-control 1-clip overfit → correct/wrong oracle control →
action-to-control predictor`다. 어느 단계에서든 외형은 좋아지지만 correct action 이점이 없으면 장기 학습하지
않는다. 실제 제출 후보는 Dream 10k를 항상 포함하며, 새 후보가 불확실할 때 Dream을 덮어쓰지 않는다.

### 8-sample oracle 결과와 범위가 수정된 STOP 판정

전체 gate는 `STOP_DENSE_WARP_PATH`였다.

| 항목 | 결과 |
|---|---:|
| median full-frame L1 감소 | 0.47% |
| median motion-region L1 감소 | 7.37% |
| median cycle confidence | 0.8830 |
| median control-mask 면적 | 29.14% |
| oracle weighted / static | 0.601028 / 0.586220 |
| oracle−static weighted CI95 | `[-0.00146,+0.03430]` |
| oracle−Dream-10k weighted CI95 | `[-0.06003,-0.02255]` |

oracle warp는 local Dream보다 유의하게 좋았지만 static을 이기지 못했고 pixel gate도 실패했다. montage에서
원인은 분명했다. 로봇이 첫 프레임의 위치를 벗어나거나 화면 밖에서 들어오는 clip에서는 미래 위치에 복사할
source pixel이 없다. target→source RAFT는 이 disocclusion을 복원하지 못해 팔/그리퍼를 중복시키거나
반투명한 구멍으로 만들었다. glove clip에서는 robot이 미래에 새로 등장하므로 source-only warp의 표현력
상한 자체가 부족했다. motion mask도 그림자·테이블 경계까지 포함해 중앙값 29%로 지나치게 넓었다.

따라서 종료되는 범위는 다음과 같다.

- 첫 프레임 픽셀만 이동하는 pure warp
- 같은 제약을 가진 nearest-neighbor/retrieval-flow 최종 renderer
- 현재 artifact를 그대로 Dream additive adapter에 넣는 실험

VAP/MVA의 generative completion까지 반증된 것은 아니다. 두 방법은 future skeleton 또는 rendered robot을
조건으로 가려졌다 새로 나타나는 surface를 생성한다. 그러나 eval에서 camera/URDF-aligned future render를
만들 수 없는 문제가 그대로이므로, 이를 해결하는 별도 action→visual-prompt gate 전에는 ControlNet 장기
학습을 시작하지 않는다.

당장의 우선순위는 실제 leaderboard에서 검증된 Dream 계열이다. 기존 step screen에서 step 8k는 후반 DINO
이탈 `0.1845`, step 10k는 `0.3149`였으므로 다음 실제 제출 후보는 8k checkpoint다. 동시에 Dream output의
움직임 영역만 사용하고 첫 프레임 배경을 고정하는 보존 후처리를 train-only holdout에서 screen한다. 이는
새 생성 백본을 다시 학습하지 않고 0.28188 incumbent의 외형 붕괴를 직접 줄이는 저비용 후보다.

### Dream preservation screen과 step-8k 비교

Dream 10k의 24개 고정 holdout에 세 후처리를 적용했다.

| 후보 | DINO | Video | Action | weighted |
|---|---:|---:|---:|---:|
| raw 10k | 0.188623 | 0.053763 | 1.281031 | 0.585128 |
| global blend 0.85 | 0.185395 | 0.054746 | 1.274450 | **0.581822** |
| motion mask 0.06 | **0.170438** | 0.059933 | 1.285647 | 0.583370 |
| motion mask 0.04 + blend 0.85 | 0.181482 | 0.061007 | 1.276582 | 0.583380 |

global 0.85가 weighted point estimate는 가장 좋았지만 raw 대비 `-0.003306`, CI95
`[-0.009846,+0.003363]`으로 불확실했다. mask 0.06은 DINO를 유의하게 개선했지만 Video를 유의하게
악화시켜 제출 후보로 자동 승격하지 않는다. 즉 background locking은 외형 보존에는 효과가 있지만 motion
feature를 함께 지우는 예상된 trade-off가 확인됐다.

별도로 step-8000을 같은 설정(seed 0, DDIM 50, eta 1, no-EMA)으로 생성했다.

| 후보 | DINO | Video | Action | weighted |
|---|---:|---:|---:|---:|
| raw 8k | **0.153839** | 0.063459 | 1.294889 | **0.583145** |
| raw 10k | 0.188623 | **0.053763** | **1.281031** | 0.585128 |

8k−10k DINO는 `-0.034784`, CI95 `[-0.062972,-0.009522]`로 유의하게 좋고, Video는
`+0.009696`, CI95 `[+0.000372,+0.020201]`로 유의하게 나쁘다. Action은 `+0.013858`, weighted는
`-0.001983`이지만 두 CI 모두 0을 포함한다. local proxy가 실제 10k/static 순위를 반대로 예측했으므로
이 결과로 8k의 공개점수를 단정할 수 없다. 다음 leaderboard probe는 변화가 작고 불확실한 후처리보다
후반 외형 안정성이 명확히 개선된 **raw 8k**를 우선한다.

## 19. SO100 rendered visual-action prompt 선행 게이트 (2026-08-04)

이 절의 결론은 절대적인 계획이 아니라, 현재 데이터와 구현으로 검증 가능한 범위를 기준으로 한
**진입/중단 결정**이다. VAP/MVA 자체가 잘못된 방법이라는 뜻은 아니다. 다만 이 계열은 미래 로봇 렌더가
실제 영상과 정확히 정렬되어야 한다는 전제가 있고, 그 전제를 먼저 통과하지 못하면 생성 백본을 학습해도
action-conditioned control 대신 잘못 놓인 형상을 학습하게 된다.

공식 SO-ARM100 URDF와 ManiSkill의 수정된 SO100 URDF/mesh를 비교했다. 공식 asset에는 관절 축·limit 문제가
보고되어 있어 렌더러에는 ManiSkill 수정본을 사용했다. challenge action은 `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` 순서이며, 앞의 다섯 축은 degree 계열, gripper는 linear
계열이다. 각 로봇의 절대 calibration 파일은 dataset metadata에 없으므로 action delta만 신뢰하고,

`q(t) = q0 + scale * (action(t) - action(0))`

로 놓은 뒤 첫 pose `q0`와 camera를 train-only track에 맞췄다. CPU FK/mesh renderer와 최적화 코드는
`train/so100_renderer.py`, `tools/render_so100_prompt.py`, `tools/fit_so100_track_alignment.py`에 있다.

### Gate A — track 기반 camera/q0 정렬

7개 식별 가능한 holdout에서 홀수 시점을 fit에 쓰지 않고 normal/zero/reverse/batch-roll action을 비교했다.

| 항목 | 중앙값 또는 승률 |
|---|---:|
| normal−zero reprojection | `-0.009826 px`, normal 승률 `4/7` |
| normal−reverse reprojection | `-1.938953 px`, normal 승률 `7/7` |
| normal−batch-roll reprojection | `-1.244074 px`, normal 승률 `6/7` |

시간 방향과 다른 trajectory를 구별하는 신호는 확인됐지만, zero 대비 이점은 사실상 0이고 overlay에서 전체
로봇 skeleton이 실제 로봇 밖에 놓였다. sparse track 일부나 움직이는 물체에 맞춰진 결과라서 deployable
visual control로 볼 수 없다. 결과는 `results/so100_track_alignment_gate.json`, overlay는
`diagnostics/so100_alignment/`에 저장했다.

### Gate B — VAP 방식의 공식 MatchAnything 보정

VAP 논문이 사용한 렌더→실영상 homography correction을 재현하기 위해 공식 MatchAnything ELoFTR weight를
CPU로 실행했다. 배경을 합성한 입력에서는 identity/background match만으로 가짜 PASS가 생길 수 있음을
발견했고, rendered foreground 경계 match와 target 실제 edge까지의 거리 조건을 추가했다.

| 결과 | 값 |
|---|---:|
| 유효 샘플 | 7 |
| strict PASS | 2 |
| pass rate | **28.6%** |
| 최종 판정 | **REJECT_MATCHANYTHING_ALIGNMENT** |

결과는 `results/matchanything_alignment_gate.json`, 시각화는 `diagnostics/matchanything_alignment/`에 있다.
cropped/off-screen robot, glove/occlusion, camera 다양성 때문에 rough URDF render를 안정적으로 보정하지
못했다. 알려진 camera parameter가 있거나 정확한 render가 준비된 VAP 데이터와 현재 challenge의 차이다.

따라서 **현재 상태에서는 MVA weight smoke와 GPU 학습으로 넘어가지 않는다.** 이 판단은 계산을 아끼기 위한
추측이 아니라 서로 다른 두 정렬 게이트의 실패에 근거한다. 다음 조건을 모두 만족하는 deployable
first-frame camera/q0 estimator가 생길 때만 이 경로를 다시 연다.

1. group holdout에서 strict mesh-edge alignment pass rate `>= 80%`
2. normal action이 zero와 batch-roll을 각각 `>= 80%` 샘플에서 이김
3. 추론 시 future GT frame, GT track, GT flow를 사용하지 않음
4. overlay에서 팔 전체가 실제 robot silhouette와 일치함

현재 실제 제출 incumbent는 Dream 10k의 `0.28188`이며, renderer gate 실패가 이를 대체하지 않는다. 다음
제출 probe는 이미 정한 raw Dream 8k를 유지한다. 새로운 생성 아키텍처는 위와 같은 입력 조건 검증 없이
장기 학습하지 않는다.

## 20. 생성 경로 재시작: faithful IRASim 우선 (2026-08-04)

후보 순서는 IRASim → HMA → 1X 레시피이며 한 후보의 선행 gate가 끝나기 전에 다음 모델을 동시에
개조하지 않는다. 기존 IRASim 500-step은 공식 checkpoint의 horizon, action input layer, EMA를 동일하게
유지하지 못했으므로 faithful 재현으로 간주하지 않는다.

새 IRASim 경로는 공개 RT-1 Frame-Ada backbone 297개 tensor를 모두 shape-exact하게 불러온다. 공식
`16 frames + 15 actions`, SDXL VAE, 256×320 direct resize, epsilon diffusion, PNDM 50-step,
`final_frame_ada=False`, action dropout 0.1과 EMA 0.9999를 보존한다. SO-100 차이는 공개 7D action MLP 앞의
zero-initialized 6→7 linear adapter 하나뿐이다. challenge source PNG는 GT frame 0과 정확히 같으므로 출력도
source를 첫 프레임으로 보존하고 15개 미래 프레임만 생성한다.

CPU 구조 audit는 PASS했다. 다음 두 gate는 모델 학습보다 먼저 실행한다.

1. SDXL VAE-only reconstruction: median PSNR `>=25 dB`, gradient-energy ratio `>=0.70`
2. public RT-1 native prior, zero action, 50-step 8개: SO-100 외형이 유지되는지 육안 및 DINO 확인

VAE가 실패하면 IRASim을 종료한다. VAE는 통과하지만 native prior가 다른 robot을 즉시 생성하면 장기
full-model 학습 전에 HMA tokenizer/native dynamics와 비교한다. 둘 다 통과할 때만 faithful SO-100
short training을 실행하며 normal/zero/batch-roll과 EMA/main을 함께 평가한다.

2026-08-04 재감사에서 challenge-kit의 기존 `pad=False`가 direct resize가 아니라 aspect-preserving
center crop이라는 추가 차이를 발견했다. IRASim 경로는 별도 `pad=None` 모드로 수정했고, 공식
`torchvision.resize(..., antialias=True)`와 synthetic non-square 입력에서 bit-exact함을 CPU audit에
추가했다. 학습과 추론은 이제 동일한 helper를 사용한다. VAE gate도 공식 inference처럼 posterior
`sample()`을 주 판정값으로 사용하며, `mode()` 결과는 참고값으로 함께 기록한다.

수정 후 8-sample VAE gate는 posterior sample PSNR median `33.9813 dB`, gradient-energy ratio median
`0.9247`로 **PASS_VAE**였다. 따라서 VAE-only 경계는 닫고 공개 RT-1 native-prior 50-step gate로
진행한다.

후속 결과에서 public prior는 절반 정지/절반 붕괴였고, faithful 4-GPU 500-step main branch는 8개 모두
source robot/background 형상을 잃었다. VAE와 구현 경계는 통과했지만 short transfer가 visual prior를
SO-100으로 옮기지 못했으므로 IRASim은 종료한다. 이 계획은 절대적이지 않지만, 현재 증거에서는 LR 변경이나
2k 연장보다 다음 후보의 tokenizer/native-prior gate가 정보 대비 비용이 낮다.

## 21. 과제 정합 아키텍처 재설계: source-anchored motion renderer (2026-08-04)

이 절 역시 절대적인 계획이 아니다. 다만 Dream, Wan, Cosmos, oracle-control, IRASim에서 반복된 공통 실패가
`action 신호가 전혀 없음` 하나가 아니라 **매 프레임 전체 장면을 재생성하면서 source robot의 외형과 배경을
잃는 것**이었으므로, 다음 주력 경로는 또 다른 full-frame generator가 아니라 이 실패를 구조적으로 제한하는
모델로 바꾼다. HMA는 공개 tokenizer reconstruction/native-prior를 확인하는 저비용 challenger로 남기되,
gate 통과 전부터 주력 장기학습으로 승격하지 않는다.

출발점은 임의의 custom flow model이 아니라 FOMM/TPS 계열의 검증된 image-animation 분해다. 첫 프레임
`I0`에서 appearance feature를 한 번 추출하고, action trajectory로 미래 part keypoint/affine motion을
예측한다. dense-motion network는 이를 feature flow와 occlusion map으로 변환한다. source에서 가져올 수 있는
영역은 warped source feature로 복원하고, 가려졌다 새로 보이는 영역만 작은 residual/inpainting decoder가
그린다. 최종 합성은 개념적으로 다음 제약을 갖는다.

`I_t = (1 - M_t) * I_0 + M_t * R_t`

실제 구현은 feature-space warp를 사용하되, 정지 배경에는 identity/copy 경로가 있고 residual 생성 범위는
motion/occlusion mask에 묶는다. 모든 미래 시점은 frame 0 reference feature에 다시 anchor하여 recurrent
재도색에 따른 누적 붕괴를 막는다. pure warp만으로는 disocclusion을 만들 수 없으므로 local generator를
제거하지 않으며, 반대로 local generator가 전체 화면을 다시 그리지 못하게 제한한다.

Action encoder는 SO-100 6D absolute command, step delta, time embedding을 함께 받고 source image에서 추정한
episode calibration latent와 결합한다. camera/URDF가 없는 현재 데이터에서는 explicit rendered skeleton을
주 입력으로 강제하지 않는다. VAP 경로는 strict alignment가 2/7에 그쳤으므로, RGB에서 self-supervised하게
학습되는 keypoint/part representation을 우선한다. train-only optical flow/point track은 auxiliary target으로
쓸 수 있지만 eval에서는 first frame과 action 이외의 future 정보를 절대 사용하지 않는다.

이번에는 장기 학습 전에 아래 순서를 모두 통과해야 한다.

1. **Renderer oracle gate:** train clip의 GT-derived motion/keypoint를 넣었을 때 target을 선명하게 복원하고
   robot/background identity를 보존한다. 실패하면 표현력이 부족하므로 action predictor를 붙이지 않는다.
2. **Single-clip action overfit:** first frame+action만으로 한 clip의 16 frame을 거의 복원하고 correct action이
   zero/reverse/batch-roll보다 명확히 좋아야 한다.
3. **Small group holdout:** 서로 다른 camera/group에서 static보다 개선되고, source-copy보다 motion score를
   얻으며, 외형 붕괴율이 사전 기준 이하인 경우에만 scale-up한다.
4. 위 gate를 통과한 뒤에만 full dataset 학습과 submission probe를 수행한다.

이 방향은 기존 `flow_world_model`의 단순/bounded warp를 재사용한다는 뜻이 아니다. 공식 FOMM/TPS의
keypoint→dense motion→multi-scale occlusion→source-feature fusion 구조를 먼저 fidelity 있게 옮기고,
driving-frame keypoint를 SO-100 action predictor로 대체하는 최소 변경부터 한다. 목표는 `0.28188`보다 좋을
것이라는 선언이 아니라, 지금까지 전혀 분리되지 않았던 **renderer 상한**, **action-to-motion 학습**,
**source 보존** 실패를 각각 독립 gate로 측정할 수 있게 만드는 것이다.

## 22. DreamZero-SO101 native-prior 전환 (2026-08-04)

이 전환 역시 절대적인 최종 계획이 아니다. FOMM은 single-clip에서 source 보존과 action 추종을 분리해
검증했지만 multi-clip robot detail이 평균화됐고, 별도 action-to-keypoint predictor와 renderer를 처음부터
함께 일반화시키는 비용이 남았다. 반면 공개 `Vizuara/dreamzero-so101-lora`는 동일한 6-DOF SO-101을
Wan2.1-I2V-14B 위에서 72K step 학습한 adapter이므로, 새 모델을 장기학습하기 전에 공개 native visual prior의
실제 품질을 확인할 가치가 가장 크다.

공식 raw checkpoint 저장소의 20K checkpoint 안에서 공개 LoRA 폴더에 없던 `experiment_cfg/conf.yaml`과
`metadata.json`을 복구했다. 이에 따라 다음 입력 규칙을 근사가 아니라 학습 설정 그대로 확인했다.

- 입력은 640x480의 `front`, `gripper`, `top` 세 view이며 각각 0.95 center crop 후 176x320으로 resize한다.
- 세 view는 352x640의 2x2 canvas(front TL, gripper BL, top TR, black BR)로 합친다.
- state는 absolute 6D joint angle을 SO-101 state q01/q99로 정규화한다.
- action target은 absolute가 아니라 chunk 시작 state를 뺀 24-step relative joint angle이며 별도의 action
  q01/q99로 정규화한다.
- 공개 adapter의 action/state encoder tensor는 embodiment category가 1개이므로 내부 ID 0이 실제 SO-101이다.
- 논문/학습 config의 native inference는 4 denoising step을 사용한다. repository의 16-step acceleration mask를
  그대로 둔 채 step 수만 4로 줄이면 마지막 requested step이 캐시 재사용으로 바뀌므로 smoke에서는 네 step을
  모두 실제 DiT forward로 실행한다.

Challenge는 single front RGB와 16-step SO-100 absolute target만 제공한다. uploader별 absolute calibration이
다르며, holdout의 첫 action을 SO-101 state q01/q99에 직접 넣으면 첫 8개에서 dimension의 16.7--83.3%가
clamp 경계에 붙었다. 따라서 native video-prior gate에서는 잘못된 absolute mapping을 가장하지 않고 SO-101
metadata center(normalized zero)를 state로 사용한다. 이것은 action-conditioned submission 후보가 아니라
**공개 adapter가 challenge RGB에서 robot appearance와 자연스러운 motion을 생성할 수 있는지**만 검사한다.

`train/generate_dreamzero_so101_native.py`는 첫 challenge RGB를 front quadrant에 두고 나머지 view를 black으로
채운다. causal latent block 두 번(첫 reference+2 latent, 다음 2 latent)을 이어 17 RGB frame으로 decode하고,
front quadrant를 원 해상도로 복원해 source frame 0 + 미래 15 frame을 저장한다. 동시에 352x640 composite도
`_composite/`에 저장해 crop 실수와 model collapse를 구분한다. 이 경로는 supplied future action을 사용하지
않는다고 benchmark JSON에 명시한다.

다음 판단 순서는 고정한다.

1. native 1-sample front/composite에서 SO-101 arm과 배경이 16 frame 동안 식별 가능해야 한다.
2. 통과 시 challenge action은 absolute offset이 아니라 `A_t - A_0`로 바꿔 공식 relative-action q01/q99에
   넣는다. clean action latent를 고정하고 video latent만 denoise하는 action-inpainting을 구현한다.
3. native가 이미 붕괴하면 이 LoRA를 challenge에 추가 학습하거나 제출하지 않는다. Wan14B base I2V와 LoRA
   on/off를 같은 seed로 비교해 LoRA domain mismatch인지 base/canvas 문제인지 먼저 분리한다.
4. action-inpainting은 normal/zero/reverse/batch-roll에서 normal 우위와 native 대비 외형 보존을 모두 통과한
   경우에만 16-sample holdout 및 제출로 확장한다.

따라서 DreamZero가 논문상 유사하다는 이유만으로 성공을 보장하지 않는다. SO-101/30fps/3-view 학습과
SO-100/6fps/single-view challenge 사이의 domain·camera·calibration 차이가 실제로 남아 있으며, 이번 native
gate는 바로 그 빈틈을 장기학습 전에 측정하기 위한 단계다.

## 23. Wan2.1-I2V-14B source-conditioned spatial action control (2026-08-05)

이 절은 현재까지의 실패와 새로 얻은 native-prior 증거를 반영한 **검증 가능한 다음 후보**이지 절대적인
계획이나 성능 보장이 아니다. 특히 로컬 feature score가 실제 Dream 10k 제출 점수 `0.28188`의 순위를
정확히 예측하지 못한 전례가 있으므로, 아래 gate는 잘못된 모델을 빨리 버리기 위한 조건이지 공개 점수의
추정식이 아니다.

### 전환 근거

DreamZero-SO101 native 경로는 첫 프레임 뒤 blur와 좌상단 노이즈만 남고 움직이지 않았다. 공개 weight가
학습한 입력은 SO-101 3-view, 30 fps, 24-step relative action/state인데 challenge는 SO-100 single-view,
6 fps, 16-step absolute command이므로, 이를 그대로 재사용하는 것은 같은 embodiment가 아니다. 반면 LoRA나
action adapter가 전혀 없는 vanilla `Wan2.1-I2V-14B-480P`를 challenge `val_000009`에 실행했을 때 robot이
charger 쪽으로 명확히 움직였고 형태도 이전 후보보다 잘 유지됐다. 50-step 640x480 실행은 47.16초,
peak 43.57 GiB였으며 마지막 frame의 sharpness ratio는 0.687이었다. global zoom과 과도한 scene edit은
남았지만, 최소한 **backbone/VAE가 이 도메인의 움직이는 robot video를 생성할 수 있음**은 확인됐다.

따라서 이번 변경은 Wan을 새로 구축하거나 전체 14B를 fine-tune하지 않는다. native I2V prior는 보존하고,
액션을 source image의 공간 위치에 결합하는 작은 branch와 저율 LoRA만 학습한다. VAP의 핵심인 spatial visual
control과 AVID류의 pretrained video adapter 원칙을 취하되, VAP의 URDF skeleton을 그대로 쓰지는 않는다.
앞선 공식/ManiSkill SO100 render 정렬이 strict `2/7`에 불과했으므로 부정확한 skeleton은 오히려 틀린 위치를
강제하기 때문이다.

### 구현

구현 파일은 다음과 같다.

- `third_party/DiffSynth-Studio/diffsynth/models/wan_video_dit.py`:
  `SourceConditionedSpatialActionAdapter`
- `third_party/DiffSynth-Studio/diffsynth/pipelines/wan_video.py`: 4개 transformer depth의 residual injection
- `train/wan_spatial_action_dataset.py`: group holdout, hybrid action, motion mask, static counterexample
- `train/train_wan21_spatial_action.py`: paired correct/counterfactual flow-matching loss
- `train/generate_wan21_spatial_action.py`: challenge inference와 zero/batch-roll ablation
- `tools/audit_wan21_spatial_action.py`: GPU를 쓰지 않는 구조 계약 검사
- `train/run_wan21_spatial_action.sh`, `tools/run_wan21_spatial_action_gate.sh`: 재현 가능한 실행 경로

액션 표현은 normalized absolute 6D, 첫 command 대비 delta 6D, step delta 6D를 합친 18D hybrid이다. 16개
command를 causal temporal encoder가 Wan의 미래 latent 4개로 정확히 압축한다. 첫 프레임 Wan VAE latent
16채널에 screen coordinate를 붙여 robot/context의 위치를 추정하고, action별 FiLM으로 공간 feature를
변조한다. 이 residual을 block `0,10,20,30` 뒤에 넣는다. 마지막 projection은 weight/bias 모두 정확히 0으로
초기화되어 step 0의 출력은 vanilla Wan과 bit-level functionally 동일하다. 체크포인트 옆
`control_config.json`에 action mode와 injection layer를 저장하며 추론은 이를 자동으로 읽고 충돌하는 CLI
설정을 거부한다.

한 training sample에서 correct와 time-reverse action을 동일 clean latent, noise, timestep으로 B=2 한 번에
forward한다. 기본 reconstruction은 GT motion mask에 3배 가중하며, correct denoising error가 wrong보다
margin만큼 작지 않으면 ranking loss를 더한다. 데이터의 15%는 첫 프레임을 17번 정확히 반복하고 constant
action을 correct, 실제 움직임 action을 wrong으로 두어 고정 camera/zero motion을 직접 학습한다. 이 설계는
이전처럼 normal/zero output이 같아도 pixel loss만 내려가는 shortcut을 줄이기 위한 것이다.

CPU audit 결과는 `PASS_STRUCTURE`다. 실제 121개 group, 10,697개 base clip에서 `(17 frames, 16x18 actions)`
계약과 static exact-repeat를 확인했다. adapter init maximum은 정확히 `0.0`이고, 첫 output-projection update
뒤 action/source encoder 양쪽 gradient가 0보다 크며, reverse action sensitivity도 0보다 컸다. paired loss도
finite scalar와 action gradient를 통과했다. 이는 구현 연결의 증거일 뿐 생성 품질의 PASS는 아니다.

### 실행 및 중단 기준

긴 학습으로 바로 가지 않고 아래 순서를 지킨다.

```bash
# 0. CPU-only 구조/데이터/model-file 감사
PHASE=audit bash train/run_wan21_spatial_action.sh

# 1. H100 8장, 5-step 실행 smoke
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PHASE=smoke NPROC=8 bash train/run_wan21_spatial_action.sh

# 2. smoke가 정상일 때만 250-step screen
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PHASE=gate250 NPROC=8 bash train/run_wan21_spatial_action.sh

# 3. 250 checkpoint를 8개 holdout에서 normal/constant/batch-roll 비교
# GEN_NPROC=8이면 각 H100이 한 sample을 맡고, 14B를 GPU마다 독립 로드한다.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CHECKPOINT=open/baseline/outputs/wan21_spatial_action_250/step-250.safetensors \
LIMIT=8 GEN_NPROC=8 bash tools/run_wan21_spatial_action_gate.sh
```

5-step smoke의 목적은 품질 판정이 아니라 14B DDP, checkpoint, gradient graph가 실제 H100에서 끝까지 도는지
확인하는 것이다. 250-step에서는 먼저 영상으로 robot identity, 배경, camera zoom을 보고, 다음 조건을 함께
판정한다.

1. normal이 constant-action과 batch-roll보다 action component에서 낮아야 한다.
2. `normal_minus_static < 0`이어야 하며 Dream incumbent 대비 악화가 크면 승격하지 않는다.
3. 일부 sample만 잘 되고 나머지가 blur/morphing이면 평균 하나로 통과시키지 않는다.
4. training log의 `correct_minus_wrong`가 지속적으로 음수가 되지 않으면 action branch 학습 실패로 중단한다.
5. 위 조건에 발전 가능성이 있을 때만 `PHASE=train500`으로 500 step까지 잇는다. 2k 연장은 자동 계획이 아니다.

즉 이번 후보가 이전 방법보다 근거는 강하지만 아직 “확실히 0.1 이하”인 방법은 아니다. 확실해진 부분은
14B native generator가 challenge image에서 실제 motion을 만들 수 있다는 것, action/source/loss 연결이 구조
감사를 통과했다는 것, 그리고 실패 시 어느 경계에서 중단할지가 코드로 고정됐다는 점이다.

## 24. Dense 결합 중단과 HMA tokenizer-first gate (2026-08-05)

이 단계 역시 절대적인 계획이 아니다. Wan2.1 spatial-action 250-step의 정상 action은 zero와 batch-roll보다
좋지 않았고, 영상 일부에서는 action과 무관한 human hand/object hallucination이 발생했다. 이에 action을 잘
따라간 FOMM/dense decoder 출력을 Wan 영상과 합치는 방안을 검토했지만, 구현 전에 이미 존재하는 oracle
결과를 다시 감사했다.

- future-GT RAFT flow를 직접 사용한 dense warp는 full-frame L1을 중앙값 `0.47%`, motion 영역을 `7.37%`
  개선하는 데 그쳤고 `STOP_DENSE_WARP_PATH`였다.
- 한 clip에 overfit한 FOMM oracle은 PSNR `38.22 dB`로 통과했지만, 같은 camera/domain의 32 clips를 학습하고
  held clips에서 평가한 renderer는 PSNR `22.07 dB`, static 대비 `+3.14 dB`로 gate를 통과하지 못했다.
- future-GT point track을 넣은 Wan2.2 v2도 static보다 나았지만 Dream incumbent보다 유의하게 나빴다.

따라서 현재 dense 출력과 Wan RGB를 단순 blend/inpaint하는 것은 action predictor의 문제가 아니라
renderer의 disocclusion/일반화 실패를 다시 감추는 조합이라 판단해 중단한다. single-clip 성공은 구조적
상한이 아니라 scene memorization이 가능한지를 보여준 결과로 해석한다.

다음 후보 HMA는 전체 학습으로 바로 가지 않는다. 공개 로봇 영상용 MagViT tokenizer가 현재 SO-100의 가는
관절과 그리퍼를 quantized token round-trip 뒤에도 보존하는지 먼저 확인한다. `tools/audit_hma_magvit.py`는
공식 short-side 256 + center crop과 LFQ index decode를 그대로 사용하며, train-only 8개 group holdout의
모든 frame을 복원한다. median PSNR `>=24 dB`, sample별 p10의 중앙값 `>=20 dB`, gradient-energy ratio
`0.70--1.30`을 동시에 만족할 때만 HMA pretrained dynamics/action adaptation으로 넘어간다. 이 수치는
leaderboard 점수 추정이 아니라 tokenizer 때문에 발생하는 blur를 dynamics 학습 전에 차단하는 gate다.

8개 train-only holdout의 128 frames에서 실제 LFQ index round-trip을 CPU로 실행한 결과 median PSNR은
`28.29 dB`, sample별 p10 PSNR의 중앙값은 `27.62 dB`, gradient-energy ratio는 `0.792`로
`PASS_HMA_MAGVIT`였다. 따라서 HMA에서 이후 영상이 무너지면 공개 discrete tokenizer 자체보다 dynamics
rollout 또는 SO-100 action adaptation을 먼저 의심한다.

공개 `hma-base-disc` dynamics는 12-frame window와 4 prompt frames를 요구하지만 challenge에는 source 한 장만
있다. `train/generate_hma_native.py`의 다음 gate는 source token을 네 번 정확히 반복하고 action을 주지 않은
상태로 8 future frames를 생성한 뒤, 마지막 네 frame을 다음 window의 prompt로 사용해 7 frames를 더 만든다.
최종 영상은 supplied source 1장 + generated future 15장이다. HMA center crop 바깥은 source RGB를 그대로
복사한다. 이 gate에서 action을 생략한 이유는 checkpoint 안에 우연히 6D인 Berkeley Fanuc action stem이
있더라도 SO-100 absolute joint target과 의미·통계가 전혀 다르기 때문이다. native 결과가 로봇 형태와
자연스러운 움직임을 유지할 때만 SO-100 6D projector를 새로 만들고 group-holdout fine-tuning으로 넘어간다.

실제 native 감사에서 action token을 완전히 생략하면 첫 future frame부터 원본이 녹색/검정 blob으로 바뀌었다.
공개 checkpoint는 `drop_action_ratio=0`이고 64 action tokens를 항상 사용하므로, 이는 올바른 zero-action이
아니었다. 유일한 6D 공개 domain인 Berkeley Fanuc의 mean action을 넣어 정규화 후 정확히 zero가 되게 하자
로봇과 배경은 유지됐지만 모든 future frame이 동일해졌다. 이어 각 관절에 시간 변화 `0.75 std` sinusoid를
주어도 `temporal_std ≈ 7e-8`로 동일했다. 따라서 tokenizer의 `PASS_HMA_MAGVIT`와 별개로 공개 dynamics가
challenge source에서 action-sensitive하지 않다고 판정하여 `REJECT_HMA_DYNAMICS`로 종료한다. SO-100 action
projector만 짧게 학습하는 단계로 승격하지 않는다.

## 25. FOMM faithful multi-clip renderer 최종 판정 (2026-08-05)

이 판정 역시 전체 연구 방향이 절대적이라는 뜻은 아니지만, 현재 형태의 FOMM renderer를 더 학습하거나
action predictor를 결합할 근거는 사라졌다. 기존 32-clip checkpoint에서 공식 FOMM의 perceptual loss와
affine/TPS equivariance value·Jacobian loss를 복구한 뒤 500 step을 추가 학습하고, 학습에 쓰지 않은 8 clips의
future driving frame을 oracle로 넣어 renderer 상한을 다시 측정했다.

- median PSNR: `21.9445 dB` (기준 `>=22 dB` 미달)
- median static PSNR: `18.5566 dB`
- median gain over static: `+2.8459 dB` (기준 `>=4 dB` 미달)
- median background MAE: `0.00202` (보존 기준 통과)
- median gradient-energy ratio: `0.8420` (선명도 기준 통과)
- median motion-gradient ratio: `0.7753` (motion 선명도 기준 통과)
- 최종 판정: `REJECT_MULTICLIP_RENDERER_ORACLE`

즉 배경 copy와 전역 선명도는 문제가 아니지만, unseen clip의 실제 robot articulation/disocclusion을 static
source보다 충분히 잘 복원하지 못한다. 더 중요한 점은 faithful loss 추가 전의 static 대비 이득 `+3.14 dB`보다
후속 checkpoint의 `+2.85 dB`가 낮아졌다는 것이다. 병목은 action predictor가 아니라 renderer의 multi-clip
일반화이며, 이 상태에서 action-to-keypoint를 붙이면 oracle보다 어려운 문제를 푸는 셈이다.

따라서 현재 FOMM 계열에서는 다음을 중단한다.

- 동일 checkpoint의 step 연장 또는 loss weight 탐색
- FOMM keypoint에 SO-100 action predictor 결합
- FOMM/dense 출력을 Wan 영상과 단순 blend 또는 inpaint

이 결과는 source-anchoring 원리 자체를 반증하지는 않지만, 현재 데이터와 예산에서 FOMM의 learned
keypoint/dense-motion renderer가 제출 후보가 아니라는 점은 반증한다. 실제 제출 incumbent는 계속 Dream 10k
`0.28188`이며, 후속 제출 후보는 별도 held-out gate와 완성 영상 검사를 통과한 경우에만 만든다.

## 26. OSCAR 이전 SO-100 skeleton calibration 3-way gate (2026-08-05)

이 계획도 절대적인 최종안이 아니다. 세 방법 중 하나가 source 정합과 action counterfactual gate를 통과할
때만 OSCAR/Cosmos video generation으로 연결한다. 세 방법이 모두 실패하면 생성모델 학습량을 늘리는 대신
camera/kinematics 조건 생성 경로를 다시 설계한다. 특히 train future frame이나 `observation.state`를 쓰는
결과는 제출 파이프라인으로 승격하지 않는다.

공식 OSCAR류 조건은 알려진 URDF joint state와 camera로 black background 위에 skeleton을 렌더링하지만,
challenge는 source RGB 한 장과 16x6 absolute target command만 제공한다. 이전 SO-100 projection이 strict
`2/7`이었던 이유는 이 camera와 uploader별 absolute zero를 근사했기 때문이다. 따라서 장기 생성학습 전에
아래 세 보정 경로를 동일한 SO-100 URDF와 동일한 OSCAR RGB skeleton 포맷으로 비교한다.

1. `multiframe`: RobotArena식 train-only teacher다. 실제 `observation.state`와 RAFT motion track을 사용해
   joint zero와 weak-perspective camera를 함께 최적화한다. sparse track만 쫓는 해를 막기 위해 source robot
   mask 안의 track만 사용하고, source box·mask centerline 정합을 목적함수에 같이 둔다. `deployable=false`다.
2. `singleframe`: RoboPose식 source-only render-and-compare 경로다. local Grounding-DINO의 robot box와 RGB
   edge에 URDF chain을 맞춘다. future RGB/state를 읽지 않으므로 제출 시 사용 가능하다.
3. `mask`: EasyHeC/CtRNet식 source-only silhouette refinement 경로다. 같은 detector box에서 GrabCut robot
   mask를 만들고 mask centerline·inside fraction·box에 single-frame 결과를 재정합한다. 현재 단계는 per-image
   optimization이며, 통과 후에만 빠른 pose regressor의 teacher로 사용한다.

시간 정렬은 코드에서 하나로 고정했다. 제출 영상 frame 0은 supplied source pose이며 frame 1..15만
`action[0..14]`의 결과다. `action[15]`는 저장 horizon 밖의 context다. eval에서 source
`observation.state`를 사용할 수 없으므로 deployable 두 경로는 train artifacts에서 계산한 median
`state[0]-action[0]`만 사용해 source command lag를 추정한다. 이 통계는 future eval 정보를 포함하지 않는다.

gate는 서로 다른 두 문제를 분리한다.

- source geometry: projected chain의 frame 내부 비율 `>=0.85`, dilated robot mask 내부 비율 `>=0.50`,
  mask centerline 중앙 거리 `<=12 px`, detector box IoU `>=0.15`를 모두 요구한다.
- action geometry: offline에서만 실제 state trajectory를 FK한 고정-topology projection과 비교한다. normal이
  zero보다 최소 60%, reverse와 다른 sample의 batch-roll보다 각각 최소 75% sample에서 낮은 RMSE여야 한다.
  RAFT assignment score도 기록하지만 sparse correspondence shortcut 때문에 단독 합격 기준으로 쓰지 않는다.

1-sample CPU wiring smoke에서 세 방법 모두 source alignment gate를 통과했다. 특히 이전에 robot 밖으로
벗어났던 multiframe skeleton은 mask/box 제약 뒤 robot 위에 놓였다. state projection에서는 normal-minus-zero가
multiframe `-0.398 px`, singleframe `-2.373 px`, mask `-2.122 px`로 올바른 방향이었다. 이 수치는 sample 하나와
낮은 optimizer iteration의 결과이므로 품질 PASS나 OSCAR 성능을 의미하지 않는다.

재현 명령은 다음과 같다. 기본은 CPU detector이며, GPU 사용은 실행자가 명시적으로
`DETECTOR_DEVICE=cuda:0`을 줄 때만 발생한다.

```bash
# 빠른 wiring 확인
PHASE=smoke DETECTOR_DEVICE=cpu bash tools/run_oscar_skeleton_gate.sh

# 8개 held-out clip 정식 geometry gate
PHASE=gate DETECTOR_DEVICE=cpu bash tools/run_oscar_skeleton_gate.sh
```

결과는 `results/oscar_so100_skeleton_gate.json`, 영상과 source/skeleton/overlay 진단 이미지는
`diagnostics/oscar_so100_skeletons_gate/{multiframe,singleframe,mask}`에 저장된다. 승격 우선순위는
`mask` deployable PASS, `singleframe` deployable PASS, multiframe teacher-only PASS 순이다. multiframe만
통과하면 teacher를 source-only predictor로 distill하는 별도 단계가 필요하며 곧바로 제출에 사용하지 않는다.

### 8-sample 첫 실행의 false PASS와 gate 수정

첫 8-sample 결과는 수치상 세 방법 모두 PASS였지만 영상 감사에서 이를 철회했다. holdout 3, 4, 6의
Grounding-DINO 최고-confidence box는 로봇 옆의 검은 장갑이었고, 5는 작업판 대부분을 robot box로 잡았다.
7은 실제 state/action이 전 차원 완전 정지인 clip이지만 화면 경계에서 일부만 보이는 robot을 full-chain
box loss로 압축했다. 기존 source alignment는 detector가 만든 mask에 다시 맞는지를 측정했으므로 잘못된
물체에 정확히 맞아도 통과했고, state projection은 상대 action 변화만 검증하여 q0/camera 오정합을 잡지
못했다.

수정 selector는 robot/base/gripper와 glove/hand를 하나의 Grounding-DINO query에서 함께 검출하고,
hand/glove label과 겹치거나 과도하게 큰 positive proposal에 벌점을 준다. 수정 후 3, 4, 6은 좌측의 실제
로봇, 5는 좌측 경계의 robot-arm 영역, 7은 우측 경계의 실제 로봇을 선택했다. 그러나 partial/cropped robot에
full URDF chain을 per-image로 맞추는 문제는 여전히 ill-posed이며 5에서는 GrabCut 자체가 실패했다.

따라서 정식 gate는 8개 모두 결과가 존재해야 하며, dynamic clip에서 normal RAFT track RMSE median
`<=10 px`, `12 px` 미만 비율 `>=75%`, reverse와 batch-roll 대비 normal 승률 각각 `>=60%`를 추가로
요구한다. 이는 state-relative 변화만 맞고 source의 실제 로봇 위치가 틀린 false PASS를 차단한다. 이 수정
전의 `results/oscar_so100_skeleton_gate.json` PASS는 유효한 승격 근거로 사용하지 않는다.

## 27. Motion-first + generative completion 문헌 재검증 (2026-08-05)

이 결론 역시 절대적인 최종 아키텍처가 아니다. 다만 문헌을 다시 대조한 결과, `source appearance를
motion으로 warp하고 occlusion/disocclusion만 생성모델이 복원`하는 큰 구조는 충분한 선행 근거가 있다.
반면 우리 single-clip FOMM의 10개 unsupervised keypoint를 그대로 정밀 robot control로 취급할 근거는 없다.

- TPSMM은 source feature를 optical flow로 warp하고 multi-resolution occlusion mask를 이용해 missing region을
  inpaint한다. 논문 자체도 FOMM류의 큰 pose gap에서 occlusion 영역이 커질수록 작은 inpainting network에
  과도하게 의존해 품질이 낮아진다고 분석한다: https://arxiv.org/abs/2203.14367
- Motion-I2V는 I2V를 pixel-trajectory motion predictor와 video renderer로 분리하고, predicted trajectory로
  reference feature를 미래 frame에 전달한다. ablation에서 motion field가 없는 경우보다 안정적이었고 단순
  addition보다 attention fusion이 가장 일관적이었다: https://arxiv.org/abs/2401.15977
- MOFA-Video는 sparse trajectory/landmark를 dense flow로 확장한 뒤 source의 multi-scale feature를 warp하여
  frozen I2V diffusion에 guidance로 넣는다. 이는 우리가 의도한 `FOMM motion proposal + pretrained
  generative completion`에 가장 가까운 공개 구조다: https://arxiv.org/abs/2405.20222

최신 robot action-to-video 연구도 raw action token보다 pixel-aligned geometry를 선호하지만, 그 조건은 우리
초기 skeleton 근사보다 훨씬 강하다.

- VAP은 known camera로 action을 mesh/depth/skeleton으로 렌더링하고 MatchAnything으로 실제 영상과 어긋난
  episode를 교정 또는 제거한다. ControlNet이 가장 중요한 모듈이었고, ablation에서 skeleton은
  PSNR `21.26`, ST-IoU `0.450`인 반면 mesh는 `23.51/0.586`, depth는 `23.41/0.581`이었다:
  https://arxiv.org/abs/2508.13104
- OSCAR는 2.16M source video에서 약 180k episode만 남기는 강한 curation을 수행했고, 평가 clip은
  end-effector가 전체 horizon 동안 화면 안에 있는 경우만 선택했다. Cosmos-Predict2.5-2B를 15k-step
  robot stage 이후 robot+human mixture로 추가 학습했다. 따라서 cropped/occluded SO-100 source에
  per-image skeleton을 맞추는 현재 설정과 동일하지 않다: https://arxiv.org/abs/2606.04463
- Kinema4D는 URDF kinematics를 4D pointmap으로 투영하고 201,426개의 high-quality 4D annotation으로
  generative model을 학습한다: https://arxiv.org/abs/2603.16669
- Robot-Factored World Models는 own controller로 action을 nominal trajectory로 rollout하고 URDF rendering,
  camera-aware RGB/depth, end-effector depth를 함께 제공한다. 미래 logged state를 쓰지 않으면서 action
  realization을 world model 밖으로 분리한다: https://arxiv.org/abs/2607.22535

우리 challenge는 eval에서 camera calibration, source `observation.state`, depth가 없고 source RGB와
absolute 16x6 target command만 있다. uploader별 zero offset과 cropped robot도 존재한다. 따라서 VAP/OSCAR/
RoFacto의 정확한 geometry interface를 만들기 위한 inverse problem이 별도로 남으며, Grounding-DINO box와
orthographic 12-DoF fit은 이를 해결하지 못했다.

이에 따라 다음 주력 후보는 FOMM RGB decoder의 연장이 아니라 **MOFA/Motion-I2V형 robot motion interface**다.

1. train future RGB에서 RAFT dense flow, track, robot motion region을 teacher condition으로 만든다.
2. source RGB feature와 18D absolute/anchor-delta/step-delta action을 받아 15-frame dense flow,
   occlusion mask, semantic part heatmap을 예측한다. FOMM의 clip-dependent keypoint index를 직접 target으로
   쓰지 않는다.
3. predicted flow로 source multi-scale feature를 warp하고, pretrained I2V의 ControlNet/attention branch가
   missing region과 contact response만 생성한다. motion mask 밖은 source identity path로 합성한다.
4. 첫 gate에서는 action predictor를 쓰지 않고 train-only oracle RAFT flow를 넣어 pretrained refiner의
   renderer 상한을 측정한다. oracle에서도 DINO/video가 무너지면 이 경로를 중단한다.
5. oracle renderer 통과 뒤에만 action-to-flow를 학습하고 group holdout에서 normal/zero/reverse/batch-roll과
   official frozen action extractor loss를 비교한다.

이 후보는 과거 `flow_world_v3` 반복이 아니다. 그 모델은 작은 custom CNN이 bounded flow와 RGB residual을
처음부터 함께 예측했으며 pretrained video prior나 multi-scale warped-feature attention이 없었다. Wan spatial
adapter도 raw action residual을 DiT에 넣었을 뿐 pixel-aligned flow teacher가 없었다. 새 gate의 핵심은
**oracle motion과 generative renderer를 먼저 분리 검증**하는 것이다.

## 28. Wan2.1 oracle motion-field refiner 구현과 중단 기준 (2026-08-05)

이 구현도 절대적인 제출 계획이 아니다. 특히 future RGB에서 만든 RAFT field를 사용하므로 현재 결과는
**제출 불가능한 train-only upper bound**다. 목적은 action-to-motion predictor를 새로 학습하기 전에
`정확한 pixel motion이 주어지면 pretrained I2V가 SO-100 articulation과 disocclusion을 복원할 수 있는가`만
분리해서 판정하는 것이다.

로컬 환경에는 Wan2.1/2.2의 VACE 실행 코드는 있지만 VACE 또는 MOFA pretrained control weights가 없다.
14B VACE branch 전체를 처음부터 학습하는 것은 현재 64-clip gate와 시간 예산에 맞지 않는다. 따라서 이미
source preservation이 가장 나았던 `Wan2.1-I2V-14B-480P`를 완전히 freeze하고 약 6.3M parameter의
zero-initialized motion branch만 학습한다. 이는 official MOFA/VACE 재현이라고 부르지 않고
`MOFA-inspired oracle motion refiner`로 구분한다.

control은 RGB target 자체가 아니라 다음 7채널, 17-frame residual field다.

1. source를 target 좌표로 inverse warp한 RGB와 source RGB의 차이 3채널
2. width/height로 정규화한 target-to-source backward RAFT flow 2채널
3. forward-backward cycle visibility 1채널
4. motion support 안의 occlusion/disocclusion uncertainty 1채널

frame 0은 정확히 0이고, 16 future controls는 Wan VAE 시간 구조와 동일하게 4-frame씩 묶여 4 future latent로
압축된다. control conv, source VAE feature localization, transformer block별 독립 zero projection은 모두
spatial token을 유지한다. control path는 bias-free이고 source feature는 control feature에 곱으로만 작용하므로
학습 뒤에도 `f(zero_control)=0`이 구조적으로 보장된다. base Wan이나 LoRA는 이 gate에서 학습하지 않는다.

관련 파일은 다음과 같다.

- `tools/prepare_oracle_motion_field_gate.py`: 64 train + 8 group-holdout clip의 bidirectional RAFT field 생성
- `train/wan_oracle_motion_field_dataset.py`: correct/zero/cross-dataset field paired dataset
- `train/train_wan21_oracle_motion_refiner.py`: 동일 noise/timestep의 correct-vs-counterfactual denoising loss
- `train/generate_wan21_oracle_motion_refiner.py`: normal/zero/batch-roll oracle inference
- `tools/compare_oracle_motion_refiner_gate.py`: strict paired CI gate

실행은 자동으로 GPU를 점유하지 않는다. 아래 명령을 실행자가 직접 수행한다.

```bash
# 1. RAFT teacher artifact 생성: GPU 0 하나만 사용
CUDA_VISIBLE_DEVICES=0 PHASE=prepare \
  bash train/run_wan21_oracle_motion_refiner.sh

# 2. GPU 없이 artifact shape + structural-zero 검사
PHASE=audit bash train/run_wan21_oracle_motion_refiner.sh

# 3. 품질판정이 아닌 DDP/gradient/checkpoint 5-step smoke
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 PHASE=smoke \
  bash train/run_wan21_oracle_motion_refiner.sh

# 4. smoke가 정상 종료된 경우에만 250-step, 8 H100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 PHASE=gate250 \
  bash train/run_wan21_oracle_motion_refiner.sh

# 5. 8개 oracle normal/zero/batch-roll 생성 + official local feature gate
CUDA_VISIBLE_DEVICES=0,1,2,3 GEN_NPROC=4 \
  bash tools/run_wan21_oracle_motion_refiner_gate.sh
```

승격은 normal weighted score가 static보다 낮고, normal-minus-zero와
normal-minus-batch-roll weighted paired bootstrap CI의 상한이 모두 0보다 낮으며, DINO와 action component의
point difference도 두 counterfactual보다 모두 낮을 때만 `PROMOTE_ACTION_TO_FIELD_STAGE`다. point estimate만
맞으면 250-step을 늘리지 않고 holdout을 24개로 다시 평가한다. 위 조건을 못 맞추면
`REJECT_MOTION_FIELD_REFINER`로 종료하고 action-to-flow predictor는 구현하지 않는다. direct oracle warp가 이미
full L1 `+0.47%`, motion L1 `+7.37%` 개선에 그쳤기 때문에, 이 한 번의 generative completion gate가 이
계열에 허용하는 마지막 renderer 실험이다.

## 29. DreamZero state-anchored input-contract gate (2026-08-05)

이 단계도 절대적인 최종 계획이나 성공 보장이 아니다. Wan2.1 oracle motion refiner는 250-step 뒤에도
normal weighted가 static을 이기지 못했고 correct action이 zero/batch-roll보다 일관되게 좋아지지 않아
`REJECT_MOTION_FIELD_REFINER`로 종료했다. 다음 실험은 새 adapter를 더 학습하는 대신, 공개
DreamZero-SO101가 실제로 학습한 action interface를 challenge train-only holdout에서 재현하는 가장 작은
반증 실험이다.

기존 `given` 구현은 각 8-step 구간의 첫 action을 state proxy로 빼고 24 token으로 보간했다. 공개 checkpoint
설정을 다시 대조하면 원 학습은 첫 action이 아니라 각 chunk anchor의 `observation.state`를 모든 24개 absolute
target에서 뺀다. 또한 영상은 8 frame을 action offset `[0, 3, ..., 21]`에서 뽑으므로 challenge의 8개
6fps target을 24개 action token으로 선형 확장하는 시간 변환은 유지한다. 새 구현은 train artifact의
`states[0]`, `states[8]`을 두 chunk anchor로 사용한다.

다만 이것을 완전한 동일 입력이라고 부르지는 않는다. Challenge SO-100의 absolute state를 SO-101 q01/q99로
직접 정규화하면 선택 holdout에서 50--83% 차원이 clamp되어 state token calibration은 호환되지 않는다.
따라서 state token은 metadata center를 유지하고, oracle state는 **relative action을 정확히 만드는 데만** 쓴다.
이 gate가 검증하는 것은 full policy contract가 아니라 state-anchored action이 released joint video/action
DiT에 실제 영향을 주는지다.

`train/generate_dreamzero_contract_gate.py`와 `tools/run_dreamzero_contract_gate.sh`는 다음을 한 번의 14B model
load 안에서 비교한다.

- `front-only`: 실제 challenge 입력처럼 front TL만 채우고 gripper/top view는 black
- `replicate-three`: 동일 이미지를 세 view에 복제하는 missing-camera 원인 분리용 OOD ablation
- `normal`: source clip의 action과 실제 chunk-anchor state
- `zero`: 각 chunk의 action을 해당 anchor state로 고정하여 물리 relative action을 정확히 0으로 만듦
- `batch-roll`: 다른 dataset clip의 action과 anchor state를 함께 이동하여 calibration 혼합을 방지

CPU audit에서는 서로 다른 dataset의 `holdout_0000`, `holdout_0003`을 고정했다. 두 샘플 모두 normal과
batch-roll의 relative action q99 saturation이 0%였고, normal-zero normalized mean absolute difference는
각각 `0.0371`, `0.0449`, normal-batch-roll은 둘 다 `0.0549`였다. 따라서 이후 정지 결과를 action clamp나
동일 counterfactual 탓으로 설명할 수 없다. audit 결과는
`diagnostics/dreamzero_so101_contract_gate/input_audit.json`에 저장된다.

실행은 GPU를 자동 선택하지 않으며 다음의 두 단계로 제한한다.

```bash
# 1. 두 canvas에서 normal만 생성: 총 4개 영상
CUDA_VISIBLE_DEVICES=0 PHASE=view \
  bash tools/run_dreamzero_contract_gate.sh

# 2. 더 나은 canvas 하나에서 zero/batch-roll을 추가 생성
CUDA_VISIBLE_DEVICES=0 PHASE=counterfactual VIEW_MODE=front-only \
  bash tools/run_dreamzero_contract_gate.sh

# 3. 동일 두 train-only clip의 공식 local feature 비교
CUDA_VISIBLE_DEVICES=0 PHASE=score VIEW_MODE=front-only \
  bash tools/run_dreamzero_contract_gate.sh
```

1단계에서 두 canvas 모두 정지/blur이면 추가 생성 없이 DreamZero-SO101 직접 전이를 중단한다. 한 canvas가
robot 형태와 motion을 보존할 때만 2단계로 진행한다. 2-sample CI는 승격 근거로 충분하지 않으므로, normal이
시각적으로 zero/roll과 다르고 DINO/action point estimate도 둘보다 좋은 경우에만 8-sample confirmatory gate를
허용한다. 이 조건을 통과하기 전에는 DreamZero fine-tune이나 state predictor를 학습하지 않는다.

## 30. 데이터 계약 우선 재시작: action grounding before pixels (2026-08-05)

이 절은 절대적인 최종 계획이나 특정 백본의 성공 보장이 아니다. 다만 기존 생성 실험 대부분이 서로 다른
논문의 action 의미를 challenge action에 정확히 대응하지 않은 채 진행됐다는 공통 문제를 수정한다. 새 RGB
생성 학습은 아래 grounding gate를 통과하기 전까지 중단한다.

Challenge의 frame 0은 입력 이미지와 같고 `action[t]`는 자체 train audit에서 `state[t+1]`에 가장 가깝다.
따라서 16-frame 제출에서 직접 관측되는 계약은 다음과 같다.

```text
source = frame[0]
targets = frame[1:16]
commands = action[0:15]
action[15] = 제출 horizon 밖의 상태를 향하므로 제외
```

기존 `train/irasim_so100.py`는 변환 뒤 `[1:16]`을 사용하여 첫 command를 버리고 action 15를 포함했다.
이를 공통 `train/so100_transition_contract.py`로 교체했다. 첫 deployable step은
`action[0]-state[0]`, 이후 step은 `action[t]-action[t-1]`이다. eval에는 state가 없으므로 relative IRASim
추론은 source-state estimate가 없을 때 즉시 중단하고, absolute command만 기본 deployable 모드로 둔다.

train metadata만으로 uploader 8개를 통째로 holdout한 22,073-window manifest를 만들었다. 2,400개 고정
window 감사 결과는 다음과 같다.

- action-to-current MAE: `2.8293` degree
- action-to-next MAE: `1.8618` degree
- next가 더 가까운 window: `92.08%`
- `action[t]-state[t]`와 실제 state delta의 MAE: `1.8618` degree
- source state + 이전 command만 쓰는 deployable step과 실제 state delta의 MAE: `0.8386` degree
- train/holdout uploader overlap: 0, invalid record: 0

따라서 새 표현은 단순 anchor delta가 아니라 세 스트림을 사용한다.

```text
absolute target       = action[t]
source-anchor target  = action[t] - estimated_state[0]
deployable step       = action[0]-estimated_state[0] (t=0)
                        action[t]-action[t-1]          (t>0)
```

관련 산출물과 코드는 다음과 같다.

- `results/so100_contract_manifest.jsonl`: 19,779 train + 2,294 uploader-holdout window
- `results/so100_contract_manifest.meta.json`: split과 contract metadata
- `results/so100_contract_audit.json`: 2,400-window 수치 감사 (`PASS_DATA_CONTRACT`)
- `tools/cache_so100_grounding_features.py`: 공식 DINO component와 action/state cache
- `train/train_so100_grounding_probe.py`: source-state, future-state, future-DINO 동시 probe
- `train/run_so100_grounding_gate.sh`: 단계별 실행 wrapper

probe 승격 조건은 source image state estimator가 action0 proxy를 CI95 기준으로 이기고, future state와 DINO가
각각 static을 이기며, correct action이 zero와 batch-roll보다 모두 CI95 기준으로 좋아지는 것이다. 하나라도
실패하면 RGB 생성기를 더 학습하지 않고 실패한 경계(visual state 또는 action dynamics)를 먼저 수정한다.

실행 wrapper는 GPU를 자동 선택하지 않는다. manifest/audit는 이미 CPU로 완료했다. DINO cache와 probe를
GPU에서 실행하려면 실행자가 명시적으로 다음처럼 지정한다.

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda PHASE=cache \
  bash train/run_so100_grounding_gate.sh

CUDA_VISIBLE_DEVICES=0 DEVICE=cuda PHASE=probe \
  bash train/run_so100_grounding_gate.sh
```

## 31. Grounding probe v1 결과와 anchored-residual v2 (2026-08-05)

5,120-window DINO cache로 v1 probe를 2,000 step 학습했지만 최종 판정은 `REJECT_GROUNDING`이었다.
다만 이는 action 자체가 시각 변화와 무관하다는 결과가 아니다. uploader-holdout 1,024개에서 best step 750은
correct action이 zero와 batch-roll보다 future state 및 future DINO 양쪽에서 CI95 기준으로 좋았다. 실패한
항목은 absolute predictor가 이미 강한 deployable baseline을 훼손한 두 부분이었다.

- action0을 source state로 쓰는 normalized MAE: `0.0983`
- source DINO CLS로 source state를 새로 회귀한 MAE: `0.7785`
- `action[t]`를 `state[t+1]`로 직접 쓰는 MAE: `0.0699`
- learned future-state decoder MAE: `0.2630`
- source DINO를 반복하는 static future-DINO cosine error: `0.1122`
- absolute future-DINO decoder cosine error: `0.2590`

따라서 v1을 더 오래 학습하지 않는다. v2는 source state에 action0 proxy를 사용하고 future state는 감사에서
확인한 `action[t] -> state[t+1]` 매핑을 그대로 보존한다. 시각 branch만 다음의 bounded residual을 학습한다.

```text
future_state[t] = action[t]
future_dino[t]  = normalize(source_dino + 0.05 * tanh(residual[t]))
```

visual residual head는 zero-init이므로 학습 시작점은 정확히 static DINO baseline이다. correct action이
zero-motion 및 batch-roll보다 좋아지도록 ranking loss도 함께 사용한다. v2 gate는 direct physical mapping을
훼손하지 않고, future DINO가 static을 이기며, correct action이 두 counterfactual보다 CI95 기준으로 좋은
경우에만 통과한다. 이는 성공 보장이 아니라 v1에서 확인된 잘못된 absolute-decoding 가정을 제거한 다음
반증 실험이다. v2까지 실패하면 global DINO CLS를 motion target으로 쓰는 경로를 종료하고 spatial token,
keypoint 또는 optical-flow target으로 전환한다.

v2의 1,000-step 실행에서는 마지막 checkpoint가 `future_dino=0.1249`로 static `0.1122`보다 나빠
`REJECT_GROUNDING`이었지만, checkpoint selection으로 저장된 best는 **step 100**이었다. best의
`future_dino=0.1064`는 static보다 낮았고, normal-minus-zero DINO CI95는
`[-0.00856, -0.00647]`, normal-minus-roll DINO CI95는 `[-0.01708, -0.01399]`로 모든 gate를 통과했다.
따라서 v2 결과는 `PASS_GROUNDING`으로 판정하되, 200 step 이후 지속적으로 악화된 명확한 과학습 때문에
step 100만 사용한다. 이 holdout을 checkpoint 선택에도 사용했으므로 renderer 승격 전에 uploader 단위의
별도 confirmatory split에서 한 번 더 재현해야 한다.

confirmatory protocol은 기존 holdout uploader를 고정 seed로 4/4 분리한다. selection uploader
`CSCSXX, pierfabre, pranavsaroha, vladfatu`의 524개만 checkpoint 선택에 사용하고, confirmation uploader
`DorayakiLin, VoicAndrei, sixpigs1, therarelab`의 500개는 선택된 checkpoint를 마지막에 한 번 평가한다.
정식 실행 전에 gate를 고정하여, future DINO가 static보다 paired bootstrap CI95 기준으로 좋아야 할 뿐 아니라
상대 개선도 최소 2%여야 한다. correct-vs-zero와 correct-vs-roll 역시 기존처럼 CI95 상한이 0보다 작아야 한다.
정식 결과를 본 뒤 이 기준이나 split을 변경하지 않는다.

## 32. Grounded source-warp renderer: 일주일 fast path (2026-08-05)

confirmation과 병행해 생성 단계는 새 foundation model로 다시 시작하지 않고 기존 flow renderer를 최소
수정한다. 과거 `flow_world_v3`는 hybrid action을 처음부터 해석하는 별도 Transformer를 사용했기 때문에,
action semantics가 시각 motion에 grounding되기 전에 static pixel shortcut으로 수렴할 수 있었다. 새 경로는
v2 grounding checkpoint를 동결하고 다음 visual residual token을 renderer의 유일한 motion condition으로 쓴다.

```text
motion_token[t] = predicted_future_dino[t] - source_dino
source spatial feature + motion_token[t]
    -> bounded backward flow + occlusion/edit mask + bounded RGB residual
    -> source-copy / warp composite
```

frame 0은 입력을 그대로 반환한다. future도 source pixel warp가 기본이며 RGB residual은 `0.08` 범위로
제한된다. 따라서 Dream/Wan처럼 전체 latent/RGB를 다시 그려 배경과 로봇 외형을 동시에 잃는 자유도가 없다.
RAFT-small teacher는 frame 1, 8, 15의 backward flow를 직접 감독하고, correct grounding token과 batch-roll
token의 pixel reconstruction ranking도 유지한다. ImageNet ResNet spatial encoder는 500-step screen 동안
동결한다.

관련 코드는 `train/train_grounded_flow_renderer.py`, `train/generate_grounded_flow_videos.py`,
`train/run_grounded_flow_renderer.sh`이다. CPU 구조 감사에서 cache record 4,096개가 원본 video/action과
재결합됐고, motion token shape `(1,15,384)`, renderer output `(1,16,3,H,W)`, finite output 및 frame-0 exact
copy를 확인했다. 이 역시 성공 보장은 아니다. 250/500-step 영상에서 normal motion이 보이지 않거나
normal이 zero/batch-roll과 다르지 않으면 추가 학습 없이 종료한다.

첫 500-step 결과는 eval 영상의 마지막-frame edit fraction이 대체로 2--10%였지만 실제 관절 이동 대신
국소 왜곡에 머물렀다. train sample에서도 predicted flow는 마지막 frame 가로 약 `0.30px`, 세로
`0.04px`였고, prediction L1 `0.02625`는 static `0.02655`와 거의 같았다. 반면 같은 sample의 RAFT teacher는
motion pixel에서 frame 15 평균 `2.68px`를 제공했다. 즉 grounding 실패가 아니라 mixed RGB objective가
spatial flow supervision을 눌러 static shortcut으로 수렴한 것으로 판정한다. 마지막 deterministic screen인
v2는 RGB reconstruction weight를 `1.0 -> 0.25`, RAFT flow weight를 `0.5 -> 5.0`, motion-pixel boost를
`4 -> 20`, mask supervision을 `0.1 -> 0.5`로 고정한다. v2 500-step도 명확한 관절 이동을 만들지 못하면
flow/warp renderer를 종료하고 pretrained Dream의 spatial-control 경로만 남긴다.

## 33. BWM-5B SO100 이식: 장기 학습 challenger (2026-08-06)

이 절은 절대적인 최종 계획이 아니라, 지금까지의 static shortcut 실패를 반영한 **검증 가능한 challenger**다.
Boundless World Model(BWM)의 공개 `step-12000` 가중치와 Wan2.2-TI2V-5B를 초기값으로 사용한다. 새 video
backbone이나 얕은 외부 adapter를 처음부터 학습하지 않고, BWM이 이미 학습한 first-frame 보존, action
cross-attention 및 temporal AdaLN 경로를 그대로 가져온다. SO100 이식 시에는 BWM action MLP 전체와 DiT
30개 블록의 `q,k,v,o,ffn.0,ffn.2` LoRA(r=32)를 함께 학습한다.

대회 action 16x6은 다음 17x18 frame token으로 한 곳에서만 변환한다.

```text
token[0]     = [source action0 proxy 6D, zero anchor delta 6D, zero step delta 6D]
token[1:17]  = [future absolute 6D, source-relative delta 6D, step delta 6D]
```

공식 SO100 z-score와 delta scale을 적용한 뒤, BWM의 원래 percentile-normalized `[-1,1]` 입력 범위에 맞춰
각 성분을 3-sigma clipping 및 scaling한다. 공개 BWM action encoder의 14D 입력 projection 두 개만 18D로
늘리며 새 weight는 zero-init한다. pretrained bias와 이후 MLP weight는 보존한다. AdaLN 경로에서 호출되지
않는 구형 `action_embedding`은 동결하여 DDP unused-parameter 문제를 막는다. text condition은 공식 BWM
profile과 같이 사용하지 않는다.

CPU audit 결과 공개 checkpoint 829 tensor, DiT block 30개, 로컬 Wan shard 3개와 VAE, 119개 train dataset의
10,640 clip, `(1,3,17,H,W)` video와 `(1,17,18)` action contract가 모두 통과했다
(`results/bwm_so100_contract_audit.json`). 학습은 384x512, 8-GPU, 10,000 optimizer step을 기본 장기 실행으로
두되 1,000 step마다 checkpoint를 남긴다. 긴 학습 자체가 성공 근거는 아니다. 1k/3k/5k/10k에서 동일 seed로
normal, zero-motion, batch-roll을 비교하며, normal이 두 counterfactual과 구분되지 않거나 화질이 지속적으로
나빠지면 마지막 step 대신 앞선 checkpoint를 선택하거나 이 경로를 종료한다.

구현은 `train/bwm_so100.py`, `train/train_bwm_so100_lora.py`,
`train/generate_bwm_so100_videos.py`, `train/run_bwm_so100.sh`에 고정했다. frame 0은 제출 시 원본 RGB로
복원하고, mixed-resolution challenge image는 모델 입력에서만 letterbox한 뒤 생성 frame을 원래 content
영역과 해상도로 역변환한다.

## 34. BWM EEF14 faithful reset: 원본 action contract 보존 (2026-08-06)

이 계획 역시 절대적인 답이나 성공 보장이 아니다. 33절의 hybrid18 모델은 4,000 step에서 일부 큰 motion을
만들었지만, sample 0/1/2/4/5에는 ghosting과 관절 기하 왜곡이 생겼고 3/6/7은 거의 정지했다. 새 18D 입력
projection의 학습값뿐 아니라 공개 action MLP의 깊은 weight도 크게 이동했다. 따라서 장기 학습을 계속하지
않고 checkpoint는 비교용으로만 보존한다.

새 경로는 공개 BWM이 실제로 사용한 dual-arm EEF 14D 계약을 바꾸지 않는다.

```text
BWM token = [left xyz, left roll/pitch/yaw, left gripper,
             right xyz, right roll/pitch/yaw, right gripper]
SO100      = [source-anchored FK EEF 7D, inactive-arm zero 7D]
```

전체 SO100 데이터에서 raw joint target의 절대 servo offset은 서로 일치하지 않았다. 기존 absolute 변환을
그대로 적용하면 shoulder/elbow 약 33%, gripper 약 23%가 corrected URDF joint limit 밖이었다. 따라서
알 수 없는 절대 offset을 EEF로 위장하지 않는다. 각 17-frame clip에서 첫 action을 joint-limit 안쪽의
비특이 canonical pose `[0,-0.5,0.5,0.5,0,0]`에 맞추고, 공통으로 검증된 degree-to-radian 부호/scale을
시간 변화량에만 적용한다. 실제 초기 자세와 외형은 source RGB가 담당한다. canonical pose를 all-zero로
두지 않은 이유는 `Fixed_Jaw` orientation이 pitch `pi/2`의 Euler gimbal lock이 되어 작은 joint motion이
roll/yaw 불연속으로 변하기 때문이다.

train uploader holdout을 제외한 119개 dataset, 960,429 raw frame에서 stride 4로 201,142개 window를
샘플링했고, 총 3,419,414 EEF token의 p01/p99를 계산했다. 변환 전 limit 초과는 가장 큰 shoulder lift도
6.15%이며 실제 입력에서는 joint limit으로 bounded된다. CPU audit 결과는 다음과 같다.

- 공개 action input shape 보존: `3072x14`, `12288x56`
- train sample tensor: video `(1,3,17,64,96)`, action `(1,17,14)`
- inactive arm maximum absolute value: `0.0`
- holdout normal-minus-zero token absolute mean: `0.09350`
- holdout normal-minus-batch-roll token absolute mean: `0.14584`
- verdict: `PASS_EEF14_CONTRACT`

관련 구현은 `train/bwm_so100_eef.py`, `tools/compute_so100_eef_stats.py`,
`tools/audit_bwm_so100_eef_contract.py`, `train/train_bwm_so100_eef_lora.py`,
`train/generate_bwm_so100_eef_videos.py`, `train/run_bwm_so100_eef.sh`이며 결과는
`results/bwm_so100_eef14_audit.json`에 저장했다. 원본 action input projection은 재초기화하거나 확장하지
않는다. 공개 action MLP와 전 DiT block LoRA만 낮은 learning rate `2e-5`로 학습한다.

실험 순서는 zero-shot 8개 시각 감사, 5-step 분산 smoke, 1,000-step normal/zero/batch-roll gate 순이다.
zero-shot이 완전 붕괴하거나 1k에서 action counterfactual이 분리되지 않으면 12k를 강행하지 않는다. 그 경우
우선순위는 (1) 이미 점수 재현 근거가 있는 Cosmos action-conditioned SO100 recipe, (2) single-arm 또는
SO100/SO101 사전학습 checkpoint의 native input contract다. 이 fallback 순서도 현재 증거에 따른 계획일 뿐,
새 결과가 더 나은 방법을 지지하면 변경한다.

### EEF14 zero-shot 결과

공개 BWM checkpoint만 사용한 8-sample zero-shot은 전체 장면과 로봇 외형을 대체로 유지하여 scratch
generator의 즉시 붕괴와는 달랐다. 그러나 정답 action에 대응하는 관절 운동은 거의 보이지 않았고, 일부
sample에서는 화면 경계에서 검은 로봇/물체가 새로 들어오는 hallucination이 생겼다. source-to-last RGB MAE
median은 normal `0.1091`, zero-motion `0.1121`로 오히려 zero가 비슷하거나 더 컸다. 같은 seed에서
normal-minus-zero median은 `0.0127`, normal-minus-batch-roll은 `0.0217`이었다. 즉 generator prior와
condition sensitivity는 남아 있지만 SO100 action grounding은 아직 없다. 따라서 zero-shot 제출 후보는
아니며, 형태 보존 prior를 훼손하지 않는 낮은 LR의 1,000-step screen만 허용한다. 250/500/750/1000에서
normal motion이 zero/roll과 분리되는 checkpoint가 하나도 없으면 12k로 연장하지 않는다.

## 35. Ctrl-World native SO100 port (2026-08-06)

이 절 역시 절대적인 최종 계획이나 성공 보장이 아니다. World Model Arena의 정성 영상이 좋다는 이유만으로
모델 이름을 흉내 내지 않고, [Ctrl-World 논문](https://arxiv.org/abs/2510.10125)과
[공개 코드](https://github.com/Robert-gyj/Ctrl-World)의 실제 tensor/loss/rollout 계약을 확인한 뒤 구현했다.
논문은 7개 history frame과 15-step action chunk라고 기술한다. 공개 구현에서는 15개 action을 5개 pose로
시간 다운샘플하고, tensor를 `6 sparse history + current reconstruction + 4 future = 11 frames`로 구성한다.
16-frame 제출 영상은 이 5-frame 예측을 네 번 autoregressive rollout하고 각 chunk의 current reconstruction을
버린 뒤 source RGB + future 15개를 저장한다.

원본과 유지한 부분은 다음과 같다.

- `stabilityai/stable-video-diffusion-img2vid` temporal UNet, scheduler, temporal VAE
- 프레임별 pose embedding을 spatial transformer cross-attention에 넣는 방식
- 3-layer action projection MLP, history/condition noise, 5% condition dropout
- EDM preconditioning과 current+future `x0` weighted MSE
- VAE/image encoder 동결, **전체 UNet + action MLP 학습**: LoRA나 얕은 외부 adapter가 아니다.
- 첫 rollout에서 과거가 없으면 source latent를 반복하고 마지막 생성 latent로 AR 진행

데이터/대회 계약 때문에 바꾼 부분은 세 가지뿐이다.

1. DROID의 세 카메라 joint prediction을 SO100 단일 외부 카메라 prediction으로 바꾼다. 논문 자체에도
   `Ctrl-World-Single-View` ablation이 있으므로 새 아키텍처를 발명한 것은 아니다. 단, 논문 결과상 single-view는
   multi-view보다 접촉 상호작용과 hallucination에 약했다는 한계가 남는다.
2. Franka Cartesian pose 7D 대신 SO100 native joint state/target 6D를 쓴다. 임의 calibration/FK로 7D처럼
   보이게 하지 않는다. train에서는 `observation.state`를 pose로 쓰고, challenge에서는 로컬 audit에서
   `action[t]`가 `state[t+1]`에 가장 가까운 target임을 확인했으므로 future pose로 쓴다.
3. train caption은 있지만 eval instruction은 없으므로 text condition을 끈다. 평가 때 없는 정보를 학습
   shortcut으로 만들지 않기 위한 변경이다.

공개 DROID Ctrl-World checkpoint는 7D Franka/3-view 모델이지만 robot-video temporal prior와 frame-level
control weight를 갖고 있으므로 초기값으로 사용한다. 로드 시 shape가 다른
`action_encoder.action_encode.0.weight` 하나만 새 6D projection으로 남기고 나머지 호환 tensor를 모두
로드한 뒤, 원 논문처럼 전체 UNet을 미세조정한다. SVD-only 초기화보다 DROID의 robot-domain prior를 먼저
활용하는 선택이다. HF 공개 파일명은 `checkpoint-10000.pt`인데 논문 본 실험은 100k step이라고 명시하므로,
공개 checkpoint가 논문의 최종 100k 가중치와 동일하다고 간주하지 않는다.

CPU contract audit는 119 train dataset, 8 dataset-level holdout, 11,071 episode-camera record, 누락 0,
split overlap 0, frozen `valset_holdout` 6개 dataset의 train 유입 0, 네 개 rollout condition `(11,6)`을
확인해 `PASS_CTRL_WORLD_DATA_CONTRACT`가 나왔다. train-only 10,644 episode / 955,261 frame의 state
p01/p99도 계산했다. 결과는
`results/ctrl_world_so100_contract_audit.json`과 `results/ctrl_world_so100_pose_stats.json`에 저장했다.

```bash
# SVD는 gated license 동의와 hf login이 먼저 필요하다.
PHASE=assets bash train/run_ctrl_world_so100.sh
PHASE=audit bash train/run_ctrl_world_so100.sh
PHASE=stats bash train/run_ctrl_world_so100.sh
NPROC=8 PHASE=cache bash train/run_ctrl_world_so100.sh

# 5-step 실행/메모리/DDP 확인
NPROC=8 PHASE=smoke bash train/run_ctrl_world_so100.sh

# 1k screen. full UNet이므로 250 step만으로 결론 내리지 않는다.
NPROC=8 MAX_STEPS=1000 SAVE_STEPS=250 LABEL=ctrl_world_so100_1k \
  PHASE=train bash train/run_ctrl_world_so100.sh

CHECKPOINT=open/baseline/outputs/ctrl_world_so100_1k/step-1000.safetensors \
LABEL=ctrl_world_so100_1k LIMIT=8 PHASE=gate bash train/run_ctrl_world_so100.sh
```

1k에서 normal/zero-motion/batch-roll 영상이 분리되고 normal이 static을 이기며 action counterfactual 점수도
올바른 방향일 때만 3k→10k로 이어간다. 반대로 세 조건이 같거나 source morphology가 빠르게 붕괴하면 논문의
100k 학습량을 근거로 무작정 연장하지 않는다. 논문의 영상은 95k DROID trajectory, 2x8 H100, batch 64,
100k step, multi-view/wrist observation으로 얻은 결과이므로 단일-view SO100에서 동일 품질을 보장하지 않는다.
제출 전에는 SVD Community License와 대회 규정의 호환성을 다시 확인한다.

관련 코드는 `train/ctrl_world_so100.py`, `tools/prepare_ctrl_world_so100.py`,
`train/train_ctrl_world_so100.py`, `train/generate_ctrl_world_so100.py`,
`train/run_ctrl_world_so100.sh`이다.

## 36. Cosmos unified spatial-action v2 (2026-08-06)

이 절은 Ctrl-World와 FOMM의 제출 영상이 모두 열화된 뒤 정한 **현재 우선 계획**이며, 절대적인 최종
방법이나 성능 보장이 아니다. 새 결과가 가정을 반박하면 손실 가중치를 계속 올리지 않고 해당 축을 폐기한다.
목표는 공개 Cosmos-Predict2.5-2B action-conditioned 모델의 robot-video 생성 prior를 유지하면서, 기존
SO100 레시피의 공간적으로 균일한 Frame-AdaLN 조건을 frame-level action token으로 보완하는 것이다.

### 36.1 아키텍처

기존 경로는 한 글자도 제거하지 않는다.

```text
SO100 action (B,16,6)
  ├─ 기존: 4-frame chunk (B,4,24) → action MLP → timestep + Frame-AdaLN
  └─ 신규: 16 frame tokens
       [absolute(6), one-step delta(6), source-relative delta(6)]
       → MLP 18→1024→1024 + learned temporal position
       → 각 28 DiT block의 별도 action cross-attention softmax
       → zero-init channel gate를 통해 기존 cross-attention residual에 가산
```

신규 action attention은 기존 text cross-attention의 k/v/output projection과 그 LoRA를 공유한다. text 512토큰과
action 16토큰을 한 softmax에 합치지 않고 별도 softmax를 사용해 action 질량이 긴 empty-text context에 묻히는
것을 막는다. 신규 파라미터는 tokenizer 1,087,488개와 28×2048 gate 57,344개, 총 **1,144,832개**다.
`action_ctx_scale=0`이면 신규 계산을 건너뛰므로 기존 모델로 정확히 돌아갈 수 있다.

이 선택은 다음 연구의 공통 부분만 취한다.

- Ctrl-World의 frame-level action conditioning: 4-frame chunk 하나보다 시간별 조건을 보존한다.
- Visual Action Prompts와 OSCAR의 교훈: action의 변화·기하 정보를 절대 pose 하나에 묻지 않는다.
- zero-initialized residual adapter 계열: pretrained generator를 초기 시점에 훼손하지 않는다.
- Cosmos action-conditioned 모델: 검증된 flow-matching video prior와 Frame-AdaLN은 계속 사용한다.

VAP/OSCAR의 2D skeleton 자체는 채택하지 않았다. SO100 전체 데이터의 카메라 내·외부 파라미터가 없어서
여러 uploader/camera에 공통된 정확한 2D skeleton을 만들 수 없고, 이전 pseudo-skeleton 실험도 3~7번
sample에서 형상이 깨졌다. 구조를 참고하는 것과 검증되지 않은 control image를 넣는 것은 구분한다.

### 36.2 하나의 연속 학습에서 사용하는 손실

이 버전은 35k→90k→DRaFT처럼 목적함수를 단계별 실행으로 나누지 않는다. 하나의 optimizer/run에서
flow matching은 항상 켜고, 나머지 항을 첫 10k exposure 동안 0에서 목표 가중치까지 선형 ramp한다.

```text
L = L_flow
  + ramp(s) [0.25 L_x0 + 0.10 L_temporal + 0.10 L_preserve]
  + ramp(s) 0.05 L_counterfactual       (확률 0.25)
```

- `L_flow`: 기존 rectified flow velocity MSE. 생성 prior 학습의 중심이다.
- `L_x0`: 같은 noisy latent에서 복원한 paired GT latent의 Smooth-L1. sigma가 너무 큰 구간은 가중치를 0으로
  낮춰 pure-noise 상태에서 불안정한 one-step 복원을 강제하지 않는다.
- `L_temporal`: 예측 latent의 인접 시간차가 GT 시간차와 같도록 한다. 정지 영상으로 수렴하는 지름길을 막는다.
- `L_preserve`: GT에서 첫 frame 대비 변화가 작은 위치에만 추가 identity 압력을 준다. 배경은 보존하되 실제
  robot/object motion 영역을 source로 덮어쓰지 않는다.
- `L_counterfactual`: 동일한 noisy latent에 reverse 또는 8-step rolled action을 넣었을 때, 올바른 action의
  velocity 예측이 paired GT에 margin 0.02 이상 더 가깝게 만든다. action을 무시하는 해를 직접 겨냥한다.

제공 코드의 latent IDM/DRaFT 보상은 이 unified 경로에서 사용하지 않는다. 회고에서 자체 IDM은 eval scene에서
상수 예측보다 나빴고, 그것을 결합 손실로 쓰면 잘못된 OOD gradient를 항상 주게 된다. 공식 action extractor도
학습에 사용하지 않는다. 네 보조손실은 모두 train paired RGB에서 만든 latent만 사용한다.

### 36.3 구현 및 감사 결과

주요 파일은 다음과 같다.

- `inha_worldmodel_scratch_training/cosmos/train/action_token_dit.py`
- `inha_worldmodel_scratch_training/cosmos/train/unified_losses.py`
- `inha_worldmodel_scratch_training/cosmos/train/train_lora.py`
- `inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh`
- `inha_worldmodel_scratch_training/cosmos/train/audit_action_token_dit.py`
- `inha_worldmodel_scratch_training/cosmos/train/audit_unified_losses.py`
- `inha_worldmodel_scratch_training/cosmos/train/audit_training_inputs.py`

CPU 계약 감사 결과:

- action token `(2,16,1024)`, finite
- 신규 파라미터 정확히 `1,144,832`
- 실제 2B graph를 meta device에서 구성해 28 blocks, 총 2,127,846,400 parameters 계약 확인
- action path disabled / zero gate에서 base attention max difference `0.0`
- gate를 열면 출력 차이 `0.05544`, tokenizer와 gate 모두 gradient 수신
- perfect velocity에서 paired losses 최대 `1.86e-8`
- corrupted velocity에서 x0/temporal/preserve 모두 증가하고 finite gradient 수신
- verdict: `PASS_ATOK_CPU_AUDIT`, `PASS_UNIFIED_LOSS_AUDIT`

### 36.4 실행 순서

아래는 학습을 여러 모델로 나누는 것이 아니다. assets/index/latent는 데이터 준비이고,
`unified_train` 한 번이 실제 단일 학습이다.

명령을 하나씩 입력할 필요는 없다. `PHASE=prepare`는 모든 데이터 준비와 감사를 순서대로 실행하고,
`PHASE=all`은 준비 뒤 같은 unified run의 8-step smoke와 본 학습까지 자동으로 이어간다.

```bash
cd /nfsdata/home/lexxsh/workspace/inha

PHASE=assets bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh
PHASE=index bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh
NPROC=8 PHASE=latents bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh
PHASE=audit bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh
PHASE=preflight bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh

# 같은 run에서 정확히 optimizer 1회만 실행하는 배관 smoke.
CUDA_VISIBLE_DEVICES=0 BATCH=2 STOP_AT=4 PHASE=unified_train \
  bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh

# 본 학습: 위 step 8의 weight와 Adam state를 자동 복원해 같은 run을 계속한다.
CUDA_VISIBLE_DEVICES=0 PHASE=unified_train \
  bash inha_worldmodel_scratch_training/cosmos/train/run_spatial_action_v2.sh
```

기본 학습량은 seed가 고정된 중복 없는 train window 104,536 exposure 한 번이다(가용 104,538개 중
effective batch 8로 나누어떨어지는 최대 prefix). 기존 LoRA/action MLP와
신규 action-token 경로를 함께 학습하되 Cosmos 2B base는 동결한다. 5k마다 checkpoint를 남기는 것은 모델을
단계별로 바꾸기 위한 것이 아니라 collapse를 조기에 발견하고 최선점을 고르기 위해서다.

`latents` 단계는 train 104,828개 창뿐 아니라 episode-level held-out validation 236개 에피소드도 별도
`val_latents`에 전처리한다. `preflight`는 manifest와 실제 train latent filename 집합이 정확히 일치하고
validation latent가 존재하는지 검사한다. 하나라도 빠지면 `unified_train`도 즉시 중단하여, latent 누락 때문에
학습이 조용히 raw-video VAE 경로로 바뀌는 일을 막는다. 현재 코드 전달 시점에는 latent 파일이 아직 생성되지
않았으므로 입력 감사 결과는 의도대로 `REJECT_UNIFIED_INPUTS`이며, 위 `PHASE=latents` 완료 뒤 PASS가 되어야 한다.

H100 latent 전처리는 GPU별 `LATENT_BATCH=8`로 VAE encode를 묶는다. 메모리가 부족하면 해당 rank가 4→2→1로
자동 감소해 같은 window를 재시도한다. 저장 형식은 계속 window당 `.pt` 하나이며, resize는 dataset과 동일한
CPU float32 letterbox를 유지한다.

긴 실행 전에는 `STOP_AT=8`로 동일한 run을 한 optimizer update만큼 점검한 뒤 같은 명령으로 재개할 수 있다.
재개 시 checkpoint의 micro-step만큼 deterministic sampler prefix를 건너뛰므로 앞 창을 다시 학습하지 않는다.
H100 기본은 `BATCH=2`, gradient accumulation 4다. `EFFECTIVE_BATCH=8`을 고정한 채 `BATCH=4`이면
accumulation 2, `BATCH=8`이면 accumulation 1로 자동 조정된다. checkpoint에는 두 값을
기록하고 재개 시 `resume_step × BATCH`만큼 sampler prefix를 건너뛰므로 batch 설정이 달라도 노출량 계약이
유지된다. 동일 run 중간에 batch 값을 바꾸는 것은 `run_config.json` 불일치로 차단한다.

`UNIFIED_WARMUP=10000`, `SAVE_EXPOSURES=5000`, 전체 104,536은 모두 loader iteration이 아니라 sample
exposure 단위다. 따라서 micro-batch를 바꿔도 loss ramp, checkpoint 간격, 총 데이터 양은 변하지 않는다.

중단 기준도 절대 규칙은 아니지만, 최소한 다음은 지킨다.

1. 5k~10k에서 `action_ctx_gate`가 계속 0이고 correct/reverse 출력이 같으면 배관 실패로 중단한다.
2. 10k/25k/50k에서 source morphology가 연속적으로 악화하면 100k까지 loss 하락만 보고 강행하지 않는다.
3. normal/zero/batch-roll 분리가 생겨도 normal 화질이 static보다 나빠지면 action 성공으로 판정하지 않는다.
4. 최종 선택은 loss 하나가 아니라 고정 holdout의 영상, normal-vs-counterfactual, weighted score를 함께 본다.

## 37. Cosmos3-Edge native Action FD full-SFT (2026-08-07)

이 절은 36절의 Cosmos-Predict2.5 전 블록 LoRA를 반복하는 계획이 아니다. 공식 Cosmos3 Action
forward-dynamics 구현이 공개된 뒤 로컬 공식 코드와 입력 계약을 다시 대조해 정한 **현재 최우선 screen**이다.
이 역시 절대적인 계획이나 성능 보장이 아니며, 250/500-step gate가 가정을 반박하면 20k까지 무조건 연장하지
않고 입력 표현 또는 모델 계열을 다시 선택한다.

공식 기준은 `third_party/cosmos-framework/docs/action_fd_droid_posttrain.md`와
`cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_droid_posttrain.py`다. 핵심 계약은
첫 RGB frame과 16-step embodiment action으로 17-frame video를 예측하는 `forward_dynamics`다. 공식 DROID
action은 10D `[body-frame dxyz(3), column rot6d(6), absolute gripper(1)]`이며, 64 action channel로 zero-pad하기
전에 domain별 affine normalization을 수행한다.

### 37.1 이전 구현에서 확인한 계약 오류

기존 Edge zero-shot converter는 challenge `action[0]`을 source state로도 복제했다. 그 결과
`frame0 -> frame1`에 대응하는 첫 action이 항상 identity가 되었고, 실제 `state[0] -> action[0] -> frame1`
시간축과 어긋났다. 또한 고정 외부 카메라를 `ego_view`로 표기했고 공식 ActionProcessor와 달리 normalized
action을 `[-1,1]`로 clamp했다. 이 세 항목을 다음처럼 고쳤다.

- train: parquet의 실제 `observation.state[start]`에서 `action[start]`로 첫 transition을 계산한다.
- eval: state가 없으므로 train-only median `state0-action0` offset으로 source state를 추정한다.
- camera/action processing: `third_person_view`, Bridge domain ID 7, unclamped q01/q99 affine normalization을 쓴다.
- gripper: relative 변화가 아니라 URDF limit에 대응하는 absolute `clip(0.5 + raw/100, 0, 1)`을 쓴다.
- frozen holdout의 24개 episode는 학습 index에서 제외한다.

CPU audit 결과 203,579개 stride-4 train window, 10,712개 episode, holdout overlap 0, video/action shape
`[3,17,H,W]`/`[16,10]`, finite action을 확인했다. 수정된 첫 action의 identity error median은
`0.02186`이고 과거 방식은 수치적으로 0이므로 시간축 수정이 실제로 작동한다. 결과는
`results/cosmos3_edge_so100_native_audit.json`과 `results/cosmos3_edge_so100_action_audit.json`에 저장한다.

### 37.2 학습 대상과 loss

별도 AdaLN/action adapter나 LoRA를 추가하지 않는다. 공개 Edge의 pretrained action head를 유지하고 공식
full-SFT parameter allowlist를 그대로 따른다.

```text
moe_gen, time_embedder, vae2llm, llm2vae, k_norm_und_for_gen,
action2llm, llm2action, action_modality_embed
```

loss도 임의의 DINO/preserve/counterfactual 항을 섞지 않고 공식 rectified flow-matching objective를 먼저 쓴다.
action bridge 세 모듈만 base LR의 5배, base LR `1e-4`, 20k scheduler horizon, weight decay `0.05`, bf16,
EMA, grad clip 1.0, 8-GPU FSDP가 기본이다. 즉 이전 2.5 실험의 “base freeze + 모든 DiT block LoRA + custom combined loss”와
다르게, pretrained Edge 생성 경로와 action bridge 자체를 함께 갱신한다.

### 37.3 실행과 중단 기준

```bash
cd /nfsdata/home/lexxsh/workspace/inha

PHASE=audit bash train/run_cosmos3_edge_so100_native.sh
PHASE=setup bash train/run_cosmos3_edge_so100_native.sh
PHASE=convert-base bash train/run_cosmos3_edge_so100_native.sh
PHASE=dryrun bash train/run_cosmos3_edge_so100_native.sh

# 8-GPU graph/메모리/I/O 확인
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PHASE=smoke \
  bash train/run_cosmos3_edge_so100_native.sh

# 첫 screen: global batch 8, checkpoint 250/500
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_STEPS=500 SAVE_STEPS=250 PHASE=train \
  bash train/run_cosmos3_edge_so100_native.sh

PHASE=export bash train/run_cosmos3_edge_so100_native.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=8 PHASE=gate bash train/run_cosmos3_edge_so100_native.sh
```

500 step은 최종 수렴량이 아니라 배관·방향 screen이다. 500에서 LR이 끝나지 않도록 scheduler는 처음부터 공식
20k horizon으로 고정한다. normal 영상의 morphology가 유지되고 static을 이기며,
normal/zero-motion/batch-roll이 올바른 방향으로 분리될 때만 2k, 이후 공식 DROID 규모인 20k까지 단계적으로
늘린다. 500 step에서 세 action variant가 같으면 학습량 부족으로 단정하지 않고 action sensitivity와 첫 전이
계약부터 다시 감사한다. normal이 움직여도 화질이 static보다 나쁘거나 6/7번처럼 scene 생성이 붕괴하면
제출하지 않는다.

주요 구현은 `train/cosmos3_edge_so100_dataset.py`, `train/cosmos3_so100_action.py`,
`train/generate_cosmos3_edge_so100.py`, `train/run_cosmos3_edge_so100_native.sh`,
`tools/audit_cosmos3_edge_so100_training.py`, 공식 experiment config
`action_fd_so100_edge.py`다.

## 38. 제출 회고와 Wan fallback 재설계 (2026-08-07)

이 절은 지금까지의 모든 실험을 한 모델이 성공했다는 이야기로 합치는 것이 아니다. 반복해서 확인된 실패
패턴을 압축하고, 37절 Cosmos3-Edge가 실패할 경우 어떤 Wan 실험만 남길지를 정하는 **조건부 계획**이다.
새 gate가 아래 가정을 반박하면 Wan2.2도 폐기할 수 있으며, 이 계획은 절대적인 최종안이나 성능 보장이 아니다.

### 38.1 지금까지 얻은 신호

| 계열 | 실제 관찰 | 판정/교훈 |
|---|---|---|
| Dream/DynamiCrafter 10k | 공개 `0.28188`; 움직임은 있으나 후반 blur와 관절 뒤틀림 | 최초 제출 기준선. 더 많은 step만으로 형태 보존이 해결되지 않음 |
| Wan2.2-5B action v1 1k | normal-vs-zero `-0.00005`, vs-roll `+0.00404`, 모두 CI가 0 포함 | AdaLN형 action 주입이 사실상 무시됨, `REJECT` |
| Wan2.2-5B action v2 2k | vs-zero `-0.00786`, vs-roll `+0.00078`; incumbent보다 유의하게 나쁨 | 2k 연장만으로 action 식별성 회복 안 됨, `REJECT` |
| Wan oracle track v1/v2 250 | GT track을 줘도 static보다 `+0.032/+0.064` 나쁨 | control 신호가 있어도 생성 prior와 결합/복원이 틀리면 실패 |
| IRASim 500 | static보다 `+0.132`, correct/zero/roll CI 분리 없음 | robot-video prior만으로 SO100 action grounding이 생기지 않음 |
| FOMM oracle | 단일 clip PSNR gain `+8.93 dB`, multi-clip median gain `+2.85 dB`이나 화질 gate 실패 | pose/warp는 동작을 옮길 수 있지만 unseen appearance renderer가 병목 |
| Wan2.1 oracle motion refiner 250 | 영상 일부는 양호하나 static을 이기지 못하고 counterfactual 불안정 | motion field와 생성 prior의 결합만으로 action correctness 보장 안 됨 |
| Wan2.1 spatial action 250 | 육안 품질 우수, 공개 `0.25`, Action MAE `0.48580` | 좋은 I2V prior는 보존됐지만 action grounding이 병목 |
| Cosmos Unified v2 13,067 | 영상은 더 흐리지만 공개 `0.22`, Action MAE `0.32250` | native robot-action prior와 충분한 exposure가 총점에서 유리 |

여러 실패의 공통점은 조건 텐서가 모델 출력에 수치적으로 영향을 준다는 structural gate와, 올바른 action의
영상을 생성한다는 semantic gate가 다르다는 것이다. adapter output이 nonzero이거나 normal/zero 영상이 조금
다르다는 사실만으로 action을 이해했다고 판정하면 안 된다. 올바른 action이 wrong action보다 paired GT와
가까워야 하고, 그 이득이 영상 보존 손해보다 커야 한다.

### 38.2 adapter 대 full-SFT는 잘못된 이분법

Cosmos Unified v2도 전체 2B를 모두 갱신한 모델이 아니다. 공식 action-conditioned backbone 위에 기존
action AdaLN, 16개 action token과 rank-32 LoRA를 학습했다. 반대로 Wan2.1 spatial 모델은 2.69M adapter와
76.68M LoRA만 학습해 14B backbone의 1% 미만을 바꿨다. 두 결과의 차이를 단순히 “Cosmos는 full-SFT,
Wan은 adapter”로 해석할 수 없다.

numeric action을 일반 I2V 모델에 넣으려면 최소한 hidden/token space로 바꾸는 bridge는 항상 필요하다. 중요한
구분은 bridge의 존재 여부가 아니라 다음 두 항목이다.

1. action이 일부 block의 약한 residual인지, 여러 block의 attention context에서 시간별 token으로 쓰이는지
2. pretrained generator가 고정된 채 새 branch만 배우는지, action과 접하는 생성 projection도 함께 갱신되는지

Wan 14B 전체를 무조건 unfreeze하는 것은 채택하지 않는다. 약 10k base clip에서 전 파라미터를 높은 LR로
학습하면 기존 I2V 화질 prior를 잃고 blur/형태 붕괴가 재발할 위험이 크며, optimizer/checkpoint 비용도 과도하다.
37절 Cosmos3-Edge도 “전 파라미터 무차별 full-SFT”가 아니라 LoRA 없이 공식 allowlist의 실가중치를 갱신한다.
`moe_gen/time_embedder/vae2llm/llm2vae`와 native action bridge를 함께 학습하는 선택적 full-SFT다.

### 38.3 조건부 Wan2.2-TI2V-5B fallback

37절의 Cosmos3-Edge native 모델이 morphology와 action gate를 함께 통과하지 못할 때만 Wan으로 돌아간다.
그 경우 Wan2.1-I2V-14B 전체 학습보다 이미 로컬에 있는 `Wan2.2-TI2V-5B`를 우선한다. 5B는 14B보다
선택적 backbone 공동 학습과 반복 gate가 현실적이고, TI2V의 첫 이미지 latent 경로를 그대로 이용할 수 있다.
다만 과거 Wan2.2 AdaLN v1/v2를 재실행하지 않는다.

예정 구조는 다음과 같다.

```text
source RGB → frozen Wan VAE → first-frame latent anchor

SO100 normalized action (B,16,6)
  → [absolute, one-step delta, source-relative delta] + temporal position
  → 16 frame-aligned action tokens
  → separate action cross-attention in multiple/all Wan2.2 DiT blocks

video tokens ↔ action tokens
  → selected generation attention/FFN full weights 또는 넓은 LoRA
  → 16 future frames; source frame은 제출 시 exact replacement
```

- action tokenizer/bridge와 action K/V projection은 전체 학습한다.
- VAE와 text encoder는 동결한다.
- generation attention/FFN은 처음부터 5B 전체를 풀지 않고 action과 직접 접하는 projection부터 선택적으로
  학습한다. action gate가 살아도 capacity 부족이 확인될 때만 범위를 넓힌다.
- text context와 action token을 한 긴 softmax에 섞어 16개 action이 묻히게 하지 않고, Cosmos v2처럼 별도
  action attention residual을 사용한다.
- 첫 프레임 조건은 clean latent로 고정하며 source morphology를 보존한다.

초기 loss는 Cosmos Unified에서 작동 원리가 확인된 paired 항을 사용한다.

```text
L = L_flow + λx0 L_x0 + λt L_temporal + λp L_preserve + λcf L_counterfactual
```

`L_counterfactual`은 같은 noisy latent/seed에 correct, zero-motion, batch-roll action을 넣어 correct가 paired
GT velocity 또는 x0에 더 가깝도록 한다. 외부 평가 extractor는 학습 loss로 사용하지 않는다. 이는 현재 Wan
spatial loss와 이름은 비슷하지만, action token이 여러 block attention에 직접 들어가고 action 접점의 생성
가중치도 함께 학습된다는 점이 다르다.

### 38.4 단계별 중단 기준

Wan2.2 fallback을 실행할 경우 step 수를 성공의 대리값으로 쓰지 않는다.

1. CPU/meta audit: action token shape, zero-gate base equivalence, nonzero-gate output difference, tokenizer와 각
   injection gate의 gradient를 확인한다.
2. 250 step은 배관 확인만 한다. reconstruction이 내려가도 correct-minus-wrong이 0이면 성공으로 보지 않는다.
3. 1k screen에서 train-only group holdout의 correct-vs-zero 및 correct-vs-roll 분리가 둘 다 올바른 방향이어야
   2k/5k로 연장한다.
4. 제출 CSV와 같은 action extraction을 사후 진단할 때 Wan2.1의 `0.4858`보다 내려가야 하며, 우선 목표는
   `0.40` 이하, Cosmos v2 수준은 `0.3225`다. 단, eval 전체 수치는 포렌식이며 checkpoint 선택은 독립
   train-only holdout으로 한다.
5. action이 좋아져도 DINO/video morphology가 지속적으로 나빠지면 full unfreeze 범위를 줄이거나 중단한다.
6. 10k는 위 gate를 통과한 모델에만 허용한다. 250이 짧다는 이유만으로 같은 실패 구조를 10k까지 강행하지 않는다.

현재 실행 우선순위는 `Cosmos3-Edge native 선택적 full-SFT → gate → 필요할 때 Wan2.2-5B action-token
fallback`이다. Wan2.1-14B spatial 250은 화질 기준 참고 모델로 보존하되, 현재 체크포인트 자체를 장기 연장하는
것은 action ranking 추세가 생긴다는 별도 증거가 나오기 전까지 기본 계획으로 두지 않는다.

### 38.5 1순위 구현: native BWM/Wan2.2-5B full post-training (2026-08-07)

World Model Arena 상위권이라는 이유만으로 서로 다른 모델의 loss와 adapter를 다시 섞지 않고, 먼저 BWM의
공개 구조 한 가지를 완결된 기준선으로 구현한다. 이 계획 역시 절대적이지 않으며 250/500-step gate가 반박하면
같은 구조를 10k까지 연장하지 않는다.

기존 `bwm_so100_lora` 및 `bwm_so100_eef14`와 다른 점은 다음과 같다.

- 공개 BWM `step-12000` robot checkpoint를 사용하지 않고 vanilla `Wan2.2-TI2V-5B`에서 시작한다.
- 14D dual-arm EEF를 근사하거나 18D hybrid input으로 확장하지 않는다. SO100의 정규화된 6D joint target을
  받는 BWM action encoder를 새로 초기화한다.
- DiT에 rank-32 LoRA만 붙이지 않는다. VAE는 동결하고 8-GPU FSDP로 action encoder와 실제 DiT 가중치를
  함께 post-train한다. text-off BWM forward에서 사용되지 않는 text projection 및 legacy action MLP만 동결한다.
- 첫 프레임 latent를 denoising 모든 단계에서 고정하고, 공식 BWM처럼 action을 cross-attention context와
  timestep AdaLN 두 경로에 동시에 넣는다.
- 우선 loss는 공식 BWM과 같은 flow-matching SFT 하나만 쓴다. preserve/DINO/ranking loss는 이 기준선의
  action grounding이 확인되기 전에는 섞지 않는다.

시간 정렬은 다음처럼 고정한다. 대회의 16개 command 앞에 `action[0]`을 history proxy로 하나 붙여 17개 token을
만든다. BWM encoder 내부의 3개 복제까지 포함하면 Wan VAE latent group은 정확히
`[a0×4] / [a0..a3] / [a4..a7] / [a8..a11] / [a12..a15]`가 된다. 따라서 마지막 action도 네 번째 future
latent에 들어가며 과거 구현처럼 평균 pooling으로 순서가 사라지지 않는다.

구현 파일은 다음과 같다.

- `train/bwm_native_so100.py`: train-only group split, 17-frame/17x6 action 계약
- `train/train_bwm_native_so100.py`: fresh 6D encoder + DiT full post-training, BWM-compatible checkpoint
- `train/generate_bwm_native_so100.py`: first-frame exact replacement와 normal/zero/batch-roll 생성
- `tools/audit_bwm_native_so100.py`: GPU를 쓰지 않는 asset/data/temporal-group 감사
- `train/run_bwm_native_so100.sh`: 8-GPU FSDP smoke/overfit/screen/train 및 gate

CPU 감사는 실제 119개 train group, 10,640 clips에서 video `(1,3,17,64,96)`, action `(1,17,6)`, context
`(1,17,32)`, temporal modulation `(1,5,32)`를 확인해 `PASS_BWM_NATIVE_SO100`이다. 이는 수렴이나 영상 품질을
증명하지 않으며 배관 계약만 증명한다.

실행 순서는 `audit → 5-step smoke → single-clip 250-step overfit → 500-step fixed holdout screen`이다. 250에서
loss만 내려가고 동일 clip을 복원하지 못하면 구현 또는 최적화 문제로 중단한다. 500에서 normal/zero/batch-roll이
분리되지 않거나 normal이 static보다 나쁘면 학습량 부족으로 단정하지 않는다. 두 gate를 통과할 때만 2k, 5k,
10k로 연장한다. 480x640은 BWM 공개 해상도에 맞춘 본 설정이고, 320x512는 smoke/overfit 비용 절감용이다.

### 38.6 native BWM 2k 시각 감사와 Wan2.1-I2V-14B 비교 (2026-08-07)

이 절은 38.5의 계획을 실제 실행한 뒤 생긴 **정정 기록**이다. 아래 결과는 BWM 구조 자체가 부적합하다는
일반적 결론도, Wan2.1-14B가 반드시 최종 승자라는 보장도 아니다. 현재 실행이 공개 BWM의 성능 조건을 어디까지
재현했고 어디에서 달랐는지를 구분하기 위한 기록이다.

#### 실제 실행과 체크포인트 무결성

본 5k run은 계획과 달리 `320x512`로 시작됐다. 8-GPU FSDP, global batch 8, action encoder LR `5e-5`,
DiT LR `1e-5`, gradient accumulation 1로 학습했다. 정상 저장된 체크포인트는 다음 두 개다.

| checkpoint | 크기 | 감사 결과 |
|---|---:|---|
| `open/baseline/outputs/bwm_native_so100_5k/step-1000.safetensors` | 20.36 GB | 30 DiT blocks + action encoder 포함 |
| `open/baseline/outputs/bwm_native_so100_5k/step-2000.safetensors` | 20.36 GB | `PASS_BWM_NATIVE_CHECKPOINT` |
| `step-3000.safetensors`, `step-4000.safetensors` | 각 약 462 MB | 불완전 저장, 추론 및 재개 금지 |

2k까지 8 GPU가 본 clip exposure는 약 16,000개다. repeat를 포함한 loader 길이 21,280개 대비 `0.752 epoch`,
원래 episode/camera stream 10,640개 대비 약 `1.50 pass`다. 따라서 step 숫자만 공개 BWM의 `step-12000`과
비교할 수 없다. 공개 예시 설정은 gradient accumulation 8을 사용하지만 실제 BWM 학습 레시피는 저장소에서
아직 `Training: Coming soon`으로 표시되어 있어 effective batch, 데이터 양, curriculum을 재현했다고 말할 수 없다.

2k checkpoint로 eval 216개를 30-step 생성한 뒤, 해상도와 sampler가 원인인지 분리하기 위해 같은 checkpoint를
`480x640`, 50-step으로 8개 재생성했다. 후자는 업스케일을 제거하고 공개 Wan 추론 step 수에 맞춘 검사지만,
robot morphology 붕괴가 계속됐다. 따라서 저해상도 업스케일과 30-step은 초기 결과의 blur를 더할 수 있으나
로봇의 소실·재등장·관절 변형을 설명하는 단독 원인은 아니다.

육안상 배경과 큰 정적 물체는 비교적 유지되지만 작은 로봇 영역에서 다음 실패가 반복됐다.

- 첫 입력 frame은 정확히 교체됐지만 첫 생성 future frame부터 불연속이 생긴다.
- 로봇이 덩어리로 변하거나 관절/그리퍼가 늘어나고, 일부 sample에서는 사라졌다가 다른 위치와 형태로 나타난다.
- 이는 단순 VAE reconstruction blur보다 source-conditioned generation의 identity/morphology 실패에 가깝다.

eval sample별 관찰은 이미 생성된 제출 후보를 설명하기 위한 포렌식이다. 특정 eval 장면을 학습 데이터 선택,
loss, hyperparameter 또는 checkpoint 선택에 사용하지 않는다. 중단/승격은 별도의 train-only group holdout과
normal/zero/batch-roll gate로 결정한다.

#### 14B가 선명하고 5B가 무너진 구조적 이유

`submissions/wan21_spatial_action_250`과 이번 BWM-native 2k는 단순히 parameter 수만 다른 같은 모델이 아니다.

| 항목 | Wan2.1-I2V-14B spatial 250 | native BWM/Wan2.2-TI2V-5B 2k |
|---|---|---|
| base | 전용 `Wan2.1-I2V-14B-480P` | vanilla `Wan2.2-TI2V-5B` |
| source image | `has_image_input=True`; VAE image condition + CLIP image embedding | `has_image_input=False`; clean first-frame latent prefix fusion |
| semantic condition | 고정 robot prompt, deformation negative prompt, CFG 5 | text encoder 미사용, CFG 1 |
| backbone update | 14B base 보존, 2.69M adapter + 76.68M LoRA | fresh 47.5M action encoder + 실제 4.41B DiT 전체 갱신 |
| inference | 480x640, 50 step | 최초 320x512, 30 step; 480x640/50도 별도 실패 확인 |
| 확인된 장점 | source robot의 형태와 질감이 상대적으로 선명 | 빠르고 action 접점은 넓지만 morphology가 아직 불안정 |
| 확인된 병목 | 공개 `0.25`, Action MAE `0.48580`; action grounding 약함 | 공개 제출 전; 시각 gate를 통과하지 못함 |

14B는 첫 이미지를 latent와 CLIP semantic context로 중복 고정하고, prompt/negative prompt까지 사용한다. 또한
학습 중 원본 생성 prior의 99% 이상을 직접 바꾸지 않았다. 반면 현재 5B는 첫 latent를 경계조건으로 융합하지만
별도 image encoder가 없고, 공개 robot world-model weight 없이 text-off 상태에서 SO100 dynamics와 action
mapping을 동시에 새로 학습했다. 그러므로 14B의 높은 완성도와 5B의 붕괴 차이를 `14B > 5B`라는 용량 하나로
환원할 수 없으며, **source-conditioning 방식, 초기 checkpoint, 학습 범위**가 함께 다르다.

#### 공개 BWM을 그대로 재현한 것이 아닌 이유

공개 BWM inference는 vanilla Wan이 아니라 `BLM-Lab/Boundless-World-Model`의 `step-12000.safetensors`를
추가로 로드한다. 공개 profile도 14D dual-arm `eef_abs/state_pose`, 57 RGB frames, 9 history frames,
672x896을 사용한다. 현재 대회 경로는 6D SO100 absolute joint command, 17 RGB frames, 1 history image다.
아키텍처의 `encode_ti2v2`와 AdaLN/context 주입 코드는 가져왔지만 다음은 재현하지 못했다.

1. RoboTwin robot dynamics를 이미 학습한 BWM DiT/action weight
2. 14D EEF/state와 SO100 6D joint command 사이의 embodiment/action semantics
3. 9-frame history가 주는 속도·접촉·가림 단서
4. 비공개 training data mixture, optimizer schedule, effective batch와 curriculum

따라서 현재 모델의 정확한 명칭은 **`BWM-style architecture initialized from vanilla Wan2.2-5B`**다. 공개
BWM 자체를 SO100에 그대로 fine-tune한 모델로 기록하지 않는다.

#### 조건부 다음 방향

현재 5B full-SFT를 step 수만 늘리는 것은 기본 계획에서 내린다. 다만 2k가 1 epoch 미만이므로 학습 부족 가능성
자체를 부정하지도 않는다. 장기 연장은 train-only holdout에서 morphology와 action counterfactual이 함께
개선된다는 증거가 있을 때만 허용한다.

현 시점의 우선 후보는 다음 두 갈래다.

1. 공개 BWM `step-12000`의 DiT와 14D action path를 보존하고, SO100 6D를 14D BWM action space로 연결하는
   bridge만 먼저 학습한다. 이 경우에도 9-history/dual-arm domain gap은 별도 gate 대상이다.
2. 이미 source fidelity가 확인된 Wan2.1-I2V-14B를 유지하고, base 전체가 아니라 action branch와 낮은 비율의
   LoRA를 train-only holdout에서 1k~2k까지 검증한다. 목표는 화질을 다시 만드는 것이 아니라 Action MAE와
   correct-vs-wrong 분리를 개선하는 것이다.

둘 중 어느 것도 절대적인 최종안이 아니다. 다만 이번 비교로 `vanilla TI2V-5B + fresh action encoder +
text-off full-SFT`가 공개 BWM 영상 품질을 자동으로 재현한다는 가정은 반박됐다.
