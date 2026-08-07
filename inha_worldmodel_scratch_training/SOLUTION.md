# 최종 해법 — SO-100 액션 조건 영상 생성

초기 프레임 1장(640×480)과 16스텝 6축 관절 명령을 받아 **16프레임 영상**을 생성한다.
점수(낮을수록 좋음) = `0.3 × DINO + 0.3 × Video(r3d_18) + 0.4 × Action MAE`.

**최종 성적 0.1872440957** — 제출물 `submission_auto06_s30.csv`

---

## 1. 최종 구성

```
기반 모델   Cosmos-Predict2.5-2B (action-cond, ema_bf16)  — 전부 동결
학습 대상   LoRA r32 + 액션 임베더 = 113.4M 파라미터
가중치      cosmos/train/runs/v6_draft/step_095500.pt     (95,500 샘플 노출)

추론        30스텝 · shift 5.0 · 오토가이던스 0.6 · seed 7 · LoRA 배율 1.0
            액션 CFG 없음 · 일반 CFG 없음 · 시드 평균 없음
인코딩      libx264 · crf 12 · preset slow · yuv420p · 6fps
```

### 왜 이 값들인가

10개 축을 리더보드로 전수 측정했고 **전부 위 값이 최적**이었다(상세는 `docs/EXPERIMENTS.md`).

| 축 | 재본 값 | 확정 |
|---|---|---|
| 학습 길이·레시피 | 6방향 | 95,500 |
| LoRA 어댑터 배율 | 0.844 / **1.0** / 1.156 | 1.0 |
| 샘플링 스텝 | 20 / **30** / 45 | 30 |
| shift | 3.0 / **5.0** | 5.0 |
| 오토가이던스 | 0.45 / **0.6** / 0.75 | 0.6 |
| 액션 CFG | 0 / 0.5 / 1.0 / 2.0 / 구간제한 | 사용 안 함 |
| 인코딩 색차 | **yuv420p** / yuv444p | yuv420p |
| 시드 평균 · 후보선택 | 시도 후 폐기 | 사용 안 함 |

---

## 2. 모델 구조

### 액션 주입

사전학습 모델의 액션 임베더는 7축용이라 SO-100(6축)과 입력 차원이 맞지 않는다.
`fc1`은 새로 초기화하고 `fc2`만 사전학습 가중치를 쓴다.

액션 `(16, 6)`은 **`(4, 24)`로 재배열**된다 — 잠재 프레임 4개가 각각 **연속 액션 4개**를
받는다(`temporal_compression_ratio=4`). 즉 프레임별로 다른 액션이 주입된다.

| 경로 | 어디로 |
|---|---|
| `action_embedder_B_D` | 타임스텝 임베딩에 가산 |
| `action_embedder_B_3D` | adaLN 변조에 가산 (28블록 전부의 정규화를 조절) |

### LoRA 대상

`q_proj`·`k_proj`·`v_proj`·`output_proj`(각 56개) + `layer1`·`layer2`(각 28개) = 280개 모듈.
`r=32`, `alpha=32`, dropout 0.

---

## 3. 추론 절차

1. 초기 프레임을 letterbox로 320×512에 맞추고 VAE 인코딩 → `gt_latent`
2. 액션을 **train 통계**로 z-정규화 (`data/train/so100_action_statistics.json`)
3. 순수 노이즈에서 30스텝 반복
   - **첫 잠재 프레임을 정답으로 덮어씀** (초기 프레임이 조건이므로)
   - 속도 예측 `v_cond`
   - **오토가이던스**: LoRA를 끈 기본 모델로 `v_base`를 구해
     `v = v_cond + 0.6 × (v_cond − v_base)`
   - `FlowUniPCMultistepScheduler`로 한 걸음 이동
4. VAE 디코딩 → 16프레임 → mp4

오토가이던스는 **미세조정이 배운 방향을 증폭**한다. `net.disable_adapter()`로 LoRA만 끄면
되므로 별도 모델이 필요 없고, 스텝당 순전파가 2회로 늘어난다.

---

## 4. 재현 절차

```powershell
python cosmos\train\build_index.py            # 1) 에피소드 목록
python cosmos\train\precompute_latents.py     # 2) 잠재 사전계산 (반나절, 40GB)
python cosmos\train\compute_motion.py         # 3) 움직임 가중치
python cosmos\train\train_latent_idm2.py      # 4) 자체 판독기 v2 (보조손실용)
python cosmos\train\score_train_windows.py    # 5) 데이터 정제 채점
powershell -File cosmos\train\run_train.ps1   # 6) 본 학습 (3단계, 95,500 노출)
powershell -File cosmos\train\run_infer.ps1   # 7) 추론 + 제출 CSV
```

### 학습 사슬

```
사전학습 베이스
  → v1_lora          흐름정합만
  → v3_aux           + 판독기 보조손실(W_AUX 0.3, P_AUX 0.3) + 데이터 정제 + 움직임 가중
  → soup 75k~90k     가중치 평균 4개 (make_soup.py)
  → v6_draft         DRaFT 보상 미세조정 → step_095500.pt   ★최종
```

### 손실 구성

기본은 **흐름정합 MSE**다. `x₀`와 노이즈를 섞은 `x_t`에서 예측 속도와 `(noise − x₀)`의
차이를 잰다. **첫 잠재 프레임은 항상 정답으로 고정**한다.

30% 확률로 **판독기 보조손실**이 붙는다. 예측된 `x₀`에서 액션을 읽어내는 자체 판독기를
통과시켜 명령 액션과의 L1 오차를 벌점으로 건다(가중치 0.3).

마지막 단계는 **DRaFT** — 저노이즈에서 3스텝 오일러 전개로 영상을 만든 뒤 판독기 보상과
원본 앵커를 건다. 미분은 마지막 스텝만 한다(전체 미분은 29GB로 OOM).

---

## 5. 비용 (규정 한도 대비)

| | 로컬 (RTX 5070 Ti 16GB) | 기준장비 환산 | 한도 |
|---|---|---|---|
| 학습 전체 | 약 2일 | 약 1.3일 | 4일 |
| 추론 216샘플 | 57.9분 | 약 29분 | 1시간 |

기준장비(RTX PRO 6000 96GB) 환산 = 하드웨어 2.5배 × 배치 1.5~2배, 보수적으로 3배.
학습은 `BATCH=1` + 8누적인데 96GB에서는 `BATCH=8`을 한 번에 올릴 수 있다.

**로컬 환경 주의**: 우리 카드는 16.3GB라 학습(15.1GB)과 데스크톱 앱이 경합하면 Windows가
학습 VRAM을 시스템 RAM으로 밀어내 0.88 → 1.45초/it로 65% 느려진다. nvidia-smi는 정상으로
보이며 **전력이 253W → 186W로 떨어지는 것**이 유일하게 빠른 신호다. 기준장비에는 없는 문제다.

---

## 6. 규정 준수

- **제출킷은 최종 mp4 확정 후 CSV 변환에만, 무수정 1회 실행한다.** 학습·추론·후처리
  어디에도 킷의 모델·코드·checkpoint를 쓰지 않는다.
- 보조손실 판독기(`latent_idm2`)는 **대회 train 잠재와 정답 액션만으로 자체 학습**한 모델이다.
- 성분 분해값(Action/Visual)을 산출하지 않는다. 모든 판단은 리더보드 총점으로만 한다.
- eval 데이터는 **주어진 조건 입력**(초기 프레임, 명령 액션)을 추론에 쓰는 것 외에
  사용하지 않는다. 8/2~8/4 사이의 경위는 `docs/EXPERIMENTS.md` 자진 기재 절 참조.
- 과거 규정 위반 우려로 격리한 코드는 `_archive/`에 있으며 실행 경로에 없다.

---

## 7. 폴더 구조

```
cosmos/
  checkpoints/          기반 모델·VAE·텍스트 임베딩 (4.5GB)
  cosmos-predict2.5/    모델 코드 (NVIDIA 원본)
  port/                 Windows 호환 스텁 + 모델 빌더
  train/
    *.py *.ps1          재현에 필요한 것만
    latents/            사전계산 잠재 104,828창 (40GB)
    runs/
      v6_draft/         ★챔피언 가중치
      latent_idm2/      보조손실 판독기
      data_clean/       정제 가중치
      gen_auto06_s30/   챔피언 생성물 216개
      input_videos_backup/   초기 프레임 백업
data/                   train 10,735 + val 236 에피소드, eval 216
submission_kit/         공식 킷 (무수정)
docs/EXPERIMENTS.md     실험 전체 기록
submission_auto06_s30.csv   ★최종 제출물
```
