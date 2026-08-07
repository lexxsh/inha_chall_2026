# Artifact policy and local paths

이 저장소는 소스와 Markdown 연구 기록을 보존하고, 대용량/민감 가능성이 있는 artifact는 Git에서 제외합니다. 문서에 적힌 절대 경로는 원 연구 환경의 provenance이며 다른 환경에서는 해당 경로를 재설정해야 합니다.

## Git에 포함하지 않은 항목

- competition train/eval/holdout 데이터와 LeRobot 원본 데이터
- Wan, Cosmos, DreamZero, IRASim, HMA 등 공개 base weight
- 학습 checkpoint (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`)
- VAE latent, DINO cache, precompute cache
- dataset/window index parquet와 derived manifest (코드로 재생성)
- 생성 MP4, diagnostics frame, 제출 CSV
- Python virtual environment, package/download cache, logs

## 보존된 핵심 artifact provenance

| 후보 | 원 checkpoint | 원 제출 디렉터리 |
|---|---|---|
| Cosmos Unified Action v2 13,067 | `inha_worldmodel_scratch_training/cosmos/train/runs/cosmos_unified_action_v2/latest.pt` | `submissions/cosmos_unified_action_v2_13067/` |
| Wan2.1 Spatial Action 250 | `open/baseline/outputs/wan21_spatial_action_250/step-250.safetensors` | `submissions/wan21_spatial_action_250/` |
| BWM-style Wan2.2-5B 2K | `open/baseline/outputs/bwm_native_so100_5k/step-2000.safetensors` | 로컬 2K 생성/CSV, 제출 보류 |

두 retained 후보의 checkpoint와 CSV SHA-256은 [RETAINED_CANDIDATES.md](RETAINED_CANDIDATES.md)에 기록했습니다. 결과 지표 중 1MB 미만 JSON/CSV는 `results/`에 포함했고, feature tensor와 공식 제출 CSV처럼 큰 파일은 제외했습니다.

공식 제출을 재현할 때는 영상을 복원한 뒤 competition submission kit의 `make_submission_csv.py`를 그대로 사용해야 합니다. CSV를 수동 수정하거나 비공식 feature extractor로 대체하지 않습니다.
