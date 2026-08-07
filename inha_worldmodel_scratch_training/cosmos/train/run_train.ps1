# 챔피언 가중치를 처음부터 재현하는 학습 3단계.
#
# 사슬:
#   사전학습 베이스
#     -> v1_lora        흐름정합만
#     -> v3_aux         + 판독기 보조손실 + 데이터 정제 + 움직임 가중
#     -> soup 75k~90k   가중치 평균 4개 (make_soup.py)
#     -> v6_draft       DRaFT 보상 미세조정 -> step_095500.pt   ★최종
#
# 선행 조건 (SOLUTION.md 4절 순서대로):
#   build_index.py -> precompute_latents.py -> compute_motion.py
#   -> train_latent_idm2.py -> score_train_windows.py
#
# ★ BATCH=1 필수. 잠재 모드 기본값 2 는 16GB 를 넘겨 Windows 가 시스템 RAM 으로 넘기면서
#   죽지 않고 40배 느려진다(1.24 -> 60.7초/스텝). 시작 3분 안에 s/it 과 mem 을 확인할 것.
#   정상값: 약 0.95초/it, mem 14.4GB. 전력 240W 안팎.

param(
    [ValidateSet("all", "v1", "v3", "soup", "draft")]
    [string]$Stage = "all"
)

$ErrorActionPreference = "Continue"
$R  = "C:\Users\m\Desktop\인하인공지능챌린지"
$PY = "$R\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED  = "1"
Set-Location "$R\cosmos\train"

# 학습 로그 인코딩: Tee-Object 는 PS 5.1 에서 UTF-16 으로 써서 grep 이 못 읽는다.
# 파이프라인 인코딩을 UTF-8 로 맞추고 Out-File 로 저장한다.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Reset-Env {
    foreach ($v in "DRAFT", "ATOK", "ACT_DROP", "PERBLOCK", "FULLFT", "RUN_CKPT",
                   "DELTA", "ABS_AUG", "ABS_AUG_SCALE", "RESUME_FROM", "MAX_STEPS") {
        Set-Item "Env:\$v" ""
    }
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:BATCH      = "1"          # 빠뜨리면 40배 느려진다
    $env:LR         = "5e-5"
    $env:MOTION_POW = "1"          # 움직임 가중 (균등 샘플링은 후퇴 확인됨)
    $env:SAVE_EVERY = "2500"
}

function Run-Stage([string]$name, [int]$stop) {
    $log = "$R\cosmos\train\runs\${name}_train.log"
    New-Item -ItemType Directory -Force "$R\cosmos\train\runs\$name" | Out-Null
    $env:RUN_NAME = $name
    $env:STOP_AT  = "$stop"
    Write-Output "[$name] 시작 $(Get-Date -Format 'HH:mm:ss') -> $stop 노출"
    & $PY train_lora.py 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    Write-Output "[$name] 종료 $(Get-Date -Format 'HH:mm:ss')"
}

# --- 1단계: 흐름정합만 ---
if ($Stage -in @("all", "v1")) {
    Reset-Env
    Run-Stage "v1_lora" 35000
}

# --- 2단계: 보조손실 + 정제 + 움직임 가중 ---
if ($Stage -in @("all", "v3")) {
    Reset-Env
    $env:AUX = "1"; $env:AUX_JUDGE = "v2"; $env:W_AUX = "0.3"; $env:P_AUX = "0.3"
    $env:CLEAN = "1"
    $env:RESUME_FROM = "$R\cosmos\train\runs\v1_lora\latest.pt"
    Run-Stage "v3_aux" 90000
}

# --- 3단계: 가중치 평균(soup) ---
if ($Stage -in @("all", "soup")) {
    Write-Output "[soup] 75k/80k/85k/90k 평균"
    & $PY make_soup.py 2>&1 | Select-String "saved|soup"
}

# --- 4단계: DRaFT 보상 미세조정 -> 챔피언 ---
if ($Stage -in @("all", "draft")) {
    Reset-Env
    $env:AUX = "1"; $env:AUX_JUDGE = "v2"; $env:W_AUX = "0.3"; $env:P_AUX = "0.3"
    $env:CLEAN = "1"
    $env:DRAFT = "1"; $env:DRAFT_K = "3"; $env:DRAFT_P = "0.5"
    $env:W_R = "1.0"; $env:W_ANCHOR = "1.0"
    $env:RESUME_FROM = "$R\cosmos\train\runs\v3_aux\soup_75000_80000_85000_90000.pt"
    Run-Stage "v6_draft" 95500
    Write-Output ""
    Write-Output "★ 최종 가중치: runs\v6_draft\step_095500.pt"
    Write-Output "  추론은 run_infer.ps1 로."
}
