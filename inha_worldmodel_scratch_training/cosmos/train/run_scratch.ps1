# 처음부터 학습 — 챔피언까지 쌓아온 기법을 한 사슬로 합쳐 새로 돌린다.
#
# 챔피언(run_train.ps1)과 다른 점 세 가지:
#   1) A구간은 중복 없이 전 창을 정확히 한 번씩 본다 (커버리지 100%).
#      챔피언은 중복 뽑기라 창의 54%, 프레임의 73% 만 봤다.
#   2) 판독기 벌점과 데이터 정제를 step 0 부터 켠다.
#      챔피언은 35,000 부터였다. 대신 벌점 가중치를 10,000 노출에 걸쳐 0 -> 0.3 으로 올린다
#      (초반에는 모델이 아직 장면을 못 만들어 판독기 채점이 잡음이다).
#   3) 샘플러 시드를 박는다. 챔피언까지는 시드가 없어서, 같은 설정으로 다시 돌려도
#      다른 창을 뽑았다. 그래서 8번의 가중치 실험을 원인까지 가르지 못했다.
#
# DRaFT 만은 끝에 남긴다. 3스텝 전개로 실제 영상을 만들어 채점하는 방식이라
# 아직 아무것도 못 만드는 모델에 걸면 보상이 순수 잡음이다. 램프로도 못 피한다.
#
# 총 150,038 노출 / 약 37시간 (로컬 RTX 5070 Ti 실측 0.88~0.95초/노출).
#
# ★ 챔피언 자산은 건드리지 않는다. RUN_NAME 은 v17_scratch 고정이고
#   runs/v6_draft/ 와 submission_auto06_s30.csv 는 읽지도 쓰지도 않는다.
#
# 사용법:
#   powershell -File run_scratch.ps1              # A -> B -> soup -> D 전부
#   powershell -File run_scratch.ps1 -Stage A     # 한 구간만
#   powershell -File run_scratch.ps1 -MotionPow 0.3   # B구간 편향 조절

param(
    [ValidateSet("all", "A", "B", "soup", "D")]
    [string]$Stage = "all",
    [double]$MotionPow = 0.5,          # B구간 움직임 가중. 1.0=챔피언(편향 140%), 0.5=120%, 0=100%
    [int]$Seed = 1234
)

$ErrorActionPreference = "Continue"
$R  = "C:\Users\m\Desktop\인하인공지능챌린지"
$PY = "$R\.venv\Scripts\python.exe"
$RUN = "v17_scratch"

# 노출 경계. A는 정제로 배제된 창 290개를 뺀 실제 뽑기 수와 일치해야 한다.
$A_END = 104538
$B_END = 144538
$D_END = 150038

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED  = "1"
Set-Location "$R\cosmos\train"

# 로그 인코딩: Tee-Object 는 PS 5.1 에서 UTF-16 으로 써서 grep 이 못 읽는다.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Reset-Env {
    foreach ($v in "DRAFT", "ATOK", "ACT_DROP", "PERBLOCK", "FULLFT", "RUN_CKPT",
                   "DELTA", "ABS_AUG", "ABS_AUG_SCALE", "RESUME_FROM", "MAX_STEPS",
                   "NOREPL", "SCRATCH", "AUX_WARMUP", "WIDEN_FROM") {
        Set-Item "Env:\$v" ""
    }
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:BATCH        = "1"        # 빠뜨리면 40배 느려진다 (시스템 RAM 스필)
    $env:LR           = "5e-5"
    $env:SAVE_EVERY   = "5000"     # soup 재료로 쓴다
    $env:SAMPLER_SEED = "$Seed"
}

function Run-Stage([string]$tag, [int]$stop) {
    $log = "$R\cosmos\train\runs\${RUN}_${tag}.log"
    New-Item -ItemType Directory -Force "$R\cosmos\train\runs\$RUN" | Out-Null
    $env:RUN_NAME = $RUN
    $env:STOP_AT  = "$stop"
    Write-Output "[$tag] 시작 $(Get-Date -Format 'MM-dd HH:mm:ss')  ->  $stop 노출"
    & $PY train_lora.py 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    Write-Output "[$tag] 종료 $(Get-Date -Format 'MM-dd HH:mm:ss')"
    Write-Output "  로그: $log"
}

# --- A: 전수 순회 104,538 (중복 없음) ---
# 벌점·정제는 켜되 벌점 가중치는 10,000 까지 램프. DRaFT 없음.
if ($Stage -in @("all", "A")) {
    Reset-Env
    $env:SCRATCH    = "1"          # 웜스타트 금지 (조용한 폴백 차단)
    $env:NOREPL     = "1"          # 전 창 정확히 한 번씩
    $env:AUX        = "1"; $env:AUX_JUDGE = "v2"; $env:W_AUX = "0.3"; $env:P_AUX = "0.3"
    $env:AUX_WARMUP = "10000"
    $env:CLEAN      = "1"
    Run-Stage "A_fullpass" $A_END
}

# --- B: 가중 뽑기 40,000 ---
if ($Stage -in @("all", "B")) {
    Reset-Env
    $env:AUX        = "1"; $env:AUX_JUDGE = "v2"; $env:W_AUX = "0.3"; $env:P_AUX = "0.3"
    $env:CLEAN      = "1"
    $env:MOTION_POW = "$MotionPow"
    $env:RESUME_FROM = "$R\cosmos\train\runs\$RUN\latest.pt"
    Run-Stage "B_weighted" $B_END
}

# --- soup: B구간 네 지점 평균 ---
if ($Stage -in @("all", "soup")) {
    $env:SOUP_DIR = "$R\cosmos\train\runs\$RUN"
    Write-Output "[soup] 129538 / 134538 / 139538 / 144538 평균"
    & $PY make_soup.py 129538 134538 139538 144538 2>&1 | Select-String "added|saved"
}

# --- D: DRaFT 5,500 ---
if ($Stage -in @("all", "D")) {
    Reset-Env
    $env:AUX        = "1"; $env:AUX_JUDGE = "v2"; $env:W_AUX = "0.3"; $env:P_AUX = "0.3"
    $env:CLEAN      = "1"
    $env:MOTION_POW = "$MotionPow"
    $env:DRAFT      = "1"; $env:DRAFT_K = "3"; $env:DRAFT_P = "0.5"
    $env:W_R        = "1.0"; $env:W_ANCHOR = "1.0"
    $env:RESUME_FROM = "$R\cosmos\train\runs\$RUN\soup_129538_134538_139538_144538.pt"
    Run-Stage "D_draft" $D_END
    Write-Output ""
    Write-Output "완료. 제출 후보 4개:"
    Write-Output "  A끝   runs\$RUN\step_$($A_END.ToString('000000')).pt"
    Write-Output "  B끝   runs\$RUN\step_$($B_END.ToString('000000')).pt"
    Write-Output "  soup  runs\$RUN\soup_129538_134538_139538_144538.pt"
    Write-Output "  D끝   runs\$RUN\step_$($D_END.ToString('000000')).pt"
    Write-Output ""
    Write-Output "생성은 run_infer.ps1 -Ckpt <경로> -Tag <이름> 으로."
}

# --- 시작 3분 안에 반드시 확인할 것 ---
Write-Output ""
Write-Output "확인: 로그 첫 줄에 'SCRATCH=1' 과 'sampler: 전수 1회(중복 없음) seed=$Seed' 가 찍혔는지."
Write-Output "      s/it 약 0.88~0.95, mem 14.4GB, 전력 240W 안팎이면 정상."
