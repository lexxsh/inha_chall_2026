# 최종 추론 — eval 216샘플 mp4 생성 후 제출 CSV 변환.
#
# 이 스크립트 하나가 제출물을 만든다. 리더보드 0.1872440957 을 낸 설정 그대로다.
# 검증: 정리 후 3샘플 재생성 결과가 기존 산출물과 비트 단위로 동일함을 확인했다(2026-08-05).
#
# 순서는 반드시 이래야 한다 (대회 규정):
#   참가자 코드로 mp4 생성 -> mp4 확정 -> 공식 제출킷을 **수정 없이** 실행 -> CSV
# 킷의 모델·코드·checkpoint 는 학습·추론·후처리 어디에도 쓰지 않는다.

param(
    [string]$Ckpt   = "",          # 비우면 챔피언
    [string]$Tag    = "s30",       # 출력 폴더/CSV 접미사
    [int]$Steps     = 30,
    [double]$Shift  = 5.0,
    [double]$AutoG  = 0.6,
    [int]$Subset    = 0            # >0 이면 N개마다 하나만 (설정 점검용, 제출 금지)
)

$ErrorActionPreference = "Continue"
$R  = "C:\Users\m\Desktop\인하인공지능챌린지"
$PY = "$R\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED  = "1"

if (-not $Ckpt) { $Ckpt = "$R\cosmos\train\runs\v6_draft\step_095500.pt" }
if (-not (Test-Path $Ckpt)) { Write-Output "중단: 가중치가 없다 — $Ckpt"; exit 1 }

Set-Location "$R\cosmos\train"
$env:RUN_CKPT     = $Ckpt
$env:STEPS        = "$Steps"
$env:SHIFT        = "$Shift"
$env:AUTO_G       = "$AutoG"
$env:SEED         = "7"
$env:LORA_SCALE   = "1.0"
$env:SUBSET       = "$Subset"
$env:OUTDIR       = "$R\cosmos\train\runs\gen_auto06_$Tag"
# 아래는 전부 "쓰지 않음". 측정으로 확정된 값이므로 바꾸지 말 것.
$env:ACFG = "0"; $env:ANULL = "0"; $env:ACFG_LO = "0"; $env:ACFG_HI = "1"
$env:ACFG_RESCALE = "0"; $env:GUIDANCE = "0"; $env:NAVG = "1"
$env:PIX_FMT = ""            # 빈 값 = libx264 기본(yuv420p). 444p 는 후퇴 확인됨
foreach ($v in "DELTA", "ABS_AUG", "ABS_AUG_SCALE", "ATOK", "PERBLOCK", "FULLFT") {
    Set-Item "Env:\$v" ""
}

$est = [math]::Round((1.5 + 0.48 * $Steps) * 216 / 60, 0)
Write-Output "생성 시작 $(Get-Date -Format 'HH:mm:ss')  — $Steps 스텝, 예상 $est 분"
Write-Output "  가중치 $(Split-Path $Ckpt -Leaf)   shift $Shift   오토가이던스 $AutoG"
& $PY generate_eval.py 2>&1 | Select-String -Pattern "설정|elapsed|done|Traceback|Error"

if ($Subset -gt 0) {
    Write-Output "정찰 모드($Subset) — CSV 를 만들지 않는다"
    exit 0
}

# 제출킷은 여기서만, 수정 없이 실행한다
Set-Location "$R\submission_kit"
& $PY make_submission_csv.py --prediction-root "$R\cosmos\train\runs\gen_auto06_$Tag" `
    --output-csv "$R\submission_auto06_$Tag.csv" 2>&1 | Select-String "saved to"

$csv = "$R\submission_auto06_$Tag.csv"
if (Test-Path $csv) {
    $rows = (Get-Content $csv | Measure-Object -Line).Lines
    $ok = if ($rows -eq 649) { "정상" } else { "★비정상★ (649 여야 한다)" }
    Write-Output "완료 $(Get-Date -Format 'HH:mm:ss') — $csv  $rows 행  $ok"
} else {
    Write-Output "[경고] CSV 가 만들어지지 않았다"
}
