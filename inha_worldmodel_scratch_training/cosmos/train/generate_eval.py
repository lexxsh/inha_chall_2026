# 평가 216샘플 전체 추론 → <OUTDIR>/*.mp4 (16프레임)
#
# 설정은 전부 환경변수로 받고 manifest.json 에 기록한다. 예전에는 출력 폴더가 고정이고
# 기존 파일을 조용히 건너뛰어서, 설정을 바꿔 다시 돌려도 이전 영상이 그대로 남아 있었다.
# 어떤 설정이 어떤 영상을 만들었는지 사후에 알 수 없었던 것이 그 버그의 실질적 피해다.
#
#   OUTDIR   출력 폴더 (기본 submission_kit/input_videos)
#   STEPS    샘플링 스텝 (기본 20 — 챔피언 submission_single_draft 와 동일)
#   SHIFT    스케줄러 shift (기본 5.0)
#   SEED     노이즈 시드 (기본 7)
#   GUIDANCE 조건프레임 널 CFG (기본 0 = 미사용, 이미지 앵커를 깨므로 권장하지 않음)
#   ACFG     액션 CFG 배율 (기본 0). val 최적 0.5. 추론 시간 2배
#   ANULL    액션 CFG 의 널 (0=0벡터, 1=초기자세 유지). 실측상 0 이 더 좋다
#   RESUME   1이면 이미 있는 mp4 를 건너뛴다 (중단 후 이어받기 전용, 기본 0)
#   OVERWRITE 1이면 설정이 다른 기존 mp4 를 지우고 진행 (기본은 중단)
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.distributed as dist
import imageio
from PIL import Image

from validate import load_finetuned, generate, to_uint8_16, write_mp4, CKPT_DIR, TEXT_EMB, RUN_CKPT

_PACKAGED_EVAL_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eval"
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = Path(os.environ.get(
    "SO100_EVAL_ROOT",
    str(_PACKAGED_EVAL_ROOT if _PACKAGED_EVAL_ROOT.exists() else _WORKSPACE_ROOT / "open" / "data" / "eval"),
))
DEFAULT_OUT = _WORKSPACE_ROOT / "open" / "submission_kit" / "input_videos"
OUT_DIR = Path(os.environ.get("OUTDIR", str(DEFAULT_OUT)))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "port"))
from generate_zeroshot import letterbox  # noqa: E402


def main():
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    torch.cuda.set_device(local_rank)
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl")
    is_main = rank == 0

    cfg = {
        "ckpt": str(RUN_CKPT),
        "steps": int(os.environ.get("STEPS", "20")),
        "shift": float(os.environ.get("SHIFT", "5.0")),
        "seed": int(os.environ.get("SEED", "7")),
        "guidance": float(os.environ.get("GUIDANCE", "0")),
        # 액션 CFG. val 40클립 기준 0.5 가 최적(시각 쌍대차분 -0.0026±0.0013), 널은 0벡터(anull=0).
        # 스텝당 순전파가 2배라 추론 시간도 2배다.
        "acfg": float(os.environ.get("ACFG", "0")),
        "anull": int(os.environ.get("ANULL", "0")),
        # 유도를 거는 노이즈 구간과 분산 되돌리기. 기본값은 무효과(전 구간·되돌리기 없음)라
        # LB 0.1925 를 낸 설정(ACFG=1.0 ANULL=0)이 그대로 재현된다.
        "acfg_lo": float(os.environ.get("ACFG_LO", "0")),
        "acfg_hi": float(os.environ.get("ACFG_HI", "1")),
        "acfg_rescale": float(os.environ.get("ACFG_RESCALE", "0")),
        # 오토가이던스: 널 = LoRA 끈 기본 모델. 스텝당 순전파 +1.
        "auto": float(os.environ.get("AUTO_G", "0")),
        # 시드 평균: N개 표본을 픽셀공간에서 평균한다.
        # 채점 지표는 사후분포의 **평균**에 가까울수록 좋은데 확산 샘플러는 **표본**을 준다.
        # 그 차이(시드 난수 성분)는 전부 손실이다. 잠재 평균이 아니라 픽셀 평균인 이유는
        # 디코더가 비선형이라 잠재 평균의 디코드는 "평균 영상"이 아니기 때문이다.
        "navg": int(os.environ.get("NAVG", "1")),
        # LoRA 어댑터 실효 강도(alpha/r). 1.0 = 학습된 그대로. manifest 에 반드시 남긴다 —
        # 이 값이 다르면 같은 체크포인트라도 완전히 다른 모델이고, 안 적으면 나중에 구분이 안 된다.
        "lora_scale": float(os.environ.get("LORA_SCALE", "1.0")),
        "atok": int(os.environ.get("ATOK", "0")),
        "atok_gate_scale": float(os.environ.get("ATOK_GATE_SCALE", "1.0")),
        "pix_fmt": os.environ.get("PIX_FMT", "") or "libx264-default(yuv420p)",
        "encode": "crf12/preset-slow",
    }
    resume = os.environ.get("RESUME", "0") == "1"
    if is_main:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    if is_main:
        stale = sorted(OUT_DIR.glob("sample_*.mp4"))
        if stale and not resume:
            prev = OUT_DIR / "manifest.json"
            old = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else None
            if old != cfg:
                # 설정이 다른 mp4 를 그냥 덮어쓰면 어떤 설정의 산출물인지 알 수 없게 된다.
                # 그렇다고 말없이 지우면 출처 불명의 과거 제출본을 날릴 수 있으므로 멈춘다.
                if os.environ.get("OVERWRITE") != "1":
                    raise SystemExit(
                        f"[중단] {OUT_DIR} 에 설정이 다른 mp4 {len(stale)}개가 있다.\n"
                        f"  기존 설정: {old}\n  요청 설정: {cfg}\n"
                        f"  → OUTDIR 로 다른 폴더를 지정하거나, 지워도 되면 OVERWRITE=1 을 주라."
                    )
                print(f"[정리] OVERWRITE=1 — 기존 mp4 {len(stale)}개 삭제 (이전={old})", flush=True)
                for p in stale:
                    p.unlink()
        print(f"[설정] out={OUT_DIR} {cfg} resume={resume} world_size={world_size}", flush=True)
    if distributed:
        dist.barrier()

    device = torch.device("cuda", local_rank)
    net = load_finetuned(device)
    vae = WanVAE(z_dim=16, vae_pth=str(CKPT_DIR / "tokenizer.pth"), dtype=torch.bfloat16, device="cuda")
    text_emb = torch.load(TEXT_EMB, map_location="cpu", weights_only=False).to(device, torch.bfloat16)

    stats = json.loads((EVAL_ROOT.parent / "train" / "so100_action_statistics.json").read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    ids = sorted(p.stem for p in (EVAL_ROOT / "images").glob("sample_*.png"))
    # SUBSET=k 면 k개마다 하나만 — 설정 정찰용(움직임·떨림만 재고 제출하지 않는다).
    # own_bon.py 와 같은 규약이다.
    stride = int(os.environ.get("SUBSET", "0"))
    if stride:
        ids = ids[::stride]
        if is_main:
            print(f"[정찰] {len(ids)}개만 생성 (SUBSET={stride})", flush=True)
    total_ids = len(ids)
    ids = ids[rank::world_size]
    print(f"[rank {rank}] assigned {len(ids)}/{total_ids} samples on cuda:{local_rank}", flush=True)
    t_start = time.time()
    for n, sid in enumerate(ids):
        out = OUT_DIR / f"{sid}.mp4"
        if resume and out.exists() and out.stat().st_size > 0:
            continue
        image = np.asarray(Image.open(EVAL_ROOT / "images" / f"{sid}.png").convert("RGB"))
        frame0 = letterbox(image, 320, 512) * 2.0 - 1.0  # (3,320,512) [-1,1]
        act = (np.load(EVAL_ROOT / "actions" / f"{sid}.npy").astype(np.float32) - mean) / std
        acts = torch.from_numpy(act)
        gens = []
        for j in range(max(1, cfg["navg"])):
            gens.append(generate(net, vae, text_emb, frame0, acts, device,
                                 guidance=cfg["guidance"], seed=cfg["seed"] + 997 * j,
                                 num_steps=cfg["steps"], shift=cfg["shift"],
                                 action_guidance=cfg["acfg"], anull=cfg["anull"],
                                 acfg_lo=cfg["acfg_lo"], acfg_hi=cfg["acfg_hi"],
                                 acfg_rescale=cfg["acfg_rescale"], auto_guidance=cfg["auto"])[0])
        gen = gens[0] if len(gens) == 1 else torch.stack(gens).float().mean(0)
        write_mp4(out, to_uint8_16(gen).cpu().numpy())  # (16,320,512,3)
        if (n + 1) % 20 == 0:
            el = time.time() - t_start
            print(
                f"[rank {rank}] {n + 1}/{len(ids)} elapsed {el / 60:.1f}min "
                f"eta {el / (n + 1) * (len(ids) - n - 1) / 60:.1f}min",
                flush=True,
            )
    if distributed:
        dist.barrier()
    if is_main:
        (OUT_DIR / "manifest.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print(f"done — {OUT_DIR}/manifest.json 에 설정 기록", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
