# 학습 데이터 정제 1단계: 잠재 판독기로 train 창 전수 채점 (판독액션 vs 라벨액션 MAE).
# 오차가 유난히 큰 에피소드 = 영상-라벨 불일치(원격조작 노이즈·캘리브레이션 오류) 후보 → v4 학습에서 감량.
# train 데이터 + 자체 모델만 사용 (규정 청정). DEVICE=cpu로 생성 작업과 병행 가능.
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import LatentClipDataset

RUNS = Path(__file__).resolve().parent / "runs"
OUT = RUNS / "data_clean"
DEVICE = os.environ.get("DEVICE", "cpu")
BATCH = int(os.environ.get("BATCH", "32"))
JUDGE = os.environ.get("JUDGE", "v1")  # v2면 잠재 판독기 v2로 재채점 (정제 2회전)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("THREADS", "6")))
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    if JUDGE == "v2":
        from train_latent_idm2 import LatentIDM2 as _J

        model = _J().to(device).eval()
        ck = torch.load(RUNS / "latent_idm2" / "lidm2_best.pt", map_location="cpu", weights_only=False)
    else:
        from train_latent_idm import LatentIDM as _J

        model = _J().to(device).eval()
        ck = torch.load(RUNS / "latent_idm" / "lidm_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"])

    ds = LatentClipDataset()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, persistent_workers=True)
    maes, t0 = [], time.time()
    with torch.no_grad():
        for i, item in enumerate(loader):
            z = item["latent"].to(device)
            a = item["act"].to(device)
            m = torch.mean(torch.abs(model(z) - a), dim=(1, 2))  # (B,)
            maes.append(m.cpu().numpy())
            if (i + 1) % 100 == 0:
                done = (i + 1) * BATCH
                el = time.time() - t0
                print(f"{done}/{len(ds)} eta {el / done * (len(ds) - done) / 60:.0f}min", flush=True)
    mae = np.concatenate(maes)[: len(ds)]

    suffix = "" if JUDGE == "v1" else f"_{JUDGE}"
    df = pd.DataFrame({"path": ds.files, "mae": mae, "motion": ds.motion})
    df["episode"] = df.path.str.split("_").str[0]
    df.to_parquet(OUT / f"window_mae{suffix}.parquet")

    ep = df.groupby("episode")["mae"].agg(["median", "mean", "count"]).sort_values("median", ascending=False)
    ep.to_parquet(OUT / f"episode_mae{suffix}.parquet")
    q = np.percentile(mae, [50, 75, 90, 95, 99])
    print(f"windows {len(df)} | window MAE p50 {q[0]:.3f} p75 {q[1]:.3f} p90 {q[2]:.3f} p95 {q[3]:.3f} p99 {q[4]:.3f}", flush=True)
    print("worst episodes (median MAE):", flush=True)
    print(ep.head(15).to_string(), flush=True)


if __name__ == "__main__":
    main()
