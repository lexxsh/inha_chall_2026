# 잠재공간 액션 판독기: (16,5,40,64) 잠재 → (16,6) 액션. v3 벌점용 대리 채점기 (train 데이터만 사용)
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import LatentClipDataset

OUT = Path(__file__).resolve().parent / "runs" / "latent_idm" / "lidm_best.pt"
BATCH = 32
LR = 3e-4
MAX_STEPS = 6000
VAL_EVERY = 500


class LatentIDM(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(16, 64, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(128, hidden, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden * 5, 512), nn.SiLU(), nn.Linear(512, 16 * 6))

    def forward(self, z):  # (B,16,5,40,64)
        h = self.conv(z.float())              # (B,H,5,5,8)
        h = h.mean(dim=(-1, -2)).flatten(1)   # (B,H*5)
        return self.head(h).view(-1, 16, 6)


def main() -> None:
    device = torch.device("cuda")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = LatentIDM().to(device)
    ds = LatentClipDataset()
    # 파일명 해시로 내부 2% val 분리
    val_mask = np.array([zlib.crc32(f.encode()) % 100 < 2 for f in ds.files])
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0][:512]
    tr = torch.utils.data.Subset(ds, tr_idx.tolist())
    loader = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    def val_mae() -> float:
        model.eval()
        maes = []
        with torch.no_grad():
            for i in range(0, len(va_idx), 64):
                items = [ds[int(j)] for j in va_idx[i : i + 64]]
                z = torch.stack([it["latent"] for it in items]).to(device)
                a = torch.stack([it["act"] for it in items]).to(device)
                maes.append(float(torch.mean(torch.abs(model(z) - a))))
        model.train()
        return float(np.mean(maes))

    best, step, t0 = 1e9, 0, time.time()
    while step < MAX_STEPS:
        for item in loader:
            z = item["latent"].to(device)
            a = item["act"].to(device)
            loss = F.l1_loss(model(z), a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step} l1 {loss.item():.4f} {(time.time() - t0) / 100:.2f}s/it", flush=True)
                t0 = time.time()
            if step % VAL_EVERY == 0:
                v = val_mae()
                marker = ""
                if v < best:
                    best = v
                    torch.save({"step": step, "val_mae": v, "state": model.state_dict()}, OUT)
                    marker = " (saved)"
                print(f"[val] step {step} MAE {v:.4f}{marker}", flush=True)
            if step >= MAX_STEPS:
                break
    print(f"done. best val MAE {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
