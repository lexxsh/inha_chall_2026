# 잠재 판독기 v2: 차분채널 + 프레임별 토큰 트랜스포머 + 노이즈 증강 + EMA.
# v1(소형 conv3d, val 0.2171) 대체 목표. 대회 train 잠재+정답 액션만 사용 (킷 무관, 규정 청정)
from __future__ import annotations

import math
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

OUT = Path(__file__).resolve().parent / "runs" / "latent_idm2" / "lidm2_best.pt"
BATCH = 64
LR = 3e-4
MAX_STEPS = 25000
VAL_EVERY = 1000
NOISE_MAX = 0.1
EMA_DECAY = 0.999


class LatentIDM2(nn.Module):
    def __init__(self, ch: int = 256, layers: int = 4):
        super().__init__()
        # 시간축은 섞지 않는 (1,3,3) conv — 시간 관계는 트랜스포머 담당
        self.stem = nn.Sequential(
            nn.Conv3d(32, 128, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), nn.SiLU(),
            nn.Conv3d(128, ch, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), nn.SiLU(),
            nn.Conv3d(ch, ch, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), nn.SiLU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, 5, ch))
        enc = nn.TransformerEncoderLayer(ch, 8, ch * 4, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, layers)
        # 잠재프레임 1~4가 각각 액션 4개 담당 (프레임 0=초기조건은 어텐션으로만 기여)
        self.head = nn.Linear(ch, 4 * 6)

    def forward(self, z):  # (B,16,5,40,64)
        z = z.float()
        d = torch.diff(z, dim=2, prepend=z[:, :, :1])
        h = self.stem(torch.cat([z, d], dim=1))       # (B,ch,5,5,8)
        h = h.mean(dim=(-1, -2)).permute(0, 2, 1)     # (B,5,ch)
        h = self.tf(h + self.pos)
        return self.head(h[:, 1:]).reshape(-1, 16, 6)


def main() -> None:
    device = torch.device("cuda")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = LatentIDM2().to(device)
    ema = LatentIDM2().to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    ds = LatentClipDataset()
    val_mask = np.array([zlib.crc32(f.encode()) % 100 < 2 for f in ds.files])
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0][:512]
    tr = torch.utils.data.Subset(ds, tr_idx.tolist())
    loader = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=6, persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEPS, eta_min=1e-5)

    def val_mae(m) -> float:
        m.eval()
        maes = []
        with torch.no_grad():
            for i in range(0, len(va_idx), 64):
                items = [ds[int(j)] for j in va_idx[i : i + 64]]
                z = torch.stack([it["latent"] for it in items]).to(device)
                a = torch.stack([it["act"] for it in items]).to(device)
                maes.append(float(torch.mean(torch.abs(m(z) - a))))
        m.train()
        return float(np.mean(maes))

    best, step, t0 = 1e9, 0, time.time()
    while step < MAX_STEPS:
        for item in loader:
            z = item["latent"].to(device)
            a = item["act"].to(device)
            sigma = torch.rand(z.shape[0], 1, 1, 1, 1, device=device) * NOISE_MAX
            loss = F.l1_loss(model(z + sigma * torch.randn_like(z)), a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - EMA_DECAY)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            step += 1
            if step % 200 == 0:
                print(f"step {step} l1 {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 200:.2f}s/it", flush=True)
                t0 = time.time()
            if step % VAL_EVERY == 0:
                v = val_mae(ema)
                marker = ""
                if v < best:
                    best = v
                    torch.save({"step": step, "val_mae": v, "state": ema.state_dict()}, OUT)
                    marker = " (saved)"
                print(f"[val] step {step} EMA MAE {v:.4f}{marker}", flush=True)
            if step >= MAX_STEPS:
                break
    print(f"done. best val MAE {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
