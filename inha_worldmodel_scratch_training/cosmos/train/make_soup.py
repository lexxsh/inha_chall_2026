# Model Soup: 여러 LoRA 체크포인트의 학습 파라미터를 평균해 soup.pt 생성
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

RUNS = Path(os.environ.get("SOUP_DIR", str(Path(__file__).resolve().parent / "runs" / "v1_lora")))


def main() -> None:
    steps = [int(s) for s in (sys.argv[1:] or ["10000", "12500", "15000"])]
    acc, count = None, 0
    for s in steps:
        ck = torch.load(RUNS / f"step_{s:06d}.pt", map_location="cpu", weights_only=False)
        state = ck["state"]
        if acc is None:
            acc = {k: v.clone().double() for k, v in state.items()}
        else:
            for k in acc:
                acc[k] += state[k].double()
        count += 1
        print(f"added step {s}")
    soup = {k: (v / count).float() for k, v in acc.items()}
    out = RUNS / f"soup_{'_'.join(str(s) for s in steps)}.pt"
    torch.save({"step": f"soup{steps}", "state": soup}, out)
    print("saved:", out)


if __name__ == "__main__":
    main()
