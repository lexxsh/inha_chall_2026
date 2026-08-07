"""메트릭이 어떤 예측에 어떻게 반응하는지 실측한다.

궁금한 것:
  - 아무것도 안 한 정지 영상(첫 프레임 반복)의 점수 = 모델이 반드시 넘어야 할 하한선
  - 블러/시간축 스무딩이 cosine distance에 유리한가, action MAE를 얼마나 망치는가
  - 여러 샘플 평균(앙상블)이 개별 샘플보다 실제로 좋은가
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scorer import Scorer, cosine_distance

REPO = Path(__file__).resolve().parents[1]
TRAJ_LEN = 16


def gaussian_blur(video: torch.Tensor, sigma: float) -> torch.Tensor:
    """(T,H,W,C) uint8에 공간 가우시안 블러."""
    if sigma <= 0:
        return video
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x = video.permute(0, 3, 1, 2).float()
    channels = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kx, groups=channels)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), ky, groups=channels)
    return x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def temporal_smooth(video: torch.Tensor, window: int) -> torch.Tensor:
    """시간축 이동평균. 움직임을 뭉갠다."""
    if window <= 1:
        return video
    x = video.float()
    pad = window // 2
    padded = torch.cat([x[:1].repeat(pad, 1, 1, 1), x, x[-1:].repeat(pad, 1, 1, 1)], dim=0)
    out = torch.stack([padded[i : i + window].mean(dim=0) for i in range(x.shape[0])])
    return out.clamp(0, 255).to(torch.uint8)


def simulate_sample(gt: torch.Tensor, rng: np.random.Generator, strength: float) -> torch.Tensor:
    """디퓨전 샘플의 편차를 흉내낸다: 시간 지터 + 공간 이동 + 노이즈.

    같은 조건에서 seed만 다른 샘플들이 팔 위치·타이밍이 조금씩 어긋나는 상황을 모사한다.
    """
    x = gt.float()
    shift = int(rng.integers(-1, 2))
    if shift != 0:
        idx = np.clip(np.arange(TRAJ_LEN) + shift, 0, TRAJ_LEN - 1)
        x = x[idx]
    dx, dy = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
    x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    x = x + torch.from_numpy(rng.normal(0, 8.0 * strength, size=tuple(x.shape)).astype(np.float32))
    return x.clamp(0, 255).to(torch.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valset", default=str(REPO / "valset"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--out", default=str(REPO / "results/metric_probe.json"))
    args = parser.parse_args()

    valset = Path(args.valset)
    manifest = json.loads((valset / "manifest.json").read_text())[: args.limit]
    scorer = Scorer()
    rng = np.random.default_rng(0)

    variants: dict[str, list[torch.Tensor]] = {}
    gt_eval_all, actions_all = [], []

    for record in manifest:
        sample_id = record["sample_id"]
        gt_raw = np.load(valset / "gt_videos" / f"{sample_id}.npy")
        actions = np.load(valset / "actions" / f"{sample_id}.npy")
        gt = scorer.to_eval(gt_raw)
        gt_eval_all.append(gt)
        actions_all.append(actions)

        static = gt[:1].repeat(TRAJ_LEN, 1, 1, 1)
        samples = [simulate_sample(gt, rng, strength=1.0) for _ in range(4)]
        ensemble4 = torch.stack([s.float() for s in samples]).mean(0).clamp(0, 255).to(torch.uint8)
        ensemble2 = torch.stack([s.float() for s in samples[:2]]).mean(0).clamp(0, 255).to(torch.uint8)

        for name, video in [
            ("gt", gt),
            ("static_first_frame", static),
            ("blur_sigma1", gaussian_blur(gt, 1.0)),
            ("blur_sigma3", gaussian_blur(gt, 3.0)),
            ("blur_sigma6", gaussian_blur(gt, 6.0)),
            ("temporal_smooth3", temporal_smooth(gt, 3)),
            ("temporal_smooth5", temporal_smooth(gt, 5)),
            ("sim_sample_single", samples[0]),
            ("sim_ensemble2", ensemble2),
            ("sim_ensemble4", ensemble4),
        ]:
            variants.setdefault(name, []).append(video)

    # 다른 샘플의 GT를 섞어 넣어 '완전히 틀린 예측'의 상한을 잡는다.
    variants["shuffled_gt"] = gt_eval_all[1:] + gt_eval_all[:1]

    gt_batch = torch.stack(gt_eval_all)
    gt_features = scorer.features(gt_batch)

    results = {}
    for name, videos in variants.items():
        batch = torch.stack(videos)
        feats = scorer.features(batch)
        dino = cosine_distance(feats["dino"], gt_features["dino"])
        video_d = cosine_distance(feats["video"], gt_features["video"])
        maes = torch.cat(
            [scorer.action_mae(batch[i : i + 1], actions_all[i][None]) for i in range(batch.shape[0])]
        )
        results[name] = {
            "dino_cos_dist": float(dino.mean()),
            "video_cos_dist": float(video_d.mean()),
            "action_mae": float(maes.mean()),
        }
        print(
            f"{name:22s} DINO {results[name]['dino_cos_dist']:.4f}  "
            f"VIDEO {results[name]['video_cos_dist']:.4f}  "
            f"ACTION_MAE {results[name]['action_mae']:.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
