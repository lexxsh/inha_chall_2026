"""Action Component가 실제로 무엇을 측정하는지 진단한다.

GT 영상인데도 MAE가 1.2 근처로 나왔다. 정규화 공간에서 평균(=0)을 찍는 것보다도 나쁜 값이라
추출기가 절대 관절각을 못 맞추는 것인지, 아니면 우리 쪽 사용법이 틀린 것인지 가려야 한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scorer import Scorer

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valset", default=str(REPO / "valset"))
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    valset = Path(args.valset)
    manifest = json.loads((valset / "manifest.json").read_text())[: args.limit]
    scorer = Scorer()

    preds, targets, datasets = [], [], []
    for record in manifest:
        sample_id = record["sample_id"]
        gt_raw = np.load(valset / "gt_videos" / f"{sample_id}.npy")
        actions = np.load(valset / "actions" / f"{sample_id}.npy")
        gt = scorer.to_eval(gt_raw)[None]
        frames = gt.to(scorer.device).permute(0, 4, 1, 2, 3).float().div(255.0).mul(2.0).sub(1.0)
        with torch.no_grad():
            pred = scorer.action_model(frames).float().cpu()[0]
        target = (torch.from_numpy(actions) - scorer.action_mean) / scorer.action_std
        preds.append(pred)
        targets.append(target)
        datasets.append(record["dataset"])

    pred = torch.stack(preds)
    target = torch.stack(targets)

    mae_model = (pred - target).abs().mean()
    mae_zero = target.abs().mean()  # 정규화 공간에서 전역 평균을 찍는 자명한 예측
    per_dim = (pred - target).abs().mean(dim=(0, 1))

    print(f"샘플 {pred.shape[0]}개 / 프레임 {pred.shape[1]} / 차원 {pred.shape[2]}")
    print(f"추출기 MAE          : {mae_model:.4f}")
    print(f"전역 평균 예측 MAE  : {mae_zero:.4f}   <- 이보다 나쁘면 추출기가 정보를 못 준다는 뜻")
    print(f"차원별 MAE          : {[round(v, 3) for v in per_dim.tolist()]}")
    print(f"타깃 차원별 std     : {[round(v, 3) for v in target.std(dim=(0, 1)).tolist()]}")
    print(f"예측 차원별 std     : {[round(v, 3) for v in pred.std(dim=(0, 1)).tolist()]}")

    # 차원별 상관계수: 추출기가 방향성이라도 잡는지 확인
    print("\n차원별 피어슨 상관:")
    for d in range(pred.shape[2]):
        p = pred[..., d].flatten()
        t = target[..., d].flatten()
        r = float(torch.corrcoef(torch.stack([p, t]))[0, 1])
        print(f"  dim {d}: r = {r:+.3f}")

    # 클립 내부의 '변화'만 볼 때의 정확도 (카메라/캘리브레이션 오프셋을 제거)
    pred_rel = pred - pred[:, :1]
    target_rel = target - target[:, :1]
    print(f"\n첫 프레임 기준 상대 변화 MAE : {(pred_rel - target_rel).abs().mean():.4f}")
    print(f"상대 변화의 자명한 예측(0) MAE: {target_rel.abs().mean():.4f}")

    # 샘플별 오차가 데이터셋(=카메라 세팅)에 묶여 있는지
    per_sample = (pred - target).abs().mean(dim=(1, 2))
    order = torch.argsort(per_sample)
    print("\n가장 잘 맞춘 5개:")
    for i in order[:5].tolist():
        print(f"  {per_sample[i]:.3f}  {datasets[i]}")
    print("가장 못 맞춘 5개:")
    for i in order[-5:].tolist():
        print(f"  {per_sample[i]:.3f}  {datasets[i]}")

    # 편향(bias)을 제거하면 얼마나 남는지 = 추출기가 절대값만 못 맞추는 것인지 판별
    bias = (pred - target).mean(dim=(0, 1))
    print(f"\n전역 편향 제거 후 MAE: {(pred - target - bias).abs().mean():.4f}")
    sample_bias = (pred - target).mean(dim=1, keepdim=True)
    print(f"샘플별 편향 제거 후 MAE: {(pred - target - sample_bias).abs().mean():.4f}")


if __name__ == "__main__":
    main()
