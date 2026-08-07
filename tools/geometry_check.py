"""출력 영상의 기하(letterbox)가 어긋나면 얼마나 손해인지 측정한다.

평가 파이프라인은 어떤 해상도의 mp4든 320x512 캔버스에 비율을 유지해 letterbox 한다.
원본이 4:3(640x480)이면 좌우에 검은 띠가 생긴다.
따라서 모델이 320x512를 꽉 채워 뱉으면 정답과 기하가 어긋난다.
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


def stretch_to_canvas(video_uint8: np.ndarray, height: int = 320, width: int = 512) -> torch.Tensor:
    """검은 띠 없이 캔버스를 꽉 채우도록 늘린다(비율 왜곡)."""
    x = torch.from_numpy(video_uint8).float().permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valset", default=str(REPO / "valset"))
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    valset = Path(args.valset)
    manifest = json.loads((valset / "manifest.json").read_text())[: args.limit]
    scorer = Scorer()

    correct, stretched, cropped = [], [], []
    for record in manifest:
        gt_raw = np.load(valset / "gt_videos" / f"{record['sample_id']}.npy")
        correct.append(scorer.to_eval(gt_raw))
        stretched.append(stretch_to_canvas(gt_raw))
        # 띠 없이 중앙을 잘라 채우는 경우(pad=False 방식)
        from feature_csv_utils import preprocess_video

        cropped_video = preprocess_video(gt_raw, 320, 512, pad=False).clamp(-1, 1)
        cropped_video = ((cropped_video + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0)
        cropped.append(cropped_video)

    gt_feats = scorer.features(torch.stack(correct))
    print("정답 대비 cosine distance (기하만 다르게 했을 때)")
    for name, videos in [("letterbox 정상", correct), ("꽉 채워 늘림", stretched), ("중앙 크롭", cropped)]:
        feats = scorer.features(torch.stack(videos))
        dino = float(cosine_distance(feats["dino"], gt_feats["dino"]).mean())
        vid = float(cosine_distance(feats["video"], gt_feats["video"]).mean())
        print(f"  {name:14s} DINO {dino:.4f}   VIDEO {vid:.4f}")

    # eval 입력 기하 확인
    src_h, src_w = 480, 640
    scale = min(320 / src_h, 512 / src_w)
    content_h, content_w = round(src_h * scale), round(src_w * scale)
    bar = (512 - content_w) // 2
    print(f"\neval 원본 {src_w}x{src_h} -> 320x512 캔버스에서 실제 내용 {content_w}x{content_h}px, 좌우 검은 띠 {bar}/{512 - content_w - bar}px")


if __name__ == "__main__":
    main()
