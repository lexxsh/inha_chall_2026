"""첫 프레임을 16번 반복한 정지 영상을 eval 전체에 대해 생성한다.

학습이 전혀 필요 없는 하한선 제출물이다. 리더보드 점수를 받아보면
세 컴포넌트가 어떤 식으로 합산되는지 역추정할 수 있고,
이후 모델이 '정지보다 나은가'를 리더보드 단위로 판정할 기준이 된다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/submission_kit"
sys.path.insert(0, str(KIT))

from feature_csv_utils import (  # noqa: E402
    list_challenge_sample_ids,
    preprocess_video,
    save_video_tensor,
)

TRAJ_LEN = 16
FPS = 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", default=str(REPO / "open/data/eval"))
    parser.add_argument("--out", default=str(REPO / "open/submission_kit/input_videos"))
    args = parser.parse_args()

    challenge_root = Path(args.challenge_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_ids = list_challenge_sample_ids(challenge_root)
    for i, sample_id in enumerate(sample_ids):
        image = np.asarray(Image.open(challenge_root / "images" / f"{sample_id}.png").convert("RGB"))
        clip = np.repeat(image[None], TRAJ_LEN, axis=0)
        # 평가와 동일한 letterbox를 거쳐 (C,T,H,W) [-1,1] 로 만든 뒤 저장한다.
        video = preprocess_video(clip, 320, 512, pad=True)
        save_video_tensor(video, out_root / f"{sample_id}.mp4", FPS)
        if (i + 1) % 50 == 0 or i + 1 == len(sample_ids):
            print(f"[static baseline] {i + 1}/{len(sample_ids)}")

    print(f"saved {len(sample_ids)} videos -> {out_root}")


if __name__ == "__main__":
    main()
