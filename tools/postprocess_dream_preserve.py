"""Preserve the reference background around an existing Dream prediction.

The operation is evaluation-valid: it uses only the initial challenge image
and the already generated prediction.  No future/GT frames are read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/submission_kit"
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from feature_csv_utils import (  # noqa: E402
    list_challenge_sample_ids,
    read_video_uint8,
    save_video_tensor,
    to_eval_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--challenge-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("motion-mask", "global-blend"), default="motion-mask")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.06)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--dilate", type=int, default=13)
    parser.add_argument("--feather", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def clean_motion_mask(
    difference: np.ndarray,
    threshold: float,
    min_area: int,
    dilate: int,
    feather: int,
) -> np.ndarray:
    """Return a soft per-frame edit mask in [0,1]."""
    masks = []
    for frame_difference in difference:
        smooth = cv2.GaussianBlur(frame_difference, (7, 7), 0)
        binary = (smooth >= threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        kept = np.zeros_like(binary)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] >= min_area:
                kept[labels == component] = 1
        if dilate > 1:
            kept = cv2.dilate(kept, np.ones((odd(dilate), odd(dilate)), np.uint8), iterations=1)
        soft = kept.astype(np.float32)
        if feather > 1:
            soft = cv2.GaussianBlur(soft, (odd(feather), odd(feather)), 0)
        masks.append(np.clip(soft, 0.0, 1.0))
    return np.stack(masks)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1]")
    prediction_root = Path(args.prediction_root)
    challenge_root = Path(args.challenge_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_ids = list_challenge_sample_ids(challenge_root)
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    missing = [sample_id for sample_id in sample_ids if not (prediction_root / f"{sample_id}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} Dream videos, e.g. {missing[:5]}")

    mask_fractions = []
    for index, sample_id in enumerate(sample_ids):
        destination = output_root / f"{sample_id}.mp4"
        if destination.exists() and not args.overwrite:
            continue
        prediction = read_video_uint8(prediction_root / f"{sample_id}.mp4", expected_frames=16).numpy()
        prediction = to_eval_uint8(prediction, 320, 512, pad=True).numpy()
        source_raw = np.asarray(
            Image.open(challenge_root / "images" / f"{sample_id}.png").convert("RGB")
        ).copy()
        source = to_eval_uint8(source_raw[None], 320, 512, pad=True)[0].numpy()

        prediction_float = prediction.astype(np.float32) / 255.0
        source_float = source.astype(np.float32) / 255.0
        if args.mode == "global-blend":
            mask = np.ones(prediction.shape[:3], dtype=np.float32)
        else:
            difference = np.abs(prediction_float - source_float[None]).mean(axis=-1)
            mask = clean_motion_mask(
                difference,
                args.threshold,
                args.min_component_area,
                args.dilate,
                args.feather,
            )
        mix = np.clip(args.alpha * mask, 0.0, 1.0)[..., None]
        output = source_float[None] * (1.0 - mix) + prediction_float * mix
        output[0] = source_float
        output_uint8 = np.clip(np.round(output * 255), 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(output_uint8).permute(3, 0, 1, 2).float().div(255).mul(2).sub(1)
        save_video_tensor(tensor, destination, args.fps)
        mask_fractions.append(float(mask[1:].mean()))
        if (index + 1) % 8 == 0 or index + 1 == len(sample_ids):
            print(f"[{args.mode}] {index + 1}/{len(sample_ids)}")

    if mask_fractions:
        print(
            f"saved -> {output_root} | median future edit fraction="
            f"{float(np.median(mask_fractions)):.4f}"
        )


if __name__ == "__main__":
    main()
