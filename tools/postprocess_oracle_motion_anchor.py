"""Composite Wan oracle-refiner videos only inside their dense motion support.

This is a train-only oracle diagnostic: the support mask is derived from
future-GT RAFT flow and is unavailable for a competition submission.  It
completes the source-anchored renderer gate without another model inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/submission_kit"
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from feature_csv_utils import read_video_uint8, save_video_tensor, to_eval_uint8  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        default=str(REPO / "diagnostics/wan21_oracle_motion_refiner_250"),
    )
    parser.add_argument(
        "--oracle-root", default=str(REPO / "diagnostics/oracle_motion_field_gate")
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "diagnostics/wan21_oracle_motion_refiner_250_anchored"),
    )
    parser.add_argument(
        "--report", default=str(REPO / "results/wan21_oracle_motion_anchor_report.json")
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--close", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=11)
    parser.add_argument("--feather", type=int, default=17)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def letterbox_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Map T,H,W masks with the same pad-preserving geometry as RGB evaluation."""
    source_h, source_w = mask.shape[1:]
    scale = min(height / source_h, width / source_w)
    resized_h = max(1, round(source_h * scale))
    resized_w = max(1, round(source_w * scale))
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    output = np.zeros((len(mask), height, width), dtype=np.float32)
    for index, frame in enumerate(mask):
        output[index, top : top + resized_h, left : left + resized_w] = cv2.resize(
            frame.astype(np.float32), (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
    return output


def anchored_mask(
    raw_support: np.ndarray,
    height: int,
    width: int,
    threshold: float,
    close: int,
    dilate: int,
    feather: int,
) -> np.ndarray:
    support = letterbox_mask(raw_support, height, width)
    # The temporal union covers both the source location and every future
    # location of articulated parts, preventing a copied old-arm ghost.
    union = (support[1:16].max(axis=0) >= threshold).astype(np.uint8)
    if close > 1:
        union = cv2.morphologyEx(
            union, cv2.MORPH_CLOSE, np.ones((odd(close), odd(close)), np.uint8)
        )
    if dilate > 1:
        union = cv2.dilate(
            union, np.ones((odd(dilate), odd(dilate)), np.uint8), iterations=1
        )
    alpha = union.astype(np.float32)
    if feather > 1:
        alpha = cv2.GaussianBlur(alpha, (odd(feather), odd(feather)), 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    output = np.repeat(alpha[None], 16, axis=0)
    output[0] = 0.0
    return output


def load_support(record: dict) -> np.ndarray:
    with np.load(resolve(record["motion_artifact"])) as artifact:
        control = artifact["control"].astype(np.float32)
    if control.shape[0] != 7 or control.shape[1] != 17:
        raise ValueError(f"Bad oracle control for {record['sample_id']}: {control.shape}")
    # visibility + occlusion equals the cleaned binary motion support.
    return np.clip(control[5] + control[6], 0.0, 1.0)


def save_diagnostic(
    path: Path,
    source: np.ndarray,
    prediction: np.ndarray,
    composite: np.ndarray,
    alpha: np.ndarray,
) -> None:
    indices = (0, 4, 8, 12, 15)
    thumb_w, thumb_h = 256, 160
    label_w = 90
    canvas = Image.new("RGB", (label_w + len(indices) * thumb_w, 4 * thumb_h), "black")
    draw = ImageDraw.Draw(canvas)
    mask_rgb = np.repeat((alpha[..., None] * 255).round().astype(np.uint8), 3, axis=-1)
    rows = (
        ("source", np.repeat(source[None], 16, axis=0)),
        ("generated", prediction),
        ("mask", mask_rgb),
        ("anchored", composite),
    )
    for row, (name, frames) in enumerate(rows):
        draw.text((4, row * thumb_h + 5), name, fill="white")
        for column, frame_index in enumerate(indices):
            image = Image.fromarray(frames[frame_index]).resize(
                (thumb_w, thumb_h), Image.Resampling.LANCZOS
            )
            canvas.paste(image, (label_w + column * thumb_w, row * thumb_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root)
    oracle_root = Path(args.oracle_root)
    output_root = Path(args.output_root)
    valset = oracle_root / "valset"
    records = [
        record for record in json.loads((oracle_root / "manifest.json").read_text())
        if record["split"] == "holdout"
    ]
    records.sort(key=lambda record: record["sample_id"])
    if args.limit:
        records = records[: args.limit]
    if len(records) < 2:
        raise ValueError("batch-roll anchoring requires at least two holdout records")
    supports = {record["sample_id"]: load_support(record) for record in records}
    rows = []
    for mode in ("none", "zero", "batch-roll"):
        (output_root / mode).mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records):
            sid = record["sample_id"]
            source_path = prediction_root / mode / f"{sid}.mp4"
            destination = output_root / mode / f"{sid}.mp4"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            prediction = read_video_uint8(source_path, expected_frames=16).numpy()
            prediction = to_eval_uint8(prediction, 320, 512, pad=True).numpy()
            source_raw = np.asarray(
                Image.open(valset / "images" / f"{sid}.png").convert("RGB")
            ).copy()
            source = to_eval_uint8(source_raw[None], 320, 512, pad=True)[0].numpy()

            if mode == "zero":
                support = np.zeros_like(supports[sid])
            elif mode == "batch-roll":
                rolled_sid = records[(index + 1) % len(records)]["sample_id"]
                support = supports[rolled_sid]
            else:
                support = supports[sid]
            alpha = anchored_mask(
                support,
                320,
                512,
                args.threshold,
                args.close,
                args.dilate,
                args.feather,
            )
            prediction_float = prediction.astype(np.float32) / 255.0
            source_float = source.astype(np.float32) / 255.0
            composite = source_float[None] * (1.0 - alpha[..., None]) + prediction_float * alpha[..., None]
            composite[0] = source_float
            composite_uint8 = np.clip(np.round(composite * 255), 0, 255).astype(np.uint8)
            tensor = torch.from_numpy(composite_uint8).permute(3, 0, 1, 2).float().div(255).mul(2).sub(1)
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"{destination} exists; pass --overwrite")
            save_video_tensor(tensor, destination, args.fps)
            save_diagnostic(
                output_root / "montages" / mode / f"{sid}.jpg",
                source,
                prediction,
                composite_uint8,
                alpha,
            )
            row = {
                "sample_id": sid,
                "mode": mode,
                "future_edit_fraction": float(alpha[1:].mean()),
                "generated_to_anchored_mae": float(
                    np.abs(prediction_float[1:] - composite[1:]).mean()
                ),
                "output": str(destination.resolve()),
            }
            rows.append(row)
            print(
                f"[{mode}] {index + 1}/{len(records)} {sid} "
                f"edit={row['future_edit_fraction']:.3f}"
            )
    normal_fractions = [row["future_edit_fraction"] for row in rows if row["mode"] == "none"]
    report = {
        "scope": "train-only oracle support; invalid for submission",
        "samples_per_mode": len(records),
        "modes": ["none", "zero", "batch-roll"],
        "median_normal_future_edit_fraction": float(np.median(normal_fractions)),
        "min_normal_future_edit_fraction": float(np.min(normal_fractions)),
        "max_normal_future_edit_fraction": float(np.max(normal_fractions)),
        "mask_parameters": {
            "threshold": args.threshold,
            "close": args.close,
            "dilate": args.dilate,
            "feather": args.feather,
            "temporal_union": True,
        },
        "rows": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
