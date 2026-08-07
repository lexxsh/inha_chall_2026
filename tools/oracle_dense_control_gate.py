"""Build a train-only oracle dense-flow preservation/control gate.

This is deliberately an *oracle* diagnostic: future training frames are used
to estimate target-to-source RAFT flow.  It answers a narrow question before
another expensive video-model run:

    If pixel motion were already known, can moving pixels from the reference
    frame while preserving everything else beat a static video?

The generated prediction is not a valid eval submission and the dense control
is not yet available from joint actions.  A PASS therefore promotes only the
spatial representation, never the complete action-to-video method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/submission_kit"
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from feature_csv_utils import preprocess_video, save_video_tensor  # noqa: E402

from prepare_spatial_control_gate import raft_bidirectional  # noqa: E402


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--valset",
        default=str(REPO / "diagnostics/spatial_control_gate/valset"),
        help="Train-only challenge-format holdout containing gt_videos/*.npy.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO / "diagnostics/oracle_dense_control_gate"),
    )
    parser.add_argument(
        "--result-json",
        default=str(REPO / "results/oracle_dense_control_pixels.json"),
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--min-flow", type=float, default=0.5)
    parser.add_argument("--cycle-scale", type=float, default=4.0)
    parser.add_argument("--cycle-threshold", type=float, default=6.0)
    parser.add_argument("--photo-motion-threshold", type=float, default=0.04)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense_warp(
    source: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    min_flow: float,
    cycle_scale: float,
    cycle_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse-warp source [3,H,W] into each future frame.

    ``forward`` is source->future and ``backward`` is future->source.  The
    latter directly defines the grid needed to reconstruct future pixels from
    the source.  Forward/backward cycle error supplies an occlusion confidence.
    """
    time, _, height, width = backward.shape
    dtype, device = backward.dtype, backward.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(time, -1, -1, -1)
    source_coordinates = base + backward.permute(0, 2, 3, 1)
    grid = torch.empty_like(source_coordinates)
    grid[..., 0] = source_coordinates[..., 0] * 2 / max(width - 1, 1) - 1
    grid[..., 1] = source_coordinates[..., 1] * 2 / max(height - 1, 1) - 1

    source_batch = source.unsqueeze(0).expand(time, -1, -1, -1)
    warped = F.grid_sample(
        source_batch,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    forward_at_source = F.grid_sample(
        forward,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    cycle = torch.linalg.vector_norm(backward + forward_at_source, dim=1)
    in_bounds = (
        (source_coordinates[..., 0] >= 0)
        & (source_coordinates[..., 0] <= width - 1)
        & (source_coordinates[..., 1] >= 0)
        & (source_coordinates[..., 1] <= height - 1)
    )
    confidence = torch.exp(-cycle / max(cycle_scale, 1e-6))
    confidence = confidence * in_bounds * (cycle <= cycle_threshold)
    magnitude = torch.linalg.vector_norm(backward, dim=1)

    # Background flow below the RAFT noise floor is copied exactly.  Elsewhere
    # confidence blending keeps occlusions/disocclusions tied to the reference
    # instead of creating black holes or unconstrained new shapes.
    alpha = confidence * (magnitude >= min_flow)
    prediction = warped * alpha.unsqueeze(1) + source_batch * (1 - alpha.unsqueeze(1))
    return prediction, confidence, magnitude


def motion_masks(
    magnitude: np.ndarray,
    confidence: np.ndarray,
    min_flow: float,
) -> np.ndarray:
    masks = []
    close_kernel = np.ones((5, 5), np.uint8)
    dilate_kernel = np.ones((7, 7), np.uint8)
    for flow, conf in zip(magnitude, confidence):
        mask = ((flow >= min_flow) & (conf >= 0.25)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        masks.append(mask.astype(bool))
    return np.stack(masks)


def safe_masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.size else 0.0


def save_diagnostic(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    control_rgb: np.ndarray,
    control_mask: np.ndarray,
) -> None:
    indices = sorted(set([0, len(target) // 4, len(target) // 2, 3 * len(target) // 4, len(target) - 1]))
    height, width = source.shape[:2]
    label_width = 105
    canvas = Image.new("RGB", (label_width + len(indices) * width, 4 * height), "black")
    draw = ImageDraw.Draw(canvas)
    rows = (
        ("reference", np.repeat(source[None], len(target), axis=0)),
        ("ground truth", target),
        ("oracle warp", prediction),
        ("masked control", control_rgb),
    )
    for row_index, (label, video) in enumerate(rows):
        draw.text((5, row_index * height + 8), label, fill="white")
        for col_index, frame_index in enumerate(indices):
            frame = video[frame_index].copy()
            if row_index == 3:
                boundary = cv2.morphologyEx(
                    control_mask[frame_index].astype(np.uint8),
                    cv2.MORPH_GRADIENT,
                    np.ones((3, 3), np.uint8),
                )
                frame[boundary.astype(bool)] = (255, 80, 40)
            canvas.paste(Image.fromarray(frame), (label_width + col_index * width, row_index * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    args = parse_args()
    valset = Path(args.valset)
    output = Path(args.output)
    prediction_root = output / "predictions"
    artifact_root = output / "artifacts"
    montage_root = output / "montages"
    for directory in (prediction_root, artifact_root, montage_root):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((valset / "manifest.json").read_text())
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise SystemExit("train-only holdout manifest is empty")

    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    device = torch.device(args.device)
    model = raft_small(weights=weights, progress=False).eval().requires_grad_(False).to(device)

    rows: list[dict] = []
    for index, record in enumerate(manifest):
        sample_id = record["sample_id"]
        video_path = prediction_root / f"{sample_id}.mp4"
        artifact_path = artifact_root / f"{sample_id}.npz"
        if video_path.exists() and artifact_path.exists() and not args.overwrite:
            raise FileExistsError(f"{video_path} exists; pass --overwrite to rebuild the oracle gate")

        target_uint8 = np.load(valset / "gt_videos" / f"{sample_id}.npy")
        if target_uint8.ndim != 4 or target_uint8.shape[-1] != 3:
            raise ValueError(f"unexpected GT shape for {sample_id}: {target_uint8.shape}")
        if len(target_uint8) < 2:
            raise ValueError(f"{sample_id} needs at least two frames")

        forward, backward = raft_bidirectional(model, transforms, target_uint8, device)
        source = torch.from_numpy(target_uint8[0]).permute(2, 0, 1).float().div(255).to(device)
        future_prediction, confidence, magnitude = dense_warp(
            source,
            forward,
            backward,
            args.min_flow,
            args.cycle_scale,
            args.cycle_threshold,
        )
        prediction = torch.cat([source.unsqueeze(0), future_prediction], dim=0)
        prediction_uint8 = (
            prediction.permute(0, 2, 3, 1).clamp(0, 1).mul(255).round().byte().cpu().numpy()
        )
        confidence_np = confidence.cpu().numpy()
        magnitude_np = magnitude.cpu().numpy()
        future_mask = motion_masks(magnitude_np, confidence_np, args.min_flow)

        # Estimate source-frame support from source->future flow.  This is used
        # only to visualize/store an oracle masked control, not as segmentation
        # ground truth for an action-to-control predictor.
        source_magnitude = torch.linalg.vector_norm(forward, dim=1).cpu().numpy()
        source_support = motion_masks(
            source_magnitude,
            np.ones_like(source_magnitude, dtype=np.float32),
            args.min_flow,
        ).any(axis=0)
        control_mask = np.concatenate([source_support[None], future_mask], axis=0)
        control_rgb = prediction_uint8.copy()
        control_rgb[~control_mask] = 0

        target = target_uint8.astype(np.float32) / 255.0
        pred = prediction_uint8.astype(np.float32) / 255.0
        static = np.repeat(target[:1], len(target), axis=0)
        static_error = np.abs(static - target).mean(axis=-1)
        warp_error = np.abs(pred - target).mean(axis=-1)
        photo_motion = np.abs(target - static).mean(axis=-1) >= args.photo_motion_threshold
        photo_motion[0] = False
        static_l1 = float(static_error.mean())
        warp_l1 = float(warp_error.mean())
        static_motion_l1 = safe_masked_mean(static_error, photo_motion)
        warp_motion_l1 = safe_masked_mean(warp_error, photo_motion)
        full_reduction = (static_l1 - warp_l1) / max(static_l1, 1e-8)
        motion_reduction = (static_motion_l1 - warp_motion_l1) / max(static_motion_l1, 1e-8)

        np.savez_compressed(
            artifact_path,
            # C,T,H,W matches common video-control loaders in this repository.
            control=np.concatenate(
                [control_rgb.transpose(3, 0, 1, 2).astype(np.float16) / 255.0,
                 control_mask[None].astype(np.float16)],
                axis=0,
            ),
            backward_flow=backward.cpu().numpy().astype(np.float16),
            confidence=np.concatenate(
                [np.ones((1, *confidence_np.shape[1:]), dtype=np.float16), confidence_np.astype(np.float16)],
                axis=0,
            ),
            motion_mask=control_mask,
        )
        eval_video = preprocess_video(prediction_uint8, 320, 512, pad=True)
        save_video_tensor(eval_video, video_path, args.fps)
        save_diagnostic(
            montage_root / f"{sample_id}.jpg",
            target_uint8[0],
            target_uint8,
            prediction_uint8,
            control_rgb,
            control_mask,
        )

        row = {
            "sample_id": sample_id,
            "dataset": record.get("dataset", "unknown"),
            "static_l1": static_l1,
            "oracle_warp_l1": warp_l1,
            "full_l1_reduction_fraction": float(full_reduction),
            "static_motion_l1": static_motion_l1,
            "oracle_warp_motion_l1": warp_motion_l1,
            "motion_l1_reduction_fraction": float(motion_reduction),
            "mean_cycle_confidence": float(confidence_np.mean()),
            "control_mask_fraction": float(control_mask.mean()),
            "median_flow_px": float(np.median(magnitude_np)),
            "prediction": portable_path(video_path),
            "artifact": portable_path(artifact_path),
        }
        rows.append(row)
        print(
            f"[{index + 1}/{len(manifest)}] {sample_id} "
            f"full_reduction={full_reduction:+.3f} motion_reduction={motion_reduction:+.3f} "
            f"confidence={row['mean_cycle_confidence']:.3f}"
        )

    def median(key: str) -> float:
        return float(np.median([row[key] for row in rows]))

    pixel_gate = (
        median("full_l1_reduction_fraction") >= 0.10
        and median("motion_l1_reduction_fraction") >= 0.10
        and median("mean_cycle_confidence") >= 0.60
    )
    report = {
        "scope": "train-only oracle; future GT flow is unavailable at evaluation",
        "count": len(rows),
        "prediction_root": portable_path(prediction_root),
        "median_full_l1_reduction_fraction": median("full_l1_reduction_fraction"),
        "median_motion_l1_reduction_fraction": median("motion_l1_reduction_fraction"),
        "median_cycle_confidence": median("mean_cycle_confidence"),
        "median_control_mask_fraction": median("control_mask_fraction"),
        "pixel_gate": bool(pixel_gate),
        "verdict": "PASS_PIXEL_GATE_NEEDS_OFFICIAL_FEATURE_SCORE" if pixel_gate else "REJECT_PIXEL_GATE",
        "per_sample": rows,
    }
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "per_sample"}, indent=2))
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()
