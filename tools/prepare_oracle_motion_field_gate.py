"""Precompute train-only dense motion fields for the Wan oracle refiner gate.

Future RGB is intentionally used here.  These controls are therefore illegal
at competition inference time and validate only the renderer/refiner half of
the proposed method.  A learned action-to-field model is considered only if
the frozen-I2V oracle gate passes.
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
from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from ldwma.datasets.lerobot_so100 import _decode_video_clip  # noqa: E402
from prepare_spatial_control_gate import letterbox_video, raft_bidirectional  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO / "open/data/train"))
    parser.add_argument(
        "--source-manifest",
        default=str(REPO / "diagnostics/spatial_control_gate/manifest.json"),
    )
    parser.add_argument(
        "--output", default=str(REPO / "diagnostics/oracle_motion_field_gate")
    )
    parser.add_argument(
        "--result-json", default=str(REPO / "results/oracle_motion_field_prepare.json")
    )
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--min-flow", type=float, default=0.5)
    parser.add_argument("--cycle-scale", type=float, default=4.0)
    parser.add_argument("--cycle-threshold", type=float, default=6.0)
    parser.add_argument("--split", choices=("train", "holdout", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def inverse_warp(
    source: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    min_flow: float,
    cycle_scale: float,
    cycle_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return warped source, confidence, magnitude, and valid source coordinates."""
    time, _, height, width = backward.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=backward.device, dtype=backward.dtype),
        torch.arange(width, device=backward.device, dtype=backward.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(time, -1, -1, -1)
    coordinates = base + backward.permute(0, 2, 3, 1)
    grid = torch.empty_like(coordinates)
    grid[..., 0] = coordinates[..., 0] * 2 / max(width - 1, 1) - 1
    grid[..., 1] = coordinates[..., 1] * 2 / max(height - 1, 1) - 1
    warped = F.grid_sample(
        source.unsqueeze(0).expand(time, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    forward_at_source = F.grid_sample(
        forward, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    cycle = torch.linalg.vector_norm(backward + forward_at_source, dim=1)
    in_bounds = (
        (coordinates[..., 0] >= 0)
        & (coordinates[..., 0] <= width - 1)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 1] <= height - 1)
    )
    confidence = torch.exp(-cycle / max(cycle_scale, 1e-6))
    confidence = confidence * in_bounds * (cycle <= cycle_threshold)
    magnitude = torch.linalg.vector_norm(backward, dim=1)
    active = magnitude >= min_flow
    return warped, confidence, magnitude, active


def clean_mask(active: torch.Tensor) -> torch.Tensor:
    cleaned = []
    close_kernel = np.ones((5, 5), np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)
    for frame in active.cpu().numpy().astype(np.uint8):
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, close_kernel)
        frame = cv2.dilate(frame, dilate_kernel, iterations=1)
        cleaned.append(frame.astype(np.float32))
    return torch.from_numpy(np.stack(cleaned)).to(active.device)


def build_control(
    video: np.ndarray,
    forward: torch.Tensor,
    backward: torch.Tensor,
    min_flow: float,
    cycle_scale: float,
    cycle_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    source = torch.from_numpy(video[0]).permute(2, 0, 1).float().div(255).to(backward.device)
    warped, confidence, magnitude, active = inverse_warp(
        source, forward, backward, min_flow, cycle_scale, cycle_threshold
    )
    support = clean_mask(active)
    visibility = confidence * support
    occlusion = support * (1.0 - confidence)
    source_batch = source.unsqueeze(0).expand_as(warped)
    appearance_delta = (warped - source_batch) * support.unsqueeze(1)
    flow = backward.clone()
    flow[:, 0] = flow[:, 0] * (2.0 / max(video.shape[2] - 1, 1))
    flow[:, 1] = flow[:, 1] * (2.0 / max(video.shape[1] - 1, 1))
    flow = flow * support.unsqueeze(1)
    future = torch.cat(
        [appearance_delta, flow, visibility.unsqueeze(1), occlusion.unsqueeze(1)], dim=1
    )
    zero = future.new_zeros((1, 7, video.shape[1], video.shape[2]))
    control = torch.cat([zero, future], dim=0).permute(1, 0, 2, 3).contiguous()
    oracle_rgb = source_batch + appearance_delta * visibility.unsqueeze(1)
    oracle_video = torch.cat([source.unsqueeze(0), oracle_rgb], dim=0)
    target = torch.from_numpy(video).permute(0, 3, 1, 2).float().div(255).to(source.device)
    static = source.unsqueeze(0).expand_as(target)
    static_l1 = (static - target).abs().mean()
    oracle_l1 = (oracle_video - target).abs().mean()
    metrics = {
        "static_l1": float(static_l1),
        "oracle_warp_l1": float(oracle_l1),
        "oracle_l1_reduction_fraction": float((static_l1 - oracle_l1) / static_l1.clamp_min(1e-8)),
        "mean_confidence": float(confidence.mean()),
        "motion_fraction": float(support.mean()),
        "median_flow_px": float(magnitude.median()),
    }
    return control, oracle_video, metrics


def save_preview(path: Path, video: np.ndarray, oracle: torch.Tensor) -> None:
    indices = (0, 4, 8, 12, 16)
    height, width = video.shape[1:3]
    canvas = Image.new("RGB", (len(indices) * width, 2 * height), "black")
    oracle_np = oracle.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy()
    for col, index in enumerate(indices):
        canvas.paste(Image.fromarray(video[index]), (col * width, 0))
        canvas.paste(Image.fromarray(oracle_np[index]), (col * width, height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main() -> None:
    args = parse_args()
    if args.frames != 17:
        raise ValueError("Wan2.1 oracle controls are fixed to 17 RGB frames")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    source_manifest = Path(args.source_manifest)
    records = json.loads(source_manifest.read_text())
    if args.split != "both":
        records = [record for record in records if record["split"] == args.split]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records selected")

    output = Path(args.output)
    artifacts = output / "artifacts"
    previews = output / "previews"
    valset = output / "valset"
    for directory in (artifacts, previews, valset / "images", valset / "actions", valset / "gt_videos"):
        directory.mkdir(parents=True, exist_ok=True)

    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    device = torch.device(args.device)
    raft = raft_small(weights=weights, progress=False).eval().requires_grad_(False).to(device)
    data_root = Path(args.data_root)
    output_records = []
    holdout_manifest = []
    for index, record in enumerate(records):
        sid = record["sample_id"]
        artifact_path = artifacts / f"{sid}.npz"
        if artifact_path.exists() and not args.overwrite:
            with np.load(artifact_path) as item:
                metrics = json.loads(str(item["metrics_json"]))
            output_records.append({**record, "motion_artifact": portable(artifact_path), **metrics})
            if record["split"] == "holdout":
                image_path = valset / "images" / f"{sid}.png"
                action_path = valset / "actions" / f"{sid}.npy"
                video_path = valset / "gt_videos" / f"{sid}.npy"
                if not (image_path.exists() and action_path.exists() and video_path.exists()):
                    indices = [int(record["start"]) + offset for offset in range(args.frames)]
                    raw_video = _decode_video_clip(data_root / record["video_ref"], indices)
                    eval_video = letterbox_video(raw_video, 480, 640)
                    with np.load(REPO / record["artifact"]) as source_artifact:
                        actions = source_artifact["actions"].astype(np.float32)
                    Image.fromarray(eval_video[0]).save(image_path)
                    np.save(action_path, actions[:16])
                    np.save(video_path, eval_video[:16])
                holdout_manifest.append(
                    {"sample_id": sid, "dataset": record["dataset"], "split": "train-only-holdout"}
                )
            continue

        indices = [int(record["start"]) + offset for offset in range(args.frames)]
        raw_video = _decode_video_clip(data_root / record["video_ref"], indices)
        video = letterbox_video(raw_video, args.height, args.width)
        forward, backward = raft_bidirectional(raft, transforms, video, device)
        control, oracle_video, metrics = build_control(
            video,
            forward,
            backward,
            args.min_flow,
            args.cycle_scale,
            args.cycle_threshold,
        )
        np.savez_compressed(
            artifact_path,
            control=control.cpu().numpy().astype(np.float16),
            backward_flow=backward.cpu().numpy().astype(np.float16),
            metrics_json=np.asarray(json.dumps(metrics)),
        )
        save_preview(previews / f"{sid}.jpg", video, oracle_video)
        output_records.append({**record, "motion_artifact": portable(artifact_path), **metrics})

        if record["split"] == "holdout":
            with np.load(REPO / record["artifact"]) as source_artifact:
                actions = source_artifact["actions"].astype(np.float32)
            eval_video = letterbox_video(raw_video, 480, 640)
            Image.fromarray(eval_video[0]).save(valset / "images" / f"{sid}.png")
            np.save(valset / "actions" / f"{sid}.npy", actions[:16])
            np.save(valset / "gt_videos" / f"{sid}.npy", eval_video[:16])
            holdout_manifest.append(
                {"sample_id": sid, "dataset": record["dataset"], "split": "train-only-holdout"}
            )
        print(
            f"[{index + 1}/{len(records)}] {sid} confidence={metrics['mean_confidence']:.3f} "
            f"motion={metrics['motion_fraction']:.3f} warp_gain={metrics['oracle_l1_reduction_fraction']:+.3f}"
        )

    # Rebuild the holdout manifest from all selected output records when cached
    # artifacts were reused.
    if not holdout_manifest:
        holdout_manifest = [
            {"sample_id": row["sample_id"], "dataset": row["dataset"], "split": "train-only-holdout"}
            for row in output_records
            if row["split"] == "holdout"
        ]
    (output / "manifest.json").write_text(json.dumps(output_records, indent=2))
    (valset / "manifest.json").write_text(json.dumps(holdout_manifest, indent=2))

    values = lambda key: np.asarray([row[key] for row in output_records], dtype=np.float64)
    report = {
        "scope": "train-only oracle; future RGB/RAFT flow unavailable at evaluation",
        "count": len(output_records),
        "train_count": sum(row["split"] == "train" for row in output_records),
        "holdout_count": sum(row["split"] == "holdout" for row in output_records),
        "control_channels": [
            "appearance_delta_r", "appearance_delta_g", "appearance_delta_b",
            "backward_flow_x_norm", "backward_flow_y_norm", "visibility", "occlusion",
        ],
        "median_confidence": float(np.median(values("mean_confidence"))),
        "median_motion_fraction": float(np.median(values("motion_fraction"))),
        "median_oracle_warp_gain": float(np.median(values("oracle_l1_reduction_fraction"))),
        "manifest": portable(output / "manifest.json"),
        "valset": portable(valset),
    }
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()
