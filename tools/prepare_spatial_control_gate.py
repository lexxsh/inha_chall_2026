"""Prepare a train-only oracle spatial-control gate for SO-100 videos.

This script deliberately runs *before* any Wan post-training.  It extracts
cycle-consistent point tracks with the cached RAFT-Small teacher, renders a
compact three-channel control video, and writes visual/numeric diagnostics.

The output proves only that a usable pixel-aligned oracle condition can be
obtained from provided training videos.  It does not prove that joint actions
can predict those tracks, nor that Wan will obey them; those are separate gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    LeRobotSO100Dataset,
    _decode_video_clip,
    discover_lerobot_so100_datasets,
)

DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)
TRACK_COLORS = (
    (255, 80, 80), (80, 220, 120), (80, 150, 255), (255, 210, 80),
    (220, 90, 255), (80, 230, 230), (255, 140, 60), (170, 255, 80),
    (255, 100, 170), (100, 255, 190), (130, 130, 255), (255, 245, 120),
    (200, 120, 80), (120, 200, 255), (230, 160, 255), (150, 255, 150),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--output", default=str(REPO / "diagnostics/spatial_control_gate"))
    parser.add_argument("--result-json", default=str(REPO / "results/spatial_control_gate.json"))
    parser.add_argument(
        "--valset-output",
        default=None,
        help="Challenge-format train-only holdout; defaults to OUTPUT/valset.",
    )
    parser.add_argument("--split", choices=("train", "holdout", "both"), default="both")
    parser.add_argument("--train-clips", type=int, default=64)
    parser.add_argument("--holdout-clips", type=int, default=8)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--control-height", type=int, default=80)
    parser.add_argument("--control-width", type=int, default=128)
    parser.add_argument("--num-tracks", type=int, default=16)
    parser.add_argument("--grid-step", type=int, default=8)
    parser.add_argument("--cycle-threshold", type=float, default=4.0)
    parser.add_argument("--min-motion", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stable_int(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def portable_path(path: Path) -> str:
    """Prefer repo-relative manifests while still allowing /tmp smoke tests."""
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def letterbox_video(video: np.ndarray, height: int, width: int) -> np.ndarray:
    """Letterbox T,H,W,3 uint8 using the same geometry as the training loader."""
    source_h, source_w = video.shape[1:3]
    scale = min(height / source_h, width / source_w)
    resized_h = max(1, round(source_h * scale))
    resized_w = max(1, round(source_w * scale))
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    output = np.zeros((len(video), height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(video):
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        output[index, top : top + resized_h, left : left + resized_w] = resized
    return output


def build_base(data_root: Path, paths: list[str], frames: int, seed: int) -> LeRobotSO100Dataset:
    return LeRobotSO100Dataset(
        root=data_root,
        dataset_paths=paths,
        train=True,
        traj_len=frames,
        target_height=160,
        target_width=256,
        pad=True,
        camera_key="auto",
        val_fraction=0.0,
        seed=seed,
        fps=6,
        use_all_episodes=True,
    )


def select_records(base: LeRobotSO100Dataset, count: int, seed: int) -> list[dict]:
    examples = sorted(
        base.examples,
        key=lambda row: (str(row["data_ref"]), str(row["video_ref"])),
    )
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    records = []
    for example_index in order[: min(count, len(order))]:
        example = examples[example_index]
        identity = f"{example['data_ref']}|{example['video_ref']}"
        max_start = int(example["length"]) - base.traj_len
        start = stable_int(identity, seed) % (max_start + 1)
        records.append({**example, "start": int(start), "identity": identity})
    return records


def load_record(record: dict, frames: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = [record["start"] + offset for offset in range(frames)]
    table = pd.read_parquet(record["data_ref"], columns=["action", "observation.state"])
    actions = np.stack(table["action"].iloc[indices].to_numpy()).astype(np.float32)
    states = np.stack(table["observation.state"].iloc[indices].to_numpy()).astype(np.float32)
    video = _decode_video_clip(Path(record["video_ref"]), indices)
    return letterbox_video(video, height, width), actions, states


@torch.inference_mode()
def raft_bidirectional(
    model: torch.nn.Module,
    transforms,
    video: np.ndarray,
    device: torch.device,
    chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.from_numpy(video).permute(0, 3, 1, 2).float().div(255).to(device)
    source = images[0:1]
    forward_parts, backward_parts = [], []
    for begin in range(1, len(images), chunk):
        future = images[begin : begin + chunk]
        source_batch = source.expand(len(future), -1, -1, -1)
        source_in, future_in = transforms(source_batch, future)
        forward_parts.append(model(source_in, future_in)[-1])
        future_in, source_in = transforms(future, source_batch)
        backward_parts.append(model(future_in, source_in)[-1])
    return torch.cat(forward_parts), torch.cat(backward_parts)


def sample_field(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [T,2,H,W] at [N,2] or [T,N,2] pixel coordinates."""
    time, _, height, width = field.shape
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(time, -1, -1)
    x = points[..., 0] * 2 / max(width - 1, 1) - 1
    y = points[..., 1] * 2 / max(height - 1, 1) - 1
    grid = torch.stack([x, y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1)


def select_tracks(
    forward: torch.Tensor,
    backward: torch.Tensor,
    num_tracks: int,
    grid_step: int,
    min_motion: float,
    cycle_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    time, _, height, width = forward.shape
    ys = torch.arange(grid_step // 2, height, grid_step, device=forward.device)
    xs = torch.arange(grid_step // 2, width, grid_step, device=forward.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).float()
    fwd_at_points = sample_field(forward, points)
    future_points = points.unsqueeze(0) + fwd_at_points
    bwd_at_future = sample_field(backward, future_points)
    cycle = torch.linalg.vector_norm(fwd_at_points + bwd_at_future, dim=-1)
    motion = torch.linalg.vector_norm(fwd_at_points, dim=-1)
    in_bounds = (
        (future_points[..., 0] >= 0)
        & (future_points[..., 0] <= width - 1)
        & (future_points[..., 1] >= 0)
        & (future_points[..., 1] <= height - 1)
    )
    visibility = in_bounds & (cycle <= cycle_threshold)
    max_motion = motion.max(dim=0).values
    valid_fraction = visibility.float().mean(dim=0)
    median_cycle = cycle.median(dim=0).values
    score = max_motion * valid_fraction / (1 + median_cycle)
    score[(max_motion < min_motion) | (valid_fraction < 0.4)] = -1

    chosen: list[int] = []
    min_spacing = max(grid_step * 1.5, 8.0)
    for candidate in torch.argsort(score, descending=True).tolist():
        if score[candidate] < 0 or len(chosen) >= num_tracks:
            break
        point = points[candidate]
        if all(torch.linalg.vector_norm(point - points[old]).item() >= min_spacing for old in chosen):
            chosen.append(candidate)
    # Do not fabricate tracks for a static/ambiguous clip.  A previous fallback
    # filled such clips with padding-border points and made the aggregate gate
    # look healthier than the actual control signal.
    chosen_tensor = torch.tensor(chosen, device=forward.device, dtype=torch.long)
    tracks = torch.cat(
        [points[chosen_tensor].unsqueeze(0), future_points[:, chosen_tensor]], dim=0
    )
    visible = torch.cat(
        [torch.ones((1, len(chosen)), dtype=torch.bool, device=forward.device), visibility[:, chosen_tensor]],
        dim=0,
    )

    # Coverage is the fraction of dense cumulative motion near a selected track.
    dense_motion = torch.linalg.vector_norm(forward, dim=1).max(dim=0).values.cpu().numpy()
    support = np.zeros((height, width), dtype=np.uint8)
    for x, y in points[chosen_tensor].cpu().numpy():
        cv2.circle(support, (round(float(x)), round(float(y))), max(grid_step * 2, 10), 1, -1)
    coverage = float((dense_motion * support).sum() / max(dense_motion.sum(), 1e-6))
    median_displacement = (
        float(torch.linalg.vector_norm(tracks[-1] - tracks[0], dim=-1).median().item())
        if len(chosen) else 0.0
    )
    metrics = {
        "tracks_found": len(chosen),
        "cycle_valid_fraction": float(visible[1:].float().mean().item()) if len(chosen) else 0.0,
        "median_track_displacement_px": median_displacement,
        "motion_energy_coverage": coverage,
    }
    return tracks.cpu().numpy(), visible.cpu().numpy(), metrics


def render_control(
    tracks: np.ndarray,
    visible: np.ndarray,
    source_height: int,
    source_width: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Render occupancy, horizontal displacement, vertical displacement."""
    control = np.zeros((3, len(tracks), height, width), dtype=np.float32)
    weight = np.zeros((len(tracks), height, width), dtype=np.float32)
    scale_x, scale_y = width / source_width, height / source_height
    sigma = max(1.25, min(height, width) / 64)
    radius = max(2, math.ceil(3 * sigma))
    for frame_index in range(len(tracks)):
        for point_index, (x, y) in enumerate(tracks[frame_index]):
            if not visible[frame_index, point_index]:
                continue
            cx, cy = x * scale_x, y * scale_y
            x0, x1 = max(0, int(cx) - radius), min(width, int(cx) + radius + 1)
            y0, y1 = max(0, int(cy) - radius), min(height, int(cy) + radius + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
            dx = (x - tracks[0, point_index, 0]) / max(source_width, 1)
            dy = (y - tracks[0, point_index, 1]) / max(source_height, 1)
            control[0, frame_index, y0:y1, x0:x1] = np.maximum(
                control[0, frame_index, y0:y1, x0:x1], gaussian
            )
            control[1, frame_index, y0:y1, x0:x1] += gaussian * dx
            control[2, frame_index, y0:y1, x0:x1] += gaussian * dy
            weight[frame_index, y0:y1, x0:x1] += gaussian
    nonzero = weight > 1e-6
    control[1][nonzero] /= weight[nonzero]
    control[2][nonzero] /= weight[nonzero]
    return control


def state_action_metrics(actions: np.ndarray, states: np.ndarray, action_std: np.ndarray) -> dict:
    lag_errors = {}
    for lag in range(-2, 3):
        if lag >= 0:
            action_slice, state_slice = actions[: len(actions) - lag or None], states[lag:]
        else:
            action_slice, state_slice = actions[-lag:], states[: len(states) + lag]
        error = np.abs((action_slice - state_slice) / action_std).mean()
        lag_errors[str(lag)] = float(error)
    best_lag = min(lag_errors, key=lag_errors.get)
    return {"normalized_mae_by_state_lag": lag_errors, "best_state_lag": int(best_lag)}


def overlay_frame(frame: np.ndarray, tracks: np.ndarray, visible: np.ndarray, frame_index: int) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for point_index in range(tracks.shape[1]):
        color = TRACK_COLORS[point_index % len(TRACK_COLORS)]
        history = []
        for time_index in range(frame_index + 1):
            if visible[time_index, point_index]:
                history.append(tuple(float(v) for v in tracks[time_index, point_index]))
        if len(history) > 1:
            draw.line(history, fill=color, width=2)
        if visible[frame_index, point_index]:
            x, y = tracks[frame_index, point_index]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(255, 255, 255))
    return image


def save_montage(path: Path, video: np.ndarray, tracks: np.ndarray, visible: np.ndarray) -> None:
    indices = sorted(set([0, len(video) // 4, len(video) // 2, 3 * len(video) // 4, len(video) - 1]))
    panels = [overlay_frame(video[index], tracks, visible, index) for index in indices]
    canvas = Image.new("RGB", (video.shape[2] * len(panels), video.shape[1]), "black")
    for panel_index, panel in enumerate(panels):
        canvas.paste(panel, (panel_index * video.shape[2], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def aggregate_verdict(samples: list[dict]) -> dict:
    if not samples:
        return {"verdict": "ERROR_NO_SAMPLES"}
    tracks_ok = np.mean([sample["tracks_found"] >= 8 for sample in samples])
    cycle = float(np.median([sample["cycle_valid_fraction"] for sample in samples]))
    coverage = float(np.median([sample["motion_energy_coverage"] for sample in samples]))
    rho_values = [sample["action_motion_spearman"] for sample in samples if np.isfinite(sample["action_motion_spearman"])]
    rho = float(np.median(rho_values)) if rho_values else float("nan")
    lag_ok = np.mean([sample["best_state_lag"] in (0, 1) for sample in samples])
    oracle_pass = tracks_ok >= 0.75 and cycle >= 0.60 and coverage >= 0.20
    temporal_pass = lag_ok >= 0.75 and np.isfinite(rho) and rho >= 0.15
    if oracle_pass and temporal_pass:
        verdict = "PASS_ORACLE_TRACKS_AND_TEMPORAL_ALIGNMENT"
    elif oracle_pass:
        verdict = "PASS_ORACLE_TRACKS_ONLY_ACTION_MAPPING_UNPROVEN"
    else:
        verdict = "REJECT_ORACLE_TRACK_EXTRACTION"
    return {
        "samples": len(samples),
        "track_count_pass_fraction": float(tracks_ok),
        "median_cycle_valid_fraction": cycle,
        "median_motion_energy_coverage": coverage,
        "median_action_motion_spearman": rho,
        "state_lag_0_or_1_fraction": float(lag_ok),
        "oracle_track_gate": bool(oracle_pass),
        "temporal_alignment_gate": bool(temporal_pass),
        "verdict": verdict,
        "scope_warning": (
            "This gate validates train-video oracle tracks and action/state timing only. "
            "It does not validate action-to-track prediction or Wan controllability."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.frames != 17:
        raise ValueError("The competition/Wan gate is fixed to 17 source frames (first + 16 future).")
    data_root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    valset_output = Path(args.valset_output) if args.valset_output else output / "valset"
    for folder in ("images", "actions", "gt_videos"):
        (valset_output / folder).mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    paths = discover_lerobot_so100_datasets(data_root)
    paths = [path for path in paths if not any(path.endswith(name) for name in DEFAULT_EXCLUDE)]
    shuffled = list(paths)
    random.Random(args.seed).shuffle(shuffled)
    held = set(shuffled[: args.holdout_count])
    split_paths = {
        "train": [path for path in paths if path not in held],
        "holdout": [path for path in paths if path in held],
    }
    counts = {"train": args.train_clips, "holdout": args.holdout_clips}
    requested_splits = ("train", "holdout") if args.split == "both" else (args.split,)

    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    device = torch.device(args.device)
    model = raft_small(weights=weights, progress=False).eval().requires_grad_(False).to(device)
    stats = json.loads((data_root / "so100_action_statistics.json").read_text())
    action_std = np.asarray(stats["std"], dtype=np.float32)

    manifest = []
    valset_manifest = []
    all_metrics = []
    for split in requested_splits:
        base = build_base(data_root, split_paths[split], args.frames, args.seed)
        records = select_records(base, counts[split], args.seed + (0 if split == "train" else 10_000))
        for local_index, record in enumerate(records):
            sample_id = f"{split}_{local_index:04d}"
            video, actions, states = load_record(record, args.frames, args.height, args.width)
            forward, backward = raft_bidirectional(model, transforms, video, device)
            tracks, visible, track_metrics = select_tracks(
                forward,
                backward,
                args.num_tracks,
                args.grid_step,
                args.min_motion,
                args.cycle_threshold,
            )
            control = render_control(
                tracks,
                visible,
                args.height,
                args.width,
                args.control_height,
                args.control_width,
            )
            timing = state_action_metrics(actions, states, action_std)
            action_distance = np.linalg.norm((states - states[0]) / action_std, axis=1)
            flow_motion = torch.linalg.vector_norm(forward, dim=1).flatten(1).median(dim=1).values.cpu().numpy()
            motion_curve = np.concatenate([[0.0], flow_motion])
            rho = spearmanr(action_distance, motion_curve).statistic
            rho = float(rho) if np.isfinite(rho) else 0.0

            relative_data = Path(record["data_ref"]).relative_to(data_root).as_posix()
            relative_video = Path(record["video_ref"]).relative_to(data_root).as_posix()
            artifact = output / f"{sample_id}.npz"
            np.savez_compressed(
                artifact,
                control=control.astype(np.float16),
                tracks=tracks.astype(np.float32),
                visible=visible,
                actions=actions,
                states=states,
            )
            save_montage(output / f"{sample_id}_tracks.jpg", video, tracks, visible)
            if split == "holdout":
                Image.fromarray(video[0]).save(valset_output / "images" / f"{sample_id}.png")
                np.save(valset_output / "actions" / f"{sample_id}.npy", actions[:16])
                np.save(valset_output / "gt_videos" / f"{sample_id}.npy", video[:16])
                valset_manifest.append(
                    {
                        "sample_id": sample_id,
                        "dataset": Path(relative_data).parts[0] + "/" + Path(relative_data).parts[1],
                        "split": "training_group_holdout",
                    }
                )
            sample_metrics = {
                "sample_id": sample_id,
                "split": split,
                "dataset": Path(relative_data).parts[0] + "/" + Path(relative_data).parts[1],
                "data_ref": relative_data,
                "video_ref": relative_video,
                "start": int(record["start"]),
                "artifact": portable_path(artifact),
                "action_motion_spearman": rho,
                **track_metrics,
                **timing,
            }
            manifest.append(sample_metrics)
            all_metrics.append(sample_metrics)
            print(
                f"[{sample_id}] tracks={track_metrics['tracks_found']} "
                f"cycle={track_metrics['cycle_valid_fraction']:.3f} "
                f"coverage={track_metrics['motion_energy_coverage']:.3f} rho={rho:.3f}"
            )

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (valset_output / "manifest.json").write_text(json.dumps(valset_manifest, indent=2))
    split_aggregates = {
        split: aggregate_verdict([row for row in all_metrics if row["split"] == split])
        for split in requested_splits
    }
    decision_split = "holdout" if "holdout" in split_aggregates else requested_splits[0]
    aggregate = split_aggregates[decision_split]
    report = {
        "config": vars(args),
        "train_datasets": len(split_paths["train"]),
        "holdout_datasets": len(split_paths["holdout"]),
        "aggregate": aggregate,
        "aggregate_by_split": split_aggregates,
        "decision_split": decision_split,
        "manifest": portable_path(manifest_path),
        "valset": portable_path(valset_output),
        "per_sample": all_metrics,
    }
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(aggregate, indent=2, allow_nan=False))
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()
