#!/usr/bin/env python3
"""Build and gate three SO-100 -> OSCAR skeleton calibration methods.

This program does not load OSCAR and does not train a video generator.  It
only creates the spatial condition that must be trustworthy before that work.

Methods
-------
multiframe
    Train-only RobotArena-style teacher using realized states and future
    motion tracks.  This is an oracle calibration label, not an eval method.
singleframe
    RoboPose-style source-only render-and-compare fit initialized from a
    Grounding-DINO robot box.
mask
    EasyHeC-style source-only refinement against a GrabCut robot mask seeded
    by the same zero-shot box.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from so100_renderer import SO100Model  # noqa: E402
from so100_skeleton_calibration import (  # noqa: E402
    ACTION_TO_Q_SCALE,
    Calibration,
    calibration_projection,
    chain_points,
    euler_matrix,
    fit_multiframe,
    fit_singleframe_edges,
    fit_singleframe_mask,
    motion_track_score,
    morphological_skeleton,
    project_points,
    project_trajectory,
    q_trajectory,
)


DEFAULT_URDF = REPO / "third_party/ManiSkill/mani_skill/assets/robots/so100/so100.urdf"
DEFAULT_ARTIFACT_ROOT = REPO / "diagnostics/spatial_control_gate"
DEFAULT_VALSET = DEFAULT_ARTIFACT_ROOT / "valset"
DEFAULT_DATA_ROOT = REPO / "open/data/train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default="multiframe,singleframe,mask",
        help="Comma-separated subset of multiframe,singleframe,mask.",
    )
    parser.add_argument("--split", choices=("train", "holdout"), default="holdout")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multiframe-maxiter", type=int, default=60)
    parser.add_argument("--singleframe-maxiter", type=int, default=35)
    parser.add_argument("--mask-maxiter", type=int, default=180)
    parser.add_argument("--response", type=float, default=1.0)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--valset", type=Path, default=DEFAULT_VALSET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--detector", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--box-threshold", type=float, default=0.18)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--output-root", type=Path, default=REPO / "diagnostics/oscar_so100_skeletons"
    )
    parser.add_argument(
        "--result-json", type=Path, default=REPO / "results/oscar_so100_skeleton_gate.json"
    )
    return parser.parse_args()


def _resolve_cached_model(value: str) -> str:
    explicit = Path(value)
    if explicit.exists():
        return str(explicit.resolve())
    cache = Path(
        os.environ.get(
            "HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")
        )
    )
    snapshots = cache / ("models--" + value.replace("/", "--")) / "snapshots"
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"Detector is not cached: {value}. Pass --detector /absolute/snapshot/path."
        )
    return str(candidates[-1].resolve())


def load_detector(model_name: str, device: str):
    # A direct snapshot path plus offline mode prevents transformers/PEFT from
    # making an unnecessary adapter_config HEAD request.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    path = _resolve_cached_model(model_name)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        path, local_files_only=True
    ).eval().to(device)
    return processor, model


@torch.inference_mode()
def detect_robot_box(
    processor,
    detector,
    image_rgb: np.ndarray,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, float, str]:
    image = Image.fromarray(image_rgb)
    inputs = processor(
        images=image,
        # Joint positive/negative phrases make the text labels useful for
        # separating a black glove from the adjacent white robot. Asking only
        # for "robot arm" made Grounding-DINO call the glove a robot in four
        # of the eight held-out clips.
        text=(
            "robot arm. robotic manipulator. robot base. robotic gripper. "
            "black glove. human hand."
        ),
        return_tensors="pt",
    ).to(device)
    outputs = detector(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_rgb.shape[:2]],
    )[0]
    boxes = result["boxes"].detach().cpu().numpy().astype(np.float64)
    scores = result["scores"].detach().cpu().numpy().astype(np.float64)
    labels = result.get("text_labels", result.get("labels", [""] * len(boxes)))
    labels = [str(label).lower() for label in labels]
    if not len(boxes):
        raise RuntimeError("Grounding-DINO found no robot arm")
    positive_words = ("robot", "robotic", "manipulator", "gripper")
    negative_words = ("glove", "hand")
    positive = np.asarray(
        [
            any(word in label for word in positive_words)
            and not any(word in label for word in negative_words)
            for label in labels
        ],
        dtype=bool,
    )
    negative = np.asarray(
        [
            any(word in label for word in negative_words)
            and not any(word in label for word in positive_words)
            for label in labels
        ],
        dtype=bool,
    )
    if not positive.any():
        raise RuntimeError(f"Grounding-DINO found proposals but no robot-labelled box: {labels}")

    height, width = image_rgb.shape[:2]
    frame_area = float(height * width)
    utilities = np.full(len(boxes), -1e9, dtype=np.float64)
    negative_boxes = boxes[negative]
    for index in np.flatnonzero(positive):
        candidate = boxes[index]
        size = np.maximum(candidate[2:] - candidate[:2], 1.0)
        area = float(np.prod(size))
        area_fraction = area / frame_area
        glove_coverage = 0.0
        for glove in negative_boxes:
            intersection_min = np.maximum(candidate[:2], glove[:2])
            intersection_max = np.minimum(candidate[2:], glove[2:])
            intersection = float(
                np.prod(np.maximum(intersection_max - intersection_min, 0.0))
            )
            glove_coverage = max(glove_coverage, intersection / max(area, 1.0))
        # Confidence remains primary, but compact articulated-robot proposals
        # beat tall/whole-frame shortcuts and boxes occupied by a detected hand.
        utilities[index] = (
            float(scores[index])
            - 0.70 * np.sqrt(area_fraction)
            - 0.55 * glove_coverage
            - 0.80 * max(area_fraction - 0.35, 0.0)
        )
    best = int(np.argmax(utilities))
    selected = boxes[best].copy()
    selected[[0, 2]] = np.clip(selected[[0, 2]], 0, width - 1)
    selected[[1, 3]] = np.clip(selected[[1, 3]], 0, height - 1)
    return selected, float(scores[best]), labels[best]


def grabcut_mask(image_rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = np.round(box).astype(int)
    pad_x = max(3, int(0.04 * max(x1 - x0, 1)))
    pad_y = max(3, int(0.04 * max(y1 - y0, 1)))
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(width - 1, x1 + pad_x), min(height - 1, y1 + pad_y)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError(f"Degenerate robot box: {box.tolist()}")
    labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image_rgb[..., ::-1],
        labels,
        (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
        background,
        foreground,
        7,
        cv2.GC_INIT_WITH_RECT,
    )
    mask = np.where(
        (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if cv2.countNonZero(mask) < 32:
        raise RuntimeError("GrabCut produced an empty/tiny robot mask")
    return mask


def load_source_image(
    record: dict, sample_id: str, split: str, valset: Path, data_root: Path
) -> np.ndarray:
    val_image = valset / "images" / f"{sample_id}.png"
    if split == "holdout" and val_image.exists():
        image = cv2.imread(str(val_image), cv2.IMREAD_COLOR)
    else:
        video_path = data_root / record["video_ref"]
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["start"]))
        ok, image = capture.read()
        capture.release()
        if not ok:
            image = None
    if image is None:
        raise FileNotFoundError(f"Could not load source image for {sample_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_skeleton_video(
    output: Path,
    model: SO100Model,
    calibration: Calibration,
    q_values: np.ndarray,
    width: int,
    height: int,
    fps: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    rotation = euler_matrix(calibration.euler)
    try:
        for q in q_values:
            rgb = model.render_oscar_skeleton_orthographic(
                q,
                rotation,
                calibration.pixels_per_meter,
                calibration.translation,
                width=width,
                height=height,
            )
            writer.write(rgb[..., ::-1])
    finally:
        writer.release()


def diagnostic_image(
    image_rgb: np.ndarray,
    box: np.ndarray | None,
    mask: np.ndarray | None,
    skeleton_rgb: np.ndarray,
) -> np.ndarray:
    source = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if box is not None:
        x0, y0, x1, y1 = np.round(box).astype(int)
        cv2.rectangle(source, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if mask is not None:
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(source, contour, -1, (255, 0, 255), 1)
    skeleton_bgr = skeleton_rgb[..., ::-1]
    overlay = source.copy()
    active = np.any(skeleton_rgb > 0, axis=-1)
    overlay[active] = cv2.addWeighted(source, 0.35, skeleton_bgr, 0.65, 0)[active]
    return np.hstack([source, skeleton_bgr, overlay])


def estimate_source_state_offset(manifest: list[dict], artifact_root: Path) -> np.ndarray:
    """Estimate the source state/action lag from train artifacts only."""
    offsets = []
    for record in manifest:
        if record.get("split") != "train":
            continue
        artifact = np.load(artifact_root / f"{record['sample_id']}.npz")
        actions, states = artifact["actions"], artifact["states"]
        if len(actions) and len(states):
            offsets.append(states[0].astype(np.float64) - actions[0].astype(np.float64))
    if not offsets:
        return np.zeros(6, dtype=np.float64)
    return np.median(np.asarray(offsets), axis=0)


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection_min = np.maximum(left[:2], right[:2])
    intersection_max = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(intersection_max - intersection_min, 0.0)))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    return intersection / max(left_area + right_area - intersection, 1.0)


def source_alignment_metrics(
    model: SO100Model,
    calibration: Calibration,
    robot_box: np.ndarray | None,
    robot_mask: np.ndarray | None,
    width: int,
    height: int,
) -> dict:
    """Measure whether fitted source geometry actually lies on the robot."""
    points = project_points(
        chain_points(model, calibration.q0, samples_per_link=12), calibration.params
    )
    rounded = np.round(points).astype(np.int32)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    result = {"inside_frame_fraction": float(valid.mean())}
    if robot_box is not None:
        predicted_box = np.asarray(
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
        )
        result["robot_box_iou"] = _box_iou(predicted_box, np.asarray(robot_box))
    if robot_mask is not None:
        mask = (np.asarray(robot_mask) > 0).astype(np.uint8) * 255
        dilated = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8))
        inside = 0.0
        if valid.any():
            inside = float(np.mean(dilated[rounded[valid, 1], rounded[valid, 0]] > 0))
        centerline = morphological_skeleton(mask)
        distance = cv2.distanceTransform(
            (centerline == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        centerline_distance = float(max(width, height))
        if valid.any():
            centerline_distance = float(
                np.median(distance[rounded[valid, 1], rounded[valid, 0]])
            )
        result.update(
            {
                "inside_robot_mask_fraction": inside,
                "median_centerline_distance_px": centerline_distance,
            }
        )
    return result


def score_variants(
    model: SO100Model,
    calibration: Calibration,
    actions: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    states: np.ndarray,
    source_state: np.ndarray | None,
    response: float,
    other_actions: np.ndarray,
) -> dict:
    length = min(len(actions), len(tracks))
    actions = actions[:length]
    tracks, visible = tracks[:length], visible[:length]
    if len(other_actions) != length:
        indices = np.linspace(0, len(other_actions) - 1, length).round().astype(int)
        other_actions = other_actions[indices]
    variants = {
        "normal": actions,
        "zero": np.repeat(actions[:1], length, axis=0),
        "reverse": actions[::-1].copy(),
        "batch_roll": other_actions,
    }
    scores = {}
    state_scores = {}
    oracle_q = q_trajectory(
        calibration.q0,
        np.asarray(states[:length], dtype=np.float64),
        reference=np.asarray(states[0], dtype=np.float64),
    )
    oracle_projected = project_trajectory(model, oracle_q, calibration.params)
    for name, signal in variants.items():
        _, projected = calibration_projection(
            model,
            calibration,
            signal,
            source_state=source_state,
            response=response,
        )
        scores[name] = motion_track_score(projected, tracks, visible)
        # Fixed robot-link topology avoids the assignment shortcut possible in
        # sparse RAFT tracks. Frame zero is identical by construction, so only
        # the 15 action-controlled frames contribute.
        difference = projected[1:] - oracle_projected[1:]
        state_scores[name] = float(
            np.sqrt(np.mean(np.sum(difference * difference, axis=-1)))
        )
    return {
        "motion_track_rmse": scores,
        "state_projection_rmse": state_scores,
        "normal_minus_zero": scores["normal"] - scores["zero"],
        "normal_minus_reverse": scores["normal"] - scores["reverse"],
        "normal_minus_batch_roll": scores["normal"] - scores["batch_roll"],
        "state_normal_minus_zero": state_scores["normal"] - state_scores["zero"],
        "state_normal_minus_reverse": state_scores["normal"] - state_scores["reverse"],
        "state_normal_minus_batch_roll": state_scores["normal"] - state_scores["batch_roll"],
    }


def main() -> None:
    args = parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = set(methods) - {"multiframe", "singleframe", "mask"}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.artifact_root / "manifest.json").read_text())
    source_state_offset = estimate_source_state_offset(manifest, args.artifact_root)
    split_records = [row for row in manifest if row["split"] == args.split]
    records = split_records[args.start_index : args.start_index + args.limit]
    if not records:
        raise RuntimeError(f"No {args.split} records found")

    detector_processor = detector_model = None
    if any(method in methods for method in ("singleframe", "mask")):
        detector_processor, detector_model = load_detector(
            args.detector, args.detector_device
        )

    model = SO100Model(args.urdf)
    rows = []
    for index, record in enumerate(records):
        sample_id = record["sample_id"]
        artifact = np.load(args.artifact_root / f"{sample_id}.npz")
        actions = artifact["actions"].astype(np.float64)[:16]
        states = artifact["states"].astype(np.float64)
        tracks = artifact["tracks"].astype(np.float64)[:16]
        visible = artifact["visible"].astype(bool)[:16]
        image_rgb = load_source_image(
            record, sample_id, args.split, args.valset, args.data_root
        )
        image_rgb = cv2.resize(image_rgb, (args.width, args.height))
        other_id = records[(index + 1) % len(records)]["sample_id"]
        other = np.load(args.artifact_root / f"{other_id}.npz")["actions"].astype(np.float64)[:16]
        state_motion = (states[:16] - states[0]) * ACTION_TO_Q_SCALE[None]
        oracle_state_motion = float(np.linalg.norm(state_motion, axis=1).max())
        sample = {
            "sample_id": sample_id,
            "dataset": record["dataset"],
            "oracle_state_motion_radians": oracle_state_motion,
            "oracle_dynamic": bool(oracle_state_motion >= 0.05),
            "methods": {},
        }

        box = mask = None
        detection_score = detection_label = None
        if detector_model is not None:
            try:
                box, detection_score, detection_label = detect_robot_box(
                    detector_processor,
                    detector_model,
                    image_rgb,
                    args.detector_device,
                    args.box_threshold,
                    args.text_threshold,
                )
                mask = grabcut_mask(image_rgb, box)
            except Exception as error:  # Keep multiframe diagnostics usable.
                sample["detection_error"] = f"{type(error).__name__}: {error}"

        calibrations: dict[str, Calibration] = {}
        if "multiframe" in methods and tracks.shape[1] >= 3:
            fit_tracks, fit_visible = tracks, visible
            if mask is not None:
                first_xy = np.round(tracks[0]).astype(np.int32)
                valid_xy = (
                    visible[0]
                    & (first_xy[:, 0] >= 0)
                    & (first_xy[:, 0] < args.width)
                    & (first_xy[:, 1] >= 0)
                    & (first_xy[:, 1] < args.height)
                )
                robot_tracks = np.zeros(len(first_xy), dtype=bool)
                dilated_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
                robot_tracks[valid_xy] = (
                    dilated_mask[first_xy[valid_xy, 1], first_xy[valid_xy, 0]] > 0
                )
                if robot_tracks.sum() >= 3:
                    fit_tracks = tracks[:, robot_tracks]
                    fit_visible = visible[:, robot_tracks]
            calibrations["multiframe"] = fit_multiframe(
                model,
                states[: len(tracks)],
                fit_tracks,
                fit_visible,
                source_box=box,
                source_mask=mask,
                seed=args.seed + index,
                maxiter=args.multiframe_maxiter,
            )
        if "singleframe" in methods and box is not None:
            calibrations["singleframe"] = fit_singleframe_edges(
                model,
                image_rgb,
                actions[0],
                box,
                seed=args.seed + index,
                maxiter=args.singleframe_maxiter,
            )
        if "mask" in methods and mask is not None:
            initial = calibrations.get("singleframe")
            if initial is None:
                initial = fit_singleframe_edges(
                    model,
                    image_rgb,
                    actions[0],
                    box,
                    seed=args.seed + index,
                    maxiter=args.singleframe_maxiter,
                )
            calibrations["mask"] = fit_singleframe_mask(
                model,
                image_rgb,
                actions[0],
                mask,
                initial,
                maxiter=args.mask_maxiter,
            )

        for name, calibration in calibrations.items():
            # Only the train-only teacher may use measured source state.
            source_state = (
                states[0]
                if name == "multiframe"
                else actions[0] + source_state_offset
            )
            q_values, _ = calibration_projection(
                model,
                calibration,
                actions,
                source_state=source_state,
                response=args.response,
            )
            method_root = args.output_root / name
            write_skeleton_video(
                method_root / f"{sample_id}.mp4",
                model,
                calibration,
                q_values,
                args.width,
                args.height,
                args.fps,
            )
            first_skeleton = model.render_oscar_skeleton_orthographic(
                calibration.q0,
                euler_matrix(calibration.euler),
                calibration.pixels_per_meter,
                calibration.translation,
                width=args.width,
                height=args.height,
            )
            cv2.imwrite(
                str(method_root / f"{sample_id}.jpg"),
                diagnostic_image(image_rgb, box, mask, first_skeleton),
            )
            scores = score_variants(
                model,
                calibration,
                actions,
                tracks,
                visible,
                states,
                source_state,
                args.response,
                other,
            )
            alignment = source_alignment_metrics(
                model,
                calibration,
                box,
                mask,
                args.width,
                args.height,
            )
            sample["methods"][name] = {
                **calibration.to_json(),
                **scores,
                "source_alignment": alignment,
                "source_state_policy": (
                    "oracle_observation_state"
                    if name == "multiframe"
                    else "train_median_state_minus_action"
                ),
            }
            print(
                f"[{sample_id}:{name}] track={scores['motion_track_rmse']['normal']:.3f} "
                f"state={scores['state_projection_rmse']['normal']:.3f} "
                f"n-z={scores['normal_minus_zero']:.3f} "
                f"state-n-z={scores['state_normal_minus_zero']:.3f} "
                f"n-roll={scores['normal_minus_batch_roll']:.3f}"
            )
        sample["robot_box"] = box.tolist() if box is not None else None
        sample["detection_score"] = detection_score
        sample["detection_label"] = detection_label
        rows.append(sample)

    aggregate = {}
    for method in methods:
        available = [row["methods"][method] for row in rows if method in row["methods"]]
        if not available:
            aggregate[method] = {"available": 0, "verdict": "NO_RESULT"}
            continue
        comparisons = (
            "state_normal_minus_zero",
            "state_normal_minus_reverse",
            "state_normal_minus_batch_roll",
        )
        medians = {
            key: float(np.median([row[key] for row in available])) for key in comparisons
        }
        win_rates = {
            key: float(np.mean([row[key] < 0 for row in available])) for key in comparisons
        }
        dynamic_available = [
            row["methods"][method]
            for row in rows
            if row.get("oracle_dynamic", False) and method in row["methods"]
        ]
        track_values = [row["motion_track_rmse"]["normal"] for row in dynamic_available]
        track_alignment = {
            "dynamic_samples": len(dynamic_available),
            "median_normal_rmse_px": (
                float(np.median(track_values)) if track_values else float("inf")
            ),
            "fraction_below_12px": (
                float(np.mean(np.asarray(track_values) < 12.0)) if track_values else 0.0
            ),
            "normal_beats_reverse_fraction": (
                float(
                    np.mean(
                        [
                            row["motion_track_rmse"]["normal"]
                            < row["motion_track_rmse"]["reverse"]
                            for row in dynamic_available
                        ]
                    )
                )
                if track_values
                else 0.0
            ),
            "normal_beats_batch_roll_fraction": (
                float(
                    np.mean(
                        [
                            row["motion_track_rmse"]["normal"]
                            < row["motion_track_rmse"]["batch_roll"]
                            for row in dynamic_available
                        ]
                    )
                )
                if track_values
                else 0.0
            ),
        }
        track_pass = (
            track_alignment["median_normal_rmse_px"] <= 10.0
            and track_alignment["fraction_below_12px"] >= 0.75
            and track_alignment["normal_beats_reverse_fraction"] >= 0.60
            and track_alignment["normal_beats_batch_roll_fraction"] >= 0.60
        )
        complete = len(available) == len(rows)
        # One-sample smoke runs remain INSUFFICIENT. A formal gate must produce
        # every requested sample; silently dropping a failed GrabCut fit is a
        # rejection, not an easier denominator.
        enough = len(rows) >= 6 and complete
        passed = (
            enough
            and win_rates["state_normal_minus_zero"] >= 0.60
            and win_rates["state_normal_minus_reverse"] >= 0.75
            and win_rates["state_normal_minus_batch_roll"] >= 0.75
        )
        alignment_rows = [row.get("source_alignment", {}) for row in available]
        alignment_medians = {}
        for key in (
            "inside_frame_fraction",
            "inside_robot_mask_fraction",
            "median_centerline_distance_px",
            "robot_box_iou",
        ):
            values = [row[key] for row in alignment_rows if key in row]
            if values:
                alignment_medians[key] = float(np.median(values))
        alignment_pass = (
            alignment_medians.get("inside_frame_fraction", 0.0) >= 0.85
            and alignment_medians.get("inside_robot_mask_fraction", 0.0) >= 0.50
            and alignment_medians.get("median_centerline_distance_px", 1e9) <= 12.0
            and alignment_medians.get("robot_box_iou", 0.0) >= 0.15
        )
        passed = passed and alignment_pass and track_pass
        aggregate[method] = {
            "available": len(available),
            "requested": len(rows),
            "complete": bool(complete),
            "deployable": bool(available[0]["deployable"]),
            "median_differences": medians,
            "win_rates": win_rates,
            "median_source_alignment": alignment_medians,
            "source_alignment_gate": bool(alignment_pass),
            "dynamic_track_alignment": track_alignment,
            "dynamic_track_gate": bool(track_pass),
            "verdict": (
                "PASS_SKELETON_GEOMETRY"
                if passed
                else "INSUFFICIENT_SMOKE"
                if len(rows) < 6
                else "REJECT_SKELETON_GEOMETRY"
            ),
        }

    report = {
        "scope": "skeleton calibration only; OSCAR/video generator not run",
        "split": args.split,
        "methods": list(methods),
        "response": args.response,
        "temporal_contract": "frame0=source_pose; frame1..15=action0..14",
        "deployable_source_state_offset": source_state_offset.tolist(),
        "aggregate": aggregate,
        "samples": rows,
    }
    args.result_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    print(f"saved -> {args.result_json}")


if __name__ == "__main__":
    main()
