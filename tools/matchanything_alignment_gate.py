#!/usr/bin/env python3
"""VAP-style rendered-mesh to real-frame rectification gate.

Run with ``PYTHONPATH=third_party/matchanything_runtime``.  This is a CPU-first
diagnostic and never invokes the video generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from so100_renderer import SO100Model  # noqa: E402


URDF = REPO / "third_party/ManiSkill/mani_skill/assets/robots/so100/so100.urdf"
MODEL_PATH = REPO / "models/matchanything_eloftr"
ALIGNMENT_JSON = REPO / "results/so100_track_alignment_gate.json"
VALSET = REPO / "diagnostics/spatial_control_gate/valset"


def euler_matrix(angles: np.ndarray) -> np.ndarray:
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rz @ ry @ rx


def load_matcher():
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching

    processor = AutoImageProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False
    )
    model = AutoModelForKeypointMatching.from_pretrained(
        MODEL_PATH, local_files_only=True
    ).to("cpu")
    model.eval()
    return processor, model


def keypoint_matches(processor, model, render: np.ndarray, real: np.ndarray, threshold: float):
    images = [Image.fromarray(render[..., ::-1]), Image.fromarray(real[..., ::-1])]
    inputs = processor(images, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    sizes = [[(render.shape[0], render.shape[1]), (real.shape[0], real.shape[1])]]
    matches = processor.post_process_keypoint_matching(outputs, sizes, threshold=threshold)[0]
    return (
        matches["keypoints0"].cpu().numpy().astype(np.float32),
        matches["keypoints1"].cpu().numpy().astype(np.float32),
        matches["matching_scores"].cpu().numpy().astype(np.float32),
    )


def filter_foreground(
    source: np.ndarray, target: np.ndarray, scores: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    # Interior matches can be explained by identical background context in the
    # composite and produce an identity-homography shortcut.  VAP rectification
    # needs correspondences on the rendered robot's structural boundary.
    boundary_band = (dilated > 0) & (eroded == 0)
    xy = np.round(source).astype(int)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    foreground = np.zeros(len(source), dtype=bool)
    foreground[valid] = boundary_band[xy[valid, 1], xy[valid, 0]]
    return source[foreground], target[foreground], scores[foreground]


def target_edge_distance(points: np.ndarray, image: np.ndarray) -> float:
    if not len(points):
        return float("inf")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    xy = np.round(points).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, image.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, image.shape[0] - 1)
    return float(np.median(distance[xy[:, 1], xy[:, 0]]))


def estimate_homography(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    if len(source) < 4:
        return None, np.zeros(len(source), dtype=bool)
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if homography is None or inliers is None:
        return None, np.zeros(len(source), dtype=bool)
    return homography, inliers.ravel().astype(bool)


def hull_coverage(points: np.ndarray, mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(points) < 3 or not len(xs):
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    area = cv2.contourArea(hull)
    bbox_area = max(float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)), 1.0)
    return float(area / bbox_area)


def draw_diagnostic(
    render: np.ndarray,
    real: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    inliers: np.ndarray,
    homography: np.ndarray | None,
) -> np.ndarray:
    height, width = real.shape[:2]
    warped = (
        cv2.warpPerspective(render, homography, (width, height))
        if homography is not None
        else np.zeros_like(real)
    )
    overlay = cv2.addWeighted(real, 0.6, warped, 0.4, 0)
    pair = np.hstack([render, real])
    for index, (start, end) in enumerate(zip(source, target)):
        color = (0, 220, 0) if inliers[index] else (0, 0, 220)
        p0 = tuple(np.round(start).astype(int))
        p1 = tuple(np.round(end + np.array((width, 0))).astype(int))
        cv2.line(pair, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(pair, p0, 2, color, -1)
        cv2.circle(pair, p1, 2, color, -1)
    return np.vstack([pair, np.hstack([warped, overlay])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--source-mode", choices=("render", "composite"), default="composite")
    parser.add_argument(
        "--output", type=Path, default=REPO / "results/matchanything_alignment_gate.json"
    )
    parser.add_argument(
        "--diagnostic-root", type=Path, default=REPO / "diagnostics/matchanything_alignment"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignment = json.loads(ALIGNMENT_JSON.read_text())
    by_id = {row["sample_id"]: row for row in alignment["samples"]}
    renderer = SO100Model(URDF)
    processor, matcher = load_matcher()
    results = []
    args.diagnostic_root.mkdir(parents=True, exist_ok=True)
    for index in range(args.num_samples):
        sample_id = f"holdout_{index:04d}"
        rough = by_id.get(sample_id)
        if not rough or rough["status"] != "OK":
            results.append({"sample_id": sample_id, "status": "SKIP_NO_ROUGH_ALIGNMENT"})
            continue
        real = cv2.imread(str(VALSET / "images" / f"{sample_id}.png"))
        if real is None:
            results.append({"sample_id": sample_id, "status": "SKIP_NO_IMAGE"})
            continue
        self_source, _, _ = keypoint_matches(processor, matcher, real, real, 0.0)
        camera = rough["camera_ortho"]
        render, render_mask = renderer.render_orthographic(
            np.asarray(rough["q0_radians"]),
            euler_matrix(np.asarray(camera["euler"])),
            float(camera["pixels_per_meter"]),
            np.asarray(camera["translation"]),
            width=real.shape[1],
            height=real.shape[0],
        )
        source_image = render
        if args.source_mode == "composite":
            source_image = real.copy()
            foreground = render_mask > 0
            source_image[foreground] = render[foreground]
        source, target, scores = keypoint_matches(
            processor, matcher, source_image, real, args.threshold
        )
        total_matches = len(source)
        source, target, scores = filter_foreground(source, target, scores, render_mask)
        homography, inliers = estimate_homography(source, target)
        if homography is not None and inliers.any():
            projected = cv2.perspectiveTransform(source[None], homography)[0]
            reprojection = np.linalg.norm(projected - target, axis=1)
            median_reprojection = float(np.median(reprojection[inliers]))
            coverage = hull_coverage(source[inliers], render_mask)
            edge_distance = target_edge_distance(target[inliers], real)
        else:
            median_reprojection = float("inf")
            coverage = 0.0
            edge_distance = float("inf")
        inlier_count = int(inliers.sum())
        inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
        passed = (
            inlier_count >= 8
            and inlier_ratio >= 0.3
            and median_reprojection <= 4.0
            and coverage >= 0.08
            and edge_distance <= 3.0
        )
        record = {
            "sample_id": sample_id,
            "status": "PASS" if passed else "FAIL",
            "self_match_count": int(len(self_source)),
            "total_matches": int(total_matches),
            "foreground_matches": int(len(source)),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "median_reprojection_px": median_reprojection,
            "source_hull_coverage": coverage,
            "target_median_edge_distance_px": edge_distance,
            "homography": homography.tolist() if homography is not None else None,
        }
        results.append(record)
        diagnostic = draw_diagnostic(source_image, real, source, target, inliers, homography)
        cv2.imwrite(str(args.diagnostic_root / f"{sample_id}.jpg"), diagnostic)
        print(json.dumps({key: value for key, value in record.items() if key != "homography"}))
    valid = [row for row in results if row["status"] in ("PASS", "FAIL")]
    pass_rate = float(np.mean([row["status"] == "PASS" for row in valid])) if valid else 0.0
    output = {
        "method": "VAP-style MatchAnything render-to-real homography; train-only gate",
        "pass_rate": pass_rate,
        "verdict": "PASS_MATCHANYTHING_SMOKE" if valid and pass_rate >= 0.6 else "REJECT_MATCHANYTHING_ALIGNMENT",
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "samples"}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
