"""Prepare the exact latent/state inputs used by the SO-100 Ctrl-World port."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train.ctrl_world_so100 import (  # noqa: E402
    ACTION_DIM,
    HISTORY_FRAMES,
    PREDICTION_FRAMES,
    TOKENS_PER_SAMPLE,
    build_rollout_pose_tokens,
    discover_episode_records,
    split_dataset_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "stats", "cache"), default="audit")
    parser.add_argument("--data-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument(
        "--cache-root", type=Path, default=REPO / "cache/ctrl_world_so100_svd"
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=REPO / "results/ctrl_world_so100_pose_stats.json",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=REPO / "results/ctrl_world_so100_contract_audit.json",
    )
    parser.add_argument(
        "--svd-model-path",
        type=Path,
        default=REPO / "checkpoints/stable-video-diffusion-img2vid",
    )
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=20260806)
    parser.add_argument("--camera-key", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_video(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def preprocess(frames: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Letterbox to 192x320 and normalize to [-1,1]."""
    tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    source_height, source_width = tensor.shape[-2:]
    scale = min(height / source_height, width / source_width)
    resized_height = max(1, round(source_height * scale))
    resized_width = max(1, round(source_width * scale))
    tensor = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (height - resized_height) // 2
    bottom = height - resized_height - top
    left = (width - resized_width) // 2
    right = width - resized_width - left
    tensor = F.pad(tensor, (left, right, top, bottom), value=0.0)
    return tensor.mul(2.0).sub(1.0)


def unique_data_paths(records):
    seen = set()
    for record in records:
        if record.data_path in seen:
            continue
        seen.add(record.data_path)
        yield record.data_path


def compute_stats(records, output: Path) -> dict:
    arrays = []
    for index, data_path in enumerate(unique_data_paths(records), start=1):
        table = pd.read_parquet(data_path, columns=["observation.state"])
        state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        if state.ndim != 2 or state.shape[1] != ACTION_DIM:
            raise ValueError(f"Unexpected state shape {state.shape} in {data_path}")
        arrays.append(state)
        if index % 1000 == 0:
            print(f"[stats] {index} episodes")
    values = np.concatenate(arrays, axis=0)
    payload = {
        "state_p01": np.percentile(values, 1, axis=0).tolist(),
        "state_p99": np.percentile(values, 99, axis=0).tolist(),
        "state_mean": values.mean(axis=0).tolist(),
        "state_std": values.std(axis=0).tolist(),
        "frames": int(len(values)),
        "episodes": len(arrays),
        "normalization": "clip(2*(x-p01)/(p99-p01)-1,-1,1)",
        "source": "training split only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved -> {output.resolve()}")
    return payload


def run_audit(args, train_paths, holdout_paths, records) -> None:
    sample_actions = torch.arange(16 * ACTION_DIM, dtype=torch.float32).reshape(16, ACTION_DIM)
    chunks, futures = build_rollout_pose_tokens(
        sample_actions,
        torch.zeros(ACTION_DIM),
        torch.ones(ACTION_DIM) * 100,
    )
    missing_video = [str(row.video_path) for row in records if not row.video_path.is_file()]
    missing_data = [str(row.data_path) for row in records if not row.data_path.is_file()]
    manifest_path = REPO / "valset_holdout/manifest.json"
    manifest_datasets = (
        sorted({row["dataset"] for row in json.loads(manifest_path.read_text())})
        if manifest_path.is_file()
        else []
    )
    manifest_leakage = sorted(set(manifest_datasets) & set(train_paths))
    report = {
        "method": "released Ctrl-World single-view adaptation",
        "train_datasets": len(train_paths),
        "holdout_datasets": len(holdout_paths),
        "episodes_times_cameras": len(records),
        "history_frames": HISTORY_FRAMES,
        "prediction_frames": PREDICTION_FRAMES,
        "tokens_per_sample": TOKENS_PER_SAMPLE,
        "action_dim": ACTION_DIM,
        "rollout_chunks": len(chunks),
        "rollout_chunk_shapes": [list(value.shape) for value in chunks],
        "future_group_shapes": [list(value.shape) for value in futures],
        "train_holdout_overlap": sorted(set(train_paths) & set(holdout_paths)),
        "frozen_manifest_datasets": manifest_datasets,
        "frozen_manifest_leakage_into_train": manifest_leakage,
        "missing_video_count": len(missing_video),
        "missing_data_count": len(missing_data),
        "missing_examples": (missing_video + missing_data)[:10],
        "architecture_contract": {
            "backbone": "stabilityai/stable-video-diffusion-img2vid",
            "trainable": "full SVD UNet + 3-layer action projection MLP",
            "frozen": "SVD VAE and image encoder",
            "conditioning": "per-frame cross-attention",
            "loss": "released Ctrl-World EDM-preconditioned x0 MSE on current+future",
            "text_conditioning": False,
            "multi_view": False,
        },
    }
    passed = (
        not report["train_holdout_overlap"]
        and not manifest_leakage
        and not missing_video
        and not missing_data
    )
    report["verdict"] = "PASS_CTRL_WORLD_DATA_CONTRACT" if passed else "FAIL_CTRL_WORLD_DATA_CONTRACT"
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {args.audit_path.resolve()}")


@torch.inference_mode()
def cache_records(args, records) -> None:
    accelerator = Accelerator()
    if not args.svd_model_path.is_dir():
        raise FileNotFoundError(
            f"Missing SVD model: {args.svd_model_path}. Download the official base first."
        )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        str(args.svd_model_path), subfolder="vae", torch_dtype=torch.float32
    ).to(accelerator.device)
    vae.eval()
    local = records[accelerator.process_index :: accelerator.num_processes]
    if args.limit is not None:
        local = local[: args.limit]
    started = time.perf_counter()
    completed = 0
    for local_index, record in enumerate(local, start=1):
        if record.latent_path.is_file() and not args.overwrite:
            continue
        table = pd.read_parquet(record.data_path, columns=["observation.state", "action"])
        states = torch.from_numpy(
            np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        )
        actions = torch.from_numpy(np.stack(table["action"].to_numpy()).astype(np.float32))
        frames = read_video(record.video_path)
        usable = min(len(frames), len(states), len(actions), record.length)
        frames = preprocess(frames[:usable], args.height, args.width)
        encoded = []
        for start in range(0, usable, args.encode_batch_size):
            batch = frames[start : start + args.encode_batch_size].to(accelerator.device)
            latent = vae.encode(batch).latent_dist.sample()
            latent = latent.mul(vae.config.scaling_factor).to(torch.float16).cpu()
            encoded.append(latent)
        payload = {
            "latent": torch.cat(encoded),
            "state": states[:usable],
            "action": actions[:usable],
            "meta": {
                "dataset_path": record.dataset_path,
                "episode_index": record.episode_index,
                "camera_key": record.camera_key,
                "video_path": str(record.video_path),
                "height": args.height,
                "width": args.width,
                "usable_frames": usable,
                "vae_scaling_factor_applied": True,
            },
        }
        record.latent_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = record.latent_path.with_suffix(
            record.latent_path.suffix + f".rank{accelerator.process_index}.tmp"
        )
        torch.save(payload, temporary)
        os.replace(temporary, record.latent_path)
        completed += 1
        if local_index % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[cache rank {accelerator.process_index}] {local_index}/{len(local)} "
                f"new={completed} elapsed={elapsed:.1f}s"
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Ctrl-World cache complete -> {args.cache_root.resolve()}")


def main() -> None:
    args = parse_args()
    train_paths, holdout_paths = split_dataset_paths(
        args.data_root, holdout_count=args.holdout_count, seed=args.split_seed
    )
    all_paths = train_paths + holdout_paths
    records = discover_episode_records(
        args.data_root, args.cache_root, all_paths, camera_key=args.camera_key
    )
    if args.phase == "audit":
        run_audit(args, train_paths, holdout_paths, records)
    elif args.phase == "stats":
        train_records = [row for row in records if row.dataset_path in set(train_paths)]
        compute_stats(train_records, args.stats_path)
    else:
        cache_records(args, records)


if __name__ == "__main__":
    main()
