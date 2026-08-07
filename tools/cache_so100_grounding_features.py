"""Cache official DINO features plus aligned SO-100 actions/states.

The default device is CPU.  GPU use must be requested explicitly with
``--device cuda`` by the person launching the script.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
SUBMISSION_KIT = REPO / "open/submission_kit"
for path in (SUBMISSION_KIT, KIT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_csv_utils import (  # noqa: E402
    extract_dino_features,
    load_dino_model,
    resolve_dino_image_size,
)
from ldwma.datasets.lerobot_so100 import _decode_video_clip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument(
        "--manifest", type=Path, default=REPO / "results/so100_contract_manifest.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "results/so100_grounding_features.pt"
    )
    parser.add_argument("--train-records", type=int, default=4096)
    parser.add_argument("--holdout-records", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def balanced_select(records: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin uploaders so large contributors cannot dominate the probe."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["uploader"]].append(record)
    queues = {}
    for offset, (name, rows) in enumerate(sorted(groups.items())):
        random.Random(seed + offset).shuffle(rows)
        queues[name] = deque(rows)
    names = list(queues)
    random.Random(seed).shuffle(names)
    selected: list[dict] = []
    while len(selected) < count and names:
        remaining = []
        for name in names:
            if queues[name] and len(selected) < count:
                selected.append(queues[name].popleft())
            if queues[name]:
                remaining.append(name)
        names = remaining
    return selected


def load_window(data_root: Path, record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = list(range(record["start"], record["start"] + record["frames"]))
    table = pd.read_parquet(
        data_root / record["data_rel"], columns=["action", "observation.state"]
    )
    actions = np.stack(table["action"].iloc[indices].to_numpy()).astype(np.float32)
    states = np.stack(table["observation.state"].iloc[indices].to_numpy()).astype(np.float32)
    video = _decode_video_clip(data_root / record["video_rel"], indices)
    if video.shape[0] != 16 or actions.shape != (16, 6) or states.shape != (16, 6):
        raise ValueError(
            f"bad contract shapes for {record['id']}: video={video.shape}, "
            f"actions={actions.shape}, states={states.shape}"
        )
    return video, actions, states


def extract_ragged_dino_features(
    videos: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> torch.Tensor:
    """Extract features without requiring every source video to share a resolution.

    ``extract_dino_features`` performs the official aspect-preserving resize/pad,
    but its tensor input must be rectangular.  SO-100 datasets come from several
    uploaders and therefore contain multiple camera resolutions.  Grouping equal
    shapes keeps the official preprocessing bit-for-bit identical to processing
    each video independently while retaining batching where possible.
    """
    if not videos:
        raise ValueError("cannot extract DINO features from an empty video list")

    shape_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    temporal_lengths = set()
    for index, video in enumerate(videos):
        if video.ndim != 4 or video.shape[0] < 1 or video.shape[-1] != 3:
            raise ValueError(f"expected video shaped (T,H,W,3), got {video.shape}")
        temporal_lengths.add(video.shape[0])
        shape_groups[tuple(video.shape)].append(index)
    if len(temporal_lengths) != 1:
        raise ValueError(f"all videos must share a temporal length, got {sorted(temporal_lengths)}")

    ordered: list[torch.Tensor | None] = [None] * len(videos)
    for indices in shape_groups.values():
        rectangular = torch.from_numpy(np.stack([videos[index] for index in indices]))
        group_features = extract_dino_features(rectangular, model, device, image_size)
        for index, feature in zip(indices, group_features, strict=True):
            ordered[index] = feature

    if any(feature is None for feature in ordered):
        raise RuntimeError("internal error while restoring ragged DINO feature order")
    return torch.stack([feature for feature in ordered if feature is not None])


def main() -> None:
    args = parse_args()
    if args.feature_batch_size <= 0:
        raise SystemExit("--feature-batch-size must be positive")
    records = load_manifest(args.manifest)
    selected = []
    for offset, (split, count) in enumerate(
        (("train", args.train_records), ("holdout", args.holdout_records))
    ):
        rows = [record for record in records if record["split"] == split]
        selected.extend(balanced_select(rows, min(count, len(rows)), args.seed + offset))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    model_name = "vit_small_patch14_dinov2.lvd142m"
    model = load_dino_model(device, model_name, pretrained=True)
    image_size = resolve_dino_image_size(model, requested_size=0)

    ids: list[str] = []
    splits: list[int] = []
    uploaders: list[str] = []
    actions_out: list[torch.Tensor] = []
    states_out: list[torch.Tensor] = []
    features_out: list[torch.Tensor] = []
    for begin in range(0, len(selected), args.feature_batch_size):
        batch_records = selected[begin : begin + args.feature_batch_size]
        videos, actions, states = [], [], []
        for record in batch_records:
            video, action, state = load_window(args.data_root, record)
            videos.append(video)
            actions.append(action)
            states.append(state)
        features = extract_ragged_dino_features(videos, model, device, image_size)
        features_out.append(features.to(torch.float16))
        actions_out.append(torch.from_numpy(np.stack(actions)))
        states_out.append(torch.from_numpy(np.stack(states)))
        ids.extend(record["id"] for record in batch_records)
        splits.extend(0 if record["split"] == "train" else 1 for record in batch_records)
        uploaders.extend(record["uploader"] for record in batch_records)
        print(f"[DINO cache] {min(begin + len(batch_records), len(selected))}/{len(selected)}")

    payload = {
        "ids": ids,
        "split": torch.tensor(splits, dtype=torch.uint8),
        "uploaders": uploaders,
        "actions": torch.cat(actions_out),
        "states": torch.cat(states_out),
        "dino": torch.cat(features_out),
        "metadata": {
            "manifest": str(args.manifest.resolve()),
            "model": model_name,
            "image_size": image_size,
            "frame_contract": "source=0,target=1:15,actions=0:14,action15=excluded",
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "records": len(ids),
                "train": int((payload["split"] == 0).sum()),
                "holdout": int((payload["split"] == 1).sum()),
                "dino_shape": list(payload["dino"].shape),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
