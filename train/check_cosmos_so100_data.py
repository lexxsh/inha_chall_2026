"""CPU-only contract check for the Cosmos SO-100 dataloader adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_so100.dataset import CosmosSO100Dataset

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--action-mode", default="delta")
    parser.add_argument(
        "--action-variant", choices=("normal", "zero", "reverse"), default="normal"
    )
    parser.add_argument("--split", choices=("train", "holdout"), default="train")
    args = parser.parse_args()

    dataset = CosmosSO100Dataset(
        root=args.data_root,
        split=args.split,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        action_mode=args.action_mode,
        action_variant=args.action_variant,
    )
    sample = dataset[0]
    expected = {
        "video": ((3, args.num_frames, args.height, args.width), torch.uint8),
        "action": ((args.num_frames - 1, 7), torch.float32),
        "t5_text_embeddings": ((512, 1024), torch.bfloat16),
        "padding_mask": ((1, args.height, args.width), torch.bool),
    }
    for key, (shape, dtype) in expected.items():
        value = sample[key]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise SystemExit(
                f"{key}: expected shape={shape}, dtype={dtype}; "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}"
            )
    if sample["video"].min() < 0 or sample["video"].max() > 255:
        raise SystemExit("video is outside uint8 range")
    if not torch.equal(
        sample["action"][:, 5], torch.zeros(args.num_frames - 1)
    ):
        raise SystemExit("Cosmos motion padding slot 5 is not zero")
    report = {
        "split": args.split,
        "datasets": len(dataset.selected_paths),
        "clips": len(dataset),
        "video_shape": list(sample["video"].shape),
        "action_shape": list(sample["action"].shape),
        "action_abs_mean": float(sample["action"].abs().mean()),
        "source_action_abs_mean": float(sample["action"][0].abs().mean()),
        "key": sample["__key__"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
