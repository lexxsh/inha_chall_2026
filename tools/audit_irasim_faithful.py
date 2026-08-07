"""CPU audit for the faithful SO-100 IRASim transfer boundary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train.irasim_so100 import (  # noqa: E402
    IRASimSO100Dataset,
    build_irasim,
    load_irasim_checkpoint,
    preprocess_irasim_frames,
    prepare_eval_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=str(REPO / "models/IRASim/0300000.pt"))
    parser.add_argument("--dataset-root", default=str(REPO / "open/data/train"))
    parser.add_argument("--holdout-root", default=str(REPO / "valset_holdout"))
    parser.add_argument("--output", default=str(REPO / "results/irasim_faithful_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = IRASimSO100Dataset(
        args.dataset_root,
        repeat=1,
        holdout_count=6,
        action_mode="absolute",
    )
    sample = dataset[0]
    if sample["video"].shape != (16, 3, 256, 320):
        raise RuntimeError(f"Bad video shape: {tuple(sample['video'].shape)}")
    if sample["actions"].shape != (15, 6):
        raise RuntimeError(f"Bad action shape: {tuple(sample['actions'].shape)}")

    holdout = Path(args.holdout_root)
    source = np.asarray(Image.open(holdout / "images/val_000000.png").convert("RGB"))
    target = np.load(holdout / "gt_videos/val_000000.npy")[0]
    source_is_gt_frame0 = bool(np.array_equal(source, target))
    raw_eval_actions = np.load(holdout / "actions/val_000000.npy")
    eval_actions = prepare_eval_actions(raw_eval_actions, "absolute")
    relative_without_state_rejected = False
    try:
        prepare_eval_actions(raw_eval_actions, "delta_step")
    except ValueError:
        relative_without_state_rejected = True

    # A non-square synthetic frame distinguishes direct resize from both
    # letterbox and center-crop modes and guards this boundary against drift.
    synthetic = np.arange(37 * 61 * 3, dtype=np.uint8).reshape(1, 37, 61, 3)
    actual_resize = preprocess_irasim_frames(synthetic)
    expected_resize = torch.nn.functional.interpolate(
        torch.from_numpy(synthetic).float().permute(0, 3, 1, 2) / 255.0,
        size=(256, 320),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    expected_resize = ((expected_resize - 0.5) * 2.0).permute(1, 0, 2, 3)
    direct_resize_exact = bool(torch.equal(actual_resize, expected_resize))

    model = build_irasim(action_dim=6, num_frames=16)
    report = load_irasim_checkpoint(model, args.pretrained)
    adapter_zero = float(model.action_adapter.weight.detach().abs().max())

    # Prove that zero initialization preserves the public action MLP while the
    # new adapter can learn on the first backward, without a full video pass.
    normal = sample["actions"].unsqueeze(0)
    embedded = model.backbone.embed_state(model.action_adapter(normal))
    embedded.square().mean().backward()
    adapter_grad = float(model.action_adapter.weight.grad.detach().abs().sum())

    result = {
        "checkpoint": str(Path(args.pretrained).resolve()),
        "checkpoint_report": report,
        "official_video_frames": model.backbone.num_frames,
        "official_temporal_embedding_shape": list(model.backbone.temp_embed.shape),
        "official_action_mlp_input_shape": list(model.backbone.embed_state.fc1.weight.shape),
        "so100_adapter_shape": list(model.action_adapter.weight.shape),
        "so100_adapter_zero_abs_max": adapter_zero,
        "so100_adapter_first_backward_grad_abs_sum": adapter_grad,
        "train_video_shape": list(sample["video"].shape),
        "train_action_shape": list(sample["actions"].shape),
        "eval_action_shape": list(eval_actions.shape),
        "eval_action_mode": "absolute_deployable",
        "relative_without_source_state_rejected": relative_without_state_rejected,
        "source_png_exactly_equals_gt_frame0": source_is_gt_frame0,
        "official_direct_resize_exact": direct_resize_exact,
        "verdict": "PASS" if (
            report["format"] == "public_rt1"
            and not report["mismatched"]
            and not report["unexpected"]
            and adapter_zero == 0.0
            and adapter_grad > 0.0
            and source_is_gt_frame0
            and direct_resize_exact
            and relative_without_state_rejected
        ) else "FAIL",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
