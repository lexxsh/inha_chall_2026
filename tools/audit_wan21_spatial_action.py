"""CPU-only structural audit for the Wan2.1 SO100 spatial-action implementation.

This intentionally does not load the 14B backbone or reserve a GPU.  It checks
the contracts that otherwise tend to fail only after an expensive launch:
temporal alignment, exact-zero initialization, downstream/upstream gradient
flow, counterfactual sensitivity, dataset shapes, and local model completeness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
for path in (REPO, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffsynth.models.wan_video_dit import SourceConditionedSpatialActionAdapter  # noqa: E402
from train.train_wan21_spatial_action import (  # noqa: E402
    grouped_motion_mask,
    local_model_paths,
    paired_spatial_action_flow_loss,
)
from train.wan_spatial_action_dataset import WanSO100SpatialActionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", default=str(REPO / "open/data/train")
    )
    parser.add_argument(
        "--base-model-path", default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P")
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument(
        "--out", default=str(REPO / "results/wan21_spatial_action_structure_audit.json")
    )
    return parser.parse_args()


def audit_adapter() -> dict:
    torch.manual_seed(0)
    adapter = SourceConditionedSpatialActionAdapter(
        action_dim=18,
        model_dim=64,
        source_dim=16,
        hidden_dim=32,
        num_layers=40,
        injection_layers=(0, 10, 20, 30),
    )
    actions = torch.randn(2, 16, 18)
    source = torch.randn(2, 16, 8, 10)
    initial = adapter(actions, source, (5, 4, 5))
    zero_max = float(initial.abs().max())
    if zero_max != 0.0:
        raise AssertionError(f"Adapter must preserve pretrained Wan at init, max={zero_max}")

    # At exact-zero initialization the output projection learns first.  After
    # that update, gradients must reach both the action and source encoders.
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    target = torch.randn_like(initial)
    (initial - target).square().mean().backward()
    output_grad = float(adapter.output.weight.grad.abs().sum())
    if output_grad <= 0:
        raise AssertionError("Zero-initialized output projection received no gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    actions = actions.detach().requires_grad_(True)
    source = source.detach().requires_grad_(True)
    trained = adapter(actions, source, (5, 4, 5))
    trained.square().mean().backward()
    action_grad = float(actions.grad.abs().sum())
    source_grad = float(source.grad.abs().sum())
    if action_grad <= 0 or source_grad <= 0:
        raise AssertionError(
            f"Control graph is disconnected: action_grad={action_grad}, source_grad={source_grad}"
        )

    with torch.no_grad():
        normal = adapter(actions, source, (5, 4, 5))
        reversed_control = adapter(actions.flip(1), source, (5, 4, 5))
        sensitivity = float((normal - reversed_control).abs().mean())
    if sensitivity <= 0:
        raise AssertionError("Adapter output is invariant to time-reversed actions")

    try:
        adapter(torch.randn(2, 15, 18), source, (5, 4, 5))
    except ValueError:
        alignment_guard = True
    else:
        raise AssertionError("15 actions incorrectly passed the 16-action alignment guard")

    mask = torch.zeros(1, 1, 17, 32, 48)
    mask[:, :, 5:9, 8:16, 12:24] = 1
    latent = grouped_motion_mask(mask, torch.empty(1, 16, 5, 4, 6))
    if latent.shape != (1, 1, 5, 4, 6):
        raise AssertionError(f"Bad grouped mask shape {tuple(latent.shape)}")

    return {
        "exact_zero_init": zero_max == 0.0,
        "output_projection_gradient_l1": output_grad,
        "action_gradient_l1_after_warmup": action_grad,
        "source_gradient_l1_after_warmup": source_grad,
        "reverse_action_sensitivity": sensitivity,
        "temporal_alignment_guard": alignment_guard,
        "latent_mask_shape": list(latent.shape),
    }


def audit_paired_loss() -> dict:
    """Exercise the B=1 -> correct/wrong B=2 pairing without a video model."""

    class Scheduler:
        timesteps = torch.arange(1000, dtype=torch.float32)

        @staticmethod
        def add_noise(clean, noise, timestep):
            return clean + 0.25 * noise

        @staticmethod
        def training_target(clean, noise, timestep):
            return noise - clean

        @staticmethod
        def training_weight(timestep):
            return torch.ones((), device=timestep.device)

    class Pipe:
        scheduler = Scheduler()
        torch_dtype = torch.float32
        device = torch.device("cpu")
        in_iteration_models: list[str] = []

        @staticmethod
        def model_fn(latents, actions, **kwargs):
            # Differentiable stand-in with the same paired batch convention.
            control = actions.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
            return latents + control

    correct = torch.randn(16, 18, requires_grad=True)
    wrong = correct.detach().flip(0).clone()
    # Reversal preserves a global mean, so perturb it to make the synthetic
    # model distinguish the two branches while retaining the required shapes.
    wrong[0, 0] += 1
    loss = paired_spatial_action_flow_loss(
        Pipe(),
        {
            "input_latents": torch.randn(1, 4, 5, 4, 6),
            "actions": correct,
            "wrong_actions": wrong,
            "motion_mask": torch.ones(1, 17, 32, 48),
            "ranking_valid": torch.tensor(1.0),
        },
        {},
        margin=0.002,
        rank_weight=0.5,
        motion_weight=2.0,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f"Paired loss is not a finite scalar: {loss}")
    loss.backward()
    gradient = float(correct.grad.abs().sum())
    if gradient <= 0:
        raise AssertionError("Paired loss did not backpropagate into correct actions")
    return {"finite_scalar": True, "value": float(loss.detach()), "action_gradient_l1": gradient}


def audit_dataset(root: Path) -> dict:
    dataset = WanSO100SpatialActionDataset(
        root=str(root),
        height=64,
        width=80,
        repeat=1,
        holdout_count=6,
        action_mode="hybrid",
        static_probability=0.15,
    )
    holdout = WanSO100SpatialActionDataset(
        root=str(root),
        height=64,
        width=80,
        repeat=1,
        holdout_count=6,
        action_mode="hybrid",
        split="holdout",
        static_probability=0.15,
    )
    manifest_path = REPO / "valset_holdout/manifest.json"
    manifest_groups = {
        record["dataset"] for record in json.loads(manifest_path.read_text())
    }
    held_groups = set(holdout.selected_paths)
    if held_groups != manifest_groups:
        raise AssertionError(
            f"Dataset holdout differs from scoring manifest: {held_groups ^ manifest_groups}"
        )
    if set(dataset.selected_paths) & manifest_groups:
        raise AssertionError("Training groups leak into valset_holdout")
    dynamic_index = next(
        index for index in range(min(len(dataset), 100)) if not dataset._is_static_augmentation(index)
    )
    static_index = next(
        index for index in range(min(len(dataset), 100)) if dataset._is_static_augmentation(index)
    )
    dynamic = dataset[dynamic_index]
    static = dataset[static_index]
    for name, sample in (("dynamic", dynamic), ("static", static)):
        if len(sample["video"]) != 17:
            raise AssertionError(f"{name}: expected 17 frames")
        if sample["actions"].shape != (16, 18):
            raise AssertionError(f"{name}: bad action shape {tuple(sample['actions'].shape)}")
        if sample["motion_mask"].shape != (1, 17, 64, 80):
            raise AssertionError(f"{name}: bad mask shape {tuple(sample['motion_mask'].shape)}")
    static_pixels_equal = all(
        torch.equal(
            torch.from_numpy(np.asarray(static["video"][0]).copy()),
            torch.from_numpy(np.asarray(frame).copy()),
        )
        for frame in static["video"][1:]
    )
    if not static_pixels_equal or float(static["motion_mask"].sum()) != 0.0:
        raise AssertionError("Static augmentation is not an exact first-frame repeat")
    return {
        "groups": len(dataset.selected_paths),
        "holdout_groups": sorted(held_groups),
        "holdout_matches_manifest": True,
        "train_holdout_overlap": 0,
        "base_clips": len(dataset.base),
        "action_shape": list(dynamic["actions"].shape),
        "video_frames": len(dynamic["video"]),
        "motion_mask_shape": list(dynamic["motion_mask"].shape),
        "static_exact_repeat": static_pixels_equal,
        "static_mask_sum": float(static["motion_mask"].sum()),
        "dynamic_ranking_valid": float(dynamic["ranking_valid"]),
    }


def main() -> None:
    args = parse_args()
    model_paths, tokenizer = local_model_paths(Path(args.base_model_path))
    report = {
        "verdict": "PASS_STRUCTURE",
        "adapter": audit_adapter(),
        "paired_loss": audit_paired_loss(),
        "local_model_components": len(json.loads(model_paths)),
        "local_tokenizer": tokenizer,
    }
    if not args.skip_dataset:
        report["dataset"] = audit_dataset(Path(args.dataset_root))
    print(json.dumps(report, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
