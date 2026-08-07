"""Paired, evaluator-free auxiliary losses for Cosmos SO-100 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def paired_latent_losses(
    x0: torch.Tensor,
    xt: torch.Tensor,
    v_pred: torch.Tensor,
    sigma: torch.Tensor,
    condition_mask: torch.Tensor,
    timestep: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute reconstruction, dynamics, and stable-region preservation losses.

    All supervision comes from the paired ground-truth latent. ``condition_mask``
    has the same shape as ``x0`` and marks the clean source latent frame.
    """
    if x0.shape != xt.shape or x0.shape != v_pred.shape or x0.shape != condition_mask.shape:
        raise ValueError("x0, xt, v_pred, and condition_mask must have identical shapes")
    if x0.ndim != 5 or x0.shape[2] < 2:
        raise ValueError(f"expected video latents [B,C,T,H,W] with T>=2, got {tuple(x0.shape)}")

    batch, channels, frames, height, width = x0.shape
    future = 1.0 - condition_mask
    x0_pred = x0 * condition_mask + (xt.float() - sigma * v_pred.float()) * future
    useful_t = ((0.75 - timestep.float()) / 0.50).clamp(0.0, 1.0).view(batch, 1, 1, 1, 1)

    x0_den = (future * useful_t).sum().clamp_min(1.0)
    x0_loss = (F.smooth_l1_loss(x0_pred, x0, reduction="none") * future * useful_t).sum() / x0_den

    pred_delta = x0_pred[:, :, 1:] - x0_pred[:, :, :-1]
    true_delta = x0[:, :, 1:] - x0[:, :, :-1]
    temporal_weight = useful_t.expand(-1, 1, frames - 1, 1, 1)
    temporal_den = (temporal_weight.sum() * channels * height * width).clamp_min(1.0)
    temporal_loss = (
        F.smooth_l1_loss(pred_delta, true_delta, reduction="none") * temporal_weight
    ).sum() / temporal_den

    gt_motion = (x0[:, :, 1:] - x0[:, :, :1]).abs().mean(dim=1, keepdim=True)
    motion_scale = gt_motion.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-5)
    stable = torch.exp(-gt_motion / motion_scale).detach()
    stable_weight = stable * useful_t.expand(-1, 1, frames - 1, 1, 1)
    stable_den = (stable_weight.sum() * channels).clamp_min(1.0)
    preserve_loss = ((x0_pred[:, :, 1:] - x0[:, :, 1:]).abs() * stable_weight).sum() / stable_den

    return {
        "x0_prediction": x0_pred,
        "future_mask": future,
        "x0": x0_loss,
        "temporal": temporal_loss,
        "preserve": preserve_loss,
    }


def counterfactual_ranking_loss(
    correct_velocity: torch.Tensor,
    wrong_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    future_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require the paired action to explain its future better than a wrong action."""
    count = future_mask.flatten(1).sum(1).clamp_min(1.0)
    correct = (((correct_velocity - target_velocity) ** 2) * future_mask).flatten(1).sum(1) / count
    wrong = (((wrong_velocity - target_velocity) ** 2) * future_mask).flatten(1).sum(1) / count
    return torch.relu(float(margin) + correct - wrong).mean()
