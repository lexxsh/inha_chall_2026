"""CPU numerical contract test for the one-run Cosmos loss composition."""

from __future__ import annotations

import json

import torch

from unified_losses import counterfactual_ranking_loss, paired_latent_losses


def main() -> None:
    torch.manual_seed(0)
    batch, channels, frames, height, width = 2, 4, 5, 8, 8
    x0 = torch.randn(batch, channels, frames, height, width)
    noise = torch.randn_like(x0)
    timestep = torch.tensor([0.3, 0.6])
    sigma = timestep.view(batch, 1, 1, 1, 1)
    condition = torch.zeros_like(x0)
    condition[:, :, 0] = 1.0
    xt = x0 * condition + ((1 - sigma) * x0 + sigma * noise) * (1 - condition)
    target_velocity = noise - x0

    perfect = paired_latent_losses(x0, xt, target_velocity, sigma, condition, timestep)
    corrupted_velocity = (target_velocity + 0.2).detach().requires_grad_()
    corrupted = paired_latent_losses(x0, xt, corrupted_velocity, sigma, condition, timestep)
    corrupted_total = corrupted["x0"] + corrupted["temporal"] + corrupted["preserve"]
    corrupted_total.backward()

    same_cf = counterfactual_ranking_loss(
        target_velocity, target_velocity, target_velocity, 1 - condition, margin=0.02
    )
    wrong_cf = counterfactual_ranking_loss(
        target_velocity, target_velocity + 1.0, target_velocity, 1 - condition, margin=0.02
    )
    perfect_values = {key: float(perfect[key]) for key in ("x0", "temporal", "preserve")}
    corrupted_values = {key: float(corrupted[key]) for key in ("x0", "temporal", "preserve")}
    report = {
        "perfect_losses": perfect_values,
        "corrupted_losses": corrupted_values,
        "same_action_counterfactual_loss": float(same_cf),
        "wrong_action_counterfactual_loss": float(wrong_cf),
        "gradient_finite": bool(
            corrupted_velocity.grad is not None and torch.isfinite(corrupted_velocity.grad).all()
        ),
    }
    report["verdict"] = "PASS_UNIFIED_LOSS_AUDIT" if (
        max(perfect_values.values()) < 1e-6
        and min(corrupted_values.values()) > 0
        and abs(float(same_cf) - 0.02) < 1e-6
        and float(wrong_cf) == 0.0
        and report["gradient_finite"]
    ) else "REJECT_UNIFIED_LOSS_AUDIT"
    print(json.dumps(report, indent=2))
    if not report["verdict"].startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
