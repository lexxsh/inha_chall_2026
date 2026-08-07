"""Optimizer construction for SO-100 adapter-only training."""

from __future__ import annotations

from torch import nn

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer


def get_joint_adapter_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    **kwargs,
):
    """Freeze the pretrained DiT and optimize only the joint adapter."""
    model.requires_grad_(False)
    matched = []
    for name, parameter in model.named_parameters():
        if "joint_adapter" in name:
            parameter.requires_grad_(True)
            matched.append((name, parameter))
    if not matched:
        raise RuntimeError("No joint_adapter parameters were found")

    trainable = sum(parameter.numel() for _, parameter in matched)
    total = sum(parameter.numel() for parameter in model.parameters())
    log.critical(
        f"SO-100 adapter-only optimization: {trainable:,}/{total:,} trainable parameters"
    )
    return get_base_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )
