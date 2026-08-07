"""SO-100 action adapter for the pretrained Cosmos action-conditioned DiT."""

from __future__ import annotations

import torch

from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
    ActionChunkConditionedMinimalV1LVGDiT,
)
from cosmos_so100.adapter import SO100JointAdapter


class SO100AdapterActionChunkDiT(ActionChunkConditionedMinimalV1LVGDiT):
    """Cosmos action DiT with a zero-initialized joint-to-EE front end."""

    def __init__(
        self,
        *args,
        joint_action_dim: int = 6,
        joint_adapter_hidden_dim: int = 64,
        joint_adapter_output_scale: float = 0.5,
        joint_adapter_bias: bool = True,
        **kwargs,
    ) -> None:
        # Keep action_dim=7 so every pretrained Cosmos parameter retains its
        # original shape. Only the new front end consumes SO-100's six values.
        kwargs["action_dim"] = 7
        super().__init__(*args, **kwargs)
        self.joint_adapter = SO100JointAdapter(
            input_dim=joint_action_dim,
            output_dim=7,
            hidden_dim=joint_adapter_hidden_dim,
            output_scale=joint_adapter_output_scale,
            bias=joint_adapter_bias,
        )

    def init_weights(self) -> None:
        # The base constructor invokes this before joint_adapter exists; the
        # explicit build_net initialization invokes it again after materializing
        # meta tensors, at which point the adapter is initialized safely.
        super().init_weights()
        if hasattr(self, "joint_adapter"):
            self.joint_adapter.reset_parameters()

    def forward(self, *args, action: torch.Tensor | None = None, **kwargs):
        if action is None:
            raise ValueError("SO100AdapterActionChunkDiT requires action")
        action = self.joint_adapter(action)
        return super().forward(*args, action=action, **kwargs)
