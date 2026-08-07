"""Small SO-100 adapter module, kept independent of the Cosmos CUDA stack."""

from __future__ import annotations

import torch
from torch import nn


class SO100JointAdapter(nn.Module):
    """Map normalized SO-100 joint commands into Cosmos' pretrained EE space."""

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 7,
        hidden_dim: int = 64,
        output_scale: float = 0.5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # V1 used biases and consequently learned a nearly constant domain
        # token. V2 disables both biases, making f(0)=0 an exact invariant.
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.output_scale = float(output_scale)

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # Bound both the normalized input and learned EE proxy so an early
        # update cannot destroy the pretrained video prior.
        hidden = self.act(self.fc1(torch.tanh(action)))
        return self.output_scale * torch.tanh(self.fc2(hidden))
