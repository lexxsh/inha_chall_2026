"""Champion-preserving spatial action-token extension for Cosmos Predict2.5.

The public action-conditioned DiT injects four-frame action chunks only through
the timestep/AdaLN path.  This module keeps that trained path intact and adds a
second, separately-normalized cross-attention over sixteen frame-level action
tokens.  Every new residual is controlled by a zero-initialized channel gate,
so ``action_ctx_scale=0`` is an exact escape hatch to the original model.

Only the tokenizer and the 28 gates introduce parameters.  The existing
cross-attention k/v/output projections (including their trained LoRA weights)
are shared rather than duplicated.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from einops import rearrange

from load_dit import build_dit


_STEP_DELTA_SCALE = (
    0.0647,
    0.0707,
    0.0688,
    0.0793,
    0.0306,
    0.1046,
)
_START_DELTA_SCALE = (
    0.3848,
    0.4384,
    0.4080,
    0.3947,
    0.1765,
    0.5311,
)


class ActionCtxTokenizer(nn.Module):
    """Turn 16 normalized SO-100 poses into ordered 1024-D context tokens.

    Each token contains absolute pose, one-step velocity, and displacement from
    the initial pose.  The two differences are divided by train-only statistics
    so that low-motion joints are not numerically hidden by absolute pose.
    """

    def __init__(self, action_dim: int = 6, context_dim: int = 1024, max_frames: int = 16):
        super().__init__()
        if action_dim != 6:
            raise ValueError(f"spatial action tokens currently require SO-100 6D poses, got {action_dim}")
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.max_frames = max_frames
        self.fc1 = nn.Linear(action_dim * 3, context_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(context_dim, context_dim)
        self.norm = nn.LayerNorm(context_dim)
        self.action_ctx_pos = nn.Parameter(torch.zeros(1, max_frames, context_dim))
        self.register_buffer(
            "step_delta_scale",
            torch.tensor(_STEP_DELTA_SCALE, dtype=torch.float32).view(1, 1, action_dim),
            persistent=True,
        )
        self.register_buffer(
            "start_delta_scale",
            torch.tensor(_START_DELTA_SCALE, dtype=torch.float32).view(1, 1, action_dim),
            persistent=True,
        )
        nn.init.trunc_normal_(self.action_ctx_pos, std=0.02)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(f"expected action (B,T,{self.action_dim}), got {tuple(action.shape)}")
        if action.shape[1] > self.max_frames:
            raise ValueError(f"action has {action.shape[1]} frames; maximum is {self.max_frames}")

        action = action.float()
        d_step = torch.zeros_like(action)
        d_step[:, 1:] = action[:, 1:] - action[:, :-1]
        d_start = action - action[:, :1]
        features = torch.cat(
            [
                action,
                (d_step / self.step_delta_scale).clamp_(-10.0, 10.0),
                (d_start / self.start_delta_scale).clamp_(-10.0, 10.0),
            ],
            dim=-1,
        )
        tokens = self.fc2(self.activation(self.fc1(features)))
        tokens = self.norm(tokens)
        return tokens + self.action_ctx_pos[:, : action.shape[1]]


def _action_token_cross_attention_forward(
    self,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    video_size: Any = None,
    kv_cache_cfg: Any = None,
) -> torch.Tensor:
    """Original text cross-attention plus gated action-only softmax."""
    q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
    base = self.attn_op(q, k, v)

    action_tokens = getattr(self, "_action_ctx_tokens", None)
    action_scale = float(getattr(self, "_action_ctx_scale", 1.0))
    if action_tokens is not None and action_scale != 0.0:
        k_action = self.k_proj(action_tokens)
        v_action = self.v_proj(action_tokens)
        k_action, v_action = map(
            lambda tensor: rearrange(
                tensor,
                "b s (h d) -> b s h d",
                h=self.n_heads,
                d=self.head_dim,
            ),
            (k_action, v_action),
        )
        k_action = self.k_norm(k_action)
        v_action = self.v_norm(v_action)
        action_result = self.attn_op(q, k_action, v_action)
        gate = self.action_ctx_gate.to(dtype=action_result.dtype)
        base = base + action_result * gate * action_scale

    return self.output_dropout(self.output_proj(base))


def _forward_with_action_context(self, *args, **kwargs):
    action = kwargs.get("action")
    if action is None:
        # The upstream implementation will raise the canonical error.
        return self._action_ctx_original_forward(*args, **kwargs)

    enabled = bool(getattr(self, "action_ctx_enabled", True))
    scale = float(getattr(self, "action_ctx_scale", 1.0))
    tokens = None
    if enabled and scale != 0.0:
        if action.shape[-1] != 6:
            raise ValueError(
                "ATOK is intentionally incompatible with DELTA=1: the existing AdaLN path must receive 6D actions"
            )
        model_input = kwargs.get("x_B_C_T_H_W", args[0] if args else None)
        if model_input is None:
            raise ValueError("x_B_C_T_H_W is required")
        tokens = self.action_ctx_tokenizer(action).to(dtype=model_input.dtype)

    for block in self.blocks:
        block.cross_attn._action_ctx_tokens = tokens
        block.cross_attn._action_ctx_scale = scale
    try:
        return self._action_ctx_original_forward(*args, **kwargs)
    finally:
        # Do not retain a graph through module attributes after the step.
        for block in self.blocks:
            block.cross_attn._action_ctx_tokens = None


def build_action_token_dit(action_dim: int = 6):
    """Build the official action DiT and add the champion-preserving token path."""
    # The stock 2B graph wraps every block in selective activation
    # checkpointing and assumes the original fixed sequence of matrix
    # multiplications.  This extension adds action k/v projections, so that
    # cache policy is invalid and can replay a 16-token tensor where a
    # 3200-token video tensor is expected during backward.  H100 training uses
    # per-GPU batch 1 and comfortably runs the unwrapped 2B graph.
    net = build_dit(action_dim=action_dim, sac_mode="none")
    context_dim = int(net.blocks[0].cross_attn.context_dim)
    net.action_ctx_tokenizer = ActionCtxTokenizer(action_dim=action_dim, context_dim=context_dim)
    net.action_ctx_enabled = True
    net.action_ctx_scale = 1.0

    for block in net.blocks:
        cross_attn = block.cross_attn
        if not hasattr(cross_attn, "action_ctx_gate"):
            cross_attn.register_parameter(
                "action_ctx_gate",
                nn.Parameter(torch.zeros(cross_attn._inner_dim, dtype=torch.float32)),
            )
        # Reuse the initialized module and its state_dict names; only forward changes.
        cross_attn.forward = MethodType(_action_token_cross_attention_forward, cross_attn)
        cross_attn._action_ctx_tokens = None
        cross_attn._action_ctx_scale = 1.0

    net._action_ctx_original_forward = net.forward
    net.forward = MethodType(_forward_with_action_context, net)
    return net


def action_ctx_gate_stats(net: nn.Module) -> dict[str, float]:
    """Return cheap diagnostics without depending on a particular PEFT wrapper."""
    gates = [p.detach().float() for n, p in net.named_parameters() if n.endswith("action_ctx_gate")]
    if not gates:
        return {"count": 0, "mean_abs": 0.0, "max_abs": 0.0}
    flat = torch.cat([g.reshape(-1).cpu() for g in gates])
    return {
        "count": float(len(gates)),
        "mean_abs": float(flat.abs().mean()),
        "max_abs": float(flat.abs().max()),
    }
