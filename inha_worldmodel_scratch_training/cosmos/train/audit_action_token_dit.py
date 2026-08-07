"""CPU contract test for the Cosmos spatial action-token extension."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "port"))
import bootstrap  # noqa: E402,F401

import torch  # noqa: E402

from action_token_dit import (  # noqa: E402
    ActionCtxTokenizer,
    _action_token_cross_attention_forward,
    build_action_token_dit,
)
# Import the concrete class before entering the meta-device context.  Importing
# torch._dynamo while the global default device is meta can trigger a circular
# import in some PyTorch builds.
from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (  # noqa: E402,E501
    ActionChunkConditionedMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Attention  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    tokenizer = ActionCtxTokenizer()
    action = torch.randn(2, 16, 6)
    tokens = tokenizer(action)

    # A small attention layer exercises the same separate-softmax and gating code
    # without allocating the 2B model.
    tiny_tokenizer = ActionCtxTokenizer(context_dim=16)
    tiny_tokens = tiny_tokenizer(action)
    attention = Attention(32, 16, n_heads=4, head_dim=8, backend="torch")
    query = torch.randn(2, 7, 32)
    text = torch.randn(2, 9, 16)
    reference = attention(query, text)
    attention.register_parameter("action_ctx_gate", torch.nn.Parameter(torch.zeros(32)))
    attention.forward = MethodType(_action_token_cross_attention_forward, attention)
    attention._action_ctx_tokens = tiny_tokens

    attention._action_ctx_scale = 0.0
    disabled = attention(query, text)
    attention._action_ctx_scale = 1.0
    zero_gate = attention(query, text)
    with torch.no_grad():
        attention.action_ctx_gate.fill_(0.1)
    enabled = attention(query, text)
    enabled.square().mean().backward()

    # Build the real 2B graph on meta: this verifies the patch against all 28
    # production blocks without allocating model weights or touching a GPU.
    with torch.device("meta"):
        full_model = build_action_token_dit(action_dim=6)
    full_new_parameters = sum(
        p.numel() for name, p in full_model.named_parameters() if "action_ctx" in name
    )

    tokenizer_parameters = sum(p.numel() for p in tokenizer.parameters())
    gate_parameters = 28 * 2048
    report = {
        "token_shape": list(tokens.shape),
        "tokens_finite": bool(torch.isfinite(tokens).all()),
        "tokenizer_parameters": tokenizer_parameters,
        "gate_parameters": gate_parameters,
        "new_parameters": tokenizer_parameters + gate_parameters,
        "full_model_blocks": len(full_model.blocks),
        "full_model_parameters": sum(p.numel() for p in full_model.parameters()),
        "full_model_new_parameters": full_new_parameters,
        "disabled_max_abs_diff": float((reference - disabled).abs().max()),
        "zero_gate_max_abs_diff": float((reference - zero_gate).abs().max()),
        "enabled_max_abs_diff": float((reference - enabled).abs().max()),
        "gate_receives_gradient": attention.action_ctx_gate.grad is not None,
        "tokenizer_receives_gradient": tiny_tokenizer.fc1.weight.grad is not None,
    }
    report["structural_zero_gate"] = (
        report["disabled_max_abs_diff"] == 0.0 and report["zero_gate_max_abs_diff"] == 0.0
    )
    report["verdict"] = "PASS_ATOK_CPU_AUDIT" if all(
        [
            report["tokens_finite"],
            report["new_parameters"] == 1_144_832,
            report["full_model_blocks"] == 28,
            report["full_model_new_parameters"] == 1_144_832,
            report["structural_zero_gate"],
            report["enabled_max_abs_diff"] > 0.0,
            report["gate_receives_gradient"],
            report["tokenizer_receives_gradient"],
        ]
    ) else "REJECT_ATOK_CPU_AUDIT"
    print(json.dumps(report, indent=2))
    if not report["verdict"].startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
