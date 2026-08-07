"""Small CPU invariance/gradient test for the Frame-Ada ResBlock path."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/baseline/challenge_kit"
sys.path.insert(0, str(KIT / "libs/dynamicrafter"))
sys.path.insert(0, str(KIT))

from lvdm.modules.networks.openaimodel3d import ResBlock, TimestepEmbedSequential  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    block = ResBlock(
        channels=32,
        emb_channels=64,
        dropout=0.0,
        action_emb_channels=16,
        use_checkpoint=True,
    )
    # ResBlock is identity at library initialisation. A loaded pretrained block
    # has a learned output conv, so make this tiny test emulate that state.
    torch.nn.init.normal_(block.out_layers[-1].weight, std=0.01)
    torch.nn.init.zeros_(block.out_layers[-1].bias)
    seq = TimestepEmbedSequential(block)

    x = torch.randn(4, 32, 8, 8)
    emb = torch.randn(4, 64)
    act_a = torch.randn(4, 16)
    act_b = torch.randn(4, 16)

    with torch.no_grad():
        base = seq(x, emb, batch_size=1)
        zero_a = seq(x, emb, batch_size=1, action_emb=act_a)
        zero_b = seq(x, emb, batch_size=1, action_emb=act_b)
    if not torch.equal(base, zero_a) or not torch.equal(base, zero_b):
        raise SystemExit("FAIL: zero-init Frame-Ada changed the pretrained path")

    block.train()
    out = seq(x, emb, batch_size=1, action_emb=act_a)
    out.square().mean().backward()
    projection = block.action_modulation[-1]
    grad = float(projection.weight.grad.abs().sum())
    if not grad > 0:
        raise SystemExit("FAIL: action modulation did not receive a gradient")

    with torch.no_grad():
        projection.weight.add_(0.01 * torch.randn_like(projection.weight))
        changed_a = seq(x, emb, batch_size=1, action_emb=act_a)
        changed_b = seq(x, emb, batch_size=1, action_emb=act_b)
    action_delta = float((changed_a - changed_b).abs().mean())
    if not action_delta > 0:
        raise SystemExit("FAIL: learned Frame-Ada is insensitive to actions")

    print("PASS: zero-init preserves output exactly")
    print(f"PASS: modulation gradient L1={grad:.6f}")
    print(f"PASS: different-action output delta={action_delta:.6f}")


if __name__ == "__main__":
    main()
