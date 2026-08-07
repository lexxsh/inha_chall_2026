# action-cond 2B DiT를 Windows에서 로드하고 더미 forward로 검증한다.
from __future__ import annotations

import bootstrap  # noqa: F401  (sys.path + Linux 전용 의존성 스텁)

from pathlib import Path
import os

import torch

CKPT_DIR = Path(os.environ.get(
    "COSMOS_CHECKPOINT_DIR",
    str(Path(__file__).resolve().parent.parent / "checkpoints"),
))
DIT_CKPT = CKPT_DIR / "robot" / "action-cond" / "38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
TEXT_EMB = CKPT_DIR / "robot" / "action-cond" / "cr1_empty_string_text_embeddings.pt"


def build_dit(action_dim: int = 7, sac_mode: str | None = None):
    from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
        ActionChunkConditionedMinimalV1LVGDiT,
    )
    from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

    # exp_2B_action_conditioned_rectify_flow.py 의
    # cosmos_v1_2B_action_chunk_conditioned + BRIDGE_13FRAME_256X320 오버라이드와 동일 구성
    return ActionChunkConditionedMinimalV1LVGDiT(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=28,
        num_heads=16,
        concat_padding_mask=True,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        atten_backend="minimal_a2a",
        extra_per_block_abs_pos_emb=False,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        # The stock action model uses selective activation checkpointing.  A
        # caller that changes a block's operation graph (our spatial action
        # cross-attention does) must opt out, otherwise SAC's saved-op sequence
        # no longer matches backward recomputation.
        sac_config=SACConfig(mode=sac_mode) if sac_mode is not None else SACConfig(),
        use_crossattn_projection=True,
        crossattn_proj_in_channels=100352,
        crossattn_emb_channels=1024,
        action_dim=action_dim,
        temporal_compression_ratio=4,
    )


def load_pretrained(net: torch.nn.Module) -> None:
    sd = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k.removeprefix("net."): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("missing keys (up to 10):", missing[:10])
    if unexpected:
        print("unexpected keys (up to 10):", unexpected[:10])


def main() -> None:
    from cosmos_predict2._src.predict2.conditioner import DataType

    device = torch.device("cuda")
    net = build_dit()
    load_pretrained(net)
    net = net.to(device, dtype=torch.bfloat16).eval()
    print("param count: %.2fB" % (sum(p.numel() for p in net.parameters()) / 1e9))

    text_emb = torch.load(TEXT_EMB, map_location="cpu", weights_only=False).to(device, torch.bfloat16)

    B, T_lat, H_lat, W_lat = 1, 4, 32, 40  # 256x320, 13프레임 학습 구성
    x = torch.randn(B, 16, T_lat, H_lat, W_lat, device=device, dtype=torch.bfloat16)
    cond_mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, device=device, dtype=torch.bfloat16)
    cond_mask[:, :, 0] = 1.0  # 첫 잠재 프레임 = 조건(초기 이미지)
    timesteps = torch.full((B, T_lat), 0.5, device=device, dtype=torch.bfloat16)
    action = torch.randn(B, 12, 7, device=device, dtype=torch.bfloat16)
    padding_mask = torch.zeros(B, 1, 256, 320, device=device, dtype=torch.bfloat16)
    fps = torch.full((B,), 4, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = net(
            x_B_C_T_H_W=x,
            timesteps_B_T=timesteps,
            crossattn_emb=text_emb.expand(B, -1, -1),
            condition_video_input_mask_B_C_T_H_W=cond_mask,
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            action=action,
        )
    print("forward OK, output:", tuple(out.shape), out.dtype)
    print("output stats: mean %.4f std %.4f" % (out.float().mean().item(), out.float().std().item()))


if __name__ == "__main__":
    main()
