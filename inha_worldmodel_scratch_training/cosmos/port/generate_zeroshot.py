# 사전학습 action-cond 2B로 제로샷 17프레임 생성 (Bridge 7D 임베더 그대로, 영액션=정지 기대).
# 포팅 검증 목적: 샘플링 수식(공식 RF 모델의 denoise/CFG/UniPC)과 동일 로직을 단일 파일로 재현.
from __future__ import annotations

import bootstrap  # noqa: F401

import time
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from load_dit import build_dit, load_pretrained, CKPT_DIR, TEXT_EMB

EVAL_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eval"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

NUM_FRAMES = 17  # 초기 1 + 생성 16 (4k+1)
HEIGHT, WIDTH = 320, 512
NUM_STEPS = 35
GUIDANCE = 0.0  # 제로샷 검증은 cond 단일 경로


def letterbox(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """제출킷 preprocess_video(pad=True)와 동일한 scale+pad."""
    tensor = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255.0
    _, _, h, w = tensor.shape
    scale = min(target_h / h, target_w / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    tensor = F.interpolate(tensor, size=(rh, rw), mode="bilinear", align_corners=False)
    pt = (target_h - rh) // 2
    pb = target_h - rh - pt
    pl = (target_w - rw) // 2
    pr = target_w - rw - pl
    tensor = F.pad(tensor, (pl, pr, pt, pb), value=0.0)
    return tensor[0]  # (3,H,W), 0..1


@torch.no_grad()
def generate(sample_id: str, actions_7d: torch.Tensor, dit, vae, text_emb, device) -> torch.Tensor:
    from cosmos_predict2._src.predict2.conditioner import DataType
    from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler

    image = np.asarray(Image.open(EVAL_ROOT / "images" / f"{sample_id}.png").convert("RGB"))
    frame0 = letterbox(image, HEIGHT, WIDTH)  # (3,H,W) 0..1

    # GT 영상: 프레임0=초기 이미지, 나머지 0 (공식 추론 래퍼와 동일)
    video = torch.zeros(1, 3, NUM_FRAMES, HEIGHT, WIDTH)
    video[0, :, 0] = frame0
    video = (video * 2.0 - 1.0).to(device, torch.bfloat16)

    gt_latent = vae.encode(video).float()  # (1,16,5,40,64)
    B, C, T_lat, H_lat, W_lat = gt_latent.shape

    cond_mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, device=device)
    cond_mask[:, :, 0] = 1.0
    cond_mask_c = cond_mask.repeat(1, C, 1, 1, 1)

    padding_mask = torch.zeros(B, 1, HEIGHT, WIDTH, device=device, dtype=torch.bfloat16)
    fps = torch.full((B,), 4, device=device, dtype=torch.bfloat16)
    action = actions_7d[None].to(device, torch.bfloat16)  # (1,16,7)

    def denoise(noise, xt, t_scalar):
        xt = gt_latent * cond_mask_c + xt * (1 - cond_mask_c)
        timesteps = torch.full((B, 1), float(t_scalar), device=device, dtype=torch.bfloat16)
        v = dit(
            x_B_C_T_H_W=xt.to(torch.bfloat16),
            timesteps_B_T=timesteps,
            crossattn_emb=text_emb.expand(B, -1, -1),
            condition_video_input_mask_B_C_T_H_W=cond_mask.to(torch.bfloat16),
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            action=action,
        ).float()
        # denoise_replace_gt_frames: 조건 프레임 velocity를 (noise - gt_x0)로 치환
        return (noise - gt_latent) * cond_mask_c + v * (1 - cond_mask_c)

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(NUM_STEPS, device=device, shift=5.0)

    torch.manual_seed(1)
    noise = torch.randn(B, C, T_lat, H_lat, W_lat, device=device)
    latents = noise
    for t in scheduler.timesteps:
        v = denoise(noise, latents, t)
        latents = scheduler.step(v.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)

    frames = vae.decode(latents.to(torch.bfloat16))  # (1,3,17,H,W) in [-1,1]
    return frames


def main() -> None:
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE

    device = torch.device("cuda")
    dit = build_dit(action_dim=7)
    dit.timestep_scale = 0.001  # rectified flow 실험 설정과 동일 (필수)
    load_pretrained(dit)
    dit = dit.to(device, torch.bfloat16).eval()
    vae = WanVAE(z_dim=16, vae_pth=str(CKPT_DIR / "tokenizer.pth"), dtype=torch.bfloat16, device="cuda")
    text_emb = torch.load(TEXT_EMB, map_location="cpu", weights_only=False).to(device, torch.bfloat16)

    actions = torch.zeros(16, 7)  # Bridge delta-EEF 기준 영액션 = 정지
    t0 = time.time()
    frames = generate("sample_000000", actions, dit, vae, text_emb, device)
    print("generation took %.1fs" % (time.time() - t0))

    OUT_DIR.mkdir(exist_ok=True)
    out = ((frames[0].float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    with imageio.get_writer(OUT_DIR / "zeroshot_sample_000000.mp4", fps=6, codec="libx264", macro_block_size=1) as w:
        for f in out:
            w.append_data(f)
    print("saved:", OUT_DIR / "zeroshot_sample_000000.mp4")
    # 정지 정도 측정: 프레임0 대비 마지막 프레임 평균 절대 차이
    diff = np.abs(out[-1].astype(np.float32) - out[0].astype(np.float32)).mean()
    print("frame0 vs frame16 mean abs diff: %.2f (0에 가까울수록 정지)" % diff)


if __name__ == "__main__":
    main()
