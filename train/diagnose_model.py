"""메소드 문제인가 학습 부족인가를 가른다.

두 가지를 잰다.

A. step별 후반부 열화 추세
   같은 샘플을 여러 checkpoint로 생성해, 프레임 t가 진행될수록 나빠지는 정도가
   step 증가에 따라 줄어드는지 본다. 단조 감소면 "학습 부족", 정체면 "메소드 의심".
   척도: 각 프레임의 DINO feature가 첫 프레임 대비 얼마나 멀어지는지(자기 참조라 GT 불필요).
   붕괴 프레임은 DINO 공간에서 급격히 튄다.

B. 액션 순열 민감도 (action collapse 검사)
   같은 첫 이미지 + 같은 seed로 액션만 (정상 / 시간역순 / 셔플)로 바꿔 생성한다.
   세 영상이 서로 다르면 모델이 액션을 쓴다. 거의 같으면 collapse —
   그러면 오래 학습해도 Action 지표가 안 오르므로 처방이 완전히 다르다.

GT 없는 자기참조 진단이라 정답 점수는 계산하지 않는다. 그래도 eval 입력의 출력 양상을 보고
checkpoint/방법을 고르면 eval-informed selection이 되므로 기본 입력은 train-only group holdout을 쓴다.
이 진단은 민감도 screen일 뿐 최종 선택은 GT가 있는 holdout의 공식 세 component로 한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/baseline/challenge_kit"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_module import load_delta_scale, transform_actions  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from lvdm.models.samplers.ddim import DDIMSampler  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from scripts.eval.feature_csv_utils import (  # noqa: E402
    build_inference_batch,
    list_challenge_sample_ids,
    load_action_stats,
    load_dino_model,
    resolve_dino_image_size,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(config, checkpoint, device):
    model_cfg = config.model.copy()
    pc = model_cfg.get("pretrained_checkpoint")
    if pc and not Path(pc).is_absolute():
        model_cfg.pretrained_checkpoint = str((KIT / pc).resolve())
    model = get_model(model_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    model.use_ema = False  # 짧은 학습이라 본체 가중치를 쓴다
    model.to(device).eval()
    return model


@torch.no_grad()
def generate(model, batch, device, ddim_steps, guidance, rescale, eta, seed):
    torch.manual_seed(seed)
    sampler = DDIMSampler(model)
    z, c, uc, cond_mask, _logs, kwargs = model.prepare_batch_for_inference(batch)
    shape = (model.channels, model.temporal_length, *model.image_size)
    sample_kwargs = dict(
        unconditional_guidance_scale=guidance, timestep_spacing="uniform_trailing",
        guidance_rescale=rescale, ddim_eta=eta, verbose=False, **kwargs,
    )
    amp = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
    with model.ema_scope("gen"), amp:
        samples, _ = sampler.sample(
            ddim_steps, batch_size=z.shape[0], shape=shape,
            conditioning=c, unconditional_conditioning=uc, mask=cond_mask, x0=z, **sample_kwargs,
        )
        videos = model.decode_first_stage(samples)  # (B, C, T, H, W) in [-1,1]
    return videos


class Dino:
    def __init__(self, device):
        self.device = device
        self.model = load_dino_model(device, "vit_small_patch14_dinov2.lvd142m", pretrained=True)
        self.size = resolve_dino_image_size(self.model, requested_size=0)

    @torch.no_grad()
    def feats(self, video):  # video: (C,T,H,W) in [-1,1] -> (T, D) normalized
        x = ((video.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1)  # (T,C,H,W)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = ((x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device))
        f = self.model(x.to(self.device))
        if f.ndim == 3:
            f = f[:, 0]
        return F.normalize(f.float(), dim=-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", nargs="+",
        default=[str(REPO / "train/configs/inha_full_unet.yaml")],
        help="왼쪽부터 merge할 config 목록",
    )
    ap.add_argument("--checkpoints", nargs="+", required=True, help="비교할 체크포인트들 (step 순)")
    ap.add_argument("--challenge-root", default=str(REPO / "valset_holdout"))
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--guidance-rescale", type=float, default=0.7)
    ap.add_argument("--ddim-eta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.merge(*[OmegaConf.load(path) for path in args.config])
    action_mode = config.data.params.get("action_mode", "delta")
    action_shift = config.data.params.get("action_shift", 0)
    th, tw = config.data.params.get("target_height", 320), config.data.params.get("target_width", 512)
    pad = config.data.params.get("pad", True)

    amean, astd = load_action_stats(str(REPO / "open/data/train/so100_action_statistics.json"))
    dscale = load_delta_scale(str(REPO / "train/delta_action_stats.json"), action_mode)
    sample_ids = list_challenge_sample_ids(Path(args.challenge_root))[: args.num_samples]
    dino = Dino(device)

    def make_batch(ids):
        b = build_inference_batch(Path(args.challenge_root), ids, th, tw, pad, 6, amean, astd, device)
        return b

    # ---- A. step별 후반부 열화 ----
    print("\n=== A. step별 후반부 열화 (프레임 진행에 따른 DINO 이탈, 낮을수록 안정) ===")
    print(f"{'checkpoint':<45} {'초반평균':>8} {'후반평균':>8} {'후반-초반':>9}")
    step_curve = {}
    for ckpt in args.checkpoints:
        model = build_model(config, ckpt, device)
        b = make_batch(sample_ids)
        b["act"] = transform_actions(b["act"], dscale, action_mode, action_shift)
        vids = generate(model, b, device, args.ddim_steps, args.guidance_scale,
                        args.guidance_rescale, args.ddim_eta, args.seed)
        early, late = [], []
        for v in vids:
            f = dino.feats(v)  # (T,D)
            drift = 1 - (f @ f[0:1].T).squeeze(1)  # 첫 프레임 대비 거리, (T,)
            early.append(float(drift[1:6].mean()))   # t=1..5
            late.append(float(drift[11:16].mean()))  # t=11..15
        e, l = float(np.mean(early)), float(np.mean(late))
        step_curve[Path(ckpt).stem] = (e, l)
        print(f"{Path(ckpt).stem:<45} {e:>8.4f} {l:>8.4f} {l - e:>9.4f}")
        del model
        torch.cuda.empty_cache()

    keys = list(step_curve)
    if len(keys) >= 2:
        first_late = step_curve[keys[0]][1]
        last_late = step_curve[keys[-1]][1]
        print(f"\n후반 열화 {keys[0]} -> {keys[-1]}: {first_late:.4f} -> {last_late:.4f}", end="  ")
        if last_late < first_late - 0.005:
            print("→ 자기참조 drift 감소. 안정성 개선 후보이나 올바른 동작인지는 GT 점수로 확인해야 함")
        else:
            print("→ 자기참조 drift 정체/증가. 추가 step의 안정성 이득은 보이지 않음")

    # ---- B. 액션 순열 민감도 ----
    print("\n=== B. 액션 순열 민감도 (마지막 체크포인트, 높을수록 액션을 실제로 씀) ===")
    model = build_model(config, args.checkpoints[-1], device)
    base = make_batch(sample_ids)
    raw = base["act"].clone()  # (B,T,A) 정규화 절대 액션

    variants = {}
    variants["normal"] = transform_actions(raw, dscale, action_mode, action_shift)
    variants["reversed"] = transform_actions(raw.flip(dims=[1]), dscale, action_mode, action_shift)
    perm = torch.roll(raw, shifts=1, dims=0)  # 옆 샘플의 액션을 가져온다
    variants["shuffled"] = transform_actions(perm, dscale, action_mode, action_shift)

    outs = {}
    for name, act in variants.items():
        b = make_batch(sample_ids)
        b["act"] = act
        outs[name] = generate(model, b, device, args.ddim_steps, args.guidance_scale,
                              args.guidance_rescale, args.ddim_eta, args.seed)

    def video_dist(a, b):  # 두 영상의 DINO 프레임 거리 평균
        fa = torch.stack([dino.feats(v) for v in a])  # (N,T,D)
        fb = torch.stack([dino.feats(v) for v in b])
        return float((1 - (fa * fb).sum(-1)).mean())

    d_rev = video_dist(outs["normal"], outs["reversed"])
    d_shuf = video_dist(outs["normal"], outs["shuffled"])
    # 참조: 서로 다른 샘플끼리 비교(=외형까지 다를 때의 거리 상한)
    ref = video_dist(outs["normal"], torch.roll(outs["normal"], 1, dims=0))
    print(f"정상 vs 시간역순 액션 : {d_rev:.4f}")
    print(f"정상 vs 셔플 액션     : {d_shuf:.4f}")
    print(f"(참조) 다른 샘플끼리   : {ref:.4f}  <- 외형까지 다를 때의 상한")
    print()
    if max(d_rev, d_shuf) < 0.01:
        print("판정: ACTION COLLAPSE 의심 — 액션을 바꿔도 영상이 거의 안 변한다.")
        print("      step을 늘려도 Action 지표가 안 오를 수 있다. 조건 주입/표현을 재검토할 것.")
    elif max(d_rev, d_shuf) < 0.05:
        print("판정: 액션 반응 약함 — 조건이 걸리긴 하나 미약하다. 학습 부족일 수도, 주입 약함일 수도.")
    else:
        print("판정: 영상이 action conditioner에 강하게 반응한다.")
        print("      이는 action correctness나 학습량 부족의 증거가 아니다. 정상 action이 zero/shuffle보다")
        print("      GT Action/DINO/Video 점수에서 실제로 나은지 고정 holdout으로 확인할 것.")


if __name__ == "__main__":
    main()
