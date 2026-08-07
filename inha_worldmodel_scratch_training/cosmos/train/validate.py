# 5.5k 체크포인트 검증: val 클립에서 생성 vs 정지 vs 정답 영상의 Action MAE 비교 (대회 추출기 사용)
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "port"))
import bootstrap  # noqa: F401

import numpy as np
import torch

from load_dit import build_dit, DIT_CKPT, TEXT_EMB, CKPT_DIR
from so100_dataset import SO100ClipDataset

import os

# 이 파일은 재현 패키지의 추론 경로다. 제출킷을 참조하는 코드는 일절 두지 않는다
# (킷 채점 코드였던 main() 은 runs/rules_quarantine/validate_main_kit_eval.py 로 분리했다).
RUN_CKPT = Path(os.environ.get("RUN_CKPT", Path(__file__).resolve().parent / "runs" / "v1_lora" / "latest.pt"))
N_SAMPLES = 8
NUM_STEPS = 35

# DELTA=1 이면 액션 조건이 (16,12) = [절대 z, (z_t-z_0)/스케일] 이다.
# **학습(train_lora.to_delta12)과 한 글자도 다르면 안 된다** — 조건 표현이 어긋나면
# 로드는 성공하고 생성만 조용히 망가진다.
DELTA = os.environ.get("DELTA") == "1"
_DELTA_SCALE = None
if DELTA:
    import json as _json

    _p = Path(__file__).resolve().parent / "delta_action_stats.json"
    if not _p.exists():
        raise SystemExit("delta_action_stats.json 이 없다 — compute_delta_stats.py 를 먼저 돌려라")
    _DELTA_SCALE = torch.tensor(_json.loads(_p.read_text(encoding="utf-8"))["scale"],
                                dtype=torch.float32).view(1, 1, 6)


def action_cond(act: torch.Tensor) -> torch.Tensor:
    """(B,16,6) 절대 z -> DELTA 면 (B,16,12). train_lora.to_delta12 와 동일 수식."""
    if not DELTA:
        return act
    d = (act - act[:, :1]) / _DELTA_SCALE.to(act.dtype).to(act.device)
    return torch.cat([act, d], dim=-1)


def load_finetuned(device):
    if os.environ.get("ATOK") == "1":
        from action_token_dit import build_action_token_dit

        net = build_action_token_dit(action_dim=6)
        print("eval: ATOK model (액션 크로스어텐션 토큰)", flush=True)
    elif os.environ.get("PERBLOCK") == "1":
        from perblock_dit import build_perblock_dit

        net = build_perblock_dit(action_dim=6)
        print("eval: PERBLOCK model", flush=True)
    else:
        net = build_dit(action_dim=12 if DELTA else 6)
    net.timestep_scale = 0.001
    if hasattr(net, "action_ctx_scale"):
        net.action_ctx_scale = float(os.environ.get("ATOK_GATE_SCALE", "1.0"))
        print(f"eval: action_ctx_scale={net.action_ctx_scale:g}", flush=True)
    sd = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("net."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if "action_embedder" not in k or "fc2" in k}
    net.load_state_dict(sd, strict=False)

    for m in [m for m in sys.modules if m.startswith("transformer_engine")]:
        del sys.modules[m]
    from peft import LoraConfig, get_peft_model

    ck = torch.load(RUN_CKPT, map_location="cpu", weights_only=False)

    def _load_checked(model, state, kind):
        """strict=False 로 로드하되 실제로 몇 개가 붙었는지 세고, 0에 가까우면 멈춘다.

        체크포인트 키 접두사가 어긋나면(peft 래핑 유무 등) strict=False 는 아무것도
        로드하지 않고 조용히 통과한다. 그 상태로 생성하면 사전학습 가중치만으로 돌면서
        '로드 성공'만 찍혀서, 실험 결과 전체가 조용히 무의미해진다."""
        own = set(model.state_dict())
        matched = sum(1 for k in state if k in own)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"{kind} ckpt step={ck['step']} — 체크포인트 {len(state)}개 중 {matched}개 적용 "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
        if matched < max(1, len(state) // 2):
            raise SystemExit(
                f"[중단] 체크포인트 키가 모델과 맞지 않는다 ({matched}/{len(state)} 적용). "
                f"예상 밖 키 예: {list(unexpected)[:3]}"
            )

    lora_shapes = [v.shape[0] for k, v in ck["state"].items() if "lora_A" in k]
    if not lora_shapes:
        # 전체FT 체크포인트(코랩 v9 등): peft 없이 전 가중치 직접 로드
        _load_checked(net, ck["state"], "fullft")
        return net.to(device, torch.bfloat16).eval()
    # 체크포인트에서 LoRA 랭크 자동 감지 (r32/r64 혼용 지원)
    lora_r = lora_shapes[0]
    # LORA_SCALE: 어댑터 강도. peft 의 실효 배율은 lora_alpha/r 이므로 alpha 를 곱해 조절한다.
    # 1.0 = 학습된 그대로(챔피언). <1 이면 기본 모델 쪽으로 당기고, >1 이면 더 민다.
    #
    # 왜 이 손잡이인가: 8/4 에 val 손실과 LB 가 3점에서 완벽히 역상관인 것이 확인됐다
    # (train 에 더 맞출수록 LB 가 나빠진다). 그 방향을 **재학습 없이** 시험할 수 있는
    # 유일한 손잡이다. 학습으로는 6전 6패였다.
    #
    # ⚠ 오토가이던스와 상호작용한다. 오토는 v = v_cond + g*(v_cond - v_base) 인데
    # v_base 는 disable_adapter() 로 LoRA 를 끈 것이라 **스케일과 무관**하다.
    # 따라서 s 를 키우면 v_cond 와 (v_cond - v_base) 가 **둘 다** 커져 이중 증폭이 된다.
    # 액션 CFG + 오토가이던스가 서로 간섭해 후퇴했던 것과 같은 구조이므로 주의해서 읽어야 한다.
    lora_scale = float(os.environ.get("LORA_SCALE", "1.0"))
    alpha = int(round(lora_r * lora_scale))
    net = get_peft_model(
        net,
        LoraConfig(r=lora_r, lora_alpha=alpha, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2"]),
    )
    _load_checked(net, ck["state"], "finetune")
    if alpha != lora_r:
        # alpha 가 정수라 요청값과 실제 배율이 조금 다를 수 있다. 실제 값을 찍는다.
        print(f"LORA_SCALE 요청 {lora_scale:g} -> 실제 alpha/r = {alpha}/{lora_r} "
              f"= {alpha/lora_r:.4f}배", flush=True)
    return net.to(device, torch.bfloat16).eval()


def apply_cfg(v_cond: torch.Tensor, v_null: torch.Tensor, w: float,
              rescale: float = 0.0, mask=None) -> torch.Tensor:
    """CFG 한 스텝. rescale>0 이면 유도 후 표준편차를 조건부 예측 쪽으로 되돌린다.

    유도는 예측의 분산을 키운다(w 가 클수록 심하다). 그대로 두면 과포화·과대이동이 되는데,
    조건부 예측의 표준편차에 맞춰 되돌린 뒤 원본 유도값과 rescale 비율로 섞으면 완화된다.
    w=1.0 에서 개선이 평탄해진 원인이 이 분산 팽창이라면, 되돌리기가 더 큰 w 를 열어준다.

    mask 를 주면 그 영역(1인 곳)에서만 표준편차를 잰다. 조건 프레임 자리는 스텝 끝에서
    해석적 값으로 통째로 덮어써지므로 통계에 넣으면 배율이 틀어진다.

    rescale=0 이면 아무 일도 하지 않으므로 기존 동작과 완전히 동일하다."""
    out = v_cond + w * (v_cond - v_null)
    if rescale > 0:
        if mask is None:
            sc, so = v_cond.std(), out.std()
        else:
            m = mask > 0
            sc, so = v_cond[m].std(), out[m].std()
        out = rescale * (out * (sc / so.clamp_min(1e-8))) + (1 - rescale) * out
    return out


@torch.no_grad()
def generate(net, vae, text_emb, frame0, actions, device, guidance: float = 0.0, seed: int = 7,
             kwargs_return_latent: bool = False, num_steps: int | None = None, shift: float = 5.0,
             action_guidance: float = 0.0, anull: int = 1,
             acfg_lo: float = 0.0, acfg_hi: float = 1.0, acfg_rescale: float = 0.0,
             auto_guidance: float = 0.0):
    """frame0 (3,320,512) [-1,1], actions (16,6) 정규화 → 17프레임 (3,17,320,512).

    guidance>0        조건프레임을 0으로 둔 무조건 분기와의 CFG. 초기프레임 앵커까지 같이
                      지워버리므로 우리 과제에는 맞지 않는다. 비교용으로만 남겨둔다.
    action_guidance>0 액션만 널로 둔 분기와의 CFG. 초기프레임 조건은 양쪽 분기에 그대로
                      살아 있어서 앵커가 유지되고, 액션이 화면을 지배하는 정도만 증폭된다.
                      스텝당 순전파가 2배다.
    anull             널을 무엇으로 둘지. SO-100 액션은 **절대 관절각**이므로
                      (mean [3.0,117.6,109.8,61.7,-29.4,9.8] deg) 정규화 0벡터는
                      "움직이지 마라"가 아니라 "데이터셋 평균 자세로 가라"는 또 하나의
                      명령이다. 그래서 기본값은 1 = 첫 액션을 16스텝 반복(초기 자세 유지)이고,
                      이때 유도 방향이 정확히 "명령된 움직임 - 가만히 있기"가 된다.
                      0 은 0벡터(의미상 부정확, 비교용). **실측상 0 이 낫다** — LB 0.1925 는 anull=0.
    acfg_lo/hi        유도를 거는 노이즈 구간 (정규화 t = timestep/1000 기준, 기본 0~1 = 항상).
                      CFG 를 전 구간에 균일하게 거는 것보다 중간 구간에만 거는 편이 나은 경우가
                      많다 — 고노이즈 구간에서는 방향이 부정확하고, 저노이즈 구간에서는 이미
                      결정된 구조를 흔들어 결만 거칠어지기 쉽다.
    acfg_rescale      유도 후 분산 되돌리기 (0=끄기, 1=완전). CFG 는 예측의 표준편차를 키워
                      과포화를 만든다. 조건부 예측의 표준편차에 맞춰 되돌린 뒤 원본과 섞는다.
    auto_guidance>0   오토가이던스: 널 분기를 **LoRA 를 끈 기본 모델**(peft disable_adapter)로
                      둔다. 액션 CFG 가 액션 조건만 증폭하는 것과 달리, 이것은 미세조정이
                      배운 전부(도메인 외형 + 액션 추종)를 증폭한다. ACT_DROP 실패(LB 0.2754)가
                      역설적으로 이 카드의 근거다 — 널 분기는 충분히 "못해야" 유도가 사는데,
                      LoRA 끈 기본 모델은 이 도메인에서 확실히 못한다. 학습된 널이 아니라
                      영구히 고정된 널이라 그 실패 모드(널이 유능해져 유도가 죽는 것)도 없다.
                      액션 CFG 와 독립으로 합성 가능(각각 v0 기준 가산). 스텝당 순전파 +1.

    num_steps/shift 를 넘기지 않으면 기존 동작(전역 NUM_STEPS, shift 5.0) 그대로다.
    챔피언 경로(own_bon.py)는 validate.NUM_STEPS 를 덮어쓰는 방식이라 영향받지 않는다."""
    from cosmos_predict2._src.predict2.conditioner import DataType
    from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler

    video = torch.zeros(1, 3, 17, 320, 512)
    video[0, :, 0] = frame0
    gt_latent = vae.encode(video.to(device, torch.bfloat16)).float()
    B, C, T, H, W = gt_latent.shape
    cond = torch.zeros(B, 1, T, H, W, device=device)
    cond[:, :, 0] = 1.0
    cond_c = cond.repeat(1, C, 1, 1, 1)
    pad = torch.zeros(B, 1, 320, 512, device=device, dtype=torch.bfloat16)
    fps = torch.full((B,), 6, device=device, dtype=torch.bfloat16)
    # DELTA 면 여기서 (1,16,6) -> (1,16,12). 액션 CFG 의 널(act_null)은 이 변환 **뒤**의
    # 텐서에서 만들어지므로(아래 195행) 두 채널 모두 일관되게 널이 된다.
    act = action_cond(actions[None].to(device, torch.bfloat16))

    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(NUM_STEPS if num_steps is None else num_steps, device=device, shift=shift)
    def net_v(xt_masked, t, a=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(
                x_B_C_T_H_W=xt_masked.to(torch.bfloat16),
                timesteps_B_T=torch.full((B, 1), float(t), device=device, dtype=torch.bfloat16),
                crossattn_emb=text_emb.expand(B, -1, -1),
                condition_video_input_mask_B_C_T_H_W=cond.to(torch.bfloat16),
                fps=fps, padding_mask=pad, data_type=DataType.VIDEO,
                action=act if a is None else a,
            )
        return out.float()

    torch.manual_seed(seed)
    noise = torch.randn(B, C, T, H, W, device=device)
    lat = noise
    act_null = None
    if action_guidance > 0:
        act_null = act[:, :1].expand_as(act).contiguous() if anull else torch.zeros_like(act)
    use_auto = auto_guidance > 0
    if use_auto:
        assert hasattr(net, "disable_adapter"), "auto_guidance 는 peft 모델(LoRA)에서만 가능"
    for t in sch.timesteps:
        masked = gt_latent * cond_c + lat * (1 - cond_c)
        v = net_v(masked, t)
        v0 = v  # 유도들은 각각 v0 기준으로 가산 — 순차 합성의 교차항을 만들지 않는다
        if action_guidance > 0 and acfg_lo <= float(t) / 1000.0 <= acfg_hi:
            v_null = net_v(masked, t, act_null)  # 초기프레임 조건은 유지, 액션만 널
            v = apply_cfg(v, v_null, action_guidance, acfg_rescale, 1 - cond_c)
        if use_auto:
            action_model = getattr(getattr(net, "base_model", None), "model", net)
            had_action_ctx = hasattr(action_model, "action_ctx_enabled")
            old_action_ctx = getattr(action_model, "action_ctx_enabled", False)
            if had_action_ctx:
                action_model.action_ctx_enabled = False
            try:
                with net.disable_adapter():
                    v_base = net_v(masked, t)    # LoRA와 신규 action-token 경로를 모두 끈 base
            finally:
                if had_action_ctx:
                    action_model.action_ctx_enabled = old_action_ctx
            v = v + auto_guidance * (v0 - v_base)
        if guidance > 0:
            v_uncond = net_v(lat * (1 - cond_c), t)  # 조건프레임 0
            v = v + guidance * (v - v_uncond)
        v = (noise - gt_latent) * cond_c + v * (1 - cond_c)
        lat = sch.step(v.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False)[0].squeeze(0)
    frames = vae.decode(lat.to(torch.bfloat16))  # (1,3,17,320,512)
    if kwargs_return_latent:
        return frames, lat
    return frames


@torch.no_grad()
def generate_batch(net, vae, text_emb, frame0s, actions_b, device, seeds,
                   num_steps: int | None = None, shift: float = 5.0,
                   action_guidance: float = 0.0, anull: int = 1,
                   acfg_lo: float = 0.0, acfg_hi: float = 1.0, acfg_rescale: float = 0.0):
    """여러 샘플을 한 번에 생성. frame0s (B,3,320,512), actions_b (B,16,6), seeds 길이 B.

    반환 (frames (B,3,17,320,512), latents (B,16,5,40,64)).

    `generate()` 와 **일부러 분리**했다. 챔피언 재현 경로는 generate() 이고 그건
    비트 단위로 검증돼 있으므로 손대지 않는다. 배치 크기가 달라지면 cuBLAS 커널 선택이
    달라져 결과가 마지막 자리에서 흔들릴 수 있는데, 대량 데이터 생성에는 무해하지만
    제출물 재현에는 허용할 수 없는 차이다.

    노이즈는 샘플별로 `manual_seed(seed_i)` 를 걸고 뽑아 순차 실행과 같은 난수를 쓴다.
    후보 N개를 뽑을 때는 같은 frame0/action 을 B번 반복하고 seeds 만 다르게 주면 된다."""
    from cosmos_predict2._src.predict2.conditioner import DataType
    from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler

    B = frame0s.shape[0]
    assert len(seeds) == B and actions_b.shape[0] == B
    video = torch.zeros(B, 3, 17, 320, 512)
    video[:, :, 0] = frame0s
    gt_latent = vae.encode(video.to(device, torch.bfloat16)).float()
    _, C, T, H, W = gt_latent.shape
    cond = torch.zeros(B, 1, T, H, W, device=device)
    cond[:, :, 0] = 1.0
    cond_c = cond.repeat(1, C, 1, 1, 1)
    pad = torch.zeros(B, 1, 320, 512, device=device, dtype=torch.bfloat16)
    fps = torch.full((B,), 6, device=device, dtype=torch.bfloat16)
    act = action_cond(actions_b.to(device, torch.bfloat16))  # DELTA 면 (B,16,12)

    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(NUM_STEPS if num_steps is None else num_steps, device=device, shift=shift)

    def net_v(xt_masked, t, a=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(
                x_B_C_T_H_W=xt_masked.to(torch.bfloat16),
                timesteps_B_T=torch.full((B, 1), float(t), device=device, dtype=torch.bfloat16),
                crossattn_emb=text_emb.expand(B, -1, -1),
                condition_video_input_mask_B_C_T_H_W=cond.to(torch.bfloat16),
                fps=fps, padding_mask=pad, data_type=DataType.VIDEO,
                action=act if a is None else a,
            )
        return out.float()

    noise = []
    for s in seeds:
        torch.manual_seed(int(s))
        noise.append(torch.randn(C, T, H, W, device=device))
    noise = torch.stack(noise)
    lat = noise
    act_null = None
    if action_guidance > 0:
        act_null = act[:, :1].expand_as(act).contiguous() if anull else torch.zeros_like(act)
    for t in sch.timesteps:
        masked = gt_latent * cond_c + lat * (1 - cond_c)
        v = net_v(masked, t)
        if action_guidance > 0 and acfg_lo <= float(t) / 1000.0 <= acfg_hi:
            v = apply_cfg(v, net_v(masked, t, act_null), action_guidance, acfg_rescale, 1 - cond_c)
        v = (noise - gt_latent) * cond_c + v * (1 - cond_c)
        lat = sch.step(v.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False)[0].squeeze(0)
    return vae.decode(lat.to(torch.bfloat16)), lat


def to_uint8_16(frames_c_t_h_w: torch.Tensor) -> torch.Tensor:
    """(3,17or16,H,W) [-1,1] → (16,H,W,3) uint8"""
    x = frames_c_t_h_w[:, :16].float().clamp(-1, 1)
    return ((x + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()


def write_mp4(path, frames_u8, fps: int = 6, crf: int = 12, preset: str = "slow") -> None:
    """모든 생성 경로가 이 함수만 써서 인코딩을 통일한다.

    챔피언(own_bon.py)은 crf 12·preset slow 로 썼는데 generate_eval.py 는 imageio 기본값을
    써서, 같은 가중치·같은 설정이어도 두 경로의 영상이 등가가 아니었다. 과거 실험 간
    시각 항목 비교가 성립하지 않았던 원인이다.

    PIX_FMT 환경변수로 크로마 서브샘플링을 바꿀 수 있다(예: yuv444p). 지정하지 않으면
    libx264 기본값(yuv420p)이라 색차 해상도가 가로세로 절반이다. 다만 정답 영상도 420p 로
    인코딩됐을 수 있어 444p 가 유리하다는 보장은 없다 — A/B 로만 판정할 것."""
    import imageio

    params = ["-crf", str(crf), "-preset", preset]
    pix_fmt = os.environ.get("PIX_FMT", "")
    if pix_fmt:
        params += ["-pix_fmt", pix_fmt]
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1,
                            ffmpeg_params=params) as w:
        for f in frames_u8:
            w.append_data(f)
