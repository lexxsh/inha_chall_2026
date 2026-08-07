"""학습한 액션 조건 모델로 eval 영상을 생성한다.

베이스라인의 generate_baseline_videos.py와 같은 흐름이지만 한 가지가 다르다.
학습에서 액션을 delta로 조건화했으므로 추론에서도 **같은 변환을 적용해야 한다.**
여기가 어긋나면 예외 없이 조용히 엉뚱한 영상이 나온다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/baseline/challenge_kit"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_module import load_delta_scale, transform_actions  # noqa: E402
from lvdm.models.samplers.ddim import DDIMSampler  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from scripts.eval.feature_csv_utils import (  # noqa: E402
    build_inference_batch,
    list_challenge_sample_ids,
    load_action_stats,
    save_video_tensor,
)


def rebuild_ema(model, state: dict | None, use_ema: bool) -> None:
    """추론용 EMA를 안전하게 다시 만든다.

    문제: `LitEma`는 생성 시점에 `requires_grad=True`인 파라미터만 shadow로 등록하고,
    그 값을 **그때의 모델 파라미터로 복사**한다. 그런데 `get_model()`은 모델을 만든 뒤에
    (=EMA가 이미 만들어진 뒤에) 사전학습을 로드하므로, 추론에서 새로 생긴 full EMA의 shadow는
    랜덤 초기값이다. 학습은 동결을 적용한 뒤 EMA를 만들었으므로 체크포인트에는 trainable 40%분만 있고,
    나머지 60%는 랜덤인 채로 남는다. `ema_scope()`가 이를 본체에 복사하면 UNet 대부분이 망가진다.

    해법: 체크포인트를 본체에 로드한 **뒤** EMA를 다시 만들어 shadow를 로드된 값으로 채우고,
    체크포인트에 실제로 존재하는 shadow만 덮어쓴다. 그러면 동결 레이어는 사전학습값,
    학습 레이어는 학습된 EMA값이 된다.
    """
    if not getattr(model, "use_ema", False):
        return
    if not use_ema:
        model.use_ema = False
        print("[EMA] 사용하지 않음 — 체크포인트 본체 가중치로 생성한다")
        return

    from lvdm.ema import LitEma

    model.model_ema = LitEma(model.model)  # 본체가 이미 로드된 상태이므로 shadow가 올바른 값으로 시작
    total = len(model.model_ema.m_name2s_name)
    overlaid = 0
    if state:
        ema_state = {k[len("model_ema.") :]: v for k, v in state.items() if k.startswith("model_ema.")}
        own = dict(model.model_ema.named_buffers())
        with torch.no_grad():
            for name, value in ema_state.items():
                if name in own and own[name].shape == value.shape:
                    own[name].copy_(value)
                    overlaid += 1
        ema_action = [
            k for k in ema_state
            if "action_embed" in k or "null_action_emb" in k or "action_modulation" in k
        ]
        if not ema_action:
            raise SystemExit(
                "체크포인트에 EMA action 키가 없다. EMA 경로로 생성하면 액션 조건이 반영되지 않는다. "
                "--no-ema 로 본체 가중치를 쓰거나 EMA를 포함해 저장된 체크포인트를 쓸 것."
            )
        mode = getattr(model.model.diffusion_model, "action_injection", "additive")
        if mode == "frame_adaln" and not any("action_modulation" in k for k in ema_action):
            raise SystemExit(
                "frame_adaln 체크포인트에 EMA action_modulation 키가 없다. "
                "--no-ema로 본체를 쓰거나 올바른 config/checkpoint 조합을 확인할 것."
            )
    print(f"[EMA] shadow {total}개 중 체크포인트에서 {overlaid}개 덮어씀 "
          f"(나머지는 로드된 본체 값 = 학습 시 동결된 레이어)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", nargs="+",
        default=[str(REPO / "train/configs/inha_full_unet.yaml")],
        help="왼쪽부터 merge할 config. Frame-Ada는 base와 overlay 두 파일을 순서대로 지정한다.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="학습으로 나온 체크포인트(.ckpt). 생략하면 config의 사전학습 가중치만으로 돈다 "
        "— 추론 시간 측정(dry-run)용이며 영상 품질은 의미 없다.",
    )
    parser.add_argument("--limit", type=int, default=0, help="앞의 N개만 생성 (시간 측정용)")
    parser.add_argument("--no-ema", action="store_true", help="EMA를 쓰지 않고 체크포인트 본체 가중치로 생성")
    parser.add_argument("--challenge-root", default=str(REPO / "open/data/eval"))
    parser.add_argument("--prediction-root", default=str(REPO / "generated_videos"))
    parser.add_argument("--action-stats-path", default=str(REPO / "open/data/train/so100_action_statistics.json"))
    parser.add_argument("--delta-stats-path", default=str(REPO / "train/delta_action_stats.json"))
    parser.add_argument("--action-mode", default=None, help="기본값은 학습 설정에서 읽는다")
    parser.add_argument("--action-shift", type=int, default=None, help="기본값은 학습 설정에서 읽는다")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ddim-eta", type=float, default=1.0)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-ablation",
        choices=("none", "zero", "reverse-time", "batch-roll"),
        default="none",
        help="action conditioner 반증 실험. batch-roll은 같은 batch 안에서 다른 샘플의 action을 넣는다.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.merge(*[OmegaConf.load(path) for path in args.config])
    print(f"[config] {' + '.join(args.config)}")
    # 학습과 같은 표현이어야 한다. 기본값을 학습 설정에서 그대로 읽어 온다.
    action_mode = args.action_mode or config.data.params.get("action_mode", "delta")
    action_shift = args.action_shift if args.action_shift is not None else config.data.params.get("action_shift", 0)
    target_height = config.data.params.get("target_height", 320)
    target_width = config.data.params.get("target_width", 512)
    pad = config.data.params.get("pad", True)

    # 학습 때 쓴 체크포인트 대신 우리가 학습한 가중치를 얹는다.
    model_cfg = config.model.copy()
    # config의 pretrained_checkpoint는 challenge_kit 기준 상대경로라 다른 위치에서 실행하면
    # 못 찾는다. 어디서 실행하든 backbone을 찾도록 config 파일 위치 기준으로 절대경로화한다.
    kit_dir = REPO / "open/baseline/challenge_kit"
    pc = model_cfg.get("pretrained_checkpoint")
    if pc and not Path(pc).is_absolute():
        resolved = (kit_dir / pc).resolve()
        if not resolved.exists():
            resolved = Path(pc).resolve()
        model_cfg.pretrained_checkpoint = str(resolved)
        print(f"[경로] backbone -> {model_cfg.pretrained_checkpoint}")
    model = get_model(model_cfg)
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # strict=False라 액션 관련 키가 통째로 빠져도 계속 돈다. 그 경우 액션 조건이 없는
        # 영상을 만들면서도 정상처럼 보이므로 여기서 잡는다.
        main_action = [k for k in state if k.startswith("model.") and ("action_embed" in k or "null_action_emb" in k)]
        if not main_action:
            raise SystemExit(
                f"{args.checkpoint} 에 본체 action_embed/null_action_emb 키가 없다. "
                "액션 조건 없이 생성될 수 있으므로 중단한다."
            )
        injection = getattr(model.model.diffusion_model, "action_injection", "additive")
        if injection == "frame_adaln":
            main_modulation = [
                k for k in state if k.startswith("model.") and "action_modulation" in k
            ]
            if not main_modulation:
                raise SystemExit(
                    f"{args.checkpoint}에 frame_adaln action_modulation 키가 없다. "
                    "additive checkpoint와 Frame-Ada config를 섞었을 가능성이 있다."
                )
        print(f"[로드] {args.checkpoint}: 누락 {len(missing)} / 예상외 {len(unexpected)} / 본체 액션 키 {len(main_action)}")
    else:
        print("[로드] 체크포인트 없음 — 사전학습 가중치만으로 실행한다(추론 시간 측정 전용, 품질 무의미)")

    rebuild_ema(model, state, use_ema=not args.no_ema)
    model.to(device).eval()
    sampler = DDIMSampler(model)

    action_mean, action_std = load_action_stats(args.action_stats_path)
    delta_scale = load_delta_scale(args.delta_stats_path, action_mode)
    print(f"[액션] 표현={action_mode}, shift={action_shift:+d}, delta_scale={delta_scale.tolist()}")

    challenge_root = Path(args.challenge_root)
    prediction_root = Path(args.prediction_root)
    prediction_root.mkdir(parents=True, exist_ok=True)
    sample_ids = list_challenge_sample_ids(challenge_root)
    total_samples = len(sample_ids)
    if args.limit:
        sample_ids = sample_ids[: args.limit]
        print(f"[제한] {len(sample_ids)}개만 생성해 전체 {total_samples}개 소요시간을 외삽한다")

    amp = torch.cuda.amp.autocast() if args.precision == 16 and device.type == "cuda" else nullcontext()
    ddim_kwargs = dict(
        unconditional_guidance_scale=args.guidance_scale,
        timestep_spacing="uniform_trailing",
        guidance_rescale=args.guidance_rescale,
        ddim_eta=args.ddim_eta,
        verbose=False,
    )

    started = time.perf_counter()
    done = 0
    for start in range(0, len(sample_ids), args.batch_size):
        batch_ids = sample_ids[start : start + args.batch_size]
        paths = [prediction_root / f"{sid}.mp4" for sid in batch_ids]
        if not args.overwrite and all(p.exists() for p in paths):
            done += len(batch_ids)
            continue

        batch = build_inference_batch(
            challenge_root, batch_ids, target_height, target_width, pad, args.fps,
            action_mean, action_std, device,
        )
        # ★ 학습과 동일한 액션 표현으로 변환 (data_module과 같은 함수를 쓴다)
        batch["act"] = transform_actions(batch["act"], delta_scale, action_mode, action_shift)
        if args.action_ablation == "zero":
            batch["act"].zero_()
        elif args.action_ablation == "reverse-time":
            batch["act"] = batch["act"].flip(1)
        elif args.action_ablation == "batch-roll":
            if batch["act"].shape[0] < 2:
                raise SystemExit("--action-ablation=batch-roll은 모든 batch에 샘플이 2개 이상 있어야 한다.")
            batch["act"] = batch["act"].roll(1, dims=0)
        if args.action_ablation != "none":
            print(f"[action ablation] {args.action_ablation}: {batch_ids}")

        z, c, uc, cond_mask, _logs, kwargs = model.prepare_batch_for_inference(batch)
        sample_kwargs = dict(ddim_kwargs)
        sample_kwargs.update(kwargs)
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), model.ema_scope("generate"):
            with amp:
                samples, _ = sampler.sample(
                    args.ddim_steps, batch_size=z.shape[0], shape=shape,
                    conditioning=c, unconditional_conditioning=uc,
                    mask=cond_mask, x0=z, **sample_kwargs,
                )
            videos = model.decode_first_stage(samples)

        for sid, video in zip(batch_ids, videos):
            save_video_tensor(video, prediction_root / f"{sid}.mp4", args.fps)
        done += len(batch_ids)
        elapsed = time.perf_counter() - started
        print(f"[생성] {done}/{len(sample_ids)}  경과 {elapsed / 60:.1f}분  "
              f"예상 총 {elapsed / max(done, 1) * len(sample_ids) / 60:.1f}분")

    total = time.perf_counter() - started
    per_sample = total / max(done, 1)
    projected = per_sample * total_samples
    print(f"\n완료: {len(sample_ids)}개 / {total / 60:.1f}분  (샘플당 {per_sample:.1f}초)")
    print(f"전체 {total_samples}개 환산: {projected / 60:.1f}분 / 한도 60분")
    if projected > 3600:
        print(f"경고: 1시간 제한 초과 예상({projected / 60:.1f}분). ddim-steps·batch-size를 줄이거나 "
              "더 작은 백본이 필요하다.")
    else:
        print(f"여유: {(3600 - projected) / 60:.1f}분")
    print(f"제출하려면 영상을 open/submission_kit/input_videos/ 로 옮기고 make_submission_csv.py 실행")


if __name__ == "__main__":
    main()
