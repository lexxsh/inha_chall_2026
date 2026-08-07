"""사전학습 UNet이 실제로 몇 개나 로드되는지 센다.

load_checkpoints()가 strict=False로 부르기 때문에 설정이 어긋나도 예외가 나지 않는다.
베이스라인이 붕괴한 이유가 사전학습 미사용이었던 만큼, 여기서 로드율을 확인하지 않으면
"사전학습을 쓴다고 믿지만 실은 대부분 랜덤 초기화"인 상태로 4일을 태울 수 있다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", nargs="+",
        default=[str(REPO / "train/configs/inha_full_unet.yaml")],
        help="왼쪽부터 merge할 config 목록",
    )
    parser.add_argument("--checkpoint", default=str(REPO / "open/baseline/checkpoints/backbone.ckpt"))
    args = parser.parse_args()

    from lvdm.utils.utils import instantiate_from_config

    config = OmegaConf.merge(*[OmegaConf.load(path) for path in args.config])
    unet_cfg = config.model.params.unet_config

    print("UNet 생성 중...")
    unet = instantiate_from_config(unet_cfg)
    model_sd = unet.state_dict()
    n_params = sum(v.numel() for v in model_sd.values())
    print(f"생성된 UNet: 키 {len(model_sd)}개 / 파라미터 {n_params / 1e6:.1f}M")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt = ckpt.get("state_dict", ckpt)
    prefix = "model.diffusion_model."
    ckpt_unet = {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}
    print(f"체크포인트 UNet: 키 {len(ckpt_unet)}개 / 파라미터 {sum(v.numel() for v in ckpt_unet.values()) / 1e6:.1f}M")

    matched, shape_mismatch, missing_in_ckpt = [], [], []
    for key, value in model_sd.items():
        if key not in ckpt_unet:
            missing_in_ckpt.append(key)
        elif tuple(ckpt_unet[key].shape) != tuple(value.shape):
            shape_mismatch.append((key, tuple(ckpt_unet[key].shape), tuple(value.shape)))
        else:
            matched.append(key)
    unused_in_ckpt = [k for k in ckpt_unet if k not in model_sd]

    matched_params = sum(model_sd[k].numel() for k in matched)
    action_prefixes = ("action_embed.", "null_action_emb")
    def is_action_key(key: str) -> bool:
        return key.startswith(action_prefixes) or ".action_modulation." in f".{key}."

    backbone_keys = [k for k in model_sd if not is_action_key(k)]
    backbone_params = sum(model_sd[k].numel() for k in backbone_keys)
    matched_backbone_params = sum(model_sd[k].numel() for k in matched if not is_action_key(k))
    print()
    print("=" * 70)
    print(f"로드 성공     : 키 {len(matched)}개 / 파라미터 {matched_params / 1e6:.1f}M "
          f"({100 * matched_params / n_params:.1f}%)")
    print(f"backbone 일치 : 파라미터 {matched_backbone_params / 1e6:.1f}M / "
          f"{backbone_params / 1e6:.1f}M ({100 * matched_backbone_params / backbone_params:.1f}%)")
    print(f"형상 불일치   : {len(shape_mismatch)}개  <- 0이어야 한다")
    print(f"체크포인트 없음: {len(missing_in_ckpt)}개  <- 액션 임베딩 등 신규 모듈만이어야 한다")
    print(f"모델에 없음   : {len(unused_in_ckpt)}개  <- 버려지는 사전학습 가중치")
    print("=" * 70)

    if shape_mismatch:
        print("\n[형상 불일치 — 반드시 고칠 것]")
        for key, ckpt_shape, model_shape in shape_mismatch[:20]:
            print(f"  {key}\n     체크포인트 {ckpt_shape} vs 모델 {model_shape}")
        if len(shape_mismatch) > 20:
            print(f"  ... 외 {len(shape_mismatch) - 20}개")

    if missing_in_ckpt:
        print("\n[새로 학습될 모듈]")
        for key in missing_in_ckpt[:20]:
            print(f"  {key} {tuple(model_sd[key].shape)}")
        if len(missing_in_ckpt) > 20:
            print(f"  ... 외 {len(missing_in_ckpt) - 20}개")

    if unused_in_ckpt:
        print("\n[버려지는 사전학습 가중치]")
        for key in unused_in_ckpt[:20]:
            print(f"  {key} {tuple(ckpt_unet[key].shape)}")
        if len(unused_in_ckpt) > 20:
            print(f"  ... 외 {len(unused_in_ckpt) - 20}개")

    unexpected_missing = [k for k in missing_in_ckpt if not is_action_key(k)]
    ok = not shape_mismatch and not unexpected_missing and matched_backbone_params / backbone_params > 0.99
    print("\n판정:", "통과 — 사전학습 UNet이 제대로 로드된다" if ok else "실패 — 설정을 고칠 것")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
