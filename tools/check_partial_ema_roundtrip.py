"""partial-EMA checkpoint를 full inference EMA로 안전하게 복원하는지 검사한다.

실제 대형 checkpoint를 만들지 않고도 학습 때 동결된 파라미터가 EMA scope에서
오염되지 않는지, 학습된 파라미터에는 저장된 EMA가 적용되는지 재현한다.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/baseline/challenge_kit"
sys.path.insert(0, str(KIT / "libs/dynamicrafter"))
sys.path.insert(0, str(REPO / "train"))

from generate_videos import rebuild_ema  # noqa: E402
from lvdm.ema import LitEma  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(3, 3)
        self.action_embed = nn.Sequential(nn.Linear(3, 3))


class InferenceWrapper:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.use_ema = True
        # 실제 추론 초기화 순서처럼 main checkpoint를 읽기 전에 만들어진 EMA.
        self.model_ema = LitEma(copy.deepcopy(model))


def main() -> None:
    torch.manual_seed(20260720)

    trained = TinyModel()
    for parameter in trained.frozen.parameters():
        parameter.requires_grad_(False)
    partial_ema = LitEma(trained)

    # 본체와 EMA가 서로 달라지도록 한 번의 가상 학습/EMA update를 만든다.
    with torch.no_grad():
        for parameter in trained.action_embed.parameters():
            parameter.add_(2.0)
    partial_ema(trained)

    checkpoint_ema = {
        f"model_ema.{name}": value.clone()
        for name, value in partial_ema.state_dict().items()
    }
    expected_action = {
        name: value.clone()
        for name, value in partial_ema.named_buffers()
        if "action_embed" in name
    }

    # 추론에서는 UNet의 학습 당시 동결 여부를 모르므로 모든 경로가 trainable인 상태다.
    inference_model = TinyModel()
    wrapper = InferenceWrapper(inference_model)
    inference_model.load_state_dict(trained.state_dict())
    for parameter in inference_model.parameters():
        parameter.requires_grad_(True)

    main_before = {
        name: parameter.detach().clone()
        for name, parameter in inference_model.named_parameters()
    }
    rebuild_ema(wrapper, checkpoint_ema, use_ema=True)

    wrapper.model_ema.store(wrapper.model.parameters())
    wrapper.model_ema.copy_to(wrapper.model)
    ema_applied = dict(wrapper.model.named_parameters())

    for name, before in main_before.items():
        if name.startswith("frozen."):
            torch.testing.assert_close(ema_applied[name], before, rtol=0, atol=0)

    shadow_names = wrapper.model_ema.m_name2s_name
    for name in ("action_embed.0.weight", "action_embed.0.bias"):
        shadow = shadow_names[name]
        torch.testing.assert_close(
            ema_applied[name], expected_action[shadow], rtol=0, atol=0
        )

    wrapper.model_ema.restore(wrapper.model.parameters())
    for name, before in main_before.items():
        torch.testing.assert_close(
            dict(wrapper.model.named_parameters())[name], before, rtol=0, atol=0
        )

    print("PASS: frozen main weights 보존, trainable partial EMA 적용, scope 복원 성공")


if __name__ == "__main__":
    main()
