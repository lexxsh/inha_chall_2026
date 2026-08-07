# Windows에서 cosmos-predict2.5 코드를 import하기 위한 Linux 전용 의존성 스텁.
# 실제 사용 경로(minimal_a2a=SDPA, 단일 GPU)에서는 호출되지 않는 모듈만 가짜로 채운다.
from __future__ import annotations

import sys
import types
import os
from pathlib import Path

_BUNDLED_REPO = Path(__file__).resolve().parent.parent / "cosmos-predict2.5"
_WORKSPACE_REPO = Path(__file__).resolve().parents[3] / "third_party" / "cosmos-predict2.5"
REPO_ROOT = Path(os.environ.get("COSMOS_REPO", "")) if os.environ.get("COSMOS_REPO") else (
    _BUNDLED_REPO if _BUNDLED_REPO.exists() else _WORKSPACE_REPO
)
if not (REPO_ROOT / "cosmos_predict2" / "__about__.py").exists():
    raise FileNotFoundError(
        "Cosmos Predict2.5 source tree not found. Set COSMOS_REPO or place it at "
        f"{_BUNDLED_REPO} / {_WORKSPACE_REPO}"
    )


def _fake_module(name: str, **attrs) -> types.ModuleType:
    import importlib.machinery

    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def install_stubs() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # torch 2.7의 sdpa_kernel은 set_priority_order 대신 set_priority 인자를 사용.
    # cosmos 코드가 구 인자명으로 호출하면 TypeError → FLASH 단독 폴백 → Windows에서 커널 없음.
    import torch.nn.attention as _torch_attn

    _orig_sdpa_kernel = _torch_attn.sdpa_kernel

    def _sdpa_kernel_compat(backends, *args, set_priority_order=None, **kwargs):
        if set_priority_order is not None:
            try:
                return _orig_sdpa_kernel(backends, *args, set_priority=set_priority_order, **kwargs)
            except TypeError:
                return _orig_sdpa_kernel(backends, *args, **kwargs)
        return _orig_sdpa_kernel(backends, *args, **kwargs)

    _torch_attn.sdpa_kernel = _sdpa_kernel_compat

    # cosmos_cuda: uv 설치 마커 패키지. 버전 문자열만 맞으면 통과
    if "cosmos_cuda" not in sys.modules:
        about: dict = {}
        exec((REPO_ROOT / "cosmos_predict2" / "__about__.py").read_text(encoding="utf-8"), about)
        _fake_module("cosmos_cuda", __version__=about["__version__"])

    # transformer_engine: attention 백엔드 "transformer_engine" 선택 시에만 실제 사용됨
    # 단, RMSNorm/apply_rotary_pos_emb는 minimal_a2a 경로에서도 쓰이므로 동일 수식으로 대체
    if "transformer_engine" not in sys.modules:
        import torch
        import torch.nn as _nn

        def apply_rotary_pos_emb(t, freqs, tensor_format="bshd", fused=True):
            """TE 규약 재현: freqs (S,1,1,D)는 각도(하프 2회 반복), rotate-half 회전."""
            rot_dim = freqs.shape[-1]
            if tensor_format == "bshd":
                f = freqs.squeeze(2).squeeze(1)[None, :, None, :]  # (1,S,1,D)
            elif tensor_format == "sbhd":
                f = freqs  # (S,1,1,D)
            else:
                raise ValueError(f"unsupported tensor_format: {tensor_format}")
            t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            cos = f.cos()
            sin = f.sin()
            x1, x2 = t_rot.float().chunk(2, dim=-1)
            rotated = torch.cat((-x2, x1), dim=-1)
            t_rot = (t_rot.float() * cos + rotated * sin).to(t.dtype)
            return torch.cat((t_rot, t_pass), dim=-1) if t_pass.numel() else t_rot

        te = _fake_module("transformer_engine")
        te_pytorch = _fake_module("transformer_engine.pytorch", RMSNorm=_nn.RMSNorm)
        te.pytorch = te_pytorch
        attn_mod = _fake_module(
            "transformer_engine.pytorch.attention",
            apply_rotary_pos_emb=apply_rotary_pos_emb,
        )
        rope_mod = _fake_module(
            "transformer_engine.pytorch.attention.rope",
            apply_rotary_pos_emb=apply_rotary_pos_emb,
        )
        attn_mod.rope = rope_mod
        te_pytorch.attention = attn_mod

    # multistorageclient: NVIDIA 내부 스토리지 백엔드. 로컬 파일 IO만 사용하므로 이름만 제공
    if "multistorageclient" not in sys.modules:
        _fake_module(
            "multistorageclient",
            StorageClient=object,
            StorageClientConfig=object,
        )
        msc_types = _fake_module(
            "multistorageclient.types",
            MSC_PROTOCOL="msc://",
            Range=object,
        )
        sys.modules["multistorageclient"].types = msc_types

    # megatron: context parallel(멀티 GPU) 전용. 단일 GPU에서는 parallel_state 조회만 통과하면 됨
    if "megatron" not in sys.modules:
        megatron = _fake_module("megatron")

        class _ParallelState:
            @staticmethod
            def is_initialized() -> bool:
                return False

            @staticmethod
            def get_context_parallel_world_size() -> int:
                return 1

            @staticmethod
            def get_context_parallel_group():
                return None

            @staticmethod
            def get_context_parallel_rank() -> int:
                return 0

        core = _fake_module("megatron.core", parallel_state=_ParallelState())
        megatron.core = core


install_stubs()
