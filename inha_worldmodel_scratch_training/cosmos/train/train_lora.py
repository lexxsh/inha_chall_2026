# SO-100 fine-tune v1: 액션 임베더(6D, full) + DiT LoRA(r32). 완전 조건부(uncond 드롭아웃 없음).
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "port"))
import bootstrap  # noqa: F401

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from load_dit import build_dit, DIT_CKPT, TEXT_EMB, CKPT_DIR
from so100_dataset import SO100ClipDataset
from unified_losses import counterfactual_ranking_loss, paired_latent_losses

import os

OUT_DIR = Path(__file__).resolve().parent / "runs" / "v1_lora"
LATENT_DIR = Path(os.environ.get("COSMOS_LATENT_DIR", str(Path(__file__).resolve().parent / "latents")))
VAL_LATENT_DIR = Path(os.environ.get(
    "COSMOS_VAL_LATENT_DIR",
    str(Path(__file__).resolve().parent / "val_latents"),
))
DELTA_STATS = Path(__file__).resolve().parent / "delta_action_stats.json"


def load_val_set(n: int):
    """에피소드 단위로 완전히 분리된 val 창을 고정된 순서로 n개 읽는다.

    왜 필요한가: 지금까지 학습 중 val 손실을 재지 않았다. 그런데 8/3 균등샘플링 실행은
    train 손실이 22.5시그마로 떨어지는 동안 리더보드는 0.1872 -> 0.1964 로 나빠졌다.
    train 손실 하락은 점수 개선을 보장하지 않으며, 과적합을 알아챌 수단이 필요하다.
    """
    if not VAL_LATENT_DIR.exists():
        return None, None
    files = sorted(VAL_LATENT_DIR.glob("*.pt"))
    if not files:
        return None, None
    # 항상 같은 창을 쓴다(고정 간격 추출). 매번 다른 표본이면 비교가 성립하지 않는다.
    step = max(1, len(files) // n)
    files = files[::step][:n]
    lats, acts = [], []
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        lats.append(d["latent"].float())
        acts.append(d["act"].float())
    return torch.stack(lats), torch.stack(acts)


def to_delta12(act: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    """(B,16,6) 절대 z 액션 -> (B,16,12) = [절대 z, (z_t - z_0)/스케일].

    왜: 이 과제에서 모델이 조건으로 받아야 하는 것은 **어디서 어디로 움직이는가**인데,
    절대 관절각만 주면 그 정보가 시작 자세와 뒤섞인다. 시작 자세는 초기 프레임 이미지가
    이미 담고 있으므로, 액션 쪽에는 **자세 기준에 불변인 표현**(상대 변화량)을 함께 주는 것이
    자연스럽다. 로봇 정책에서 상대/델타 액션 표현은 표준적인 선택이다.

    델타를 **덧붙이기만** 한다(절대를 빼지 않는다). 첫 프레임 이미지가 실제 자세를 보여주므로
    절대 정보를 제거하면 두 조건이 모순되기 때문이다. 신규 채널은 0 초기화라 학습 시작
    시점에는 챔피언과 완전히 동일하게 동작한다(verify_delta_warmstart.py, 최대차 0.000e+00).

    스케일은 train 잠재에서만 계산한다(compute_delta_stats.py -> delta_action_stats.json).
    """
    d = (act - act[:, :1]) / scale.to(act.dtype).to(act.device)
    return torch.cat([act, d], dim=-1)
LATENT_MANIFEST = Path(os.environ.get(
    "COSMOS_LATENT_MANIFEST",
    str(Path(__file__).resolve().parent / "latent_manifest.parquet"),
))
BATCH = 1
ACCUM = 8
LR = 5e-5  # phase2: 과적합 억제 위해 인하
UNCOND_P = 0.0  # CFG 폐기로 드롭아웃 제거 (Codex 감사 지적 반영 — 조건 앵커 낭비 방지)
SHIFT = 5.0
LOG_EVERY = 20
SAVE_EVERY = 500


class LatentClipDataset(torch.utils.data.Dataset):
    """사전계산 잠재 창 로더. 반환: latent (16,5,40,64) bf16, act (16,6) fp32"""

    def __init__(self):
        import pandas as pd

        if LATENT_MANIFEST.exists():
            mf = pd.read_parquet(LATENT_MANIFEST)
        else:
            rows = []
            for p in sorted(LATENT_DIR.glob("*.pt")):
                d = torch.load(p, map_location="cpu", weights_only=False)
                rows.append({"path": p.name, "motion": float(d["motion"])})
            mf = pd.DataFrame(rows)
            mf.to_parquet(LATENT_MANIFEST, index=False)
        self.files = mf.path.tolist()
        self.motion = mf.motion.to_numpy()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(LATENT_DIR / self.files[idx], map_location="cpu", weights_only=False)
        return {"latent": d["latent"], "act": d["act"].float()}


class ResumeOnceSampler(Sampler[int]):
    """Skip a deterministic sampler prefix only on its first iteration.

    Checkpoints store the micro-step but not WeightedRandomSampler's generator
    state.  Rebuilding it with the same seed and skipping ``offset`` indices
    restores the exact position for the single no-replacement pass.  Subsequent
    epochs, if explicitly requested, use the complete newly sampled order.
    """

    def __init__(self, sampler: Sampler[int], offset: int):
        self.sampler = sampler
        self.offset = max(0, int(offset))
        self._first = True

    def __iter__(self):
        import itertools

        offset = self.offset if self._first else 0
        self._first = False
        return itertools.islice(iter(self.sampler), offset, None)

    def __len__(self) -> int:
        return max(0, len(self.sampler) - (self.offset if self._first else 0))


class DistributedResumeSampler(Sampler[int]):
    """Shard one deterministic global sampler order across DDP ranks.

    Every rank reconstructs the same ``WeightedRandomSampler`` order from the
    same explicit generator seed.  Rank ``r`` consumes positions
    ``r, r + world_size, ...`` after an optional *global* resume offset.  This
    preserves the single no-replacement pass and makes an 8-GPU resume skip
    exactly the windows already consumed by all eight ranks, rather than eight
    independent prefixes.
    """

    def __init__(self, sampler: Sampler[int], rank: int, world_size: int, offset: int = 0):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid DDP shard rank={rank}, world_size={world_size}")
        self.sampler = sampler
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.offset = max(0, int(offset))
        self._first = True

    def __iter__(self):
        import itertools

        offset = self.offset if self._first else 0
        self._first = False
        global_order = itertools.islice(iter(self.sampler), offset, None)
        return itertools.islice(global_order, self.rank, None, self.world_size)

    def __len__(self) -> int:
        offset = self.offset if self._first else 0
        remaining = max(0, len(self.sampler) - offset)
        if remaining <= self.rank:
            return 0
        return 1 + (remaining - self.rank - 1) // self.world_size


def main() -> None:
    from cosmos_predict2._src.predict2.conditioner import DataType
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE

    import os as _os

    rank = int(_os.environ.get("RANK", "0"))
    local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    # Sampler ordering has its own explicit CPU generator.  Per-rank CUDA seeds
    # intentionally differ so different windows do not receive identical noise.
    torch.manual_seed(20260807 + rank)
    # 액션을 크로스어텐션 컨텍스트 토큰으로도 넣는 구조 (공간 접지). 게이트 zero-init 이라
    # 챔피언 가중치를 얹으면 초기 출력이 사실상 동일하다 — action_token_dit.py 참조.
    # v3: 자체 판독기(latent IDM) 보조손실. 이 판독기는 **대회 train 데이터의 잠재와 정답
    # 액션만으로 우리가 직접 학습한 모델**이다(train_latent_idm2.py). 제출킷의 모델·코드·
    # checkpoint 와는 무관하며, 킷에서 유래한 가중치나 특징값을 일절 쓰지 않는다.
    aux = _os.environ.get("AUX") == "1"
    w_aux = float(_os.environ.get("W_AUX", "0.3"))
    p_aux = float(_os.environ.get("P_AUX", "0.3"))
    # AUX_WARMUP: 벌점 가중치를 0에서 w_aux 까지 이 노출수에 걸쳐 선형으로 올린다.
    #   처음부터 학습할 때 초반 수천 스텝은 LoRA 가 0 초기화라 모델이 아직 로봇 장면을 못 만든다.
    #   그 구간의 판독기 채점은 신호가 아니라 잡음이라, 바로 0.3 을 걸면 기울기가 흐려진다.
    #   0(기본)이면 램프 없음 = 기존 동작과 완전히 동일하다.
    aux_warmup = int(_os.environ.get("AUX_WARMUP", "0"))
    # v6 mini-DRaFT: K스텝 오일러 전개 전체를 미분해 최종 복원물에 판독기 보상+실영상 앵커
    draft = _os.environ.get("DRAFT") == "1"
    draft_k = int(_os.environ.get("DRAFT_K", "3"))
    draft_p = float(_os.environ.get("DRAFT_P", "0.5"))
    w_r = float(_os.environ.get("W_R", "1.0"))
    w_anchor = float(_os.environ.get("W_ANCHOR", "1.0"))
    # 액션 드롭아웃: 확률 p로 액션을 0벡터로 바꿔 학습한다. 0벡터가 "액션 없음"의 학습된 표현이
    # 되므로 추론 때 액션 CFG(validate.generate 의 action_guidance)의 널 분기가 분포 안에 놓인다.
    # 드롭아웃 없이 쓰는 액션 CFG는 널 분기가 미학습 입력이라 증폭 방향을 신뢰할 수 없다.
    act_drop = float(_os.environ.get("ACT_DROP", "0"))
    # 관절별 보상 가중 (기본 균등). 근거: 생성물에서 wrist_roll 오차가 정답 대비 14.6× 열화 — 생성 사각지대
    joint_w = torch.tensor(
        [float(x) for x in _os.environ.get("JOINT_W", "1,1,1,1,1,1").split(",")], dtype=torch.float32
    )
    joint_w = (joint_w / joint_w.mean()).view(1, 1, 6)
    if draft:
        aux = True  # 판독기 로딩 재사용
    lora_r = int(_os.environ.get("LORA_R", "32"))  # v7: r64 용량 확장 (제로패드 웜스타트)
    # DELTA=1: 액션 조건에 (z_t - z_0) 채널을 덧붙인다. to_delta12 주석에 근거 측정치.
    delta = _os.environ.get("DELTA") == "1"
    delta_scale = None
    if delta:
        import json as _json

        if not DELTA_STATS.exists():
            raise SystemExit(f"delta_action_stats.json 이 없다 — compute_delta_stats.py 를 먼저 돌려라")
        delta_scale = torch.tensor(_json.loads(DELTA_STATS.read_text(encoding="utf-8"))["scale"],
                                   dtype=torch.float32).view(1, 1, 6)
        print(f"DELTA ON — 액션 축 6->12, 델타 스케일 "
              f"{[round(float(x), 3) for x in delta_scale.view(-1)]}", flush=True)
    # 절대 채널 오프셋 증강 — **불변성 학습**이 목적이다.
    #
    # 원리: 이 과제의 조건은 (초기 프레임 이미지, 액션 시퀀스) 쌍인데, 팔의 절대 자세는
    # 이미지가 이미 담고 있다. 액션의 절대 성분은 그와 중복이고, 정작 필요한 것은 **움직임**이다.
    # 따라서 좋은 조건 표현이라면 액션 절대값에 임의의 상수 오프셋이 실려도 출력이 흔들리지
    # 않아야 한다. 그 불변성을 직접 가르치는 것이 이 증강이다.
    #
    # (z_t - z_0) 는 모든 z 에 같은 상수를 더해도 정확히 불변이므로(verify_abs_aug.py 로 확인,
    # 변화 0.000e+00), 절대 채널에만 오프셋을 넣으면 "절대는 흔들리고 델타는 그대로"가 된다.
    # 모델은 이미지+델타로 우회하는 법을 배운다.
    #
    # 폭 2.0 의 근거: 액션 정규화 z=(a-mean)/std 의 mean/std 가 **train 통계**
    # (data/train/so100_action_statistics.json, count 974,661)이므로 z 는 train 위에서 정의상
    # 단위분산이다. 따라서 U(-2,2) = **±2 train 표준편차**로, 표준적인 증강 폭 규칙에서
    # 그대로 나온다. 평가 데이터를 참조하지 않고 결정된다.
    #
    # 필요성 근거(측정): 델타 채널만 추가하면 모델이 거의 쓰지 않는다. v12 실측에서
    # 델타/절대 가중치 비율이 30,000 노출에도 0.0245 에 그쳤다. 학습 분포 안에서는 절대
    # 채널만으로 손실이 충분히 내려가 델타를 쓸 압력이 없기 때문이다. 증강이 그 압력을 만든다.
    abs_aug = float(_os.environ.get("ABS_AUG", "0"))        # 적용 확률
    abs_aug_scale = float(_os.environ.get("ABS_AUG_SCALE", "2.0"))  # train z 표준편차 배수
    if abs_aug > 0 and not delta:
        raise SystemExit("ABS_AUG 는 DELTA=1 에서만 의미가 있다(델타가 있어야 대체 경로가 생긴다)")
    if abs_aug > 0:
        print(f"ABS_AUG ON — 확률 {abs_aug:g} 로 절대 채널에 U(-{abs_aug_scale:g},{abs_aug_scale:g}) "
              f"오프셋 = ±{abs_aug_scale:g} train 표준편차 (델타 채널은 불변). "
              f"목적: 절대 자세 오프셋에 대한 불변성", flush=True)
    # 코랩(A100 40GB) 전용 모드들
    fullft = _os.environ.get("FULLFT") == "1"      # peft 없이 전 파라미터 FT (bnb 8-bit Adam 권장)
    atok = _os.environ.get("ATOK") == "1"
    actx_only = atok and _os.environ.get("ACTX_ONLY", "1") == "1"
    # One continuous multi-objective recipe. Every auxiliary target is computed
    # from the paired ground-truth latent; no self-trained IDM/evaluator is used.
    unified_v2 = _os.environ.get("UNIFIED_V2") == "1"
    unified_warmup = int(_os.environ.get("UNIFIED_WARMUP", "10000"))
    w_x0 = float(_os.environ.get("W_X0", "0.25"))
    w_temporal = float(_os.environ.get("W_TEMPORAL", "0.10"))
    w_preserve = float(_os.environ.get("W_PRESERVE", "0.10"))
    w_counterfactual = float(_os.environ.get("W_COUNTERFACTUAL", "0.05"))
    counterfactual_p = float(_os.environ.get("COUNTERFACTUAL_P", "0.25"))
    counterfactual_margin = float(_os.environ.get("COUNTERFACTUAL_MARGIN", "0.02"))
    if atok and delta:
        raise SystemExit("ATOK=1 과 DELTA=1 은 함께 쓰지 않는다 — 기존 AdaLN 6D 경로를 보존해야 한다")
    if atok and fullft:
        raise SystemExit("ATOK=1 은 챔피언 동결 실험이다 — FULLFT=1 과 함께 쓰지 않는다")
    if unified_v2 and (aux or draft):
        raise SystemExit("UNIFIED_V2 는 자체 IDM/DRaFT를 사용하지 않는다 — AUX/DRAFT를 끄라")
    if unified_v2:
        print(
            "UNIFIED_V2 ON — flow + paired x0 + temporal + stable-region preservation "
            "+ action counterfactual ranking; one continuous run",
            flush=True,
        )
        print(
            f"weights x0={w_x0:g} temporal={w_temporal:g} preserve={w_preserve:g} "
            f"counterfactual={w_counterfactual:g} p={counterfactual_p:g} "
            f"warmup={unified_warmup}",
            flush=True,
        )
    pixanchor = _os.environ.get("PIXANCHOR") == "1"  # DRaFT에 DINO 픽셀 앵커 (VAE 디코딩 경유 — 16GB 불가, A100용)
    w_pix = float(_os.environ.get("W_PIX", "1.0"))
    save_every = int(_os.environ.get("SAVE_EVERY", str(SAVE_EVERY)))
    run_name = _os.environ.get(
        "RUN_NAME",
        "v6_draft" if draft else ("v3_aux" if aux else "v1_lora"),
    )
    out_dir = Path(__file__).resolve().parent / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "run_name": run_name,
        "architecture": "cosmos_action_chunk_adaln_plus_16_action_tokens" if atok else "cosmos_action_chunk_adaln",
        "activation_checkpointing": "none_for_action_token_graph" if atok else "stock_mm_only",
        "unified_v2": unified_v2,
        "actx_only": actx_only,
        "lora_rank": lora_r,
        "loss": {
            "flow": 1.0,
            "x0": w_x0 if unified_v2 else 0.0,
            "temporal": w_temporal if unified_v2 else 0.0,
            "preserve": w_preserve if unified_v2 else 0.0,
            "counterfactual": w_counterfactual if unified_v2 else 0.0,
            "counterfactual_probability": counterfactual_p if unified_v2 else 0.0,
            "counterfactual_margin": counterfactual_margin if unified_v2 else 0.0,
            "warmup": unified_warmup if unified_v2 else 0,
        },
        "uses_external_evaluator_loss": bool(aux or draft),
        "sampler_seed": int(_os.environ.get("SAMPLER_SEED", "1234")),
        "micro_batch": int(_os.environ.get("BATCH", "1")) if unified_v2 else None,
        "effective_batch": int(_os.environ.get("EFFECTIVE_BATCH", "8")) if unified_v2 else None,
        "world_size": world_size,
    }
    recipe_path = out_dir / "run_config.json"
    if is_main:
        if recipe_path.exists():
            previous_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            if previous_recipe != recipe:
                # A pre-DDP dry start could leave only run_config.json.  With no
                # checkpoint there is no trajectory to protect, so migrate it.
                if not (out_dir / "latest.pt").exists():
                    print(f"checkpoint 없는 run_config를 {world_size}-GPU 설정으로 갱신", flush=True)
                    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
                else:
                    raise SystemExit(f"기존 run_config와 현재 설정이 다르다: {recipe_path}")
        else:
            recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()
    # All ranks validate the file written by rank 0 before allocating the model.
    if json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
        raise SystemExit(f"run_config 동기화 실패: {recipe_path}")

    # DELTA=1 이면 액션 축을 6 -> 12 로 늘린다([절대 z, 델타]). 모델 코드 수정은 필요 없다 —
    # 이 클래스는 in_features = action_dim x temporal_compression_ratio(=4) 로 임베더를 만들고
    # forward 에서 (B,16,D) -> (B,4,4D) 로 재배열하므로 action_dim 만 바꾸면 맞아떨어진다.
    # (챔피언은 DELTA 를 쓰지 않는다. 6축 그대로다.)
    if atok:
        from action_token_dit import build_action_token_dit

        net = build_action_token_dit(action_dim=6)
        print("ATOK ON — 기존 AdaLN 유지 + 16개 spatial action token (zero-init gate)", flush=True)
    else:
        net = build_dit(action_dim=12 if delta else 6)
    net.timestep_scale = 0.001
    sd = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("net."): v for k, v in sd.items()}
    # 액션 임베더 fc1은 28→24차원이라 사전학습 가중치 제외(재초기화 유지)
    sd = {k: v for k, v in sd.items() if "action_embedder" not in k or "fc2" in k}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"load: missing={len(missing)} (embedder fc1 포함 정상) unexpected={len(unexpected)}")
    net = net.to(device, torch.bfloat16)

    if fullft:
        # 전 파라미터 FT: LoRA 챔피언 ckpt를 가중치에 병합(웜스타트) 후 전체 학습
        for p in net.parameters():
            p.requires_grad_(True)
        merge_from = _os.environ.get("MERGE_FROM", "")
        if merge_from:
            ck_m = torch.load(merge_from, map_location="cpu", weights_only=False)
            st = ck_m["state"]
            params = dict(net.named_parameters())
            merged = 0
            for k, v in st.items():
                if "lora_A" not in k:
                    continue
                base_key = k.replace("base_model.model.", "").replace(".lora_A.default.weight", ".weight")
                Bm = st[k.replace("lora_A", "lora_B")].float()
                tgt = params.get(base_key)
                if tgt is None:
                    continue
                tgt.data.add_((Bm @ v.float()).to(tgt.dtype).to(tgt.device))  # alpha/r=1
                merged += 1
            emb = {k.replace("base_model.model.", ""): v for k, v in st.items() if "lora_" not in k}
            net.load_state_dict(emb, strict=False)
            print(f"FULLFT merge warm-start: {merged} lora pairs from {Path(merge_from).name}", flush=True)
    else:
        # cosmos 모델 빌드 완료 후 TE 스텁 제거 → peft가 TE 통합 경로로 빠지지 않게 함
        for mod_name in [m for m in sys.modules if m.startswith("transformer_engine")]:
            del sys.modules[mod_name]
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r,  # alpha/r 비율 유지 (기존 32/32)
            target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2"],
            lora_dropout=0.0,
        )
        net = get_peft_model(net, lora_cfg)
        for name, p in net.named_parameters():
            if "action_embedder" in name or "block_action" in name or "action_ctx" in name:
                p.requires_grad_(True)

    resume_step = 0
    latest = out_dir / "latest.pt"
    widen_from = _os.environ.get("WIDEN_FROM", "")
    if fullft:
        widen_from = ""  # fullft는 MERGE_FROM 웜스타트 사용
    if _os.environ.get("SCRATCH") == "1" and not latest.exists():
        # 처음부터 학습. 아래 자동 폴백(aux면 v1_lora, draft면 v3_aux)이 조용히 걸리면
        # "처음부터"가 아니게 되는데, 로그가 없으면 그 사실이 몇 시간 뒤에나 드러난다.
        print("SCRATCH=1: 웜스타트 없음 — 사전학습 원본 + LoRA 0초기화에서 시작", flush=True)
    elif not latest.exists() and widen_from:
        # r32 체크포인트를 제로패드로 r64에 이식: A는 행, B는 열 패딩 → BA 곱 불변(기능 보존 웜스타트)
        ck = torch.load(widen_from, map_location="cpu", weights_only=False)
        padded = {}
        for k, v in ck["state"].items():
            if "lora_A" in k and v.shape[0] < lora_r:
                padded[k] = torch.cat([v, torch.zeros(lora_r - v.shape[0], v.shape[1], dtype=v.dtype)], dim=0)
            elif "lora_B" in k and v.shape[1] < lora_r:
                padded[k] = torch.cat([v, torch.zeros(v.shape[0], lora_r - v.shape[1], dtype=v.dtype)], dim=1)
            else:
                padded[k] = v
        missing_w, unexpected_w = net.load_state_dict(padded, strict=False)
        resume_step = int(ck["step"])
        print(f"widened r{lora_r} warm-start from {Path(widen_from).name} step {resume_step}", flush=True)
    elif not latest.exists() and _os.environ.get("RESUME_FROM"):
        # RESUME_FROM 은 레시피와 무관하게 최우선으로 존중한다. 예전에는 draft 분기에서만
        # 읽어서, AUX 로 돌리면 조용히 무시되고 **사전학습 원본에서 시작**했다.
        # 웜스타트 실패는 로그가 없으면 몇 시간 뒤에나 드러난다.
        latest = Path(_os.environ["RESUME_FROM"])
        if not latest.exists():
            raise SystemExit(f"RESUME_FROM 이 가리키는 파일이 없다: {latest}")
    elif not latest.exists() and draft:
        latest = Path(__file__).resolve().parent / "runs" / "v3_aux" / "latest.pt"
    elif not latest.exists() and aux:
        latest = Path(__file__).resolve().parent / "runs" / "v1_lora" / "latest.pt"  # v1 35k에서 출발
    opt_resume_path = None
    skip_opt_resume = False
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if delta:
            # 액션 축 6->12 웜스타트. 임베더 입력은 (t d) 평탄화라 열 순서가
            #   기존 24열: [t0_j0..t0_j5, t1_j0.., t2_.., t3_..]      (index = t*6 + j)
            #   신규 48열: [t0_abs(6), t0_델타(6), t1_abs(6), ...]     (index = t*12 + j)
            # 로 **엇갈린다.** 앞 24열에 그냥 복사하면 t1 이후가 전부 어긋난 채로
            # 몇 시간을 학습하게 된다. 델타 열은 0 으로 둬 초기 출력이 챔피언과 동일해야 한다.
            n_exp = 0
            for k in list(ck["state"]):
                if "action_embedder" in k and k.endswith("fc1.weight"):
                    old = ck["state"][k]                       # (H, 24)
                    if old.shape[1] == 24:
                        new = torch.zeros(old.shape[0], 48, dtype=old.dtype)
                        for t in range(4):
                            new[:, t * 12: t * 12 + 6] = old[:, t * 6: t * 6 + 6]
                        ck["state"][k] = new
                        n_exp += 1
            if n_exp:
                print(f"DELTA 웜스타트: 임베더 fc1 {n_exp}개를 24->48 열로 확장 "
                      f"(절대 열 복사, 델타 열 0 초기화 → 초기 출력은 챔피언과 동일)", flush=True)
                # 확장했으면 옵티마이저 상태는 쓸 수 없다. 그 파일의 fc1 모멘트는 24열이라
                # 48열 파라미터에 못 얹는다. 조용히 예외로 흘리지 말고 명시적으로 건너뛴다.
                skip_opt_resume = True
        # strict=False 는 키 접두사가 어긋나면 아무것도 안 붙이고 조용히 통과한다.
        # 웜스타트가 실패한 채로 몇 시간을 학습하면 그 결과는 통째로 무의미해진다.
        own = set(net.state_dict())
        matched = sum(1 for k in ck["state"] if k in own)
        net.load_state_dict(ck["state"], strict=False)
        resume_step = ck["step"]
        print(f"resumed from step {resume_step} ({latest.name}) — "
              f"체크포인트 {len(ck['state'])}개 중 {matched}개 적용", flush=True)
        if matched < max(1, len(ck["state"]) // 2):
            raise SystemExit(f"[중단] 웜스타트 키 불일치 ({matched}/{len(ck['state'])} 적용)")
        # 옵티마이저 상태는 같은 폴더의 latest_opt.pt 에만 있고 스텝이 맞을 때만 쓴다.
        # 다른 스텝의 Adam 모멘트를 얹으면 재개 전이보다 더 나쁘다.
        cand = latest.parent / "latest_opt.pt"
        if cand.exists() and not skip_opt_resume:
            opt_resume_path = cand
        elif skip_opt_resume:
            print("DELTA 확장으로 파라미터 모양이 바뀌었다 — 옵티마이저 상태는 0에서 시작한다", flush=True)
    if actx_only:
        # LR=0만으로도 수치상 업데이트는 막을 수 있지만, gradient와 Adam moments는 여전히
        # 113M warm parameters에 할당된다. requires_grad까지 끄는 것이 메모리와 계약 양쪽에서
        # 정확한 champion-preserving 실험이다.
        for name, p in net.named_parameters():
            p.requires_grad_("action_ctx" in name)
        opt_resume_path = None
        frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
        scratch_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"ACTX_ONLY ON — champion warm path frozen {frozen/1e6:.1f}M, "
              f"new action context {scratch_count/1e6:.3f}M", flush=True)
        if scratch_count != 1_144_832:
            raise SystemExit(f"ATOK 신규 파라미터 계약 불일치: expected 1,144,832 got {scratch_count:,}")
    trainable = [p for p in net.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()  # LoRA/임베더는 fp32 유지
    print("trainable params: %.1fM" % (sum(p.numel() for p in trainable) / 1e6))

    text_emb = torch.load(TEXT_EMB, map_location="cpu", weights_only=False).to(device, torch.bfloat16)

    import os

    latent_mode = LATENT_DIR.exists() and any(LATENT_DIR.glob("*.pt"))
    batch = int(os.environ.get("BATCH", 2 if latent_mode else 1))
    effective_batch = int(_os.environ.get("EFFECTIVE_BATCH", str(BATCH * ACCUM)))
    global_micro_batch = batch * world_size
    if effective_batch < global_micro_batch or effective_batch % global_micro_batch:
        raise SystemExit(
            f"EFFECTIVE_BATCH={effective_batch} must be an integer multiple of "
            f"BATCH({batch}) * WORLD_SIZE({world_size})={global_micro_batch}"
        )
    accum = effective_batch // global_micro_batch
    if is_main:
        print(
            f"DDP world_size={world_size}, per_gpu_batch={batch}, "
            f"global_micro_batch={global_micro_batch}, accum={accum}, "
            f"effective_batch={effective_batch}",
            flush=True,
        )
    # 설정이 실제로 켜졌는지 로그로 남긴다. ACT_DROP 이 0인 채로 돌면 이 실행은 그냥
    # "DRaFT 연장"이 되는데(기록상 후퇴 조건), 로그가 없으면 결과를 잘못 해석하게 된다.
    if act_drop > 0:
        eligible = (1 - draft_p) * (1 - p_aux) if draft else (1 - p_aux if aux else 1.0)
        print(f"ACT_DROP={act_drop} — 흐름정합 비-aux 스텝({eligible:.0%})에만 적용 "
              f"→ 전체 스텝의 약 {act_drop * eligible:.1%} 가 널 액션을 본다", flush=True)
    else:
        print("ACT_DROP=0 (액션 드롭아웃 없음)", flush=True)
    lidm = None
    if aux:
        # 잠재공간 대리 채점기(동결): 디코드 없이 잠재→액션 벌점. 픽셀 경로는 OOM으로 기각됨
        # AUX_JUDGE=v2면 판독기 v2(차분+트랜스포머, val 0.17대)를 보상으로 사용
        if _os.environ.get("AUX_JUDGE", "v1") == "v2":
            from train_latent_idm2 import LatentIDM2 as _LJ, OUT as LIDM_PATH
        else:
            from train_latent_idm import LatentIDM as _LJ, OUT as LIDM_PATH

        lidm = _LJ().to(device).eval()
        ck_l = torch.load(LIDM_PATH, map_location="cpu", weights_only=False)
        lidm.load_state_dict(ck_l["state"])
        for p in lidm.parameters():
            p.requires_grad_(False)
        print(f"AUX ON ({_LJ.__name__}, val {ck_l['val_mae']:.4f}): w={w_aux} p={p_aux}", flush=True)

    if latent_mode:
        # PIXANCHOR면 디코딩용 VAE + DINO 로드 (동결)
        vae = WanVAE(z_dim=16, vae_pth=str(CKPT_DIR / "tokenizer.pth"), dtype=torch.bfloat16, device="cuda") if pixanchor else None
        dino = None
        if pixanchor:
            import timm

            dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0)
            dino = dino.to(device).eval()
            for p in dino.parameters():
                p.requires_grad_(False)
            _im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            _im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        ds = LatentClipDataset()
        p50 = max(1e-6, float(np.median(ds.motion)))
        # 움직임 가중의 세기. 1.0 = 기존(저움직임 창을 최대 10배 덜 봄), 0.0 = 균등.
        # 기존 값 1.0 은 "움직임이 클수록 학습 신호가 많다"는 가정이었고 LB 로 검증된 적이 없다.
        # 평가셋 216개의 액션 움직임을 학습 분포에 대보면 중앙값이 41.5분위에 있고
        # 학습 상위 5분위에는 1.9% 뿐이다. 즉 가중치가 **평가에 없는 구간**을 키우고 있었다.
        # (평가 '조건 입력'의 분포만 본 것이다 — 정답 영상도 채점기도 쓰지 않았다.)
        motion_pow = float(_os.environ.get("MOTION_POW", "1.0"))
        w_np = np.power((ds.motion / p50).clip(0.3, 3.0), motion_pow)
        if _os.environ.get("CLEAN") == "1":  # 데이터 정제: 판독기 전수채점 기반 불량 에피소드 감량/배제
            import pandas as _pd

            # 기본을 r1 으로 둔다. 예전 기본이던 episode_weight.parquet 은 10735개 중 63개(0.59%)만
            # 감량하는 사실상 무효 필터였고, 그 위에서 "CLEAN 무효과" 판정이 나왔다.
            # r1 은 322개(3.0%)를 감량한다. CLEAN_FILE 로 파일명을 바꿔 비교할 수 있다.
            cf = _os.environ.get("CLEAN_FILE", "episode_weight_r1.parquet")
            ew = _pd.read_parquet(Path(__file__).resolve().parent / "runs" / "data_clean" / cf)
            epw = dict(zip(ew.index, ew.weight.astype(float)))
            w_np = w_np * np.array([epw.get(f.split("_")[0], 1.0) for f in ds.files])
            n_dn = int((ew.weight.astype(float) < 1).sum())
            print(f"clean-weighted [{cf}]: 에피소드 {n_dn}개 감량 → 윈도우 {(w_np == 0).sum()}개 제외", flush=True)
        w = torch.from_numpy(w_np).double()
        # 샘플러 시드. 이게 없으면 같은 설정으로 다시 돌려도 다른 창을 뽑아서, 학습 A/B 의
        # 차이가 "바꾼 것 때문인지 다른 창을 뽑아서인지" 구분되지 않는다. 챔피언까지의 모든
        # 학습에 이 시드가 없었고, 그래서 8번의 가중치 실험 결과를 원인까지 가르지 못했다.
        smp_seed = int(_os.environ.get("SAMPLER_SEED", "1234"))
        g_smp = torch.Generator().manual_seed(smp_seed)
        # NOREPL=1: 중복 없이 전 창을 정확히 한 번씩 뽑는다(커버리지 100%).
        #   이때 가중치는 순서에만 관여하고 커버리지에는 영향이 없다 — 움직임 편향이 구조적으로 사라진다.
        #   정제로 가중치 0 이 된 창(에피소드 39개 = 창 290개)은 뽑을 수 없으므로 뽑기 수를 그만큼 줄인다.
        #   줄이지 않으면 torch.multinomial 이 "not enough non-negative category" 로 죽는다.
        norepl = _os.environ.get("NOREPL") == "1"
        n_pos = int((w_np > 0).sum())
        n_draw = n_pos if norepl else len(ds)
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=n_draw, replacement=not norepl, generator=g_smp
        )
        if unified_v2 and norepl:
            resume_examples = resume_step * global_micro_batch
            if resume_examples >= n_draw:
                raise SystemExit(
                    f"UNIFIED_V2 one-pass checkpoint consumed {resume_examples} >= available windows {n_draw}; "
                    "use a new run name for another pass"
                )
            if distributed:
                sampler = DistributedResumeSampler(
                    sampler, rank=rank, world_size=world_size, offset=resume_examples
                )
            elif resume_examples:
                sampler = ResumeOnceSampler(sampler, resume_examples)
            if resume_examples and is_main:
                print(
                    f"sampler resume: deterministic global prefix {resume_examples:,} windows skipped",
                    flush=True,
                )
        elif distributed:
            sampler = DistributedResumeSampler(sampler, rank=rank, world_size=world_size)
        loader = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=4, persistent_workers=True, drop_last=True)
        if is_main:
            print(f"latent mode ON: {len(ds)} windows, motion-weighted (p50={p50:.4f}, "
                  f"MOTION_POW={motion_pow:g}, 가중치 {w_np[w_np > 0].min():.2f}~{w_np.max():.2f}배)", flush=True)
            print(f"sampler: {'전수 1회(중복 없음)' if norepl else '중복 허용'} "
                  f"seed={smp_seed} 전역 뽑기 {n_draw:,}회/epoch, rank별 분할", flush=True)
    else:
        vae = WanVAE(z_dim=16, vae_pth=str(CKPT_DIR / "tokenizer.pth"), dtype=torch.bfloat16, device="cuda")
        ds = SO100ClipDataset("train")
        motion_path = Path(__file__).resolve().parent / "episode_motion.parquet"
        if motion_path.exists():
            # 움직임 가중 샘플링: 정지 위주 클립 대신 동작 구간을 더 자주 학습 (train 통계만 사용)
            import pandas as pd

            mdf = pd.read_parquet(motion_path)
            motion = dict(zip(mdf.parquet, mdf.motion))
            ep_motion = ds.episodes.parquet.map(motion).fillna(0.0).to_numpy()
            p50 = max(1e-6, float(pd.Series(ep_motion).median()))
            ep_w = (ep_motion / p50).clip(0.3, 3.0)
            counts = np.diff(ds.cumsum).astype(int)
            clip_w = torch.from_numpy(np.repeat(ep_w, counts)).double()
            g_smp = torch.Generator().manual_seed(int(_os.environ.get("SAMPLER_SEED", "1234")))
            sampler = torch.utils.data.WeightedRandomSampler(
                clip_w, num_samples=len(ds), replacement=True, generator=g_smp
            )
            loader = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=4, persistent_workers=True, drop_last=True)
            print(f"motion-weighted sampling ON (p50={p50:.4f})", flush=True)
        else:
            loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
    _lr = float(_os.environ.get("LR", str(LR)))
    # 0에서 출발하는 신규 모듈(action_ctx)과 챔피언에서 웜스타트한 기존 가중치를 같은 LR로
    # 돌리면 신규 쪽이 사실상 학습되지 않는다. LR_NEW 로 따로 준다(기본 10배).
    scratch = [p for n, p in net.named_parameters() if p.requires_grad and "action_ctx" in n]
    warm = [p for n, p in net.named_parameters() if p.requires_grad and "action_ctx" not in n]
    lr_new = float(_os.environ.get("LR_NEW", str(_lr * 10)))
    groups = []
    if warm:
        groups.append({"params": warm, "lr": _lr})
    if scratch:
        groups.append({"params": scratch, "lr": lr_new})
        print(f"신규 파라미터 {sum(p.numel() for p in scratch)/1e6:.2f}M 에 LR {lr_new:g} "
              f"(기존 {sum(p.numel() for p in warm)/1e6:.1f}M 는 {_lr:g})", flush=True)
    if not groups:
        raise SystemExit("학습 가능한 파라미터가 없다")
    if fullft:
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(groups, lr=_lr, weight_decay=0.1)
            print("FULLFT: bitsandbytes AdamW8bit", flush=True)
        except ImportError:
            opt = torch.optim.AdamW(groups, lr=_lr, weight_decay=0.1)
            print("FULLFT: bnb 없음 — fp32 AdamW (메모리 주의)", flush=True)
    else:
        opt = torch.optim.AdamW(groups, lr=_lr, weight_decay=0.1)

    # Adam 모멘트 복원. 이게 없으면 재개할 때마다 1·2차 모멘트가 0에서 다시 쌓이면서
    # 수백 스텝 동안 실효 LR 이 튄다. 며칠짜리 학습에서 크래시가 한 번만 나도
    # 재현 실행(무중단 1회)과 궤적이 갈라진다. 실패해도 학습은 계속한다(예전 동작).
    if opt_resume_path is not None:
        try:
            ock = torch.load(opt_resume_path, map_location="cpu", weights_only=False)
            if int(ock.get("step", -1)) != int(resume_step):
                print(f"[경고] {opt_resume_path.name} 스텝 {ock.get('step')} != 가중치 스텝 "
                      f"{resume_step} — 옵티마이저 상태 무시", flush=True)
            else:
                opt.load_state_dict(ock["opt"])
                print(f"옵티마이저 상태 복원 (step {resume_step})", flush=True)
        except Exception as e:  # 그룹 구조가 바뀌면 여기로 온다
            print(f"[경고] 옵티마이저 상태 복원 실패 ({type(e).__name__}: {e}) — 0에서 시작", flush=True)

    # Wrap only after loading the unprefixed checkpoint and constructing the
    # optimizer.  ``raw_net`` remains the canonical object for checkpoint keys,
    # validation, and the optional second (wrong-action) forward.  The latter
    # must not call DDP twice before one backward pass.
    raw_net = net
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        net = DistributedDataParallel(
            raw_net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        if is_main:
            print(f"DDP ON ({world_size} GPUs): gradient average per optimizer update", flush=True)

    # --- val 손실 (과적합 감지) ---
    # 고정 창 · 고정 노이즈 · 고정 타임스텝으로 재야 체크포인트 간 차이가 잡음이 아니라
    # 실제 차이가 된다. 흐름정합 손실은 무작위 t 에서 재면 분산이 커서 추세가 안 보인다.
    val_n = int(_os.environ.get("VAL_N", "96"))
    val_every = int(_os.environ.get("VAL_EVERY", str(save_every)))
    val_lat, val_act = load_val_set(val_n) if val_n > 0 else (None, None)
    if val_lat is None:
        print("val 잠재 없음 — val 손실 측정 생략 (precompute_val_latents.py 를 먼저 돌려라)",
              flush=True)
    else:
        print(f"val 손실 측정 ON: 창 {len(val_lat)}개, {val_every} 스텝마다 "
              f"(에피소드 단위 분리, 고정 노이즈)", flush=True)

    @torch.no_grad()
    def eval_val() -> float:
        # Every rank evaluates the same fixed set concurrently.  Using the raw
        # module avoids altering DDP reducer state between training backwards.
        was_training = raw_net.training
        raw_net.eval()
        g = torch.Generator(device=device).manual_seed(20260804)  # 매 호출 동일한 노이즈
        tot, cnt = 0.0, 0
        for i in range(len(val_lat)):
            x0 = val_lat[i:i + 1].to(device)
            a6 = val_act[i:i + 1].to(device, torch.bfloat16)
            a = to_delta12(a6, delta_scale) if delta else a6
            Bv, Cv, Tv, Hv, Wv = x0.shape
            cmv = torch.zeros(Bv, 1, Tv, Hv, Wv, device=device)
            cmv[:, :, 0] = 1.0
            mcv = cmv.repeat(1, Cv, 1, 1, 1)
            for tv in (0.25, 0.5, 0.75):
                noise_v = torch.randn(x0.shape, device=device, generator=g)
                xt_v = (1 - tv) * x0 + tv * noise_v
                xt_v = x0 * mcv + xt_v * (1 - mcv)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v_v = raw_net(
                        x_B_C_T_H_W=xt_v.to(torch.bfloat16),
                        timesteps_B_T=torch.full((Bv, 1), tv * 1000, device=device,
                                                 dtype=torch.bfloat16),
                        crossattn_emb=text_emb.expand(Bv, -1, -1),
                        condition_video_input_mask_B_C_T_H_W=cmv.to(torch.bfloat16),
                        fps=torch.full((Bv,), 6, device=device, dtype=torch.bfloat16),
                        padding_mask=torch.zeros(Bv, 1, 320, 512, device=device,
                                                 dtype=torch.bfloat16),
                        data_type=DataType.VIDEO, action=a)
                tgt_v = noise_v - x0
                tot += float((((v_v.float() - tgt_v) ** 2) * (1 - mcv)).sum()
                             / (1 - mcv).sum())
                cnt += 1
        if was_training:
            raw_net.train()
        return tot / max(1, cnt)

    padding_mask = torch.zeros(batch, 1, 320, 512, device=device, dtype=torch.bfloat16)
    fps = torch.full((batch,), 6, device=device, dtype=torch.bfloat16)

    step, t0 = resume_step, time.time()
    # 종료 조건. 예전에는 `while True` 로 무한히 돌아서, 스크립트로 여러 작업을 이어붙이면
    # 뒤 작업이 영영 시작되지 않았다(2026-07-30 밤 실제로 겪음). MAX_STEPS 는 이 실행에서
    # 추가로 돌 스텝 수이고, STOP_AT 은 절대 스텝 번호다. 둘 다 없으면 예전처럼 무한히 돈다.
    max_steps = int(_os.environ.get("MAX_STEPS", "0"))
    stop_at = int(_os.environ.get("STOP_AT", "0")) or (resume_step + max_steps if max_steps else 0)
    print(f"시작 스텝 {resume_step}" + (f" → {stop_at} 에서 종료" if stop_at else " (종료 조건 없음)"),
          flush=True)
    aux_log: list[float] = []
    rf_log: list[float] = []
    draft_log: list[float] = []
    x0_log: list[float] = []
    temporal_log: list[float] = []
    preserve_log: list[float] = []
    counterfactual_log: list[float] = []
    net.train()
    while not (stop_at and step >= stop_at):
        for item in loader:
            act = item["act"].to(device, torch.bfloat16)  # (B,16,6) 절대 z
            # 모델에 넣는 조건. DELTA 면 (B,16,12). **판독기(lidm)와 보상은 6차원 절대 그대로**
            # 써야 한다 — 판독기는 6축 관절각을 예측하도록 학습됐다.
            act_cond = to_delta12(act, delta_scale) if delta else act
            if abs_aug > 0:
                # 클립마다 관절별 상수 오프셋. t 에 대해 상수여야 델타가 불변으로 남는다.
                # act(6차원 원본)는 건드리지 않는다 — 판독기 보상의 정답이기 때문이다.
                Bn = act_cond.shape[0]
                hit = (torch.rand(Bn, 1, 1, device=act_cond.device) < abs_aug).to(act_cond.dtype)
                off = (torch.rand(Bn, 1, 6, device=act_cond.device) * 2 - 1) * abs_aug_scale
                act_cond = torch.cat([act_cond[..., :6] + off * hit, act_cond[..., 6:]], dim=-1)
            if "latent" in item:
                x0 = item["latent"].to(device).float()  # (B,16,5,40,64)
            else:
                video = item["video"].to(device, torch.bfloat16)  # (B,3,17,320,512)
                with torch.no_grad():
                    x0 = vae.encode(video).float()  # (B,16,5,40,64)
            B, C, T, H, W = x0.shape
            cond_mask = torch.zeros(B, 1, T, H, W, device=device)
            cond_mask[:, :, 0] = 1.0
            mask_c = cond_mask.repeat(1, C, 1, 1, 1)

            draft_step = draft and (float(torch.rand(())) < draft_p)
            aux_step = (not draft_step) and aux and (float(torch.rand(())) < p_aux)
            if draft_step:
                # mini-DRaFT: 저노이즈 K스텝 오일러 전개 → 최종 복원물에 보상+앵커
                s0 = 0.35 + 0.30 * float(torch.rand(()))
                noise = torch.randn_like(x0)
                xcur = (1 - s0) * x0 + s0 * noise
                xcur = x0 * mask_c + xcur * (1 - mask_c)
                sig = torch.linspace(s0, 0.0, draft_k + 1, device=device)

                def _vcall(xin, tval):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = net(
                            x_B_C_T_H_W=xin.to(torch.bfloat16),
                            timesteps_B_T=tval.to(torch.bfloat16),
                            crossattn_emb=text_emb.expand(B, -1, -1),
                            condition_video_input_mask_B_C_T_H_W=cond_mask.to(torch.bfloat16),
                            fps=fps,
                            padding_mask=padding_mask,
                            data_type=DataType.VIDEO,
                            action=act_cond,
                        )
                    return out.float()

                # DRaFT-1: 앞 스텝은 무기울기 전개, 마지막 스텝만 미분 (16GB 제약 — 전체 미분은 29GB OOM 실측)
                for i in range(draft_k):
                    tv = torch.full((B, 1), float(sig[i]) * 1000.0, device=device)
                    if i < draft_k - 1:
                        with torch.no_grad():
                            v_i = _vcall(xcur, tv)
                    else:
                        # 마지막 스텝만 일반 경로와 동일한 직접 호출로 미분 (외부 checkpoint는 내부 SAC와 충돌해 22GB)
                        v_i = _vcall(xcur.detach(), tv)
                    xcur = xcur - (sig[i] - sig[i + 1]) * v_i
                    xcur = x0 * mask_c + xcur * (1 - mask_c)
                reward = torch.mean(torch.abs(lidm(xcur) - act.float()) * joint_w.to(device))
                anchor = torch.mean(torch.abs(xcur - x0))
                total = w_r * reward + w_anchor * anchor
                if pixanchor:
                    # DINO 픽셀 앵커: 디코딩된 생성 프레임 vs 실영상(VAE 왕복) 프레임의 DINO 거리 — 시각 항목 정조준
                    dec_g = vae.decode(xcur.to(torch.bfloat16)).float()
                    with torch.no_grad():
                        dec_r = vae.decode(x0.to(torch.bfloat16)).float()
                    idxf = [3, 8, 12, 16]  # 표류가 생기는 후반 프레임 포함

                    def _dfeat(dec):
                        xf = (dec[0, :, idxf].permute(1, 0, 2, 3).clamp(-1, 1) + 1) / 2
                        xf = torch.nn.functional.interpolate(xf, size=(224, 224), mode="bilinear", align_corners=False)
                        return torch.nn.functional.normalize(dino((xf - _im_mean) / _im_std), dim=-1)

                    fg = _dfeat(dec_g)
                    with torch.no_grad():
                        frf = _dfeat(dec_r)
                    pix = (1 - (fg * frf).sum(-1)).mean()
                    total = total + w_pix * pix
                rf_loss = reward  # 로깅 호환
                aux_log.append(float(anchor))
            else:
                if aux_step:
                    # 벌점 스텝: x0 복원이 유의미한 저노이즈 구간에서 샘플링, 조건 유지
                    t = torch.rand(B, device=device) * 0.40 + 0.05
                else:
                    t_raw = torch.sigmoid(torch.randn(B, device=device))
                    t = SHIFT * t_raw / (1 + (SHIFT - 1) * t_raw)  # (B,) sigma in (0,1)
                noise = torch.randn_like(x0)
                sigma = t.view(B, 1, 1, 1, 1)
                xt = (1 - sigma) * x0 + sigma * noise
                # CFG 드롭아웃: 일부 샘플은 조건 프레임을 0으로 (공식 uncond 경로와 동일)
                cond_src = x0 if (aux_step or torch.rand(()) >= UNCOND_P) else torch.zeros_like(x0)
                xt = cond_src * mask_c + xt * (1 - mask_c)  # 조건 프레임은 클린 잠재(또는 0)
                v_target = noise - x0
                # 액션 드롭아웃은 흐름정합 스텝에서만. aux 스텝은 act 가 벌점의 정답이라
                # 입력을 널로 두면 "없는 액션을 복원하라"는 모순이 된다.
                act_in = act_cond
                if act_drop > 0 and not aux_step:
                    keep = (torch.rand(B, 1, 1, device=device) >= act_drop).to(act_cond.dtype)
                    act_in = act_cond * keep

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v_pred = net(
                        x_B_C_T_H_W=xt.to(torch.bfloat16),
                        timesteps_B_T=(t * 1000).view(B, 1).to(torch.bfloat16),
                        crossattn_emb=text_emb.expand(B, -1, -1),
                        condition_video_input_mask_B_C_T_H_W=cond_mask.to(torch.bfloat16),
                        fps=fps,
                        padding_mask=padding_mask,
                        data_type=DataType.VIDEO,
                        action=act_in,
                    )
                v_pred = v_pred.float()

                rf_loss = (((v_pred - v_target) ** 2) * (1 - mask_c)).sum() / (1 - mask_c).sum()
                total = rf_loss
                if unified_v2:
                    paired = paired_latent_losses(x0, xt, v_pred, sigma, mask_c, t)
                    future = paired["future_mask"]
                    x0_loss = paired["x0"]
                    temporal_loss = paired["temporal"]
                    preserve_loss = paired["preserve"]

                    # Schedules are expressed in sample exposures, not loader
                    # iterations, so changing the H100 micro-batch does not
                    # silently change the optimization recipe.
                    examples_seen = step * global_micro_batch
                    ramp = min(1.0, examples_seen / max(1, unified_warmup))
                    total = total + ramp * (
                        w_x0 * x0_loss + w_temporal * temporal_loss + w_preserve * preserve_loss
                    )
                    x0_log.append(float(x0_loss.detach()))
                    temporal_log.append(float(temporal_loss.detach()))
                    preserve_log.append(float(preserve_loss.detach()))

                    # Same noisy latent and target, but a deliberately wrong trajectory.
                    # This directly tests condition use without an external action predictor.
                    # All DDP ranks must take the same number of model forwards
                    # before the shared backward.  Use a deterministic step hash
                    # instead of an independent random draw on each rank.
                    cf_uniform = (
                        ((step * 1_103_515_245 + 12_345) & 0x7FFFFFFF) / float(0x80000000)
                    )
                    do_cf = (
                        examples_seen >= unified_warmup // 2
                        and cf_uniform < counterfactual_p
                    )
                    if do_cf:
                        wrong_action = (
                            act_cond.flip(1) if (step // max(1, accum)) % 2 == 0
                            else act_cond.roll(shifts=8, dims=1)
                        )
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v_wrong = raw_net(
                                x_B_C_T_H_W=xt.to(torch.bfloat16),
                                timesteps_B_T=(t * 1000).view(B, 1).to(torch.bfloat16),
                                crossattn_emb=text_emb.expand(B, -1, -1),
                                condition_video_input_mask_B_C_T_H_W=cond_mask.to(torch.bfloat16),
                                fps=fps,
                                padding_mask=padding_mask,
                                data_type=DataType.VIDEO,
                                action=wrong_action,
                            )
                        cf_loss = counterfactual_ranking_loss(
                            v_pred, v_wrong.float(), v_target, future, counterfactual_margin
                        )
                        total = total + ramp * w_counterfactual * cf_loss
                        counterfactual_log.append(float(cf_loss.detach()))
                if aux_step:
                    # x0 복원 → 잠재 대리 채점기 액션오차 벌점 (동결, 기울기만 통과)
                    x0_pred = x0 * mask_c + (xt.float() - sigma * v_pred) * (1 - mask_c)
                    aux_l1 = torch.mean(torch.abs(lidm(x0_pred) - act.float()))
                    ramp = min(1.0, step / aux_warmup) if aux_warmup else 1.0
                    total = rf_loss + w_aux * ramp * aux_l1
                    aux_log.append(float(aux_l1))
            (total / accum).backward()
            # draft 스텝의 rf_loss 는 보상 MAE, 일반 스텝은 흐름정합 MSE 로 단위가 다르다.
            # 한 평균에 섞으면 draft_p 비율 변화가 손실 변화로 보여서 추이를 읽을 수 없다.
            (draft_log if draft_step else rf_log).append(float(rf_loss))

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1024**3
                parts = [f"rf {np.mean(rf_log):.4f}({len(rf_log)})" if rf_log else "rf -",
                         f"draft {np.mean(draft_log):.4f}({len(draft_log)})" if draft_log else None,
                         f"aux {np.mean(aux_log):.4f}" if aux_log else None,
                         f"x0 {np.mean(x0_log):.4f}" if x0_log else None,
                         f"temp {np.mean(temporal_log):.4f}" if temporal_log else None,
                         f"pres {np.mean(preserve_log):.4f}" if preserve_log else None,
                         f"cf {np.mean(counterfactual_log):.4f}" if counterfactual_log else None]
                body = " ".join(p for p in parts if p)
                gate_log = ""
                if atok:
                    gates = [p.detach().float() for n, p in net.named_parameters()
                             if n.endswith("action_ctx_gate")]
                    gate_mean = torch.cat([g.reshape(-1) for g in gates]).abs().mean().item()
                    gate_log = f" action_ctx_gate {gate_mean:.5f}"
                if is_main:
                    print(
                        f"step {step} exposure {step * global_micro_batch} {body} "
                        f"{dt / LOG_EVERY:.2f}s/it mem/rank {mem:.1f}GB{gate_log}",
                        flush=True,
                    )
                t0 = time.time()
                aux_log, rf_log, draft_log = [], [], []
                x0_log, temporal_log, preserve_log, counterfactual_log = [], [], [], []
            if val_lat is not None and step % val_every == 0:
                _t = time.time()
                vl = eval_val()
                if is_main:
                    print(f"VAL step {step} val_loss {vl:.5f}  ({time.time()-_t:.0f}초)", flush=True)
                t0 = time.time()   # 검증 시간이 s/it 통계를 오염시키지 않게 리셋
            if step % save_every == 0 and is_main:
                if fullft:
                    state = {k: v.to(torch.bfloat16) for k, v in raw_net.state_dict().items()}
                else:
                    state = {
                        k: v for k, v in raw_net.state_dict().items()
                        if "lora" in k or "action_embedder" in k or "block_action" in k
                        or "action_ctx" in k
                    }
                payload = {"step": step, "exposures": step * global_micro_batch, "state": state}
                torch.save(payload, out_dir / "latest.pt")
                torch.save(payload, out_dir / f"step_{step:06d}.pt")
                # 옵티마이저 상태는 번호 붙은 사본을 만들지 않는다(개당 ~1GB, 재개용으로만 쓴다).
                torch.save({"step": step, "opt": opt.state_dict()}, out_dir / "latest_opt.pt")
                print(f"saved checkpoint at step {step}", flush=True)
            if stop_at and step >= stop_at:
                # 안쪽 for 루프도 끊어야 한다. 바깥 while 조건만으로는 에폭이 끝날 때까지 돈다.
                if is_main:
                    if step % save_every != 0:   # 저장 지점과 어긋나게 끝나면 마지막을 남긴다
                        st = ({k: v.to(torch.bfloat16) for k, v in raw_net.state_dict().items()} if fullft
                              else {k: v for k, v in raw_net.state_dict().items()
                                    if "lora" in k or "action_embedder" in k or "block_action" in k
                                    or "action_ctx" in k})
                        payload = {
                            "step": step,
                            "exposures": step * global_micro_batch,
                            "state": st,
                        }
                        torch.save(payload, out_dir / "latest.pt")
                        torch.save(payload, out_dir / f"step_{step:06d}.pt")
                        torch.save({"step": step, "opt": opt.state_dict()}, out_dir / "latest_opt.pt")
                    print(f"목표 스텝 {stop_at} 도달 — 종료 (현재 {step})", flush=True)
                break

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
