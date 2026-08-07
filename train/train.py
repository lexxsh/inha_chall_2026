"""사전학습 DynamiCrafter UNet을 액션 조건부로 파인튜닝한다.

challenge_kit의 scripts/train_diffusion.py를 바탕으로 세 가지를 더했다.

1. 사전학습 로드 검증 — load_checkpoints가 strict=False라 조용히 실패할 수 있어
   실제 로드율을 찍고 임계값 미만이면 중단한다.
2. 모드별 zero-init — 기존 additive는 action embedding을 0으로, Frame-Ada는 각 ResBlock의
   action scale/shift projection을 0으로 둔다. 두 모드 모두 학습 시작점은 사전학습 I2V와 같다.
3. 학습 대상 선택 — 사전학습 prior 보존 여부를 train-only group holdout에서 비교할 수 있도록
   공간 레이어를 얼리고 시간·액션 경로만 학습하는 후보를 둔다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lvdm.ema import LitEma  # noqa: E402
from lvdm.modules.attention import TemporalTransformer  # noqa: E402
from lvdm.modules.networks.openaimodel3d import TemporalConvBlock  # noqa: E402
from lvdm.utils.train import (  # noqa: E402
    get_env_vars,
    get_model,
    get_parser,
    get_trainer,
    prepare_logger,
    set_model_lr,
)
from lvdm.utils.utils import instantiate_from_config  # noqa: E402

TRAINABLE_POLICIES = ("all", "temporal", "action_only")


def is_action_parameter(name: str) -> bool:
    return (
        name.startswith("action_embed")
        or name == "null_action_emb"
        or ".action_modulation." in f".{name}."
        or name.startswith("action_modulation.")
    )


def report_pretrained_coverage(model, checkpoint_path: str, min_ratio: float = 0.95) -> None:
    """사전학습 UNet이 실제로 들어왔는지 확인한다."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt = ckpt.get("state_dict", ckpt)
    prefix = "model.diffusion_model."
    ckpt_unet = {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}

    unet_sd = model.model.diffusion_model.state_dict()
    # New action modules are intentionally absent from the generic pretrained
    # checkpoint. Excluding only those keys keeps the coverage test strict for
    # every actual backbone parameter while allowing different conditioners.
    backbone_sd = {k: v for k, v in unet_sd.items() if not is_action_parameter(k)}
    total = sum(v.numel() for v in backbone_sd.values())
    loaded = 0
    for key, value in backbone_sd.items():
        ref = ckpt_unet.get(key)
        if ref is not None and tuple(ref.shape) == tuple(value.shape) and torch.equal(ref.to(value.dtype), value.cpu()):
            loaded += value.numel()
    ratio = loaded / total
    print(f"[사전학습] UNet 파라미터 {loaded / 1e6:.1f}M / {total / 1e6:.1f}M 일치 ({100 * ratio:.1f}%)")
    if ratio < min_ratio:
        raise SystemExit(
            f"사전학습 로드율이 {100 * ratio:.1f}%로 임계값 {100 * min_ratio:.0f}% 미만이다. "
            "train/verify_checkpoint_load.py로 설정을 점검할 것."
        )


def zero_init_action_head(model) -> None:
    """Conditioner를 mode에 맞게 0-init해 pretrained 함수를 보존한다."""
    unet = model.model.diffusion_model
    if not getattr(unet, "action_conditioned", False):
        print("[zero-init] action_conditioned=False라 건너뜀")
        return
    mode = getattr(unet, "action_injection", "additive" if unet.add_act_time_emb else "concat")
    if mode == "frame_adaln":
        count = 0
        for module in unet.modules():
            modulation = getattr(module, "action_modulation", None)
            if modulation is None:
                continue
            last = modulation[-1]
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)
            count += 1
        if count == 0:
            raise RuntimeError("frame_adaln 모드인데 action_modulation ResBlock이 없다")
        message = f"ResBlock action scale/shift {count}개"
    else:
        last = unet.action_embed[-1]
        torch.nn.init.zeros_(last.weight)
        if last.bias is not None:
            torch.nn.init.zeros_(last.bias)
        message = "action_embed 마지막 층"
    with torch.no_grad():
        unet.null_action_emb.zero_()
    print(f"[zero-init] mode={mode}: {message}과 null_action_emb을 0으로 초기화 "
          "(학습 초기 출력이 사전학습 모델과 동일)")


def load_continuation_checkpoint(model, checkpoint_path: str) -> None:
    """Load a prior action-conditioned run without resetting its action head."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    action_keys = [
        key
        for key in state
        if key.startswith("model.diffusion_model.action_embed")
        or key == "model.diffusion_model.null_action_emb"
    ]
    if not action_keys:
        raise RuntimeError(f"{checkpoint_path} has no trained Dream action head")
    incompatible = model.load_state_dict(state, strict=False)
    missing_action = [
        key
        for key in incompatible.missing_keys
        if key.startswith("model.diffusion_model.action_embed")
        or key == "model.diffusion_model.null_action_emb"
    ]
    if missing_action:
        raise RuntimeError(f"continuation action keys were not loaded: {missing_action}")
    print(
        f"[continue] loaded {checkpoint_path}; action tensors={len(action_keys)}, "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
    )


def apply_trainable_policy(model, policy: str) -> None:
    """학습 대상을 정한다. 공간 레이어를 얼리면 사전학습 외형 prior가 보존된다."""
    if policy not in TRAINABLE_POLICIES:
        raise ValueError(f"policy must be one of {TRAINABLE_POLICIES}, got {policy!r}")

    unet = model.model.diffusion_model
    if policy == "all":
        for p in unet.parameters():
            p.requires_grad_(True)
    else:
        for p in unet.parameters():
            p.requires_grad_(False)
        # 액션 경로는 어떤 정책에서든 학습한다.
        for name, module in unet.named_modules():
            if name.startswith("action_embed") or "action_modulation" in name:
                for p in module.parameters():
                    p.requires_grad_(True)
        unet.null_action_emb.requires_grad_(True)

        if policy == "temporal":
            for name, module in unet.named_modules():
                if isinstance(module, (TemporalTransformer, TemporalConvBlock)):
                    for p in module.parameters():
                        p.requires_grad_(True)
            # ResBlock의 emb_layers는 시간·액션 임베딩이 실제로 작용하는 지점이라 함께 연다.
            for name, param in unet.named_parameters():
                if "emb_layers" in name or name.startswith("time_embed") or name.startswith("init_attn"):
                    param.requires_grad_(True)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[학습대상] 정책={policy}: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.1f}%)")

    # get_param_list()가 requires_grad를 거르지 않으므로 인스턴스 단에서 걸러 준다.
    original = model.get_param_list

    def filtered_param_list():
        params = original()
        if params and isinstance(params[0], dict):
            return params
        return [p for p in params if p.requires_grad]

    model.get_param_list = filtered_param_list


def run_benchmark(
    model,
    data,
    steps: int,
    device: str = "cuda:0",
    accumulate_grad_batches: int = 1,
    world_size: int = 1,
) -> None:
    """실제 optimizer step(accumulation 포함)의 시간과 메모리를 잰다."""
    if steps < 3:
        raise ValueError("--bench-steps는 warmup 2회를 제외할 수 있도록 3 이상이어야 한다")
    device = torch.device(device)
    model = model.to(device).train()
    loader = data.train_dataloader()
    # configure_optimizers는 구현에 따라 optimizer / dict / (optimizers, schedulers)를 준다.
    optimizer_cfg = model.configure_optimizers()
    if isinstance(optimizer_cfg, dict):
        optimizer = optimizer_cfg["optimizer"]
    elif isinstance(optimizer_cfg, (tuple, list)):
        first = optimizer_cfg[0]
        optimizer = first[0] if isinstance(first, (tuple, list)) else first
        if isinstance(optimizer, dict):
            optimizer = optimizer["optimizer"]
    else:
        optimizer = optimizer_cfg
    scaler = torch.amp.GradScaler(device.type)

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    it = iter(loader)
    for step in range(steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(accumulate_grad_batches):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16):
                # LatentVisualDiffusion.shared_step은 (loss, loss_dict, info)를 준다.
                loss = model.shared_step(batch, random_uncond=model.classifier_free_guidance)[0]
                scaled_loss = loss / accumulate_grad_batches
            scaler.scale(scaled_loss).backward()
            micro_losses.append(float(loss.detach()))
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if step >= 2:  # 초기 몇 스텝은 워밍업이라 제외
            times.append(elapsed)
        print(f"  optimizer step {step + 1}/{steps}  {elapsed:.2f}s  "
              f"loss {sum(micro_losses) / len(micro_losses):.4f}")

    per_step = sum(times) / max(1, len(times))
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    effective_batch = data.batch_size * accumulate_grad_batches * world_size
    print()
    print(f"[벤치마크] optimizer step당 {per_step:.2f}s "
          f"(local_batch={data.batch_size}, accumulation={accumulate_grad_batches}, "
          f"world={world_size}, effective_batch={effective_batch}), 최대 메모리 {peak:.1f}GB")
    budget_hours = 96
    print(f"[벤치마크] 대회 한도 {budget_hours}시간이면 약 {int(budget_hours * 3600 / per_step):,} 스텝, "
          f"샘플 {int(budget_hours * 3600 / per_step) * effective_batch:,}개")


def main() -> None:
    now, local_rank, global_rank, num_rank = get_env_vars()

    parser = get_parser()
    parser.add_argument("--trainable", default="temporal", choices=TRAINABLE_POLICIES)
    parser.add_argument(
        "--continue-checkpoint",
        help="Continue from an action-conditioned checkpoint; preserves its trained action head.",
    )
    parser.add_argument("--bench-steps", type=int, default=0, help="N스텝 벤치마크만 하고 종료")
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    seed_everything(args.seed)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    logger, workdir, ckptdir, cfgdir, loginfo = prepare_logger(lightning_config, config, global_rank, now)

    ## 모델
    model = get_model(config.model, workdir)
    report_pretrained_coverage(model, config.model.pretrained_checkpoint)
    if args.continue_checkpoint:
        load_continuation_checkpoint(model, args.continue_checkpoint)
    else:
        zero_init_action_head(model)
    apply_trainable_policy(model, args.trainable)
    model = set_model_lr(model, config.model, num_rank, config.data.params.batch_size)

    if model.use_ema:
        model.model_ema = LitEma(model.model)

    ## 데이터
    data = instantiate_from_config(config.data)
    data.setup()

    if args.bench_steps:
        accumulation = int(trainer_config.get("accumulate_grad_batches", 1))
        run_benchmark(
            model,
            data,
            args.bench_steps,
            device=f"cuda:{local_rank}",
            accumulate_grad_batches=accumulation,
            world_size=num_rank,
        )
        return

    trainer = get_trainer(
        lightning_config=lightning_config,
        trainer_config=trainer_config,
        config=config,
        args=args,
        workdir=workdir,
        ckptdir=ckptdir,
        logger=logger,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
