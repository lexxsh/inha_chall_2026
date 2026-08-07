"""delta 액션 조건화를 적용한 SO-100 데이터모듈.

challenge_kit의 코드는 손대지 않고 감싸기만 한다.

두 가지를 바꾼다.
1. 액션 표현: 절대 관절각 대신 첫 스텝 대비 delta.
   데이터셋마다 서보 캘리브레이션 오프셋이 달라 절대값은 "같은 각도 = 다른 자세"가 된다
   (EMPIRICAL.md 4-3: 오프셋 제거 시 추출기 MAE 1.25 -> 0.30).
   delta는 상수 오프셋을 상쇄하고, 절대 자세는 어차피 시작 이미지가 알려준다.
2. 검증 분할: 무작위 클립이 아니라 데이터셋을 통째로 홀드아웃.
   같은 장면이 학습과 검증에 함께 들어가는 group leakage는 일반화를 과대평가한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset
from ldwma.lightning.data_modules.lerobot_so100 import SO100DataModule

REPO = Path(__file__).resolve().parents[1]

# 6fps로 통일된 다른 데이터셋과 달리 10fps라 움직임 속도가 어긋난다. eval은 6fps다.
DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)


def load_delta_scale(path: str | None, mode: str = "delta", action_dims: int = 6) -> torch.Tensor:
    """delta 정규화용 스케일. 없으면 1로 두어 원 스케일을 유지한다.

    anchor delta와 step delta는 진폭이 5배 가까이 다르므로 **모드마다 다른 scale**을 써야
    조건 신호의 크기가 비슷해지고 표현 비교가 공정해진다(METHOD.md 10 P0).
    """
    if not path or not Path(path).exists():
        return torch.ones(12 if mode == "hybrid" else action_dims, dtype=torch.float32)
    stats = json.loads(Path(path).read_text())
    by_mode = stats.get("scale_by_mode", {})
    if mode == "hybrid":
        missing = [name for name in ("delta", "delta_step") if name not in by_mode]
        if missing:
            raise SystemExit(f"{path} 에 hybrid용 scale이 없다: {missing}")
        return torch.cat([
            torch.tensor(by_mode["delta"], dtype=torch.float32),
            torch.tensor(by_mode["delta_step"], dtype=torch.float32),
        ])
    if mode in by_mode:
        return torch.tensor(by_mode[mode], dtype=torch.float32)
    if mode == "delta_step":
        # 옛 통계 파일에는 step scale이 없다. anchor scale로 대체하면 진폭이 체계적으로
        # 작아져 불공정한 비교가 되므로 명시적으로 막는다.
        raise SystemExit(
            f"{path} 에 delta_step용 scale이 없다. "
            "train/compute_delta_stats.py 를 다시 실행해 scale_by_mode를 만들 것."
        )
    return torch.tensor(stats["delta_std"], dtype=torch.float32)


VALID_ACTION_MODES = ("absolute", "delta", "delta_step", "delta_anchor", "hybrid")


def action_dims_for(mode: str) -> int:
    if mode == "delta_anchor":
        return 12
    if mode == "hybrid":
        return 18
    return 6


def transform_actions(
    act: torch.Tensor,
    delta_scale: torch.Tensor | None,
    mode: str = "delta",
    shift: int = 0,
) -> torch.Tensor:
    """정규화된 절대 액션 (..., T, 6)을 학습에 쓸 조건 표현으로 바꾼다.

    학습과 추론이 **반드시 같은 함수**를 거치도록 여기 한 군데에만 둔다.
    어긋나면 예외 없이 조용히 엉뚱한 영상이 나온다.

    mode:
      absolute    원본 그대로 (비교 실험용)
      delta       (a_t - a_0) / scale — 앵커 대비 누적 변위.
                  프레임을 한꺼번에 생성하므로 각 프레임이 "시작 대비 어디"인지 바로 알 수 있다.
      delta_step  (a_t - a_{t-1}) / scale — 매 스텝 증분. 정책 논문에서 말하는 delta.
                  모델이 시간축으로 적분해야 프레임 위치가 나온다.
      delta_anchor delta에 a_0를 이어붙인 12차원
      hybrid      absolute, anchor delta, step velocity를 이어붙인 18차원.
                  시작 이미지에서 관절 상태를 역추론하는 우회 부담을 줄이면서 데이터셋별
                  calibration offset에 강한 delta 정보도 함께 보존한다.

    shift:
      0   프레임 t에 action[t]. LeRobot의 action[t]는 목표값이라 실제로는 state[t+1]에 가깝다
          (실측 |a_t - s_t| 9.135 vs |a_t - s_{t+1}| 6.799). 즉 조건이 반 스텝 앞선다.
      -1  프레임 t에 action[t-1]. action[t-1] ~ state[t]이므로 조건이 그 프레임의 자세를 가리킨다.
          첫 프레임은 앞선 액션이 없어 action[0]으로 채운다.
      대회 action extractor로 ±2 shift를 재봤으나 데이터셋별로 결론이 갈려(results/alignment_check.csv)
      어느 쪽이 유리한지 판정되지 않았다. ablation 대상으로 남긴다.
    """
    if mode not in VALID_ACTION_MODES:
        raise ValueError(f"mode must be one of {VALID_ACTION_MODES}, got {mode!r}.")

    if shift:
        index = torch.arange(act.shape[-2], device=act.device) + shift
        index = index.clamp(0, act.shape[-2] - 1)
        act = act.index_select(-2, index)

    if mode == "absolute":
        return act

    anchor = act[..., :1, :]
    if mode == "hybrid":
        anchor_delta = act - anchor
        step_delta = act - torch.cat([anchor, act[..., :-1, :]], dim=-2)
        if delta_scale is not None:
            if delta_scale.numel() != 12:
                raise ValueError(f"hybrid scale must have 12 values, got {delta_scale.numel()}.")
            scale = delta_scale.to(act.device).clamp(min=1e-6)
            anchor_delta = anchor_delta / scale[:6]
            step_delta = step_delta / scale[6:]
        return torch.cat([act, anchor_delta, step_delta], dim=-1)
    if mode == "delta_step":
        delta = act - torch.cat([anchor, act[..., :-1, :]], dim=-2)
    else:
        delta = act - anchor
    if delta_scale is not None:
        delta = delta / delta_scale.to(act.device).clamp(min=1e-6)
    if mode == "delta_anchor":
        return torch.cat([delta, anchor.expand_as(act)], dim=-1)
    return delta


class ActionRepresentationWrapper(Dataset):
    """act 필드만 바꿔 끼우는 얇은 래퍼."""

    def __init__(
        self,
        dataset: Dataset,
        mode: str = "delta",
        delta_scale: torch.Tensor | None = None,
        shift: int = 0,
    ) -> None:
        if mode not in VALID_ACTION_MODES:
            raise ValueError(f"mode must be one of {VALID_ACTION_MODES}, got {mode!r}.")
        self.dataset = dataset
        self.mode = mode
        self.delta_scale = delta_scale
        self.shift = shift

    @property
    def action_dims(self) -> int:
        return action_dims_for(self.mode)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        sample["act"] = transform_actions(sample["act"], self.delta_scale, self.mode, self.shift)
        return sample


class SO100DeltaDataModule(SO100DataModule):
    def __init__(
        self,
        *args,
        action_mode: str = "delta",
        action_shift: int = 0,
        delta_stats_path: str | None = None,
        exclude_datasets: Sequence[str] = DEFAULT_EXCLUDE,
        holdout_datasets: Sequence[str] | int = 6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_mode = action_mode
        self.action_shift = action_shift
        self.delta_stats_path = delta_stats_path or str(REPO / "train/delta_action_stats.json")
        self.exclude_datasets = tuple(exclude_datasets or ())
        self.holdout_datasets = holdout_datasets
        self.delta_scale = load_delta_scale(self.delta_stats_path, self.action_mode)

    def _resolve_dataset_paths(self) -> list[str]:
        """auto 탐색 결과에서 제외 목록을 걸러낸다."""
        paths = self.dataset_paths
        if paths == "auto":
            from ldwma.datasets.lerobot_so100 import discover_lerobot_so100_datasets

            paths = discover_lerobot_so100_datasets(self.root)
        paths = [str(p) for p in paths]
        if self.exclude_datasets:
            before = len(paths)
            paths = [p for p in paths if not any(p.rstrip("/").endswith(x) for x in self.exclude_datasets)]
            if before != len(paths):
                print(f"[data] 제외 규칙으로 데이터셋 {before - len(paths)}개 제거")
        return sorted(paths)

    def _split_datasets(self, paths: list[str]) -> tuple[list[str], list[str]]:
        """데이터셋 단위 홀드아웃. eval처럼 '처음 보는 장면'에서 검증한다."""
        if isinstance(self.holdout_datasets, int):
            if self.holdout_datasets <= 0:
                return paths, []
            # seed 고정 셔플로 재현 가능하게 뽑는다.
            import random

            order = list(paths)
            random.Random(self.seed).shuffle(order)
            held = sorted(order[: self.holdout_datasets])
        else:
            held = [p for p in paths if any(p.rstrip("/").endswith(x) for x in self.holdout_datasets)]
        train = [p for p in paths if p not in set(held)]
        return train, held

    def setup(self, stage: str | None = None) -> None:
        stats = self._load_or_compute_action_stats() if self.normalize_actions else None
        action_mean = stats["mean"] if stats else None
        action_std = stats["std"] if stats else None

        paths = self._resolve_dataset_paths()
        train_paths, val_paths = self._split_datasets(paths)
        print(f"[data] 학습 데이터셋 {len(train_paths)}개 / 검증 홀드아웃 {len(val_paths)}개")
        if val_paths:
            print(f"[data] 홀드아웃: {[Path(p).parent.name + '/' + Path(p).name for p in val_paths]}")

        common = dict(
            root=self.root,
            traj_len=self.traj_len,
            target_height=self.target_height,
            target_width=self.target_width,
            pad=self.pad,
            camera_key=self.camera_key,
            seed=self.seed,
            downsample=self.downsample,
            use_language=self.use_language,
            fps=self.fps,
            frame_stride=self.frame_stride,
            action_mean=action_mean,
            action_std=action_std,
            remote=self.remote,
            repo_id=self.repo_id,
            cache_dir=self.cache_dir,
            temporary_downloads=self.temporary_downloads,
            hf_token=self.hf_token,
        )

        def build(paths: list[str]) -> Dataset:
            # 데이터셋 단위로 홀드아웃하므로 클립 단위 분할은 끄고 전량을 쓴다.
            base = LeRobotSO100Dataset(
                dataset_paths=paths, train=True, val_fraction=0.0, use_all_episodes=True, **common
            )
            return ActionRepresentationWrapper(base, self.action_mode, self.delta_scale, self.action_shift)

        self.train_dataset = build(train_paths)
        self.val_dataset = build(val_paths) if val_paths else None
        print(f"[data] 학습 클립 {len(self.train_dataset)}개 / 검증 클립 {len(self.val_dataset) if self.val_dataset else 0}개")
        print(f"[data] 액션 표현: {self.action_mode} ({self.train_dataset.action_dims}차원), shift={self.action_shift:+d}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
