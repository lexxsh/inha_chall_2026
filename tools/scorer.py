"""제출킷과 동일한 방식으로 feature를 뽑아 로컬 점수를 계산한다.

submission_kit의 함수를 그대로 import 해서 쓰기 때문에 평가와 어긋날 여지가 없다.
리더보드 공식은 공개되지 않았으므로 cosine distance(1 - cos_sim)를 프레임 평균으로 보고한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/submission_kit"
sys.path.insert(0, str(KIT))

from feature_csv_utils import (  # noqa: E402
    extract_action_features,
    extract_dino_features,
    extract_video_features,
    load_action_stats,
    load_action_extractor,
    load_dino_model,
    load_video_feature_model,
    read_video_uint8,
    resolve_dino_image_size,
    save_video_tensor,
    to_eval_uint8,
)

TARGET_H, TARGET_W = 320, 512
TRAJ_LEN = 16
ACTION_STATS = REPO / "open/data/train/so100_action_statistics.json"
ACTION_CKPT = KIT / "checkpoints/action_extractor.ckpt"


class Scorer:
    """DINO / Video / Action 세 컴포넌트를 계산하는 평가기."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.video_model = load_video_feature_model(self.device, pretrained=True)
        self.dino_model = load_dino_model(self.device, "vit_small_patch14_dinov2.lvd142m", pretrained=True)
        self.dino_size = resolve_dino_image_size(self.dino_model, requested_size=0)
        self.action_model = load_action_extractor(str(ACTION_CKPT), self.device)
        self.action_mean, self.action_std = load_action_stats(str(ACTION_STATS))

    def to_eval(self, video_uint8: np.ndarray) -> torch.Tensor:
        """원본 해상도 (T,H,W,C) uint8 -> 평가 규격 (T,320,512,C) uint8."""
        return to_eval_uint8(video_uint8, TARGET_H, TARGET_W, pad=True)

    def features(self, videos_eval: torch.Tensor) -> dict[str, torch.Tensor]:
        """videos_eval: (B,T,H,W,C) uint8, 이미 평가 규격."""
        return {
            "dino": extract_dino_features(videos_eval, self.dino_model, self.device, self.dino_size),
            "video": extract_video_features(videos_eval, self.video_model, self.device),
        }

    def action_mae(self, videos_eval: torch.Tensor, target_actions_raw: np.ndarray) -> torch.Tensor:
        """제출킷과 동일하게 정규화 공간에서 MAE를 구한다."""
        target = torch.from_numpy(np.asarray(target_actions_raw, dtype=np.float32))
        target = (target - self.action_mean) / self.action_std
        return extract_action_features(videos_eval, self.action_model, self.device, target.to(self.device)).flatten()


def cosine_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """마지막 축 기준 1 - cos_sim. (B,D) 또는 (B,T,D) 모두 지원하며 (B,)로 줄인다."""
    dist = 1.0 - F.cosine_similarity(pred.float(), gt.float(), dim=-1)
    while dist.ndim > 1:
        dist = dist.mean(dim=-1)
    return dist


def roundtrip_mp4(video_eval_uint8: torch.Tensor, path: Path, fps: int = 6) -> torch.Tensor:
    """실제 제출 경로와 동일하게 mp4로 저장했다가 다시 읽는다(코덱 손실 포함)."""
    tensor = video_eval_uint8.permute(3, 0, 1, 2).float().div(255.0).mul(2.0).sub(1.0)
    save_video_tensor(tensor, path, fps)
    return read_video_uint8(path, expected_frames=video_eval_uint8.shape[0])
