"""Source-preserving wrapper around the official First Order Motion Model.

The upstream FOMM modules are kept untouched in ``third_party/first-order-model``.
During the renderer oracle, a target frame is used only by the keypoint detector;
target pixels are never passed to the generator.  A learned FOMM background
component and occlusion map form an explicit edit mask, so unchanged pixels have
an identity path from the challenge source image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
FOMM_ROOT = REPO / "third_party/first-order-model"
if str(FOMM_ROOT) not in sys.path:
    sys.path.insert(0, str(FOMM_ROOT))

from modules.generator import OcclusionAwareGenerator  # noqa: E402
from modules.keypoint_detector import KPDetector  # noqa: E402


class ActionKeypointPredictor(nn.Module):
    """Map a 16-step SO-100 trajectory to FOMM keypoints causally."""

    def __init__(
        self,
        num_kp: int = 10,
        action_dim: int = 18,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_kp = num_kp
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.source_projection = nn.Sequential(
            nn.LayerNorm(num_kp * 6),
            nn.Linear(num_kp * 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 16, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.value_head = nn.Linear(hidden_dim, num_kp * 2)
        self.jacobian_head = nn.Linear(hidden_dim, num_kp * 4)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.jacobian_head.weight)
        nn.init.zeros_(self.jacobian_head.bias)

    def forward(
        self,
        action_features: torch.Tensor,
        source_kp: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if action_features.shape[1:] != (16, self.action_dim):
            raise ValueError(
                f"action_features must be [B,16,{self.action_dim}], got {tuple(action_features.shape)}"
            )
        batch = action_features.shape[0]
        source_value = source_kp["value"]
        source_jacobian = source_kp["jacobian"]
        source_flat = torch.cat(
            [source_value.reshape(batch, -1), source_jacobian.reshape(batch, -1)], dim=-1
        )
        token = self.action_projection(action_features)
        token = token + self.source_projection(source_flat)[:, None] + self.position
        causal = torch.triu(
            torch.ones(16, 16, dtype=torch.bool, device=token.device), diagonal=1
        )
        token = self.temporal(token, mask=causal)
        value_delta = torch.tanh(self.value_head(token)).reshape(batch, 16, self.num_kp, 2)
        jacobian_delta = torch.tanh(self.jacobian_head(token)).reshape(
            batch, 16, self.num_kp, 2, 2
        )
        return {
            "value": source_value[:, None] + 0.75 * value_delta,
            "jacobian": source_jacobian[:, None] + 0.50 * jacobian_delta,
        }


def so100_action_features(
    actions: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    anchor_scale: torch.Tensor,
    step_scale: torch.Tensor,
) -> torch.Tensor:
    """Global absolute pose + calibration-robust anchor/step deltas."""
    absolute = (actions - action_mean) / action_std.clamp_min(1e-6)
    anchor = actions[..., :1, :]
    anchor_delta = (actions - anchor) / anchor_scale.clamp_min(1e-6)
    previous = torch.cat([anchor, actions[..., :-1, :]], dim=-2)
    step_delta = (actions - previous) / step_scale.clamp_min(1e-6)
    return torch.cat([absolute, anchor_delta, step_delta], dim=-1)


def _model_params(size: str) -> dict:
    if size == "tiny":
        return {
            "kp_block": 16,
            "kp_max": 128,
            "kp_blocks": 3,
            "gen_block": 32,
            "gen_max": 256,
            "dense_block": 32,
            "dense_max": 256,
            "dense_blocks": 3,
            "bottleneck_blocks": 2,
        }
    if size == "base":
        # Matches the official vox-256 generator/KP capacity.
        return {
            "kp_block": 32,
            "kp_max": 1024,
            "kp_blocks": 5,
            "gen_block": 64,
            "gen_max": 512,
            "dense_block": 64,
            "dense_max": 1024,
            "dense_blocks": 5,
            "bottleneck_blocks": 6,
        }
    raise ValueError(f"Unknown model size {size!r}.")


class SourceAnchoredFOMM(nn.Module):
    """Official FOMM motion renderer with an explicit source-copy path."""

    def __init__(self, num_kp: int = 10, model_size: str = "base") -> None:
        super().__init__()
        cfg = _model_params(model_size)
        common = {
            "num_kp": num_kp,
            "num_channels": 3,
            "estimate_jacobian": True,
        }
        self.kp_detector = KPDetector(
            block_expansion=cfg["kp_block"],
            max_features=cfg["kp_max"],
            num_blocks=cfg["kp_blocks"],
            temperature=0.1,
            scale_factor=0.25,
            **common,
        )
        self.generator = OcclusionAwareGenerator(
            block_expansion=cfg["gen_block"],
            max_features=cfg["gen_max"],
            num_down_blocks=2,
            num_bottleneck_blocks=cfg["bottleneck_blocks"],
            estimate_occlusion_map=True,
            dense_motion_params={
                "block_expansion": cfg["dense_block"],
                "max_features": cfg["dense_max"],
                "num_blocks": cfg["dense_blocks"],
                "scale_factor": 0.25,
                "kp_variance": 0.01,
            },
            **common,
        )
        self.num_kp = num_kp
        self.model_size = model_size
        # KPDetector first downsamples by 4, then its Hourglass downsamples by
        # 2**num_blocks. Exact divisibility avoids ambiguous skip alignment.
        self.input_divisor = 4 * (2 ** cfg["kp_blocks"])

    @staticmethod
    def _edit_mask(raw: dict[str, torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        # DenseMotionNetwork's component 0 is the official identity/background
        # transform.  Other components correspond to keypoint-induced motion.
        motion_probability = 1.0 - raw["mask"][:, :1]
        occluded = 1.0 - raw["occlusion_map"]
        edit = torch.maximum(motion_probability, occluded)
        edit = F.interpolate(edit, size=output_size, mode="bilinear", align_corners=False)
        return edit.clamp(0.0, 1.0)

    def forward(
        self,
        source: torch.Tensor,
        driving: torch.Tensor | None = None,
        driving_kp: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if (driving is None) == (driving_kp is None):
            raise ValueError("Provide exactly one of driving or driving_kp.")
        height, width = source.shape[-2:]
        if height % self.input_divisor or width % self.input_divisor:
            raise ValueError(
                f"{self.model_size} FOMM requires H,W divisible by {self.input_divisor}; "
                f"got {(height, width)}"
            )
        kp_source = self.kp_detector(source)
        if driving_kp is None:
            kp_driving = self.kp_detector(driving)
        else:
            kp_driving = driving_kp
        raw = self.generator(source, kp_driving=kp_driving, kp_source=kp_source)
        edit_mask = self._edit_mask(raw, source.shape[-2:])
        prediction = source * (1.0 - edit_mask) + raw["prediction"] * edit_mask
        return {
            "prediction": prediction,
            "raw_prediction": raw["prediction"],
            "deformed": raw["deformed"],
            "edit_mask": edit_mask,
            "occlusion_map": raw["occlusion_map"],
            "component_mask": raw["mask"],
            "kp_source": kp_source,
            "kp_driving": kp_driving,
        }


def motion_target(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft edit region covering both moved-away and newly occupied pixels."""
    change = (target - source).abs().mean(dim=1, keepdim=True)
    change = ((change - 0.015) / 0.10).clamp(0.0, 1.0)
    return F.max_pool2d(change, kernel_size=11, stride=1, padding=5)


def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[..., 1:, :] - image[..., :-1, :], image[..., :, 1:] - image[..., :, :-1]


def keypoint_separation(kp: torch.Tensor, margin: float = 0.12) -> torch.Tensor:
    distance = torch.cdist(kp.float(), kp.float())
    eye = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)[None]
    penalty = F.relu(margin - distance).masked_fill(eye, 0.0)
    return penalty.sum() / max(kp.shape[0] * kp.shape[1] * (kp.shape[1] - 1), 1)


def oracle_renderer_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    source: torch.Tensor,
    target: torch.Tensor,
    identity_prediction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Losses that separately constrain rendering, identity, and edit support."""
    prediction = output["prediction"]
    raw_prediction = output["raw_prediction"]
    edit_mask = output["edit_mask"]
    expected_edit = motion_target(source, target)

    pixel_weight = 1.0 + 4.0 * expected_edit
    reconstruction = (torch.sqrt((prediction - target).square() + 1e-6) * pixel_weight).mean()
    # The raw decoder must learn plausible occlusion filling even though the
    # composited output keeps the background on an exact source path.
    raw_reconstruction = (
        torch.sqrt((raw_prediction - target).square() + 1e-6) * (0.5 + expected_edit)
    ).mean()

    grad_pred = _image_gradients(prediction)
    grad_target = _image_gradients(target)
    gradient = sum((a - b).abs().mean() for a, b in zip(grad_pred, grad_target))
    mask = F.binary_cross_entropy(edit_mask.float().clamp(1e-4, 1 - 1e-4), expected_edit.float())
    identity = (identity_prediction - source).abs().mean()
    background = ((prediction - source).abs() * (1.0 - expected_edit)).mean()
    separation = keypoint_separation(output["kp_driving"]["value"])

    total = reconstruction + 0.25 * raw_reconstruction + 0.20 * gradient
    total = total + 0.15 * mask + 0.50 * identity + 0.20 * background + 0.10 * separation
    metrics = {
        "reconstruction": reconstruction.detach(),
        "raw_reconstruction": raw_reconstruction.detach(),
        "gradient": gradient.detach(),
        "mask": mask.detach(),
        "identity": identity.detach(),
        "background": background.detach(),
        "separation": separation.detach(),
        "edit_fraction": edit_mask.detach().mean(),
        "target_edit_fraction": expected_edit.detach().mean(),
    }
    return total, metrics
