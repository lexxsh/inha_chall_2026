"""Conservative action-conditioned flow renderer for the INHA challenge."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.SiLU(),
    )


class ConservativeFlowWorldModel(nn.Module):
    """Move only action-dependent pixels while copying appearance from frame 0.

    The renderer predicts backward sampling flow, a motion mask, and a bounded
    residual for each of 15 future frames. Frame zero is returned bitwise from
    the input image, matching the competition's 16-frame convention.
    """

    def __init__(
        self,
        action_dim: int = 18,
        grounding_dim: int | None = None,
        hidden_dim: int = 256,
        max_flow: float = 0.35,
        residual_scale: float = 0.08,
        pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.grounding_dim = grounding_dim
        self.hidden_dim = hidden_dim
        self.max_flow = max_flow
        self.residual_scale = residual_scale

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        backbone = resnet18(weights=weights)
        self.image_stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.image_layer1 = backbone.layer1
        self.image_layer2 = backbone.layer2  # 1/8 resolution, 128 channels

        self.action_input = nn.Sequential(
            nn.LayerNorm(3 * action_dim),
            nn.Linear(3 * action_dim, hidden_dim),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.action_position = nn.Parameter(torch.zeros(1, 16, hidden_dim))
        nn.init.trunc_normal_(self.action_position, std=0.02)
        self.grounding_input = (
            nn.Sequential(
                nn.LayerNorm(grounding_dim),
                nn.Linear(grounding_dim, hidden_dim),
                nn.SiLU(),
            )
            if grounding_dim is not None
            else None
        )
        if grounding_dim is not None:
            # The grounded renderer consumes frozen visual-motion tokens and
            # never executes the legacy raw-action branch.  Leaving these
            # parameters trainable makes DDP wait for gradients that cannot
            # exist, so disable them structurally instead of using
            # find_unused_parameters=True.
            self.action_input.requires_grad_(False)
            self.action_temporal.requires_grad_(False)
            self.action_position.requires_grad_(False)

        self.image_projection = nn.Conv2d(128, hidden_dim, 1)
        self.fuse = _conv_block(2 * hidden_dim, hidden_dim)
        self.decode_quarter = _conv_block(hidden_dim, 128)
        self.decode_half = _conv_block(128, 64)
        self.head = nn.Conv2d(64, 6, 3, padding=1)

        # Exact static renderer at initialization: zero flow/mask/residual.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)
        with torch.no_grad():
            self.head.bias[:2].zero_()
            self.head.bias[3:].zero_()

    @staticmethod
    def _action_features(actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [actions, torch.sin(torch.pi * actions), torch.cos(torch.pi * actions)], dim=-1
        )

    @staticmethod
    def _warp(source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = source.shape
        ys = torch.linspace(-1, 1, height, device=source.device, dtype=source.dtype)
        xs = torch.linspace(-1, 1, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        grid = base + flow.permute(0, 2, 3, 1)
        return F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(
        self,
        source: torch.Tensor,
        actions: torch.Tensor | None = None,
        return_aux: bool = False,
        motion_tokens: torch.Tensor | None = None,
    ):
        if source.ndim != 4 or source.shape[1] != 3:
            raise ValueError(f"source must be [B,3,H,W], got {tuple(source.shape)}")
        if (actions is None) == (motion_tokens is None):
            raise ValueError("Provide exactly one of actions or motion_tokens.")

        image = self.image_layer2(self.image_layer1(self.image_stem(source)))
        image = self.image_projection(image)
        if motion_tokens is not None:
            if self.grounding_input is None:
                raise ValueError("This model was not configured with grounding_dim.")
            if motion_tokens.ndim != 3 or motion_tokens.shape[1:] != (15, self.grounding_dim):
                raise ValueError(
                    f"motion_tokens must be [B,15,{self.grounding_dim}], "
                    f"got {tuple(motion_tokens.shape)}"
                )
            action = self.grounding_input(motion_tokens)
        else:
            if actions.ndim != 3 or actions.shape[1:] != (16, self.action_dim):
                raise ValueError(
                    f"actions must be [B,16,{self.action_dim}], got {tuple(actions.shape)}"
                )
            action = self.action_input(self._action_features(actions)) + self.action_position
            # Causal action context: future frame t cannot use a command after t.
            causal_mask = torch.triu(
                torch.ones(16, 16, device=actions.device, dtype=torch.bool), diagonal=1
            )
            action = self.action_temporal(action, mask=causal_mask)[:, :15]

        batch, frames, channels = action.shape
        height8, width8 = image.shape[-2:]
        image = image[:, None].expand(-1, frames, -1, -1, -1)
        action_map = action[..., None, None].expand(-1, -1, -1, height8, width8)
        fused = torch.cat([image, action_map], dim=2).reshape(
            batch * frames, 2 * channels, height8, width8
        )
        fused = self.fuse(fused)
        fused = F.interpolate(fused, scale_factor=2, mode="bilinear", align_corners=False)
        fused = self.decode_quarter(fused)
        fused = F.interpolate(fused, scale_factor=2, mode="bilinear", align_corners=False)
        fused = self.decode_half(fused)
        params = self.head(fused)
        params = F.interpolate(params, size=source.shape[-2:], mode="bilinear", align_corners=False)

        flow = torch.tanh(params[:, :2]) * self.max_flow
        mask = torch.sigmoid(params[:, 2:3])
        residual = torch.tanh(params[:, 3:]) * self.residual_scale
        source_bt = source[:, None].expand(-1, frames, -1, -1, -1).reshape(
            batch * frames, 3, *source.shape[-2:]
        )
        warped = self._warp(source_bt, flow)
        future = source_bt * (1 - mask) + (warped + residual).clamp(0, 1) * mask
        future = future.reshape(batch, frames, 3, *source.shape[-2:])
        video = torch.cat([source[:, None], future], dim=1)
        if not return_aux:
            return video
        aux = {
            "flow": flow.reshape(batch, frames, 2, *source.shape[-2:]),
            "mask": mask.reshape(batch, frames, 1, *source.shape[-2:]),
            "residual": residual.reshape(batch, frames, 3, *source.shape[-2:]),
        }
        return video, aux


def flow_world_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    aux: dict[str, torch.Tensor],
    wrong_prediction: torch.Tensor | None = None,
    ranking_margin: float = 0.01,
    teacher_flow: torch.Tensor | None = None,
    teacher_indices: torch.Tensor | None = None,
    reconstruction_weight: float = 1.0,
    flow_supervision_weight: float = 0.5,
    flow_motion_boost: float = 4.0,
    mask_supervision_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Motion-weighted reconstruction plus correct-vs-wrong action ranking."""
    source = target[:, :1]
    motion = (target - source).abs().mean(dim=2, keepdim=True)
    pixel_weight = 1.0 + 4.0 * (motion / 0.20).clamp(0, 1)
    error = torch.sqrt((prediction - target).square() + 1e-6)
    reconstruction = (error * pixel_weight).mean()

    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    temporal = torch.sqrt((pred_delta - target_delta).square() + 1e-6).mean()

    flow = aux["flow"]
    flow_tv = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    flow_tv = flow_tv + (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
    mask_sparse = aux["mask"].mean()
    # Ground-truth motion is available only during training and provides a
    # direct antidote to the very strong all-static shortcut. The soft target
    # avoids pretending that compression noise or tiny illumination changes
    # are foreground motion.
    motion_target = (target[:, 1:] - source).abs().mean(dim=2, keepdim=True)
    motion_target = ((motion_target - 0.025) / 0.10).clamp(0, 1)
    mask_supervision = F.binary_cross_entropy(
        aux["mask"].float().clamp(1e-4, 1 - 1e-4), motion_target.float()
    )
    residual_sparse = aux["residual"].abs().mean()
    background_flow = (aux["flow"].abs() * (1 - motion_target)).mean()

    flow_supervision = prediction.new_zeros(())
    if teacher_flow is not None:
        if teacher_indices is None:
            raise ValueError("teacher_indices is required with teacher_flow")
        predicted_flow = aux["flow"].index_select(1, teacher_indices)
        if predicted_flow.shape != teacher_flow.shape:
            raise ValueError(
                f"teacher flow mismatch: predicted={predicted_flow.shape}, teacher={teacher_flow.shape}"
            )
        selected_motion = motion_target.index_select(1, teacher_indices)
        flow_error = torch.sqrt((predicted_flow.float() - teacher_flow.float()).square() + 1e-6)
        flow_weight = 1 + flow_motion_boost * selected_motion.float()
        # Normalize by the weight sum so increasing foreground emphasis changes
        # allocation rather than the arbitrary global loss scale.
        flow_supervision = (flow_error * flow_weight).sum() / flow_weight.expand_as(
            flow_error
        ).sum().clamp_min(1)

    ranking = prediction.new_zeros(())
    if wrong_prediction is not None:
        correct_per_sample = error[:, 1:].mean(dim=(1, 2, 3, 4))
        wrong_error = torch.sqrt((wrong_prediction[:, 1:] - target[:, 1:]).square() + 1e-6)
        wrong_per_sample = wrong_error.mean(dim=(1, 2, 3, 4))
        ranking = F.relu(ranking_margin + correct_per_sample - wrong_per_sample).mean()

    total = reconstruction_weight * reconstruction + 0.20 * temporal + 0.01 * flow_tv
    total = total + mask_supervision_weight * mask_supervision + 0.002 * mask_sparse
    total = total + 0.10 * background_flow + flow_supervision_weight * flow_supervision
    total = total + 0.01 * residual_sparse + 0.50 * ranking
    metrics = {
        "reconstruction": reconstruction.detach(),
        "temporal": temporal.detach(),
        "flow_tv": flow_tv.detach(),
        "mask": mask_sparse.detach(),
        "mask_supervision": mask_supervision.detach(),
        "background_flow": background_flow.detach(),
        "flow_supervision": flow_supervision.detach(),
        "ranking": ranking.detach(),
    }
    return total, metrics
