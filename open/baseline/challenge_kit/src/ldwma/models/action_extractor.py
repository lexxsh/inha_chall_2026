
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from video_utils.image import preprocess_images


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1), dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        if in_channels != out_channels or stride != (1, 1, 1):
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class SO100ActionExtractor(pl.LightningModule):
    """Continuous action regressor for SO-100 videos.

    Input batches use the same format as the diffusion dataloader:
    video: (B, C, T, H, W) in [-1, 1]
    act:   (B, T, A), usually normalized with train action mean/std
    """

    def __init__(
        self,
        action_dims: int = 6,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults=(1, 2, 4, 4),
        blocks_per_stage: int = 2,
        temporal_hidden: int = 256,
        temporal_layers: int = 1,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        input_key: str = "video",
        target_key: str = "act",
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_dims = action_dims
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_key = input_key
        self.target_key = target_key

        channels = [base_channels * mult for mult in channel_mults]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(),
        )

        stages = []
        current_channels = channels[0]
        for stage_idx, out_channels in enumerate(channels):
            for block_idx in range(blocks_per_stage):
                stride = (1, 2, 2) if stage_idx > 0 and block_idx == 0 else (1, 1, 1)
                stages.append(Residual3DBlock(current_channels, out_channels, stride=stride, dropout=dropout))
                current_channels = out_channels
        self.encoder = nn.Sequential(*stages)

        self.temporal_model = nn.GRU(
            input_size=current_channels,
            hidden_size=temporal_hidden,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_hidden * 2),
            nn.Linear(temporal_hidden * 2, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, action_dims),
        )
        self.save_hyperparameters()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.stem(video)
        x = self.encoder(x)
        x = x.mean(dim=(-1, -2))
        x = rearrange(x, "b c t -> b t c")
        x, _ = self.temporal_model(x)
        return self.head(x)

    def _losses(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        video = batch[self.input_key]
        target = batch[self.target_key]
        pred = self(video)
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        per_dim_mae = torch.mean(torch.abs(pred - target), dim=(0, 1))
        metrics = {"loss": mse, "mae": mae}
        for idx, value in enumerate(per_dim_mae):
            metrics[f"mae_dim_{idx}"] = value
        return mse, metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, metrics = self._losses(batch)
        for key, value in metrics.items():
            self.log(f"train/{key}", value, prog_bar=key in {"loss", "mae"}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, metrics = self._losses(batch)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=key in {"loss", "mae"}, on_step=False, on_epoch=True)
        return loss

    def get_prediction_error(self, _cond_frames, frames: torch.Tensor, act: torch.Tensor) -> float:
        if frames.dtype == torch.uint8:
            frames = preprocess_images(frames)
        pred = self(frames)
        return F.l1_loss(rearrange(pred, "b t a -> b (t a)"), rearrange(act, "b t a -> b (t a)")).item()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def _torch_load_trusted_checkpoint(checkpoint_path: str, map_location="cpu"):
    """Load our own Lightning checkpoint on PyTorch 2.6+.

    PyTorch 2.6 changed torch.load's default to weights_only=True. Older
    Lightning checkpoints can contain config objects such as OmegaConf
    ListConfig, so trusted local checkpoints need weights_only=False.
    """
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_so100_action_extractor_checkpoint(checkpoint_path: str, map_location="cpu") -> SO100ActionExtractor:
    try:
        return SO100ActionExtractor.load_from_checkpoint(checkpoint_path, map_location=map_location)
    except Exception as first_error:
        try:
            checkpoint = _torch_load_trusted_checkpoint(checkpoint_path, map_location=map_location)
            hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
            if hparams is None:
                hparams = {}
            if not isinstance(hparams, dict):
                hparams = dict(hparams)
            model = SO100ActionExtractor(**hparams)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load SO100ActionExtractor checkpoint: {checkpoint_path}. "
                "Only use this fallback for checkpoints you trust."
            ) from second_error
