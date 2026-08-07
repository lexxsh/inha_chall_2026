import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class TemporalActionAdaLN(nn.Module):
    """Map a frame-level robot action sequence to Wan's token-wise AdaLN input.

    Wan2.2-TI2V-5B represents a 17-frame clip as one clean first-frame latent
    followed by four noisy future latents.  The 16 SO-100 actions are therefore
    pooled into four ordered groups and injected only into the future tokens.
    The final projection is zero initialized so enabling the adapter preserves
    the pretrained Wan function exactly at step zero.
    """

    def __init__(self, action_dim: int, model_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
        )
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 6 * model_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, actions: torch.Tensor, latent_frames: int, spatial_tokens: int):
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B, T, {self.action_dim}], got {tuple(actions.shape)}"
            )
        future_frames = latent_frames - 1
        if future_frames < 1:
            raise ValueError(f"Action conditioning needs at least 2 latent frames, got {latent_frames}.")

        hidden = self.encoder(actions)
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        # Exact non-overlapping groups for the intended 16 -> 4 mapping; adaptive
        # pooling also keeps short smoke-test clips and future frame counts valid.
        hidden = F.adaptive_avg_pool1d(hidden.transpose(1, 2), future_frames).transpose(1, 2)
        modulation = self.output(hidden).unflatten(-1, (6, self.model_dim))
        clean = modulation.new_zeros((modulation.shape[0], 1, 6, self.model_dim))
        modulation = torch.cat([clean, modulation], dim=1)
        return modulation.repeat_interleave(spatial_tokens, dim=1)


class TemporalActionAdaLNV2(nn.Module):
    """1X-style state/action encoder with learned 4x temporal compression.

    Continuous features are expanded sinusoidally, given an explicit temporal
    position, and compressed by two learned Conv1d layers.  Unlike V1, no
    adaptive averaging can erase the within-group order of four actions.
    """

    def __init__(self, action_dim: int, model_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.model_dim = model_dim
        feature_dim = 3 * action_dim + 2
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4),
        )
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 6 * model_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, actions: torch.Tensor, latent_frames: int, spatial_tokens: int):
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B, T, {self.action_dim}], got {tuple(actions.shape)}"
            )
        future_frames = latent_frames - 1
        expected_actions = 4 * future_frames
        if actions.shape[1] != expected_actions:
            raise ValueError(
                f"V2 requires four actions per future latent: expected T={expected_actions}, "
                f"got T={actions.shape[1]}."
            )

        # The robot values are already normalized.  A single Fourier band keeps
        # the expansion bounded while allowing the MLP to resolve fine changes.
        value_features = torch.cat(
            [actions, torch.sin(math.pi * actions), torch.cos(math.pi * actions)], dim=-1
        )
        position = torch.arange(actions.shape[1], device=actions.device, dtype=actions.dtype)
        position = position / max(actions.shape[1] - 1, 1)
        position = torch.stack(
            [torch.sin(2 * math.pi * position), torch.cos(2 * math.pi * position)], dim=-1
        )
        position = position.unsqueeze(0).expand(actions.shape[0], -1, -1)
        hidden = self.encoder(torch.cat([value_features, position], dim=-1))
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        if hidden.shape[1] != future_frames:
            raise RuntimeError(f"Temporal encoder returned {hidden.shape[1]} groups, expected {future_frames}.")

        modulation = self.output(hidden).unflatten(-1, (6, self.model_dim))
        clean = modulation.new_zeros((modulation.shape[0], 1, 6, self.model_dim))
        modulation = torch.cat([clean, modulation], dim=1)
        return modulation.repeat_interleave(spatial_tokens, dim=1)


class ActionCrossAttentionEncoder(nn.Module):
    """Turn a frame-level robot action sequence into cross-attention keys/values.

    Unlike the AdaLN adapters (v1/v2), which fold actions into a spatially
    uniform scale/shift/gate vector, this encoder emits an ordered *token*
    sequence that every DiTBlock can attend to through its own query.  The
    temporal resolution of the 16 SO-100 actions is preserved as 16 tokens by
    default (no learned compression), so spatially distinct regions of the
    frame can pull different action information.

    The encoder computes the shared key/value projections once; each DiTBlock
    supplies only its query and a zero-initialized gate (see
    ``DiTBlock.enable_action_cross_attention``), which keeps the pretrained Wan
    function bit-identical at step zero.
    """

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        num_heads: int,
        hidden_dim: int = 512,
        num_tokens: int = 16,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        # Value Fourier features plus an explicit temporal position, matching
        # the V2 AdaLN encoder's front end so the two versions see identical
        # continuous features.
        feature_dim = 3 * action_dim + 2
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
        )
        # Local temporal mixing without any stride: 16 actions stay 16 tokens.
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_dim),
        )
        # Learned positional embedding (cross-attention has no RoPE).
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, model_dim))
        # Shared key/value projections computed once for all blocks.
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.norm_k = RMSNorm(model_dim, eps=eps)

    def forward(self, actions: torch.Tensor):
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B, T, {self.action_dim}], got {tuple(actions.shape)}"
            )
        value_features = torch.cat(
            [actions, torch.sin(math.pi * actions), torch.cos(math.pi * actions)], dim=-1
        )
        length = actions.shape[1]
        position = torch.arange(length, device=actions.device, dtype=actions.dtype)
        position = position / max(length - 1, 1)
        position = torch.stack(
            [torch.sin(2 * math.pi * position), torch.cos(2 * math.pi * position)], dim=-1
        )
        position = position.unsqueeze(0).expand(actions.shape[0], -1, -1)
        hidden = self.encoder(torch.cat([value_features, position], dim=-1))
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        if hidden.shape[1] != self.num_tokens:
            # Keep short smoke-test clips valid; the intended 16 -> 16 path is a
            # no-op so no temporal information is averaged away.
            hidden = F.adaptive_avg_pool1d(
                hidden.transpose(1, 2), self.num_tokens
            ).transpose(1, 2)
        tokens = self.proj(hidden) + self.pos_emb.to(dtype=hidden.dtype)
        key = self.norm_k(self.k_proj(tokens))
        value = self.v_proj(tokens)
        return key, value


class SpatialTrackAdapter(nn.Module):
    """Zero-initialized pixel-aligned point-track control for Wan tokens.

    The input is a compact control video with occupancy and normalized x/y
    displacement channels.  It is resized to Wan's latent token grid before
    feature extraction, so callers may store controls at a cheap resolution.
    The final 1x1x1 projection is zero initialized: enabling the adapter leaves
    the pretrained model exactly unchanged until it receives an update.
    """

    def __init__(
        self,
        model_dim: int,
        input_dim: int = 3,
        hidden_dim: int = 128,
        structural_zero: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.structural_zero = structural_zero
        self.input = nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden_dim, model_dim, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _raw(self, control: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(self.input(control)))

    def forward(self, control: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if control.ndim == 4:
            control = control.unsqueeze(0)
        if control.ndim != 5 or control.shape[1] != self.input_dim:
            raise ValueError(
                f"track control must have shape [B,{self.input_dim},T,H,W], "
                f"got {tuple(control.shape)}"
            )
        control = F.interpolate(control, size=target_shape, mode="trilinear", align_corners=False)
        output = self._raw(control)
        if self.structural_zero:
            # Zero initialization only protects the pretrained model at step 0.
            # This residual parameterization guarantees f(0)=0 after training as
            # well, preventing Conv biases from becoming a domain-token shortcut.
            output = output - self._raw(torch.zeros_like(control))
        return output


class SourceConditionedSpatialActionAdapter(nn.Module):
    """Turn robot actions into source-aligned, multi-block Wan residuals.

    Numeric robot actions do not say where a joint is located in the image.  A
    global AdaLN vector therefore has to learn camera geometry implicitly and
    is easy for the video model to ignore.  This adapter combines each temporal
    action group with the first-frame VAE feature map, allowing the source image
    to localize the robot while the action controls how its features evolve.

    The final projection is exactly zero initialized.  Attaching the module to
    a pretrained Wan model therefore preserves the original I2V function at
    step zero.  One shared residual is injected after several transformer
    blocks with trainable per-injection scales; this is a lightweight
    ControlNet-style path rather than a full duplicate of the 14B backbone.
    """

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        source_dim: int = 16,
        hidden_dim: int = 192,
        num_layers: int = 40,
        injection_layers: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        if action_dim < 1 or source_dim < 1 or hidden_dim < 8:
            raise ValueError("action_dim, source_dim, and hidden_dim must be positive")
        if injection_layers is None:
            injection_layers = tuple(
                sorted(set((0, num_layers // 4, num_layers // 2, 3 * num_layers // 4)))
            )
        if not injection_layers or min(injection_layers) < 0 or max(injection_layers) >= num_layers:
            raise ValueError(
                f"Bad injection layers {injection_layers} for a {num_layers}-layer Wan model"
            )

        self.action_dim = action_dim
        self.model_dim = model_dim
        self.source_dim = source_dim
        self.hidden_dim = hidden_dim
        self.injection_layers = tuple(int(layer) for layer in injection_layers)
        self._injection_index = {
            layer: index for index, layer in enumerate(self.injection_layers)
        }

        # Value Fourier features plus an explicit temporal position.  A learned
        # stride-four compression matches Wan2.1 VAE's 4 RGB -> 1 latent layout.
        action_feature_dim = 3 * action_dim + 2
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_feature_dim),
            nn.Linear(action_feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4),
        )

        # Two normalized screen-coordinate channels make direction-dependent
        # control possible without assuming a calibrated URDF/camera projection.
        self.source_encoder = nn.Sequential(
            nn.Conv2d(source_dim + 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.source_norm = nn.GroupNorm(8, hidden_dim, affine=False)
        self.action_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.body = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden_dim, model_dim, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.layer_scales = nn.Parameter(torch.ones(len(self.injection_layers)))

    def _encode_actions(self, actions: torch.Tensor, future_latents: int) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B,T,{self.action_dim}], got {tuple(actions.shape)}"
            )
        expected = 4 * future_latents
        if actions.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} actions for {future_latents} future Wan latents, "
                f"got {actions.shape[1]}"
            )
        features = torch.cat(
            [actions, torch.sin(math.pi * actions), torch.cos(math.pi * actions)], dim=-1
        )
        position = torch.linspace(
            0.0, 1.0, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        position = torch.stack(
            [torch.sin(2 * math.pi * position), torch.cos(2 * math.pi * position)], dim=-1
        )
        position = position.unsqueeze(0).expand(actions.shape[0], -1, -1)
        hidden = self.action_encoder(torch.cat([features, position], dim=-1))
        hidden = self.temporal_encoder(hidden.transpose(1, 2)).transpose(1, 2)
        if hidden.shape[1] != future_latents:
            raise RuntimeError(
                f"Temporal action encoder returned {hidden.shape[1]} latents, "
                f"expected {future_latents}"
            )
        return hidden

    @staticmethod
    def _coordinates(
        batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(
        self,
        actions: torch.Tensor,
        source_latents: torch.Tensor,
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        latent_frames, target_height, target_width = target_shape
        future_latents = latent_frames - 1
        if future_latents < 1:
            raise ValueError("Spatial action conditioning requires at least one future latent")
        if source_latents.ndim == 5:
            source_latents = source_latents[:, :, 0]
        if source_latents.ndim != 4 or source_latents.shape[1] != self.source_dim:
            raise ValueError(
                f"source_latents must have shape [B,{self.source_dim},H,W], "
                f"got {tuple(source_latents.shape)}"
            )
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != source_latents.shape[0]:
            if source_latents.shape[0] == 1:
                source_latents = source_latents.expand(actions.shape[0], -1, -1, -1)
            else:
                raise ValueError(
                    f"action/source batch mismatch: {actions.shape[0]} vs {source_latents.shape[0]}"
                )

        action_hidden = self._encode_actions(actions, future_latents)
        coords = self._coordinates(
            source_latents.shape[0],
            source_latents.shape[2],
            source_latents.shape[3],
            source_latents.device,
            source_latents.dtype,
        )
        source_hidden = self.source_encoder(torch.cat([source_latents, coords], dim=1))
        if source_hidden.shape[-2:] != (target_height, target_width):
            source_hidden = F.interpolate(
                source_hidden,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        source_hidden = self.source_norm(source_hidden)

        scale, shift = self.action_film(action_hidden).chunk(2, dim=-1)
        scale = scale.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        shift = shift.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        future = source_hidden.unsqueeze(2) * (1.0 + scale) + shift
        clean = future.new_zeros(
            (future.shape[0], future.shape[1], 1, target_height, target_width)
        )
        hidden = torch.cat([clean, future], dim=2)
        residual = self.output(self.body(hidden))
        return rearrange(residual, "b c f h w -> b (f h w) c").contiguous()

    def residual_for_block(self, residual: torch.Tensor, block_id: int) -> Optional[torch.Tensor]:
        index = self._injection_index.get(int(block_id))
        if index is None:
            return None
        return residual * self.layer_scales[index].to(device=residual.device, dtype=residual.dtype)


class OracleMotionFieldAdapter(nn.Module):
    """Inject a source-aligned dense motion field into a pretrained Wan I2V DiT.

    The control tensor contains *motion residuals*, rather than an RGB target:
    warped-source RGB minus source RGB, normalized backward flow, visibility,
    and occlusion.  Consequently an all-zero tensor means exact no-motion and
    the adapter is structurally constrained to return zero for that input even
    after training.  This prevents the branch from learning a dataset/domain
    shortcut that ignores its spatial condition.

    Wan uses one source latent followed by four stride-four future latents for
    a 17-frame clip.  Pixel controls are grouped with the same temporal layout,
    resized to patch-token resolution, localized by the first-frame VAE
    feature map, and projected independently at several transformer depths.
    It is intentionally much smaller than a full VACE/ControlNet duplicate;
    this module is an oracle upper-bound gate before investing in a learned
    action-to-motion predictor.
    """

    def __init__(
        self,
        control_dim: int,
        model_dim: int,
        source_dim: int = 16,
        hidden_dim: int = 192,
        num_layers: int = 40,
        injection_layers: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        if control_dim < 1 or source_dim < 1 or hidden_dim < 8:
            raise ValueError("control_dim, source_dim, and hidden_dim must be positive")
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm")
        if injection_layers is None:
            injection_layers = tuple(
                sorted(set((0, num_layers // 4, num_layers // 2, 3 * num_layers // 4)))
            )
        if not injection_layers or min(injection_layers) < 0 or max(injection_layers) >= num_layers:
            raise ValueError(
                f"Bad injection layers {injection_layers} for a {num_layers}-layer Wan model"
            )

        self.control_dim = int(control_dim)
        self.model_dim = int(model_dim)
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.injection_layers = tuple(int(layer) for layer in injection_layers)
        self._injection_index = {
            layer: index for index, layer in enumerate(self.injection_layers)
        }

        # Bias-free/non-affine layers preserve the exact f(0)=0 invariant.
        self.control_encoder = nn.Sequential(
            nn.Conv3d(control_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim, affine=False),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim, affine=False),
            nn.SiLU(),
        )
        self.source_encoder = nn.Sequential(
            nn.Conv2d(source_dim + 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.body = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim, affine=False),
            nn.SiLU(),
        )
        self.outputs = nn.ModuleList(
            nn.Linear(hidden_dim, model_dim, bias=False) for _ in self.injection_layers
        )
        for output in self.outputs:
            nn.init.zeros_(output.weight)
        self.layer_scales = nn.Parameter(torch.ones(len(self.injection_layers)))

    @staticmethod
    def _coordinates(
        batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def _group_frames(control: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """Match Wan VAE's [frame0, four groups of four] temporal layout."""
        if control.shape[2] == latent_frames:
            return control
        expected_rgb_frames = 1 + 4 * (latent_frames - 1)
        if control.shape[2] != expected_rgb_frames:
            raise ValueError(
                f"Expected {expected_rgb_frames} RGB control frames for {latent_frames} "
                f"Wan latents, got {control.shape[2]}"
            )
        groups = [control[:, :, :1]]
        for group in range(latent_frames - 1):
            begin = 1 + 4 * group
            groups.append(control[:, :, begin : begin + 4].mean(dim=2, keepdim=True))
        return torch.cat(groups, dim=2)

    def forward(
        self,
        control: torch.Tensor,
        source_latents: torch.Tensor,
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        latent_frames, target_height, target_width = target_shape
        if control.ndim == 4:
            control = control.unsqueeze(0)
        if control.ndim != 5 or control.shape[1] != self.control_dim:
            raise ValueError(
                f"motion field must have shape [B,{self.control_dim},T,H,W], "
                f"got {tuple(control.shape)}"
            )
        if source_latents.ndim == 5:
            source_latents = source_latents[:, :, 0]
        if source_latents.ndim != 4 or source_latents.shape[1] != self.source_dim:
            raise ValueError(
                f"source_latents must have shape [B,{self.source_dim},H,W], "
                f"got {tuple(source_latents.shape)}"
            )
        if source_latents.shape[0] != control.shape[0]:
            if source_latents.shape[0] == 1:
                source_latents = source_latents.expand(control.shape[0], -1, -1, -1)
            elif control.shape[0] == 1:
                control = control.expand(source_latents.shape[0], -1, -1, -1, -1)
            else:
                raise ValueError(
                    f"control/source batch mismatch: {control.shape[0]} vs "
                    f"{source_latents.shape[0]}"
                )

        control = self._group_frames(control, latent_frames)
        control = F.interpolate(
            control.float(),
            size=(latent_frames, target_height, target_width),
            mode="trilinear",
            align_corners=False,
        ).to(dtype=source_latents.dtype)
        hidden = self.control_encoder(control)

        coords = self._coordinates(
            source_latents.shape[0],
            source_latents.shape[2],
            source_latents.shape[3],
            source_latents.device,
            source_latents.dtype,
        )
        source = self.source_encoder(torch.cat([source_latents, coords], dim=1))
        if source.shape[-2:] != (target_height, target_width):
            source = F.interpolate(
                source, size=(target_height, target_width), mode="bilinear", align_corners=False
            )
        # Multiplicative localization keeps f(zero_control) exactly zero.
        hidden = hidden * (1.0 + 0.5 * torch.tanh(source).unsqueeze(2))
        hidden = self.body(hidden)
        return rearrange(hidden, "b c f h w -> b (f h w) c").contiguous()

    def residual_for_block(self, hidden: torch.Tensor, block_id: int) -> Optional[torch.Tensor]:
        index = self._injection_index.get(int(block_id))
        if index is None:
            return None
        residual = self.outputs[index](hidden)
        return residual * self.layer_scales[index].to(
            device=residual.device, dtype=residual.dtype
        )


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.has_action_xattn = False

    def enable_action_cross_attention(self, num_heads: int, eps: float = 1e-6):
        """Attach a gated, separate-softmax cross-attention to action tokens.

        The keys/values are shared across all blocks and supplied by
        ``ActionCrossAttentionEncoder``; each block owns only a query
        projection, an output projection, and a scalar gate.  The gate is
        zero-initialized, so ``x + gate * action_out`` reduces to ``x`` at step
        zero and the pretrained Wan function is preserved bit-for-bit.  Only the
        gate is zero-initialized (not the output projection): zeroing both would
        create a dead-gradient region where neither can start learning.
        """
        dim = self.dim
        self.action_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.action_q = nn.Linear(dim, dim)
        self.action_norm_q = RMSNorm(dim, eps=eps)
        self.action_o = nn.Linear(dim, dim)
        self.action_attn = AttentionModule(num_heads)
        self.action_gate = nn.Parameter(torch.zeros(1))
        self.has_action_xattn = True

    def forward(self, x, context, t_mod, freqs, action_kv=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        if self.has_action_xattn and action_kv is not None:
            action_k, action_v = action_kv
            action_q = self.action_norm_q(self.action_q(self.action_norm(x)))
            action_out = self.action_attn(action_q, action_k, action_v)
            x = x + self.action_gate.to(dtype=x.dtype) * self.action_o(action_out)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.action_conditioner = None
        self.action_token_encoder = None
        self.track_adapter = None
        self.spatial_action_adapter = None
        self.oracle_motion_field_adapter = None

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def enable_action_conditioning(
        self, action_dim: int = 6, hidden_dim: int = 512, version: str = "v1"
    ):
        if not self.seperated_timestep:
            raise ValueError("Temporal action AdaLN requires a Wan model with seperated_timestep=True.")
        # New action modules are created with float32 defaults, but the Wan
        # backbone is typically bfloat16.  Capture the backbone dtype/device now
        # so every newly attached module (the returned encoder AND the per-block
        # cross-attention projections) is cast to match; otherwise the first
        # matmul against bf16 activations raises a dtype mismatch.
        reference = next(self.parameters())
        ref_kwargs = {"device": reference.device, "dtype": reference.dtype}
        if version == "xattn":
            num_heads = self.blocks[0].num_heads
            self.action_token_encoder = ActionCrossAttentionEncoder(
                action_dim, self.dim, num_heads, hidden_dim, num_tokens=16
            )
            for block in self.blocks:
                block.enable_action_cross_attention(num_heads)
                block.action_norm.to(**ref_kwargs)
                block.action_q.to(**ref_kwargs)
                block.action_norm_q.to(**ref_kwargs)
                block.action_o.to(**ref_kwargs)
                block.action_gate.data = block.action_gate.data.to(**ref_kwargs)
            self.action_token_encoder.to(**ref_kwargs)
            return self.action_token_encoder
        if version == "v1":
            cls = TemporalActionAdaLN
        elif version == "v2":
            cls = TemporalActionAdaLNV2
        else:
            raise ValueError(f"Unknown action conditioner version: {version!r}")
        self.action_conditioner = cls(action_dim, self.dim, hidden_dim).to(**ref_kwargs)
        return self.action_conditioner

    def enable_track_conditioning(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        structural_zero: bool = False,
    ):
        self.track_adapter = SpatialTrackAdapter(
            self.dim, input_dim, hidden_dim, structural_zero=structural_zero
        )
        return self.track_adapter

    def enable_spatial_action_conditioning(
        self,
        action_dim: int,
        hidden_dim: int = 192,
        source_dim: int = 16,
        injection_layers: Optional[Tuple[int, ...]] = None,
    ):
        if not self.has_image_input or not self.require_vae_embedding:
            raise ValueError(
                "Source-conditioned spatial actions require an image-conditioned Wan "
                "model with VAE embeddings (for example Wan2.1-I2V-14B)."
            )
        self.spatial_action_adapter = SourceConditionedSpatialActionAdapter(
            action_dim=action_dim,
            model_dim=self.dim,
            source_dim=source_dim,
            hidden_dim=hidden_dim,
            num_layers=len(self.blocks),
            injection_layers=injection_layers,
        )
        return self.spatial_action_adapter

    def enable_oracle_motion_field_conditioning(
        self,
        control_dim: int = 7,
        hidden_dim: int = 192,
        source_dim: int = 16,
        injection_layers: Optional[Tuple[int, ...]] = None,
    ):
        if not self.has_image_input or not self.require_vae_embedding:
            raise ValueError(
                "Oracle motion fields require an image-conditioned Wan model with "
                "VAE embeddings (for example Wan2.1-I2V-14B)."
            )
        self.oracle_motion_field_adapter = OracleMotionFieldAdapter(
            control_dim=control_dim,
            model_dim=self.dim,
            source_dim=source_dim,
            hidden_dim=hidden_dim,
            num_layers=len(self.blocks),
            injection_layers=injection_layers,
        )
        return self.oracle_motion_field_adapter

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        track_control: Optional[torch.Tensor] = None,
        enable_wantodance_global=False,
    ):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        if track_control is not None:
            if self.track_adapter is None:
                raise RuntimeError("track_control was provided but Wan track conditioning is not enabled")
            x = x + self.track_adapter(
                track_control.to(device=x.device, dtype=x.dtype), tuple(x.shape[2:])
            )
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x, track_control=kwargs.get("track_control"))
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
