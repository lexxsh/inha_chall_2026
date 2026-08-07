"""Action-identifiability objective for the SO-100 Cosmos adapter.

The stock rectified-flow loss can be minimized while ignoring action.  This
variant evaluates the same noisy latent twice: once with the aligned action and
once with another data-parallel sample's action. It therefore cannot obtain
the ranking reward from image quality, noise, or timestep differences.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
)


class SO100ActionRankingModel(ActionVideo2WorldModelRectifiedFlow):
    """Rectified-flow training with a paired counterfactual-action loss."""

    def __init__(
        self,
        config,
        action_rank_margin: float = 0.01,
        action_rank_weight: float = 1.0,
        wrong_action_mode: str = "dp_roll",
        action_log_every: int = 10,
    ) -> None:
        super().__init__(config)
        if action_rank_margin < 0:
            raise ValueError("action_rank_margin must be non-negative")
        if action_rank_weight < 0:
            raise ValueError("action_rank_weight must be non-negative")
        if wrong_action_mode not in {"dp_roll", "reverse", "roll", "zero"}:
            raise ValueError(
                "wrong_action_mode must be one of dp_roll/reverse/roll/zero, "
                f"got {wrong_action_mode!r}"
            )
        self.action_rank_margin = float(action_rank_margin)
        self.action_rank_weight = float(action_rank_weight)
        self.wrong_action_mode = wrong_action_mode
        self.action_log_every = int(action_log_every)

    def _wrong_action(self, action: torch.Tensor) -> torch.Tensor:
        # Per-rank batch size is one, so a local batch roll is exactly the same
        # action. Gather across the data-parallel group and take another rank's
        # real action, preserving all marginal and temporal statistics.
        if self.wrong_action_mode == "dp_roll":
            if torch.distributed.is_initialized() and parallel_state.is_initialized():
                group = parallel_state.get_data_parallel_group()
                world_size = torch.distributed.get_world_size(group)
                if world_size > 1:
                    gathered = [torch.empty_like(action) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered, action.contiguous(), group=group)
                    group_rank = torch.distributed.get_rank(group)
                    return gathered[(group_rank + 1) % world_size]
            # Single-GPU debugging fallback. Full training uses eight GPUs and
            # therefore never relies on this weaker temporal counterfactual.
            return torch.roll(action, shifts=1, dims=1)
        if self.wrong_action_mode == "reverse":
            return torch.flip(action, dims=(1,))
        if self.wrong_action_mode == "roll":
            return torch.roll(action, shifts=1, dims=1)
        return torch.zeros_like(action)

    def training_step(self, data_batch, iteration):
        self._update_train_stats(data_batch)
        output_batch, loss = self.forward(data_batch)
        if self.action_log_every > 0 and iteration % self.action_log_every == 0:
            metrics = torch.stack(
                [
                    loss.detach().float(),
                    output_batch["correct_edm_loss"].detach().float(),
                    output_batch["wrong_edm_loss"].detach().float(),
                    output_batch["action_loss_gap"].detach().float(),
                    output_batch["action_rank_loss"].detach().float(),
                ]
            )
            if torch.distributed.is_initialized():
                group = (
                    parallel_state.get_data_parallel_group()
                    if parallel_state.is_initialized()
                    else None
                )
                world_size = torch.distributed.get_world_size(group)
                torch.distributed.all_reduce(metrics, group=group)
                metrics /= world_size
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                log.critical(
                    "SO100 paired-action "
                    f"iter={iteration} total={metrics[0].item():.6f} "
                    f"correct={metrics[1].item():.6f} "
                    f"wrong={metrics[2].item():.6f} "
                    f"gap(correct-wrong)={metrics[3].item():.6f} "
                    f"rank={metrics[4].item():.6f}"
                )
        return output_batch, loss

    def forward(self, data_batch):
        # This mirrors the upstream RF forward path, but constructs both
        # predictions after the one shared VAE encode/noise/timestep draw.
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        _, x0, condition = self.get_data_and_condition(data_batch)
        if condition.action is None:
            raise RuntimeError("SO100ActionRankingModel requires condition.action")

        epsilon = torch.randn(x0.size(), **self.tensor_kwargs_fp32)
        batch_size = x0.size(0)
        train_time = self.rectified_flow.sample_train_time(batch_size).to(
            **self.tensor_kwargs_fp32
        )
        train_time = rearrange(train_time, "b -> b 1")

        x0, condition, epsilon, train_time = self.broadcast_split_for_model_parallelsim(
            x0, condition, epsilon, train_time
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(
            train_time, self.tensor_kwargs_fp32
        )
        if self.config.use_high_sigma_strategy:
            raise NotImplementedError(
                "Paired SO100 objective intentionally requires the standard shared timestep path"
            )

        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt, target_velocity = self.rectified_flow.get_interpolation(epsilon, x0, sigmas)

        model_input = xt.to(**self.tensor_kwargs)
        correct_pred = self.denoise(
            noise=epsilon,
            xt_B_C_T_H_W=model_input,
            timesteps_B_T=timesteps,
            condition=condition,
        )
        wrong_condition = replace(
            condition,
            action=self._wrong_action(condition.action),
        )
        wrong_pred = self.denoise(
            noise=epsilon,
            xt_B_C_T_H_W=model_input,
            timesteps_B_T=timesteps,
            condition=wrong_condition,
        )

        reduce_dims = list(range(1, correct_pred.dim()))
        correct_per_instance = torch.mean(
            (correct_pred - target_velocity) ** 2, dim=reduce_dims
        )
        wrong_per_instance = torch.mean(
            (wrong_pred - target_velocity) ** 2, dim=reduce_dims
        )
        time_weights = self.rectified_flow.train_time_weight(
            timesteps, self.tensor_kwargs_fp32
        ).reshape(-1)
        correct_weighted = time_weights * correct_per_instance
        wrong_weighted = time_weights * wrong_per_instance

        correct_loss = correct_weighted.mean()
        wrong_loss = wrong_weighted.mean()
        # Desired inequality: L(correct) + margin <= L(wrong).
        rank_loss = F.relu(
            self.action_rank_margin + correct_weighted - wrong_weighted
        ).mean()
        total_loss = correct_loss + self.action_rank_weight * rank_loss

        output_batch = {
            "x0": x0,
            "xt": xt,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": correct_pred,
            "edm_loss": total_loss,
            "correct_edm_loss": correct_loss,
            "wrong_edm_loss": wrong_loss,
            "action_rank_loss": rank_loss,
            "action_loss_gap": correct_loss - wrong_loss,
            "timesteps": timesteps,
            "per_instance_loss": correct_per_instance,
            "wrong_per_instance_loss": wrong_per_instance,
            "n_cond_frames": condition.num_conditional_frames_B,
        }
        return output_batch, total_loss
