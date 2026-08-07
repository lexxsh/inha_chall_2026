"""Canonical action/frame contract for the SO-100 video challenge.

The challenge supplies frame 0 and asks for a 16-frame clip.  LeRobot's
``action[t]`` is a target command that most closely explains ``state[t+1]``.
Consequently only actions 0..14 have an observable outcome in frames 1..15;
action 15 targets a state beyond the submitted clip.

Keep this module independent of any video backbone.  Every model should use
the same alignment and differ only in how the resulting control features are
encoded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


JOINT_DIM = 6
VIDEO_FRAMES = 16
FUTURE_FRAMES = VIDEO_FRAMES - 1


@dataclass(frozen=True)
class TransitionContract:
    """Numpy views of one 16-frame training window."""

    commands: np.ndarray
    current_states: np.ndarray
    next_states: np.ndarray
    realized_delta: np.ndarray
    target_residual: np.ndarray
    source_anchor_delta: np.ndarray
    deployable_step: np.ndarray


def _validate_numpy(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != JOINT_DIM:
        raise ValueError(f"{name} must have shape [T,{JOINT_DIM}], got {array.shape}")
    if len(array) < VIDEO_FRAMES:
        raise ValueError(f"{name} needs at least {VIDEO_FRAMES} rows, got {len(array)}")
    if not np.isfinite(array[:VIDEO_FRAMES]).all():
        raise ValueError(f"{name} contains non-finite values in the first {VIDEO_FRAMES} rows")
    return array


def build_transition_contract(
    actions: np.ndarray,
    states: np.ndarray,
) -> TransitionContract:
    """Map 16 logged commands/states to the 15 visible transitions.

    ``deployable_step`` deliberately uses only the source state and commands:

    - transition 0: ``action[0] - state[0]``
    - transition j: ``action[j] - action[j-1]`` for j > 0

    Unlike ``action[j] - state[j]``, it can be constructed at inference once a
    source-state estimate is available.  Future logged states remain targets,
    never conditioning inputs.
    """

    action = _validate_numpy("actions", actions)
    state = _validate_numpy("states", states)
    commands = action[:FUTURE_FRAMES].copy()
    current = state[:FUTURE_FRAMES].copy()
    following = state[1:VIDEO_FRAMES].copy()
    previous_command = np.concatenate([state[:1], action[: FUTURE_FRAMES - 1]], axis=0)
    return TransitionContract(
        commands=commands,
        current_states=current,
        next_states=following,
        realized_delta=following - current,
        target_residual=commands - current,
        source_anchor_delta=commands - state[:1],
        deployable_step=commands - previous_command,
    )


def transition_action_features(
    actions: torch.Tensor,
    source_state: torch.Tensor,
) -> torch.Tensor:
    """Return absolute + source-anchor + deployable-step features.

    Args:
        actions: ``[..., 16, 6]`` commands in a common normalized space.
        source_state: ``[..., 6]`` or ``[..., 1, 6]`` source state in the same
            normalized space.  At deployment this must come from a source-image
            state estimator; future logged state is never accepted here.

    Returns:
        Tensor with shape ``[..., 15, 18]``.
    """

    if actions.ndim < 2 or actions.shape[-2:] != (VIDEO_FRAMES, JOINT_DIM):
        raise ValueError(
            f"actions must end in [{VIDEO_FRAMES},{JOINT_DIM}], got {tuple(actions.shape)}"
        )
    if source_state.shape[-1] != JOINT_DIM:
        raise ValueError(f"source_state must end in {JOINT_DIM}, got {tuple(source_state.shape)}")
    if source_state.ndim == actions.ndim - 1:
        source_state = source_state.unsqueeze(-2)
    expected_prefix = actions.shape[:-2]
    if source_state.shape[:-2] != expected_prefix or source_state.shape[-2] != 1:
        raise ValueError(
            "source_state must have the same batch prefix as actions and one time row; "
            f"got actions={tuple(actions.shape)}, source_state={tuple(source_state.shape)}"
        )
    commands = actions[..., :FUTURE_FRAMES, :]
    anchor = commands - source_state
    previous = torch.cat([source_state, commands[..., :-1, :]], dim=-2)
    step = commands - previous
    return torch.cat([commands, anchor, step], dim=-1)


def irasim_transition_actions(
    actions: torch.Tensor,
    source_state: torch.Tensor | None,
    mode: str,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the 15 correctly aligned action tokens expected by IRASim.

    ``actions`` and ``source_state`` must already share the same normalization.
    Absolute mode is deployable without proprioception.  Relative modes require
    a real or predicted source state and refuse to silently substitute action 0.
    """

    if actions.ndim < 2 or actions.shape[-2:] != (VIDEO_FRAMES, JOINT_DIM):
        raise ValueError(
            f"actions must end in [{VIDEO_FRAMES},{JOINT_DIM}], got {tuple(actions.shape)}"
        )
    commands = actions[..., :FUTURE_FRAMES, :]
    if mode == "absolute":
        return commands
    if mode not in {"delta", "delta_step"}:
        raise ValueError(f"Unsupported IRASim action mode: {mode!r}")
    if source_state is None:
        raise ValueError(
            f"{mode} requires source_state. The challenge does not provide it; "
            "pass a source-image state estimate rather than using action[0] as a proxy."
        )
    features = transition_action_features(actions, source_state)
    value = features[..., 6:12] if mode == "delta" else features[..., 12:18]
    if scale is not None:
        if scale.numel() != JOINT_DIM:
            raise ValueError(f"scale must contain {JOINT_DIM} values, got {scale.numel()}")
        value = value / scale.to(device=value.device, dtype=value.dtype).clamp_min(1e-6)
    return value
