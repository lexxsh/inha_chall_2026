"""Dependency-light regression tests for the SO-100 transition contract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train.so100_transition_contract import (  # noqa: E402
    build_transition_contract,
    irasim_transition_actions,
    transition_action_features,
)


class TransitionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.states = np.arange(16 * 6, dtype=np.float32).reshape(16, 6)
        self.actions = self.states + np.linspace(1, 16, 16, dtype=np.float32)[:, None]

    def test_visible_horizon_uses_action_zero_through_fourteen(self) -> None:
        contract = build_transition_contract(self.actions, self.states)
        np.testing.assert_array_equal(contract.commands[0], self.actions[0])
        np.testing.assert_array_equal(contract.commands[-1], self.actions[14])
        self.assertFalse(np.array_equal(contract.commands[-1], self.actions[15]))

    def test_deployable_step_starts_from_source_state(self) -> None:
        contract = build_transition_contract(self.actions, self.states)
        np.testing.assert_array_equal(
            contract.deployable_step[0], self.actions[0] - self.states[0]
        )
        np.testing.assert_array_equal(
            contract.deployable_step[1:], self.actions[1:15] - self.actions[:14]
        )

    def test_torch_features_and_irasim_views_agree(self) -> None:
        actions = torch.from_numpy(self.actions)
        source = torch.from_numpy(self.states[0])
        features = transition_action_features(actions, source)
        self.assertEqual(tuple(features.shape), (15, 18))
        torch.testing.assert_close(
            irasim_transition_actions(actions, source, "absolute"), actions[:15]
        )
        torch.testing.assert_close(
            irasim_transition_actions(actions, source, "delta"), features[:, 6:12]
        )
        torch.testing.assert_close(
            irasim_transition_actions(actions, source, "delta_step"), features[:, 12:18]
        )

    def test_relative_mode_rejects_missing_source_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires source_state"):
            irasim_transition_actions(torch.from_numpy(self.actions), None, "delta_step")


if __name__ == "__main__":
    unittest.main()
