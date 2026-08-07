from __future__ import annotations

import unittest

import torch

from train.train_so100_grounding_probe import CachedSplit, GroundingProbe, partition_holdout_uploaders
from train.flow_world_model import ConservativeFlowWorldModel


class AnchoredGroundingProbeTest(unittest.TestCase):
    def test_initial_prediction_preserves_physical_and_visual_baselines(self) -> None:
        torch.manual_seed(0)
        model = GroundingProbe(dino_dim=24, hidden_dim=32, layers=1).eval()
        source = torch.randn(3, 24)
        actions = torch.randn(3, 16, 6)
        output = model(source, actions)

        expected_dino = torch.nn.functional.normalize(source, dim=-1)
        self.assertTrue(torch.equal(output["state0"], actions[:, 0]))
        self.assertTrue(torch.equal(output["future_state"], actions[:, :15]))
        self.assertTrue(
            torch.allclose(
                output["future_dino"], expected_dino[:, None].expand(-1, 15, -1), atol=1e-7
            )
        )

    def test_zero_motion_repeats_first_command(self) -> None:
        model = GroundingProbe(dino_dim=24, hidden_dim=32, layers=1).eval()
        source = torch.randn(2, 24)
        actions = torch.randn(2, 16, 6)
        output = model(source, actions, action_variant="zero")
        self.assertTrue(torch.equal(output["future_state"], actions[:, :1].expand(-1, 15, -1)))

    def test_confirmatory_uploader_partition_is_disjoint_and_deterministic(self) -> None:
        uploaders = ["train"] * 2 + [name for name in "abcdefgh" for _ in range(2)]
        payload = {
            "split": torch.tensor([0, 0] + [1] * 16, dtype=torch.uint8),
            "uploaders": uploaders,
            "actions": torch.zeros(18, 16, 6),
            "states": torch.zeros(18, 16, 6),
            "dino": torch.zeros(18, 16, 24),
        }
        selection, confirmation = partition_holdout_uploaders(payload, 123)
        repeated = partition_holdout_uploaders(payload, 123)
        self.assertEqual((selection, confirmation), repeated)
        self.assertFalse(set(selection) & set(confirmation))
        self.assertEqual(set(selection) | set(confirmation), set("abcdefgh"))
        self.assertEqual(len(CachedSplit(payload, 1, set(selection))), 8)
        self.assertEqual(len(CachedSplit(payload, 1, set(confirmation))), 8)

    def test_grounded_renderer_is_static_at_initialization(self) -> None:
        model = ConservativeFlowWorldModel(
            grounding_dim=24, hidden_dim=32, pretrained_encoder=False
        ).eval()
        source = torch.rand(2, 3, 32, 48)
        motion = torch.randn(2, 15, 24)
        prediction = model(source, motion_tokens=motion)
        self.assertEqual(tuple(prediction.shape), (2, 16, 3, 32, 48))
        self.assertTrue(torch.equal(prediction[:, 0], source))
        self.assertTrue(torch.allclose(prediction[:, 1:], source[:, None], atol=1e-6))

        prediction.sum().backward()
        unused = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(unused, [])
        self.assertFalse(model.action_position.requires_grad)


if __name__ == "__main__":
    unittest.main()
