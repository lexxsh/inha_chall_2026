from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.cache_so100_grounding_features import extract_ragged_dino_features


class MeanColorModel(torch.nn.Module):
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.mean(dim=(2, 3))


class RaggedDinoFeatureTest(unittest.TestCase):
    def test_mixed_resolutions_preserve_order_and_shape(self) -> None:
        videos = [
            np.full((16, 40, 60, 3), 20, dtype=np.uint8),
            np.full((16, 32, 32, 3), 100, dtype=np.uint8),
            np.full((16, 40, 60, 3), 220, dtype=np.uint8),
        ]
        features = extract_ragged_dino_features(
            videos, MeanColorModel(), torch.device("cpu"), image_size=28
        )

        self.assertEqual(tuple(features.shape), (3, 16, 3))
        # The output must be restored to input order, not resolution-group order.
        brightness = features.mean(dim=(1, 2))
        self.assertLess(float(brightness[0]), float(brightness[1]))
        self.assertLess(float(brightness[1]), float(brightness[2]))

    def test_mixed_resolution_single_frame_inference(self) -> None:
        images = [
            np.full((1, 48, 64, 3), 40, dtype=np.uint8),
            np.full((1, 72, 128, 3), 180, dtype=np.uint8),
        ]
        features = extract_ragged_dino_features(
            images, MeanColorModel(), torch.device("cpu"), image_size=28
        )
        self.assertEqual(tuple(features.shape), (2, 1, 3))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
