"""Local video diagnostics for Cosmos SO-100 experiments."""

from __future__ import annotations

import os

import torch

from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.callbacks.every_n_draw_sample import (
    EveryNDrawSample,
)


class LocalVideoEveryNDrawSample(EveryNDrawSample):
    """Save every generated guidance row and GT as an MP4 on local disk."""

    def run_save(self, to_show, batch_size, base_fp_wo_ext):
        stacked = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0
        labels = [f"guidance_{value:g}" for value in self.guidance] + ["gt"]
        n_samples = min(self.n_viz_sample, batch_size)
        os.makedirs(self.local_dir, exist_ok=True)
        for row, label in enumerate(labels):
            for sample_idx in range(n_samples):
                output = os.path.join(
                    self.local_dir,
                    f"{base_fp_wo_ext}_{label}_sample{sample_idx:02d}",
                )
                save_img_or_video(
                    stacked[row, sample_idx],
                    output,
                    fps=self.fps,
                )
        return super().run_save(to_show, batch_size, base_fp_wo_ext)
