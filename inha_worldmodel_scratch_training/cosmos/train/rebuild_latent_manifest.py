"""Rebuild the train latent manifest from the current episode index.

The repository may ship a manifest generated from an older episode ordering.
Latent filenames intentionally use the *filtered train-row index*, so rebuilding
episode_index.parquet can keep the same number of windows while changing many
filenames.  Deriving paths and motion from the current action parquet avoids
opening 100k small latent files and guarantees the sampler matches this cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from so100_dataset import INDEX_PATH, NUM_FRAMES, TRAIN_ROOT


HERE = Path(__file__).resolve().parent
LATENT_DIR = Path(os.environ.get("COSMOS_LATENT_DIR", HERE / "latents"))
MANIFEST = Path(os.environ.get("COSMOS_LATENT_MANIFEST", HERE / "latent_manifest.parquet"))
STRIDE = 8


def main() -> None:
    index = pd.read_parquet(INDEX_PATH)
    episodes = index[index.split == "train"].reset_index(drop=True)
    stats = json.loads((TRAIN_ROOT / "so100_action_statistics.json").read_text(encoding="utf-8"))
    action_mean = np.asarray(stats["mean"], dtype=np.float32)
    action_std = np.asarray(stats["std"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    expected: set[str] = set()
    for ep_i, row in enumerate(tqdm(episodes.itertuples(index=False), total=len(episodes), desc="manifest")):
        actions = np.stack(
            pd.read_parquet(TRAIN_ROOT / row.parquet, columns=["action"])["action"].to_numpy()
        ).astype(np.float32)
        actions = (actions - action_mean) / action_std
        for start in range(0, int(row.num_frames) - NUM_FRAMES + 1, STRIDE):
            name = f"{ep_i:06d}_{start:04d}.pt"
            act = actions[start : start + NUM_FRAMES - 1]
            motion = float(np.abs(np.diff(act, axis=0)).mean()) if len(act) > 1 else 0.0
            rows.append({"path": name, "motion": motion})
            expected.add(name)

    actual = {
        entry.name
        for entry in os.scandir(LATENT_DIR)
        if entry.is_file() and entry.name.endswith(".pt")
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    report = {
        "manifest_windows": len(rows),
        "cached_windows": len(actual),
        "missing": len(missing),
        "extra": len(extra),
        "missing_examples": missing[:5],
        "extra_examples": extra[:5],
    }
    print(json.dumps(report, indent=2), flush=True)
    if missing or extra or not rows:
        raise SystemExit("latent cache does not exactly match the current episode index")

    frame = pd.DataFrame(rows)
    temp = MANIFEST.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(MANIFEST)
    print(f"saved -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
