"""Fail-fast audit for the long Cosmos unified SO-100 run.

This intentionally avoids importing Cosmos or allocating a GPU.  Its job is to
prevent a missing/stale latent cache from silently selecting the raw-video path
after the expensive model has already been loaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
LATENT_DIR = Path(os.environ.get("COSMOS_LATENT_DIR", HERE / "latents"))
VAL_LATENT_DIR = Path(os.environ.get("COSMOS_VAL_LATENT_DIR", HERE / "val_latents"))
MANIFEST = Path(os.environ.get("COSMOS_LATENT_MANIFEST", HERE / "latent_manifest.parquet"))
INDEX = Path(os.environ.get("SO100_INDEX_PATH", HERE / "episode_index.parquet"))
CLEAN_FILE = HERE / "runs" / "data_clean" / os.environ.get("CLEAN_FILE", "episode_weight_r1.parquet")


def pt_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {entry.name for entry in os.scandir(path) if entry.is_file() and entry.name.endswith(".pt")}


def main() -> None:
    missing_files = [str(path) for path in (MANIFEST, INDEX, CLEAN_FILE) if not path.is_file()]
    if missing_files:
        raise SystemExit("missing required metadata: " + ", ".join(missing_files))

    manifest = pd.read_parquet(MANIFEST)
    if not {"path", "motion"}.issubset(manifest.columns):
        raise SystemExit(f"invalid latent manifest columns: {manifest.columns.tolist()}")
    expected = set(manifest.path.astype(str))
    actual = pt_names(LATENT_DIR)
    missing_latents = sorted(expected - actual)
    extra_latents = sorted(actual - expected)

    index = pd.read_parquet(INDEX)
    clean = pd.read_parquet(CLEAN_FILE)
    train_episodes = int((index.split == "train").sum())
    val_episodes = int((index.split == "val").sum())
    if len(clean) != train_episodes or "weight" not in clean.columns:
        raise SystemExit(
            f"clean weights do not match train episodes: clean={len(clean)} train={train_episodes}"
        )
    positive_episodes = int((clean.weight > 0).sum())
    excluded_episodes = int((clean.weight <= 0).sum())

    val_count = len(pt_names(VAL_LATENT_DIR))
    report = {
        "manifest_windows": len(expected),
        "cached_train_windows": len(actual),
        "missing_train_windows": len(missing_latents),
        "extra_train_windows": len(extra_latents),
        "missing_examples": missing_latents[:5],
        "extra_examples": extra_latents[:5],
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "positive_clean_episodes": positive_episodes,
        "excluded_clean_episodes": excluded_episodes,
        "cached_val_windows": val_count,
        "train_cache_exact": not missing_latents and not extra_latents and bool(expected),
        "val_cache_available": val_count > 0,
    }
    report["verdict"] = (
        "PASS_UNIFIED_INPUTS"
        if report["train_cache_exact"] and report["val_cache_available"]
        else "REJECT_UNIFIED_INPUTS"
    )
    print(json.dumps(report, indent=2))
    if not report["verdict"].startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
