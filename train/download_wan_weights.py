"""Resumable, serial Hugging Face download for the Wan action experiment."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# huggingface_hub constants are initialized at import time.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HOME", str(REPO / "models/.hf_cache"))
os.environ.setdefault("HF_XET_CACHE", str(REPO / "models/.hf_cache/xet"))

from huggingface_hub import snapshot_download


DOWNLOADS = (
    (
        "Wan-AI/Wan2.2-TI2V-5B",
        ["models_t5_umt5-xxl-enc-bf16.pth"],
    ),
    (
        "Wan-AI/Wan2.2-TI2V-5B",
        ["Wan2.2_VAE.pth"],
    ),
    (
        "Wan-AI/Wan2.2-TI2V-5B",
        ["diffusion_pytorch_model*.safetensors"],
    ),
    (
        "Wan-AI/Wan2.1-T2V-1.3B",
        ["google/umt5-xxl/*"],
    ),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(REPO / "models"))
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.model_root)
    root.mkdir(parents=True, exist_ok=True)

    # huggingface_hub reads these for every streamed request. A long read
    # timeout plus max_workers=1 is slower in ideal conditions but much more
    # reliable on the competition cluster/NFS route.
    for repo_id, patterns in DOWNLOADS:
        local_dir = root / repo_id
        for attempt in range(1, args.retries + 1):
            try:
                print(f"[download] {repo_id} {patterns} (attempt {attempt}/{args.retries})")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    allow_patterns=patterns,
                    max_workers=1,
                    resume_download=True,
                )
                break
            except Exception as error:
                if attempt == args.retries:
                    raise
                wait = min(5 * attempt, 60)
                print(f"[retry] {type(error).__name__}: {error}; retry in {wait}s")
                time.sleep(wait)
    print(f"Wan weights ready under {root}")


if __name__ == "__main__":
    main()
