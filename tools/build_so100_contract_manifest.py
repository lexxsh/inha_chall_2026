"""Build a deterministic uploader-held-out SO-100 transition manifest.

This script reads metadata only.  It never uses evaluation inputs and never
decodes video.  Every record identifies one 16-frame window whose visible
transitions are action[0:15] -> frame[1:16].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
KIT_SRC = REPO / "open/baseline/challenge_kit/src"
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

from ldwma.datasets.lerobot_so100 import (  # noqa: E402
    _format_lerobot_path,
    _select_video_key,
    discover_lerobot_so100_datasets,
)


DEFAULT_EXCLUDE = ("dragon-95/so100_sorting",)
FRAMES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "open/data/train")
    parser.add_argument(
        "--output", type=Path, default=REPO / "results/so100_contract_manifest.jsonl"
    )
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--holdout-uploaders", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_seed(identity: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def episode_windows(length: int, count: int, identity: str, seed: int) -> list[int]:
    max_start = length - FRAMES
    if max_start < 0:
        return []
    candidates = list(range(max_start + 1))
    rng = random.Random(stable_seed(identity, seed))
    if count >= len(candidates):
        return candidates
    return sorted(rng.sample(candidates, count))


def main() -> None:
    args = parse_args()
    if args.windows_per_episode <= 0:
        raise SystemExit("--windows-per-episode must be positive")
    paths = [
        path
        for path in discover_lerobot_so100_datasets(args.data_root)
        if not any(path.endswith(excluded) for excluded in args.exclude)
    ]
    uploaders = sorted({path.split("/", 1)[0] for path in paths})
    if not 0 < args.holdout_uploaders < len(uploaders):
        raise SystemExit(
            f"--holdout-uploaders must be in [1,{len(uploaders) - 1}], "
            f"got {args.holdout_uploaders}"
        )
    shuffled = list(uploaders)
    random.Random(args.seed).shuffle(shuffled)
    holdout = set(shuffled[: args.holdout_uploaders])

    records: list[dict] = []
    missing: list[str] = []
    for dataset_path in sorted(paths):
        dataset_root = args.data_root / dataset_path
        info = read_json(dataset_root / "meta/info.json")
        episodes_path = dataset_root / "meta/episodes.jsonl"
        if not episodes_path.exists():
            missing.append(episodes_path.as_posix())
            continue
        camera_key = _select_video_key(info["features"], "auto")
        chunks_size = int(info.get("chunks_size", 1000))
        uploader, dataset = dataset_path.split("/", 1)
        split = "holdout" if uploader in holdout else "train"
        for episode in read_jsonl(episodes_path):
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            identity = f"{dataset_path}:{episode_index}"
            data_rel = Path(dataset_path) / _format_lerobot_path(
                info["data_path"], episode_index, chunks_size
            )
            video_rel = Path(dataset_path) / _format_lerobot_path(
                info["video_path"], episode_index, chunks_size, camera_key
            )
            if not (args.data_root / data_rel).exists():
                missing.append(data_rel.as_posix())
                continue
            if not (args.data_root / video_rel).exists():
                missing.append(video_rel.as_posix())
                continue
            for start in episode_windows(
                length, args.windows_per_episode, identity, args.seed
            ):
                record_id = hashlib.sha256(f"{identity}:{start}".encode()).hexdigest()[:16]
                records.append(
                    {
                        "id": record_id,
                        "split": split,
                        "uploader": uploader,
                        "dataset": dataset,
                        "dataset_path": dataset_path,
                        "episode_index": episode_index,
                        "start": start,
                        "frames": FRAMES,
                        "fps": int(info.get("fps", 6)),
                        "camera_key": camera_key,
                        "task": (episode.get("tasks") or [""])[0],
                        "data_rel": data_rel.as_posix(),
                        "video_rel": video_rel.as_posix(),
                        "source_frame_index": start,
                        "target_frame_indices": list(range(start + 1, start + FRAMES)),
                        "command_indices": list(range(start, start + FRAMES - 1)),
                        "excluded_command_index": start + FRAMES - 1,
                    }
                )

    records.sort(key=lambda row: (row["split"], row["uploader"], row["dataset"], row["episode_index"], row["start"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "holdout")
    }
    train_uploaders = sorted({r["uploader"] for r in records if r["split"] == "train"})
    holdout_uploaders = sorted({r["uploader"] for r in records if r["split"] == "holdout"})
    summary = {
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.output.resolve()),
        "records": len(records),
        "split_counts": split_counts,
        "datasets": len(paths),
        "train_uploaders": train_uploaders,
        "holdout_uploaders": holdout_uploaders,
        "uploader_overlap": sorted(set(train_uploaders) & set(holdout_uploaders)),
        "windows_per_episode": args.windows_per_episode,
        "seed": args.seed,
        "contract": {
            "source_frame": 0,
            "target_frames": [1, 15],
            "used_actions": [0, 14],
            "excluded_action": 15,
        },
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "verdict": "PASS_MANIFEST" if records and not missing and not (set(train_uploaders) & set(holdout_uploaders)) else "FAIL_MANIFEST",
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
