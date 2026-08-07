"""Train-only DreamZero-SO101 input-contract gate.

This gate never uses evaluation data. It pairs the released SO101 prior with
held-out challenge training clips for which ``observation.state`` is known, so
the model receives the same state-anchored relative-action representation used
during its original training. It also exposes the single-view/missing-view
distribution shift and action counterfactuals without reloading the 14B model.

The oracle states are used only to construct this diagnostic. Predictions from
this script are not valid competition submissions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoTokenizer

from generate_dreamzero_so101_native import (
    REPO,
    convert_given_action_chunks,
    generate_one,
    load_model,
    make_three_view_canvas,
    save_video,
    validate_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default=str(REPO / "checkpoints/Wan2.1-I2V-14B-480P")
    )
    parser.add_argument(
        "--lora", default=str(REPO / "checkpoints/dreamzero-so101-lora")
    )
    parser.add_argument(
        "--metadata",
        default=str(
            REPO
            / "checkpoints/dreamzero-so101-checkpoints-meta/lora-100k/checkpoint-20000/experiment_cfg/metadata.json"
        ),
    )
    parser.add_argument("--tokenizer", default=str(REPO / "checkpoints/umt5-xxl"))
    parser.add_argument(
        "--challenge-root",
        default=str(REPO / "diagnostics/oracle_motion_field_gate/valset"),
    )
    parser.add_argument(
        "--oracle-manifest",
        default=str(REPO / "diagnostics/oracle_motion_field_gate/manifest.json"),
    )
    parser.add_argument(
        "--prediction-root",
        default=str(REPO / "diagnostics/dreamzero_so101_contract_gate"),
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Explicit train-only holdout ID; repeat for two or more samples.",
    )
    parser.add_argument(
        "--view-modes",
        nargs="+",
        choices=("front-only", "replicate-three"),
        default=("front-only", "replicate-three"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("normal", "zero", "batch-roll"),
        default=("normal", "zero", "batch-roll"),
    )
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--prompt", default="An SO-101 robot arm performs the manipulation task."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile-encoders", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts/normalization and write the audit without loading a GPU model.",
    )
    parser.add_argument("--benchmark-json")
    return parser.parse_args()


def resolve_records(args: argparse.Namespace) -> list[dict]:
    root = Path(args.challenge_root)
    records = json.loads(Path(args.oracle_manifest).read_text())
    by_id = {row["sample_id"]: row for row in records if row.get("split") == "holdout"}
    available = sorted(path.stem for path in (root / "images").glob("*.png"))
    selected_ids = args.sample_ids or available[: args.limit]
    if not selected_ids:
        raise ValueError("No train-only DreamZero contract samples selected")
    missing = [sid for sid in selected_ids if sid not in by_id]
    if missing:
        raise ValueError(f"Samples missing from oracle holdout manifest: {missing}")
    if "batch-roll" in args.variants and len(selected_ids) < 2:
        raise ValueError("batch-roll requires at least two selected samples")
    return [by_id[sid] for sid in selected_ids]


def load_record(record: dict, challenge_root: Path) -> dict[str, object]:
    sample_id = record["sample_id"]
    artifact_path = Path(record["artifact"])
    if not artifact_path.is_absolute():
        artifact_path = REPO / artifact_path
    with np.load(artifact_path) as artifact:
        actions = artifact["actions"].astype(np.float32)
        states = artifact["states"].astype(np.float32)
    if actions.shape[0] < 16 or actions.shape[1:] != (6,):
        raise ValueError(f"Bad oracle actions for {sample_id}: {actions.shape}")
    if states.shape[0] < 9 or states.shape[1:] != (6,):
        raise ValueError(f"Bad oracle states for {sample_id}: {states.shape}")
    val_actions = np.load(challenge_root / "actions" / f"{sample_id}.npy").astype(np.float32)
    if val_actions.shape != (16, 6) or not np.allclose(val_actions, actions[:16], atol=1e-5):
        raise ValueError(f"Valset/artifact action mismatch for {sample_id}")
    source = np.asarray(
        Image.open(challenge_root / "images" / f"{sample_id}.png").convert("RGB")
    )
    return {
        "sample_id": sample_id,
        "dataset": record["dataset"],
        "source": source,
        "actions": actions[:16],
        "states": states,
        "artifact": str(artifact_path),
    }


def condition_for(
    records: list[dict[str, object]], index: int, variant: str
) -> tuple[np.ndarray, np.ndarray, str]:
    current = records[index]
    if variant == "normal":
        return current["actions"].copy(), current["states"].copy(), current["sample_id"]
    if variant == "zero":
        states = current["states"].copy()
        action = np.empty((16, 6), dtype=np.float32)
        action[:8] = states[0]
        action[8:] = states[8]
        return action, states, current["sample_id"]
    if variant == "batch-roll":
        other = records[(index + 1) % len(records)]
        # Roll an already valid trajectory together with its state anchors. This
        # avoids manufacturing out-of-range deltas from two calibration systems.
        return other["actions"].copy(), other["states"].copy(), other["sample_id"]
    raise ValueError(variant)


def build_input_audit(
    args: argparse.Namespace, records: list[dict[str, object]]
) -> dict[str, object]:
    rows = []
    metadata = Path(args.metadata)
    for index, record in enumerate(records):
        variant_chunks: dict[str, list[np.ndarray]] = {}
        variants = {}
        for variant in args.variants:
            action, anchors, donor = condition_for(records, index, variant)
            chunks, report = convert_given_action_chunks(action, metadata, anchors)
            variant_chunks[variant] = chunks
            variants[variant] = {"donor_sample_id": donor, **report}
        comparisons = {}
        if "normal" in variant_chunks:
            normal = np.concatenate(variant_chunks["normal"], axis=1)
            for variant, chunks in variant_chunks.items():
                if variant != "normal":
                    other = np.concatenate(chunks, axis=1)
                    comparisons[f"normal_minus_{variant}_normalized_abs_mean"] = float(
                        np.abs(normal - other).mean()
                    )
        view_stats = {}
        for view_mode in args.view_modes:
            canvas = make_three_view_canvas(record["source"], view_mode)
            view_stats[view_mode] = {
                "nonzero_fraction": float(np.mean(canvas != 0)),
                "top_left_abs_mean": float(np.abs(canvas[:176, :320]).mean()),
                "bottom_left_abs_mean": float(np.abs(canvas[176:, :320]).mean()),
                "top_right_abs_mean": float(np.abs(canvas[:176, 320:]).mean()),
            }
        rows.append(
            {
                "sample_id": record["sample_id"],
                "dataset": record["dataset"],
                "artifact": record["artifact"],
                "variants": variants,
                "comparisons": comparisons,
                "views": view_stats,
            }
        )
    return {
        "scope": "train-only oracle input-contract audit; never submit these predictions",
        "samples": len(records),
        "view_modes": list(args.view_modes),
        "variants": list(args.variants),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    challenge_root = Path(args.challenge_root)
    manifest_records = resolve_records(args)
    records = [load_record(row, challenge_root) for row in manifest_records]
    output = Path(args.prediction_root)
    output.mkdir(parents=True, exist_ok=True)
    audit = build_input_audit(args, records)
    audit_path = output / "input_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"saved -> {audit_path}")
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("DreamZero-SO101 14B contract inference requires CUDA")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    # Attributes consumed by the shared native generator.
    args.action_conditioning = "given"
    args.state_mode = "metadata-center"
    validate_assets(args)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29622")
        dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = load_model(args)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    reports = {}

    try:
        for view_mode in args.view_modes:
            for variant in args.variants:
                for index, record in enumerate(records):
                    sample_id = record["sample_id"]
                    path = output / view_mode / variant / f"{sample_id}.mp4"
                    if path.exists() and not args.overwrite:
                        print(f"[DreamZero contract] skip existing {path}")
                        continue
                    condition_action, anchor_states, donor = condition_for(
                        records, index, variant
                    )
                    frames, composite, report = generate_one(
                        model,
                        tokenizer,
                        record["source"],
                        record["actions"],
                        args,
                        conditioned_action=condition_action,
                        action_anchor_states=anchor_states,
                        view_mode=view_mode,
                    )
                    save_video(frames, path, args.fps)
                    save_video(
                        composite,
                        output / "_composite" / view_mode / variant / f"{sample_id}.mp4",
                        args.fps,
                    )
                    key = f"{view_mode}/{variant}/{sample_id}"
                    reports[key] = {
                        "source_sample_id": sample_id,
                        "action_donor_sample_id": donor,
                        **report,
                    }
                    print(f"[DreamZero contract] {key} -> {path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    seconds = time.perf_counter() - started
    benchmark = {
        "scope": "train-only oracle input-contract gate; never submit these predictions",
        "generated": len(reports),
        "seconds": seconds,
        "seconds_per_video": seconds / max(1, len(reports)),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "view_modes": list(args.view_modes),
        "variants": list(args.variants),
        "num_inference_steps_per_chunk": args.num_inference_steps,
        "reports": reports,
    }
    benchmark_path = (
        Path(args.benchmark_json)
        if args.benchmark_json
        else output / "contract_benchmark.json"
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(benchmark, indent=2))
    print(json.dumps(benchmark, indent=2))
    print(f"saved -> {benchmark_path}")


if __name__ == "__main__":
    main()
