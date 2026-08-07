"""Generate competition-format SO-100 videos with native Cosmos3 forward dynamics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from cosmos3_so100_action import SO100Cosmos3ActionConverter  # noqa: E402


MODE_NAMES = ("none", "zero-motion", "reverse", "batch-roll")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--challenge-root", type=Path, default=REPO / "valset_holdout")
    parser.add_argument("--prediction-root", type=Path, default=REPO / "diagnostics/cosmos3_edge_so100")
    parser.add_argument("--limit", type=int, default=8, help="0 means all challenge samples")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--resolution-tier", type=int, choices=(256, 480), default=480)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-ablation", choices=(*MODE_NAMES, "smoke", "all"), default="none")
    parser.add_argument(
        "--prompt",
        default="A fixed-camera video of a SO-100 robot arm performing a tabletop manipulation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--benchmark-json", type=Path, default=None)
    return parser.parse_args()


def list_sample_ids(root: Path, limit: int) -> list[str]:
    image_ids = {path.stem for path in (root / "images").glob("*.png")}
    action_ids = {path.stem for path in (root / "actions").glob("*.npy")}
    ids = sorted(image_ids & action_ids)
    if not ids:
        raise FileNotFoundError(f"No paired images/actions under {root}")
    return ids[:limit] if limit > 0 else ids


def variant_actions(
    mode: str,
    sample_index: int,
    sample_ids: list[str],
    actions: dict[str, np.ndarray],
) -> np.ndarray:
    current = actions[sample_ids[sample_index]]
    if mode == "none":
        return current
    if mode == "zero-motion":
        source = SO100Cosmos3ActionConverter.estimate_source_action(current[0])
        return np.repeat(source[None], 16, axis=0).astype(np.float32)
    if mode == "reverse":
        return current[::-1].copy()
    if mode == "batch-roll":
        other = actions[sample_ids[(sample_index + 1) % len(sample_ids)]]
        return other
    raise ValueError(mode)


def normalize_result_video(video: object) -> list[Image.Image]:
    frames = video
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        frames = list(frames)
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
        frames = frames[0]
    if not isinstance(frames, list):
        raise TypeError(f"Unexpected Cosmos3 video type: {type(frames)!r}")
    output = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            output.append(frame.convert("RGB"))
        else:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            output.append(Image.fromarray(array).convert("RGB"))
    return output


def main() -> None:
    args = parse_args()
    if args.num_inference_steps <= 0:
        raise SystemExit("--num-inference-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Cosmos3 Edge generation requires CUDA; run this command on an H100 node")

    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    root = args.challenge_root.resolve()
    sample_ids = list_sample_ids(root, args.limit)
    actions = {
        sid: np.load(root / "actions" / f"{sid}.npy").astype(np.float32) for sid in sample_ids
    }
    for sid, action in actions.items():
        if action.shape != (16, 6):
            raise ValueError(f"{sid}: expected challenge actions [16,6], got {action.shape}")
    if args.action_ablation == "batch-roll" and len(sample_ids) < 2:
        raise ValueError("batch-roll requires --limit >= 2")

    converter = SO100Cosmos3ActionConverter()
    print(f"[Cosmos3 Edge] loading {args.model}", flush=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        enable_safety_checker=False,
        local_files_only=args.local_files_only,
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=10.0,
        use_karras_sigmas=False,
    )
    torch.cuda.reset_peak_memory_stats()

    if args.action_ablation == "all":
        modes = MODE_NAMES
    elif args.action_ablation == "smoke":
        modes = MODE_NAMES[:3]
    else:
        modes = (args.action_ablation,)
    started = time.perf_counter()
    mode_reports: dict[str, dict] = {}
    generated = 0
    for mode in modes:
        multi_mode = args.action_ablation in ("smoke", "all")
        output = args.prediction_root / mode if multi_mode else args.prediction_root
        output.mkdir(parents=True, exist_ok=True)
        mode_started = time.perf_counter()
        mode_count = 0
        for index, sid in enumerate(sample_ids):
            target = output / f"{sid}.mp4"
            if target.exists() and not args.overwrite:
                mode_count += 1
                generated += 1
                continue
            commands = variant_actions(mode, index, sample_ids, actions)
            # Keep the source-state estimate tied to the current source image,
            # including for foreign/reversed counterfactual action sequences.
            current = actions[sid]
            source_proxy = converter.estimate_source_action(current[0])
            cosmos_action = converter.build_actions(commands, source_action=source_proxy)
            original = Image.open(root / "images" / f"{sid}.png").convert("RGB")
            generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
            result = pipe(
                prompt=args.prompt,
                action=CosmosActionCondition(
                    mode="forward_dynamics",
                    chunk_size=16,
                    domain_name="bridge_orig_lerobot",
                    resolution_tier=args.resolution_tier,
                    raw_actions=cosmos_action,
                    image=original,
                    view_point="third_person_view",
                ),
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,
                generator=generator,
                use_system_prompt=False,
            )
            frames = normalize_result_video(result.video)
            if len(frames) != 17:
                raise RuntimeError(f"{sid}: Cosmos3 returned {len(frames)} frames, expected 17")
            # Challenge frame 0 is exactly the supplied source image. Visible
            # transitions use actions 0..14; action 15 predicts frame 16, which
            # is outside the required 16-frame submission window.
            submission_frames = [original] + [
                frame.resize(original.size, Image.Resampling.LANCZOS) for frame in frames[1:16]
            ]
            export_to_video(submission_frames, str(target), fps=args.fps, macro_block_size=1)
            mode_count += 1
            generated += 1
            elapsed = time.perf_counter() - mode_started
            print(
                f"[Cosmos3 Edge {mode}] {mode_count}/{len(sample_ids)} "
                f"{elapsed / mode_count:.1f}s/sample -> {target}",
                flush=True,
            )
        mode_reports[mode] = {
            "samples": mode_count,
            "seconds": time.perf_counter() - mode_started,
            "prediction_root": str(output.resolve()),
        }

    report = {
        "model": args.model,
        "challenge_root": str(root),
        "samples": len(sample_ids),
        "generated_mode_samples": generated,
        "seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "num_inference_steps": args.num_inference_steps,
        "resolution_tier": args.resolution_tier,
        "fps": args.fps,
        "domain_name": "bridge_orig_lerobot",
        "viewpoint": "third_person_view",
        "action_contract": "unclamped normalized backward_framewise [dxyz3, column_rot6d6, gripper1]",
        "source_state_policy": "action0 plus train-only median state0-minus-action0 offset",
        "source_frame_zero_replaced": True,
        "modes": mode_reports,
    }
    print(json.dumps(report, indent=2))
    benchmark = args.benchmark_json or args.prediction_root / "benchmark.json"
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text(json.dumps(report, indent=2))
    print(f"saved -> {benchmark.resolve()}")


if __name__ == "__main__":
    main()
