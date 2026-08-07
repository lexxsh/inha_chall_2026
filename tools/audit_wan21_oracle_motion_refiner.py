"""CPU-only structural/data audit for the Wan oracle motion refiner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/DiffSynth-Studio"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from diffsynth.models.wan_video_dit import OracleMotionFieldAdapter, WanModel  # noqa: E402
from diffsynth.pipelines.wan_video import model_fn_wan_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-root", default=str(REPO / "diagnostics/oracle_motion_field_gate")
    )
    parser.add_argument(
        "--out", default=str(REPO / "results/wan21_oracle_motion_refiner_audit.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.oracle_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing. Run PHASE=prepare before the audit."
        )
    records = json.loads(manifest_path.read_text())
    shapes = set()
    first_frame_max = []
    finite = []
    nonzero = []
    missing = []
    controls = []
    for record in records:
        path = Path(record["motion_artifact"])
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            missing.append(str(path))
            continue
        with np.load(path) as artifact:
            control = artifact["control"].astype(np.float32)
        controls.append(control)
        shapes.add(tuple(control.shape))
        first_frame_max.append(float(np.abs(control[:, 0]).max()))
        finite.append(bool(np.isfinite(control).all()))
        nonzero.append(float(np.abs(control[:, 1:]).mean()))

    adapter = OracleMotionFieldAdapter(
        control_dim=7,
        model_dim=64,
        source_dim=16,
        hidden_dim=32,
        num_layers=8,
        injection_layers=(0, 2, 4, 6),
    ).eval()
    source = torch.randn(2, 16, 8, 12)
    zero = torch.zeros(2, 7, 17, 32, 48)
    signal = torch.randn_like(zero) * 0.1
    with torch.no_grad():
        hidden_zero = adapter(zero, source, (5, 4, 6))
        hidden_signal = adapter(signal, source, (5, 4, 6))
        residual_zero = adapter.residual_for_block(hidden_zero, 0)

    # Exercise the exact pipeline hook with a tiny CPU Wan, including image
    # latent concatenation, patchification, block injection, and unpatchify.
    tiny = WanModel(
        dim=96, in_dim=36, ffn_dim=192, out_dim=16, text_dim=32, freq_dim=16,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
        has_image_input=True, require_vae_embedding=True, require_clip_embedding=False,
    ).eval()
    tiny.enable_oracle_motion_field_conditioning(
        control_dim=7, hidden_dim=32, source_dim=16, injection_layers=(0, 1)
    )
    latents = torch.randn(1, 16, 5, 8, 12)
    image_latents = torch.randn(1, 20, 5, 8, 12)
    context = torch.randn(1, 3, 32)
    with torch.no_grad():
        pipeline_output = model_fn_wan_video(
            dit=tiny,
            latents=latents,
            timestep=torch.tensor([500.0]),
            context=context,
            y=image_latents,
            motion_field_control=signal[:1],
        )
    cross_difference = 0.0
    if len(controls) >= 2:
        cross_difference = float(np.mean(np.abs(controls[0] - controls[1])))
    train_count = sum(record["split"] == "train" for record in records)
    holdout_count = sum(record["split"] == "holdout" for record in records)
    structural_pass = (
        float(hidden_zero.abs().max()) == 0.0
        and float(residual_zero.abs().max()) == 0.0
        and float(hidden_signal.abs().mean()) > 0.0
        and pipeline_output.shape == latents.shape
        and bool(torch.isfinite(pipeline_output).all())
    )
    data_pass = (
        not missing
        and shapes == {(7, 17, 320, 432)}
        and all(finite)
        and max(first_frame_max, default=1.0) == 0.0
        and np.median(nonzero) > 1e-5
        and cross_difference > 1e-5
        and train_count >= 32
        and holdout_count >= 8
    )
    report = {
        "records": len(records),
        "train_records": train_count,
        "holdout_records": holdout_count,
        "control_shapes": [list(shape) for shape in sorted(shapes)],
        "missing_artifacts": missing[:10],
        "all_finite": bool(all(finite)),
        "max_frame0_abs": max(first_frame_max, default=float("nan")),
        "median_future_abs_mean": float(np.median(nonzero)) if nonzero else 0.0,
        "first_pair_abs_difference": cross_difference,
        "adapter_parameters_test_shape": sum(parameter.numel() for parameter in adapter.parameters()),
        "zero_hidden_abs_max": float(hidden_zero.abs().max()),
        "signal_hidden_abs_mean": float(hidden_signal.abs().mean()),
        "zero_residual_abs_max": float(residual_zero.abs().max()),
        "pipeline_output_shape": list(pipeline_output.shape),
        "pipeline_output_finite": bool(torch.isfinite(pipeline_output).all()),
        "structural_zero_gate": bool(structural_pass),
        "data_gate": bool(data_pass),
        "verdict": "PASS_CPU_AUDIT" if structural_pass and data_pass else "REJECT_CPU_AUDIT",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved -> {output}")
    if report["verdict"] != "PASS_CPU_AUDIT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
