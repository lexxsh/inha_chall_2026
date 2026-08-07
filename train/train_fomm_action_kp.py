"""Fast action-to-keypoint overfit gate on a frozen FOMM renderer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import trange

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from source_anchored_fomm import ActionKeypointPredictor, SourceAnchoredFOMM, so100_action_features  # noqa: E402
from train_fomm_oracle import FixedOracleClip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer-checkpoint", required=True)
    parser.add_argument("--output", default=str(REPO / "open/baseline/outputs/fomm_action_kp_hard101"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_scales(device: torch.device) -> tuple[torch.Tensor, ...]:
    stats = json.loads((REPO / "open/data/train/so100_action_statistics.json").read_text())
    delta = json.loads((REPO / "train/delta_action_stats.json").read_text())["scale_by_mode"]
    return (
        torch.tensor(stats["mean"], device=device),
        torch.tensor(stats["std"], device=device),
        torch.tensor(delta["delta"], device=device),
        torch.tensor(delta["delta_step"], device=device),
    )


def features(actions: torch.Tensor, scales: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return so100_action_features(actions, *scales)


def kp_error(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    value = F.smooth_l1_loss(prediction["value"][:, 1:], target["value"][None, 1:])
    jacobian = F.smooth_l1_loss(
        prediction["jacobian"][:, 1:], target["jacobian"][None, 1:]
    )
    pred_velocity = prediction["value"][:, 1:] - prediction["value"][:, :-1]
    target_velocity = target["value"][1:] - target["value"][:-1]
    temporal = F.smooth_l1_loss(pred_velocity, target_velocity[None])
    return value + 0.10 * jacobian + 0.25 * temporal


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("This gate expects one CUDA GPU.")
    device = torch.device("cuda")
    renderer_path = Path(args.renderer_checkpoint)
    renderer_config = json.loads((renderer_path.parent / "config.json").read_text())
    dataset = FixedOracleClip(
        renderer_config["data_root"],
        renderer_config["dataset_index"],
        renderer_config["clip_index"],
        renderer_config["height"],
        renderer_config["width"],
        renderer_config["seed"],
    )
    renderer = SourceAnchoredFOMM(renderer_config["num_kp"], renderer_config["model_size"])
    renderer.load_state_dict(load_file(str(renderer_path), device="cpu"))
    renderer.to(device).eval().requires_grad_(False)
    video = dataset.video.to(device)
    source = video[:1]
    # These tensors are frozen targets, but SmoothL1 still saves them while
    # computing gradients for the predictor. `inference_mode` tensors forbid
    # that; `no_grad` keeps them ordinary non-requires-grad tensors.
    with torch.no_grad():
        source_kp = renderer.kp_detector(source)
        target_kp = renderer.kp_detector(video)

    predictor = ActionKeypointPredictor(
        num_kp=renderer_config["num_kp"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scales = load_scales(device)
    action = dataset.actions[None].to(device)
    correct_features = features(action, scales)
    wrong_actions = [
        action[:, :1].expand_as(action),
        action.flip(1),
        action.roll(4, dims=1),
    ]
    wrong_features = [features(item, scales) for item in wrong_actions]

    progress = trange(1, args.max_steps + 1)
    output = Path(args.output)
    for step in progress:
        predictor.train()
        correct = predictor(correct_features, source_kp)
        wrong = predictor(wrong_features[(step - 1) % len(wrong_features)], source_kp)
        correct_error = kp_error(correct, target_kp)
        wrong_error = kp_error(wrong, target_kp)
        ranking = F.relu(0.03 + correct_error - wrong_error)
        sensitivity = F.relu(0.025 - (correct["value"][:, 1:] - wrong["value"][:, 1:]).abs().mean())
        loss = correct_error + 0.50 * ranking + 0.25 * sensitivity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        progress.set_postfix(
            loss=f"{loss.item():.5f}", correct=f"{correct_error.item():.5f}",
            wrong=f"{wrong_error.item():.5f}", sens=f"{sensitivity.item():.5f}"
        )
        if step % args.save_steps == 0 or step == args.max_steps:
            output.mkdir(parents=True, exist_ok=True)
            state = {key: value.detach().cpu().contiguous() for key, value in predictor.state_dict().items()}
            save_file(state, output / f"step-{step}.safetensors")

    config = {
        **vars(args),
        "renderer_checkpoint": str(renderer_path.resolve()),
        "dataset_index": renderer_config["dataset_index"],
        "clip_index": renderer_config["clip_index"],
        "num_kp": renderer_config["num_kp"],
        "model_size": renderer_config["model_size"],
        "height": renderer_config["height"],
        "width": renderer_config["width"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved -> {output.resolve()}")


if __name__ == "__main__":
    main()
