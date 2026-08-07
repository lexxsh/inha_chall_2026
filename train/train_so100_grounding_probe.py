"""Train a cheap action-grounding gate before any RGB video generator.

The probe tests three questions on uploader-held-out data:

1. Does the dataset's direct command mapping action[t] -> state[t+1] hold on
   uploader-held-out data?
2. Can action-conditioned residuals from the source DINO feature beat a
   static-source prediction?
3. Do correct commands beat zero-motion and batch-rolled commands?

This script consumes a feature cache and never decodes or generates pixels.
The default device is CPU; pass ``--device cuda`` explicitly when desired.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[1]
TRAIN_SRC = REPO / "train"
for path in (TRAIN_SRC, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from so100_transition_contract import transition_action_features  # noqa: E402


class CachedSplit(Dataset):
    def __init__(
        self, payload: dict, split: int, uploader_allowlist: set[str] | None = None
    ) -> None:
        split_mask = payload["split"] == split
        if uploader_allowlist is not None:
            uploader_mask = torch.tensor(
                [name in uploader_allowlist for name in payload["uploaders"]], dtype=torch.bool
            )
            split_mask = split_mask & uploader_mask
        self.indices = torch.nonzero(split_mask, as_tuple=False).flatten()
        self.actions = payload["actions"]
        self.states = payload["states"]
        self.dino = payload["dino"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = int(self.indices[index])
        return {
            "actions": self.actions[source].float(),
            "states": self.states[source].float(),
            "dino": self.dino[source].float(),
        }


def partition_holdout_uploaders(payload: dict, split_seed: int) -> tuple[list[str], list[str]]:
    """Return fixed disjoint uploader sets for selection and one-shot confirmation."""
    holdout = sorted(
        {
            uploader
            for uploader, split in zip(payload["uploaders"], payload["split"].tolist(), strict=True)
            if split == 1
        }
    )
    if len(holdout) < 4:
        raise ValueError(f"confirmatory protocol needs at least 4 holdout uploaders, got {holdout}")
    shuffled = holdout.copy()
    random.Random(split_seed).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return sorted(shuffled[:midpoint]), sorted(shuffled[midpoint:])


class GroundingProbe(nn.Module):
    def __init__(self, dino_dim: int, hidden_dim: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.LayerNorm(dino_dim), nn.Linear(dino_dim, hidden_dim), nn.GELU()
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, hidden_dim), nn.GELU()
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(block, num_layers=layers)
        self.position = nn.Parameter(torch.zeros(1, 15, hidden_dim))
        self.future_dino_head = nn.Linear(hidden_dim, dino_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        # At initialization the visual prediction is exactly the static-source
        # baseline.  Training can only learn an action-conditioned residual.
        nn.init.zeros_(self.future_dino_head.weight)
        nn.init.zeros_(self.future_dino_head.bias)

    def forward(
        self,
        source_dino: torch.Tensor,
        actions: torch.Tensor,
        action_variant: str = "normal",
    ) -> dict[str, torch.Tensor]:
        source_dino = F.normalize(source_dino.float(), dim=-1)
        source = self.source_encoder(source_dino)
        # action[0] is a much stronger deployable source-state proxy than a
        # global DINO CLS token (held-out normalized MAE 0.098 vs 0.78).
        state0 = actions[:, 0]
        if action_variant == "zero":
            actions = actions[:, :1].expand(-1, 16, -1)
        elif action_variant != "normal":
            raise ValueError(f"unknown action_variant={action_variant!r}")
        features = transition_action_features(actions, state0)
        hidden = self.action_encoder(features) + source[:, None] + self.position
        causal = torch.triu(
            torch.ones(15, 15, device=hidden.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.temporal(hidden, mask=causal)
        dino_residual = 0.05 * torch.tanh(self.future_dino_head(hidden))
        return {
            "state0": state0,
            # The audit established action[t] -> state[t+1].  Do not replace
            # this strong physical prior with a learned 6D decoder.
            "future_state": actions[:, :15],
            "future_dino": F.normalize(source_dino[:, None] + dino_residual, dim=-1),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=REPO / "results/so100_grounding_features.pt"
    )
    parser.add_argument(
        "--action-stats", type=Path, default=REPO / "open/data/train/so100_action_statistics.json"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "open/baseline/outputs/so100_grounding_probe_v2"
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--ranking-margin", type=float, default=0.01)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        help="Select checkpoints on half of held-out uploaders and evaluate the other half once.",
    )
    parser.add_argument("--split-seed", type=int, default=20260805)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def normalize_batch(batch: dict, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict:
    actions = batch["actions"].to(device)
    states = batch["states"].to(device)
    return {
        "actions": (actions - mean) / std,
        "states": (states - mean) / std,
        "dino": F.normalize(batch["dino"].to(device).float(), dim=-1),
    }


def per_sample_errors(output: dict, batch: dict) -> dict[str, torch.Tensor]:
    return {
        "state0": (output["state0"] - batch["states"][:, 0]).abs().mean(dim=-1),
        "future_state": (output["future_state"] - batch["states"][:, 1:16]).abs().mean(dim=(1, 2)),
        "future_dino": (1 - (output["future_dino"] * batch["dino"][:, 1:16]).sum(dim=-1)).mean(dim=1),
    }


def ci95(values: np.ndarray, seed: int, draws: int = 2000) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return np.percentile(means, [2.5, 97.5]).tolist()


@torch.inference_mode()
def evaluate(
    model: GroundingProbe,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict:
    model.eval()
    buckets: dict[str, dict[str, list[torch.Tensor]]] = {
        variant: {metric: [] for metric in ("state0", "future_state", "future_dino")}
        for variant in ("normal", "zero", "roll")
    }
    baselines = {
        name: []
        for name in ("action0_state0", "direct_action_state", "static_state", "static_dino")
    }
    for raw in loader:
        batch = normalize_batch(raw, mean, std, device)
        normal = model(batch["dino"][:, 0], batch["actions"])
        zero = model(batch["dino"][:, 0], batch["actions"], action_variant="zero")
        rolled_actions = batch["actions"].roll(1, dims=0)
        roll = model(batch["dino"][:, 0], rolled_actions)
        for variant, output in (("normal", normal), ("zero", zero), ("roll", roll)):
            for metric, value in per_sample_errors(output, batch).items():
                buckets[variant][metric].append(value.cpu())
        baselines["action0_state0"].append(
            (batch["actions"][:, 0] - batch["states"][:, 0]).abs().mean(dim=-1).cpu()
        )
        baselines["direct_action_state"].append(
            (batch["actions"][:, :15] - batch["states"][:, 1:16])
            .abs()
            .mean(dim=(1, 2))
            .cpu()
        )
        baselines["static_state"].append(
            (batch["states"][:, :1] - batch["states"][:, 1:16]).abs().mean(dim=(1, 2)).cpu()
        )
        baselines["static_dino"].append(
            (1 - (batch["dino"][:, :1] * batch["dino"][:, 1:16]).sum(dim=-1)).mean(dim=1).cpu()
        )

    arrays = {
        variant: {metric: torch.cat(parts).numpy() for metric, parts in metrics.items()}
        for variant, metrics in buckets.items()
    }
    base = {name: torch.cat(parts).numpy() for name, parts in baselines.items()}
    deltas = {}
    for ablation in ("zero", "roll"):
        for metric in ("future_state", "future_dino"):
            value = arrays["normal"][metric] - arrays[ablation][metric]
            deltas[f"normal_minus_{ablation}_{metric}"] = {
                "mean": float(value.mean()),
                "ci95": ci95(value, seed + len(deltas)),
            }
    baseline_deltas = {}
    for metric, baseline_name in (
        ("future_state", "static_state"),
        ("future_dino", "static_dino"),
    ):
        value = arrays["normal"][metric] - base[baseline_name]
        baseline_deltas[f"normal_minus_{baseline_name}"] = {
            "mean": float(value.mean()),
            "ci95": ci95(value, seed + 200 + len(baseline_deltas)),
        }
    dino_relative_improvement = float(
        (base["static_dino"].mean() - arrays["normal"]["future_dino"].mean())
        / max(float(base["static_dino"].mean()), 1e-8)
    )
    result = {
        "samples": int(len(arrays["normal"]["future_state"])),
        "normal": {metric: float(value.mean()) for metric, value in arrays["normal"].items()},
        "baselines": {name: float(value.mean()) for name, value in base.items()},
        "baseline_deltas": baseline_deltas,
        "future_dino_relative_improvement": dino_relative_improvement,
        "counterfactual": deltas,
    }
    criteria = {
        "state0_uses_action0_proxy": abs(
            result["normal"]["state0"] - result["baselines"]["action0_state0"]
        ) < 1e-6,
        "future_state_preserves_direct_mapping": abs(
            result["normal"]["future_state"] - result["baselines"]["direct_action_state"]
        ) < 1e-6,
        "future_state_beats_static_ci95": baseline_deltas[
            "normal_minus_static_state"
        ]["ci95"][1]
        < 0,
        "future_dino_beats_static_ci95": baseline_deltas[
            "normal_minus_static_dino"
        ]["ci95"][1]
        < 0,
        "future_dino_relative_improvement_ge_2pct": dino_relative_improvement >= 0.02,
        "correct_beats_zero_state_ci95": deltas["normal_minus_zero_future_state"]["ci95"][1] < 0,
        "correct_beats_roll_state_ci95": deltas["normal_minus_roll_future_state"]["ci95"][1] < 0,
        "correct_beats_zero_dino_ci95": deltas["normal_minus_zero_future_dino"]["ci95"][1] < 0,
        "correct_beats_roll_dino_ci95": deltas["normal_minus_roll_future_dino"]["ci95"][1] < 0,
    }
    result["criteria"] = criteria
    result["verdict"] = "PASS_GROUNDING" if all(criteria.values()) else "REJECT_GROUNDING"
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    train = CachedSplit(payload, split=0)
    selection_uploaders: list[str] | None = None
    confirmation_uploaders: list[str] | None = None
    confirmation = None
    if args.confirmatory:
        selection_uploaders, confirmation_uploaders = partition_holdout_uploaders(
            payload, args.split_seed
        )
        holdout = CachedSplit(payload, split=1, uploader_allowlist=set(selection_uploaders))
        confirmation = CachedSplit(
            payload, split=1, uploader_allowlist=set(confirmation_uploaders)
        )
    else:
        holdout = CachedSplit(payload, split=1)
    if not train or not holdout or (confirmation is not None and not confirmation):
        raise SystemExit("feature cache needs both train and holdout records")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    holdout_loader = DataLoader(holdout, batch_size=args.batch_size, shuffle=False)
    confirmation_loader = (
        DataLoader(confirmation, batch_size=args.batch_size, shuffle=False)
        if confirmation is not None
        else None
    )
    stats = json.loads(args.action_stats.read_text())
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    dino_dim = int(payload["dino"].shape[-1])
    model = GroundingProbe(dino_dim, args.hidden_dim, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)

    iterator = iter(train_loader)
    best_result = None
    best_key = (2, float("inf"))
    for step in range(1, args.max_steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        batch = normalize_batch(raw, mean, std, device)
        model.train()
        output = model(batch["dino"][:, 0], batch["actions"])
        zero_output = model(batch["dino"][:, 0], batch["actions"], action_variant="zero")
        roll_output = model(batch["dino"][:, 0], batch["actions"].roll(1, dims=0))
        error = per_sample_errors(output, batch)
        zero_error = per_sample_errors(zero_output, batch)
        roll_error = per_sample_errors(roll_output, batch)
        ranking = F.relu(
            args.ranking_margin + error["future_dino"] - zero_error["future_dino"]
        ).mean() + F.relu(
            args.ranking_margin + error["future_dino"] - roll_error["future_dino"]
        ).mean()
        loss = error["future_dino"].mean() + args.ranking_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step={step}/{args.max_steps} loss={loss.item():.6f}")
        if step % args.eval_every == 0 or step == args.max_steps:
            result = evaluate(model, holdout_loader, mean, std, device, args.seed + step)
            result["step"] = step
            score = result["normal"]["future_dino"]
            selection_key = (0 if result["verdict"] == "PASS_GROUNDING" else 1, score)
            print(json.dumps(result, indent=2))
            (args.output / f"step-{step}-gate.json").write_text(json.dumps(result, indent=2))
            if selection_key < best_key:
                best_key = selection_key
                best_result = copy.deepcopy(result)
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "gate": best_result},
                    args.output / "best.pt",
                )
    (args.output / "best_gate.json").write_text(json.dumps(best_result, indent=2))
    print(
        json.dumps(
            {
                "selected_best_step": best_result["step"],
                "selected_best_verdict": best_result["verdict"],
                "selected_best_future_dino": best_result["normal"]["future_dino"],
                "static_dino": best_result["baselines"]["static_dino"],
            },
            indent=2,
        )
    )
    print(f"saved selected best -> {args.output / 'best_gate.json'}")

    if confirmation_loader is not None:
        selected_checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(selected_checkpoint["model"])
        confirmation_result = evaluate(
            model,
            confirmation_loader,
            mean,
            std,
            device,
            args.seed + 1_000_000,
        )
        confirmation_result["selected_step"] = best_result["step"]
        protocol = {
            "protocol": "uploader-disjoint selection and one-shot confirmation",
            "split_seed": args.split_seed,
            "selection_uploaders": selection_uploaders,
            "confirmation_uploaders": confirmation_uploaders,
            "selection": best_result,
            "confirmation": confirmation_result,
            "verdict": (
                "PASS_CONFIRMATORY_GROUNDING"
                if best_result["verdict"] == "PASS_GROUNDING"
                and confirmation_result["verdict"] == "PASS_GROUNDING"
                else "REJECT_CONFIRMATORY_GROUNDING"
            ),
        }
        confirm_path = args.output / "confirmatory_gate.json"
        confirm_path.write_text(json.dumps(protocol, indent=2))
        print(json.dumps(protocol, indent=2))
        print(f"saved one-shot confirmation -> {confirm_path}")


if __name__ == "__main__":
    main()
