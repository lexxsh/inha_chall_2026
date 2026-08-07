from __future__ import annotations

import csv
import json
from pathlib import Path

import av
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.video import R3D_18_Weights, r3d_18

from ldwma.datasets.lerobot_so100 import preprocess_video
from video_utils.image import preprocess_images


CSV_FIELDNAMES = ["sample_id", "feature_component", "feature_json"]
DINO_COMPONENT = "DINO Component"
VIDEO_COMPONENT = "Video Feature Component"
ACTION_COMPONENT = "Action Component"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def list_challenge_sample_ids(challenge_root: Path) -> list[str]:
    image_ids = {path.stem for path in (challenge_root / "images").glob("*.png")}
    action_ids = {path.stem for path in (challenge_root / "actions").glob("*.npy")}
    sample_ids = sorted(image_ids & action_ids)
    if not sample_ids:
        raise ValueError(f"No matching challenge image/action pairs under {challenge_root}.")
    return sample_ids


def list_video_sample_ids(video_root: Path, restrict_to: set[str] | None = None) -> list[str]:
    video_ids = {path.stem for path in video_root.glob("*.mp4")}
    if restrict_to is not None:
        video_ids &= restrict_to
    sample_ids = sorted(video_ids)
    if not sample_ids:
        raise ValueError(f"No mp4 videos found under {video_root}.")
    return sample_ids


def read_video_uint8(path: Path, expected_frames: int | None = None) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"{path}: no decoded frames.")
    if expected_frames is not None and len(frames) != expected_frames:
        raise ValueError(f"{path}: expected {expected_frames} frames, got {len(frames)}.")
    return torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)


def to_eval_uint8(video_np: np.ndarray, target_height: int, target_width: int, pad: bool) -> torch.Tensor:
    video = preprocess_video(video_np, target_height, target_width, pad).clamp(-1.0, 1.0)
    video = ((video + 1.0) / 2.0 * 255.0).to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous()


def load_video_batch(
    video_root: Path,
    sample_ids: list[str],
    temporal_length: int,
    target_height: int,
    target_width: int,
    pad: bool,
) -> torch.Tensor:
    return torch.stack(
        [
            to_eval_uint8(
                read_video_uint8(video_root / f"{sample_id}.mp4", expected_frames=temporal_length).numpy(),
                target_height,
                target_width,
                pad,
            )
            for sample_id in sample_ids
        ],
        dim=0,
    )


def save_video_tensor(video: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video.detach().cpu().clamp(-1.0, 1.0)
    frames = ((frames + 1.0) / 2.0 * 255.0).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous()
    try:
        from torchvision.io import write_video

        write_video(str(path), frames, fps=fps, video_codec="libx264")
    except Exception as error:
        raise RuntimeError(f"Failed to save generated video to {path}.") from error


def load_action_stats(path: str | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not path:
        return None, None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return torch.tensor(data["mean"], dtype=torch.float32), torch.tensor(data["std"], dtype=torch.float32)


def build_inference_batch(
    challenge_root: Path,
    sample_ids: list[str],
    target_height: int,
    target_width: int,
    pad: bool,
    fps: int,
    action_mean: torch.Tensor | None,
    action_std: torch.Tensor | None,
    device: torch.device,
) -> dict:
    videos = []
    actions = []
    for sample_id in sample_ids:
        image = np.asarray(Image.open(challenge_root / "images" / f"{sample_id}.png").convert("RGB"))
        action = np.load(challenge_root / "actions" / f"{sample_id}.npy").astype(np.float32)
        if action.ndim != 2:
            raise ValueError(f"{sample_id}: action must have shape (T, A), got {action.shape}.")
        video_np = np.zeros((action.shape[0], *image.shape), dtype=np.uint8)
        video_np[0] = image
        videos.append(preprocess_video(video_np, target_height, target_width, pad))
        action_tensor = torch.from_numpy(action).to(device)
        if action_mean is not None and action_std is not None:
            action_tensor = (action_tensor - action_mean.to(device)) / action_std.to(device)
        actions.append(action_tensor)

    return {
        "video": torch.stack(videos, dim=0).to(device),
        "act": torch.stack(actions, dim=0).to(device),
        "caption": [""] * len(sample_ids),
        "fps": torch.full((len(sample_ids),), fps, dtype=torch.long, device=device),
        "frame_stride": torch.full((len(sample_ids),), fps, dtype=torch.long, device=device),
        "start_idx": torch.zeros(len(sample_ids), dtype=torch.long, device=device),
    }


def load_video_feature_model(device: torch.device, pretrained: bool) -> torch.nn.Module:
    weights = R3D_18_Weights.DEFAULT if pretrained else None
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


def load_dino_model(device: torch.device, model_name: str, pretrained: bool) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.to(device)
    model.eval()
    return model


def load_action_extractor(path: str | None, device: torch.device):
    if not path:
        return None
    try:
        from ldwma.models.action_extractor import load_so100_action_extractor_checkpoint

        model = load_so100_action_extractor_checkpoint(path, map_location="cpu")
    except Exception as first_error:
        try:
            from lvdm.models.action_predictor import ActionPredictor

            model = ActionPredictor.load_from_checkpoint(checkpoint_path=path)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load action extractor checkpoint as SO100ActionExtractor or ActionPredictor: {path}"
            ) from second_error
    model.to(device)
    model.eval()
    return model


def _round_feature(feature: np.ndarray | torch.Tensor | list, precision: int | None):
    if isinstance(feature, torch.Tensor):
        array = feature.detach().cpu().float().numpy()
    else:
        array = np.asarray(feature, dtype=np.float32)
    if precision is not None and precision >= 0:
        array = np.round(array, precision)
    return array.tolist()


def feature_row(
    sample_id: str,
    component: str,
    feature: np.ndarray | torch.Tensor | list,
    precision: int | None = 6,
) -> dict:
    return {
        "sample_id": sample_id,
        "feature_component": component,
        "feature_json": json.dumps(_round_feature(feature, precision), separators=(",", ":")),
    }

def _resize_pad_frame_batch(frames: torch.Tensor, size: int, pad_value: float = 0.0) -> torch.Tensor:
    frames = frames.float() / 255.0
    _, _, height, width = frames.shape
    scale = min(size / height, size / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    frames = F.interpolate(frames, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_top = (size - resized_height) // 2
    pad_bottom = size - resized_height - pad_top
    pad_left = (size - resized_width) // 2
    pad_right = size - resized_width - pad_left
    return F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)


def resolve_dino_image_size(model: torch.nn.Module, requested_size: int) -> int:
    expected_size = None
    patch_embed = getattr(model, "patch_embed", None)
    img_size = getattr(patch_embed, "img_size", None)
    if isinstance(img_size, tuple) and img_size:
        expected_size = int(img_size[0])
    elif isinstance(img_size, int):
        expected_size = int(img_size)

    if requested_size <= 0:
        return expected_size or 224
    if expected_size is not None and requested_size != expected_size:
        print(f"[feature csv] DINO input size changed from {requested_size} to model expected size {expected_size}.")
        return expected_size
    return requested_size


def _normalize_image_model_output(output) -> torch.Tensor:
    if isinstance(output, dict):
        if "x_norm_clstoken" in output:
            output = output["x_norm_clstoken"]
        elif "features" in output:
            output = output["features"]
        else:
            tensor_values = [value for value in output.values() if isinstance(value, torch.Tensor)]
            if not tensor_values:
                raise TypeError("DINO model returned a dict without tensor outputs.")
            output = tensor_values[0]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3:
        output = output[:, 0]
    elif output.ndim > 3:
        output = output.flatten(2).mean(dim=-1)
    return output


def extract_video_features(videos: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    x = videos.permute(0, 4, 1, 2, 3).float() / 255.0
    x = F.interpolate(x, size=(videos.shape[1], 112, 112), mode="trilinear", align_corners=False)
    x = (x - KINETICS_MEAN.to(x.device)) / KINETICS_STD.to(x.device)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 8):
            outputs.append(model(x[start : start + 8].to(device)).cpu().float())
    return torch.cat(outputs, dim=0)


def extract_dino_features(
    videos: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> torch.Tensor:
    batch, time, height, width, channels = videos.shape
    frames = videos.permute(0, 1, 4, 2, 3).reshape(batch * time, channels, height, width)
    imagenet_mean = IMAGENET_MEAN.to(device)
    frames = _resize_pad_frame_batch(frames, image_size, pad_value=0.0).to(device)
    frames = (frames - imagenet_mean) / IMAGENET_STD.to(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], 32):
            output = model(frames[start : start + 32])
            outputs.append(_normalize_image_model_output(output).cpu().float())
    return torch.cat(outputs, dim=0).reshape(batch, time, -1)


def load_action_target_batch(
    challenge_root: Path | None,
    sample_ids: list[str],
    action_mean: torch.Tensor | None,
    action_std: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if challenge_root is None:
        return None
    targets = []
    for sample_id in sample_ids:
        action = torch.from_numpy(np.load(challenge_root / "actions" / f"{sample_id}.npy").astype(np.float32))
        if action_mean is not None and action_std is not None:
            action = (action - action_mean) / action_std
        targets.append(action)
    return torch.stack(targets, dim=0).to(device)


def extract_action_features(
    videos: torch.Tensor,
    model,
    device: torch.device,
    target_actions: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if model is None:
        return None
    frames = preprocess_images(videos.to(device))
    with torch.no_grad():
        pred = model(frames).float()
        if target_actions is not None:
            if pred.shape != target_actions.shape:
                raise ValueError(f"Action prediction/target shape mismatch: {pred.shape} vs {target_actions.shape}")
            return torch.mean(torch.abs(pred - target_actions), dim=(1, 2), keepdim=True).cpu()
        return pred.detach().cpu().flatten(1)


def write_feature_csv(
    video_root: Path,
    sample_ids: list[str],
    output_csv: Path,
    source: str,
    device: torch.device,
    temporal_length: int,
    target_height: int,
    target_width: int,
    pad: bool,
    feature_batch_size: int,
    precision: int | None,
    action_extractor_ckpt: str | None,
    challenge_root: Path | None = None,
    action_stats_path: str | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    video_feature_model = load_video_feature_model(device, pretrained=True)
    dino_model_name = "vit_small_patch14_dinov2.lvd142m"
    dino_model = load_dino_model(device, dino_model_name, pretrained=True)
    dino_image_size = resolve_dino_image_size(dino_model, requested_size=0)
    action_model = load_action_extractor(action_extractor_ckpt, device)
    action_mean, action_std = load_action_stats(action_stats_path) if challenge_root is not None else (None, None)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for start in range(0, len(sample_ids), feature_batch_size):
            batch_ids = sample_ids[start : start + feature_batch_size]
            videos = load_video_batch(video_root, batch_ids, temporal_length, target_height, target_width, pad)

            video_features = extract_video_features(videos, video_feature_model, device)
            for sample_id, feature in zip(batch_ids, video_features):
                writer.writerow(feature_row(sample_id, VIDEO_COMPONENT, feature, precision))

            dino_features = extract_dino_features(videos, dino_model, device, dino_image_size)
            for sample_id, feature in zip(batch_ids, dino_features):
                writer.writerow(feature_row(sample_id, DINO_COMPONENT, feature, precision))

            action_targets = load_action_target_batch(challenge_root, batch_ids, action_mean, action_std, device)
            action_features = extract_action_features(videos, action_model, device, action_targets)
            if action_features is not None:
                for sample_id, feature in zip(batch_ids, action_features):
                    writer.writerow(feature_row(sample_id, ACTION_COMPONENT, feature, precision))

            print(f"[feature csv] {source}: wrote {start + len(batch_ids)}/{len(sample_ids)} samples")
