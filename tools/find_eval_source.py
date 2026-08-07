"""eval 장면이 train 안에 실제로 존재하는지 DINO feature 최근접 탐색으로 검증한다.

앞선 썸네일(64x48 픽셀 MSE) 매칭은 데이터셋당 3프레임만 봤기 때문에 놓쳤을 수 있다.
여기서는 데이터셋마다 여러 에피소드를 훑고 평가와 같은 DINOv2 feature로 비교한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "open/submission_kit"
sys.path.insert(0, str(KIT))

from feature_csv_utils import load_dino_model, resolve_dino_image_size  # noqa: E402

TRAIN = REPO / "open/data/train"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def first_frame(video_path: Path) -> np.ndarray | None:
    try:
        with av.open(str(video_path)) as container:
            for frame in container.decode(container.streams.video[0]):
                return frame.to_ndarray(format="rgb24")
    except Exception:
        return None
    return None


class DinoEncoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_dino_model(self.device, "vit_small_patch14_dinov2.lvd142m", pretrained=True)
        self.size = resolve_dino_image_size(self.model, requested_size=0)

    @torch.no_grad()
    def encode(self, images: list[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        out = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            # 데이터셋마다 해상도가 달라 스택 전에 각자 리사이즈한다.
            resized = [
                F.interpolate(
                    torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255.0,
                    size=(self.size, self.size),
                    mode="bilinear",
                    align_corners=False,
                )
                for img in chunk
            ]
            x = torch.cat(resized, dim=0)
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(self.device)
            feat = self.model(x)
            if feat.ndim == 3:
                feat = feat[:, 0]
            out.append(F.normalize(feat.float(), dim=-1).cpu())
        return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-dataset", type=int, default=12)
    args = parser.parse_args()

    encoder = DinoEncoder()

    frames, labels = [], []
    datasets = sorted(d for d in TRAIN.glob("*/*") if (d / "meta/info.json").exists())
    for k, dataset_dir in enumerate(datasets):
        videos = sorted(dataset_dir.glob("videos/chunk-*/*/*.mp4"))
        if not videos:
            continue
        picks = np.linspace(0, len(videos) - 1, min(args.episodes_per_dataset, len(videos))).astype(int)
        for i in sorted(set(picks.tolist())):
            img = first_frame(videos[i])
            if img is not None:
                frames.append(img)
                labels.append(f"{dataset_dir.relative_to(TRAIN)}#{videos[i].stem}")
        if (k + 1) % 25 == 0:
            print(f"  스캔 {k + 1}/{len(datasets)} 데이터셋, 프레임 {len(frames)}개")

    print(f"train 프레임 {len(frames)}개 인코딩 중...")
    train_feats = encoder.encode(frames)

    eval_paths = sorted((REPO / "open/data/eval/images").glob("*.png"))
    eval_imgs = [np.asarray(Image.open(p).convert("RGB")) for p in eval_paths]
    eval_feats = encoder.encode(eval_imgs)

    sim = eval_feats @ train_feats.T
    best_sim, best_idx = sim.max(dim=1)

    print("\n=== eval 이미지의 train 최근접 이웃 (DINO cosine similarity) ===")
    print(f"유사도 분포: min {best_sim.min():.3f} | p25 {best_sim.quantile(0.25):.3f} | "
          f"중앙 {best_sim.median():.3f} | p75 {best_sim.quantile(0.75):.3f} | max {best_sim.max():.3f}")

    from collections import Counter

    matched_ds = Counter(labels[i].split("#")[0] for i in best_idx.tolist())
    print("\n최근접으로 지목된 데이터셋:")
    for ds, n in matched_ds.most_common(15):
        idx = [i for i, b in enumerate(best_idx.tolist()) if labels[b].split("#")[0] == ds]
        avg = float(best_sim[idx].mean())
        print(f"  {n:4d}개  평균유사도 {avg:.3f}  {ds}")

    print("\n가장 유사한 상위 10쌍:")
    order = torch.argsort(best_sim, descending=True)[:10]
    for i in order.tolist():
        print(f"  {best_sim[i]:.3f}  {eval_paths[i].stem} -> {labels[best_idx[i]]}")
    print("\n가장 안 닮은 하위 5쌍:")
    for i in torch.argsort(best_sim)[:5].tolist():
        print(f"  {best_sim[i]:.3f}  {eval_paths[i].stem} -> {labels[best_idx[i]]}")

    out = REPO / "results/eval_source_match.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            [
                {"sample": eval_paths[i].stem, "match": labels[best_idx[i]], "sim": float(best_sim[i])}
                for i in range(len(eval_paths))
            ],
            indent=2,
        )
    )
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
