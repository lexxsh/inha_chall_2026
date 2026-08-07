"""고정 train-only holdout의 생성 영상을 공식 세 component와 같은 extractor로 비교한다.

리더보드에 없는 train GT를 쓰므로 이 수치는 제출 점수의 추정치가 아니라 checkpoint/ablation의
동일조건 순위를 정하기 위한 로컬 지표다. 정지 영상 기준도 같은 manifest에서 함께 계산한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scorer import Scorer, cosine_distance

REPO = Path(__file__).resolve().parents[1]
TRAJ_LEN = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valset", default=str(REPO / "valset_holdout"))
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Score explicit manifest IDs in the supplied order; repeat as needed.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    valset = Path(args.valset)
    prediction_root = Path(args.prediction_root)
    manifest = json.loads((valset / "manifest.json").read_text())
    if args.sample_ids:
        by_id = {record["sample_id"]: record for record in manifest}
        missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in by_id]
        if missing_ids:
            raise SystemExit(f"manifest에 없는 --sample-id: {missing_ids}")
        manifest = [by_id[sample_id] for sample_id in args.sample_ids]
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise SystemExit("manifest에 평가할 샘플이 없다.")

    missing = [r["sample_id"] for r in manifest if not (prediction_root / f'{r["sample_id"]}.mp4').exists()]
    if missing:
        raise SystemExit(f"예측 영상 {len(missing)}개가 없다: {missing[:8]}")

    # submission kit의 동일 reader를 사용한다.
    from scorer import read_video_uint8

    scorer = Scorer()
    records: list[dict] = []
    static_records: list[dict] = []

    for start in range(0, len(manifest), args.batch_size):
        chunk = manifest[start : start + args.batch_size]
        pred_batch = []
        gt_batch = []
        actions = []
        for record in chunk:
            sid = record["sample_id"]
            pred_raw = read_video_uint8(prediction_root / f"{sid}.mp4", expected_frames=TRAJ_LEN)
            # Challenge samples retain their native resolution (e.g. 640x480
            # and 1280x720).  The official submission extractor always maps
            # every video to the common padded 320x512 evaluation canvas
            # before batching; mirror that contract for local scoring too.
            pred_batch.append(scorer.to_eval(pred_raw.numpy()))
            gt_raw = np.load(valset / "gt_videos" / f"{sid}.npy")
            gt_batch.append(scorer.to_eval(gt_raw))
            actions.append(np.load(valset / "actions" / f"{sid}.npy"))

        pred = torch.stack(pred_batch)
        gt = torch.stack(gt_batch)
        static = gt[:, :1].repeat(1, TRAJ_LEN, 1, 1, 1)
        gt_features = scorer.features(gt)

        for name, videos, target in (
            ("prediction", pred, records),
            ("static", static, static_records),
        ):
            features = scorer.features(videos)
            dino = cosine_distance(features["dino"], gt_features["dino"])
            video = cosine_distance(features["video"], gt_features["video"])
            action = scorer.action_mae(videos, np.stack(actions))
            for i, record in enumerate(chunk):
                d = float(dino[i])
                v = float(video[i])
                a = float(action[i])
                target.append(
                    {
                        "sample_id": record["sample_id"],
                        "dataset": record["dataset"],
                        "dino": d,
                        "video": v,
                        "action": a,
                        "weighted": 0.3 * d + 0.3 * v + 0.4 * a,
                    }
                )
        print(f"scored {min(start + args.batch_size, len(manifest))}/{len(manifest)}")

    def summary(rows: list[dict]) -> dict:
        return {key: float(np.mean([r[key] for r in rows])) for key in ("dino", "video", "action", "weighted")}

    result = {
        "valset": str(valset),
        "prediction_root": str(prediction_root),
        "count": len(records),
        "prediction": summary(records),
        "static": summary(static_records),
        "per_sample": records,
        "static_per_sample": static_records,
    }
    print(json.dumps({"prediction": result["prediction"], "static": result["static"]}, indent=2))

    out = Path(args.out) if args.out else REPO / "results" / f"{prediction_root.name}_holdout_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
