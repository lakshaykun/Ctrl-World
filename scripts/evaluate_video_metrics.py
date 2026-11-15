#!/usr/bin/env python3
"""Offline evaluation for Ctrl-World replay rollouts saved as NPZ files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_pairs(root: Path):
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {root}")

    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "ground_truth" not in archive or "prediction" not in archive:
                print(f"[WARN] Skipping {path}: required arrays missing")
                continue
            gt = np.asarray(archive["ground_truth"])
            pred = np.asarray(archive["prediction"])
            trajectory_id = (
                str(archive["trajectory_id"].item())
                if "trajectory_id" in archive else path.parent.name
            )
            interaction = (
                int(archive["interaction_index"].item())
                if "interaction_index" in archive else -1
            )

        if gt.shape != pred.shape:
            raise ValueError(f"{path}: GT {gt.shape} != prediction {pred.shape}")
        if gt.ndim != 5 or gt.shape[-1] != 3:
            raise ValueError(
                f"{path}: expected (views, frames, H, W, 3), got {gt.shape}"
            )
        yield path, trajectory_id, interaction, gt.astype(np.uint8), pred.astype(np.uint8)


def calculate_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))


def calculate_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "SSIM requires scikit-image. Install it with: pip install scikit-image"
        ) from exc
    return float(structural_similarity(gt, pred, channel_axis=-1, data_range=255))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--metrics", nargs="+", choices=["psnr", "ssim", "lpips"],
        default=["psnr", "ssim"],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)

    lpips_model = lpips_torch = None
    if "lpips" in args.metrics:
        try:
            import torch
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS requires `pip install lpips`") from exc
        lpips_torch = torch
        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_scores = {metric: [] for metric in args.metrics}
    frame_view_pairs = 0

    for path, trajectory_id, interaction, gt_clip, pred_clip in load_pairs(args.data_dir):
        clip_scores = {metric: [] for metric in args.metrics}
        for view_idx in range(gt_clip.shape[0]):
            for frame_idx in range(gt_clip.shape[1]):
                gt_frame = gt_clip[view_idx, frame_idx]
                pred_frame = pred_clip[view_idx, frame_idx]

                if "psnr" in args.metrics:
                    clip_scores["psnr"].append(calculate_psnr(gt_frame, pred_frame))
                if "ssim" in args.metrics:
                    clip_scores["ssim"].append(calculate_ssim(gt_frame, pred_frame))
                if "lpips" in args.metrics:
                    gt_tensor = (
                        lpips_torch.from_numpy(gt_frame.copy())
                        .permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 127.5 - 1
                    )
                    pred_tensor = (
                        lpips_torch.from_numpy(pred_frame.copy())
                        .permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 127.5 - 1
                    )
                    with lpips_torch.no_grad():
                        score = lpips_model(gt_tensor, pred_tensor).mean().item()
                    clip_scores["lpips"].append(float(score))
                frame_view_pairs += 1

        row = {
            "file": str(path),
            "trajectory_id": trajectory_id,
            "interaction_index": interaction,
            "num_views": int(gt_clip.shape[0]),
            "num_frames_per_view": int(gt_clip.shape[1]),
        }
        for metric, values in clip_scores.items():
            row[metric] = float(np.mean(values))
            all_scores[metric].extend(values)
        rows.append(row)
        print(
            f"[OK] {path}: "
            + ", ".join(f"{metric}={row[metric]:.5f}" for metric in args.metrics)
        )

    if not rows:
        raise RuntimeError(f"No valid NPZ evaluation files found under {args.data_dir}")

    csv_path = args.output_dir / "per_interaction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data_dir": str(args.data_dir),
        "num_interaction_files": len(rows),
        "num_frame_view_pairs": frame_view_pairs,
        "metrics": {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
            for metric, values in all_scores.items()
        },
        "notes": [
            "PSNR, SSIM and LPIPS are paired-frame metrics.",
            "Averages include every saved camera view and frame.",
            "FID/FVD are not calculated; they require dataset-level distribution protocols and enough clips.",
        ],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nOVERALL")
    for metric, values in all_scores.items():
        print(
            f"{metric.upper()}: mean={np.mean(values):.6f}, "
            f"std={np.std(values):.6f}, n={len(values)}"
        )
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
