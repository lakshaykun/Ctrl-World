#!/usr/bin/env python3
"""
Ctrl-World video evaluation.

Metrics:
  PSNR, SSIM, LPIPS: paired ground-truth/prediction frames.
  FID: frame distribution distance (Inception features).
  FVD: video distribution distance (I3D features).

Input NPZ files must contain:
  ground_truth, prediction
Optional metadata:
  trajectory_id, interaction_index, start_frame_index

Expected video array layout: (views, frames, H, W, C), or (frames, H, W, C).
This script evaluates each view independently and pools views for global metrics.

Important:
- FID/FVD are distribution metrics; they are not meaningful on only a few clips.
- FVD uses 16-frame clips by default. The script reconstructs each trajectory by
  sorting interaction files and trimming overlapping frames using start_frame_index.
  Verify that start_frame_index denotes the first frame represented in each NPZ.
- Results are written to evaluation_results/run_XXX/; an existing run is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_root", default="evaluation_results")
    p.add_argument("--metrics", nargs="+",
                   default=["psnr", "ssim", "lpips", "fid", "fvd"],
                   choices=["psnr", "ssim", "lpips", "fid", "fvd"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--fvd_clip_len", type=int, default=16)
    p.add_argument("--fvd_stride", type=int, default=16)
    p.add_argument("--max_fvd_clips_per_trajectory", type=int, default=64)
    p.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    p.add_argument("--ssim_data_range", type=float, default=255.0)
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_dir(root: Path, run_name=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        out = root / run_name
        out.mkdir(parents=False, exist_ok=False)
        return out
    nums = []
    for x in root.glob("run_*"):
        m = re.fullmatch(r"run_(\d+)", x.name)
        if m and x.is_dir():
            nums.append(int(m.group(1)))
    out = root / f"run_{max(nums, default=0) + 1:03d}"
    out.mkdir()
    return out


def as_views_frames(arr):
    a = np.asarray(arr)
    if a.ndim == 4:  # T,H,W,C
        a = a[None, ...]
    if a.ndim != 5:
        raise ValueError(f"Expected 4D or 5D video array, got shape {a.shape}")
    # Expected V,T,H,W,C. Also accept V,T,C,H,W.
    if a.shape[-1] not in (1, 3, 4) and a.shape[2] in (1, 3, 4):
        a = np.transpose(a, (0, 1, 3, 4, 2))
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            # Support [0,1] and [0,255] float arrays.
            if np.nanmax(a) <= 1.0:
                a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def load_records(data_dir):
    files = sorted(Path(data_dir).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")
    records = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if "ground_truth" not in z or "prediction" not in z:
                print(f"[SKIP] {f}: missing ground_truth/prediction")
                continue
            gt = as_views_frames(z["ground_truth"])
            pred = as_views_frames(z["prediction"])
            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch in {f}: GT {gt.shape}, pred {pred.shape}")
            meta = {}
            for k in ("trajectory_id", "interaction_index", "start_frame_index", "fps", "instruction"):
                if k in z:
                    v = z[k]
                    meta[k] = v.item() if np.asarray(v).ndim == 0 else str(v)
            traj = str(meta.get("trajectory_id", f.parent.name))
            interaction = int(meta.get("interaction_index", 0))
            start = int(meta.get("start_frame_index", interaction * gt.shape[1]))
            records.append(dict(path=str(f), gt=gt, pred=pred, traj=traj,
                                interaction=interaction, start=start, meta=meta))
    if not records:
        raise RuntimeError("No valid NPZ records found.")
    return records


def paired_metrics(records, requested, device, lpips_net, data_range):
    lpips_model = None
    if "lpips" in requested:
        try:
            import torch
            import lpips
            lpips_model = lpips.LPIPS(net=lpips_net).to(device).eval()
        except Exception as e:
            raise RuntimeError(
                "LPIPS requested but could not initialize. Install with "
                "`pip install lpips`; check PyTorch/torchvision compatibility. "
                f"Original error: {e}"
            )
    rows = []
    lpips_values = []
    for rec in records:
        per_frame = []
        V, T = rec["gt"].shape[:2]
        for v in range(V):
            for t in range(T):
                gt = rec["gt"][v, t]
                pr = rec["pred"][v, t]
                vals = {}
                if "psnr" in requested:
                    vals["psnr"] = float(peak_signal_noise_ratio(gt, pr, data_range=data_range))
                if "ssim" in requested:
                    vals["ssim"] = float(structural_similarity(
                        gt, pr, channel_axis=-1, data_range=data_range))
                if lpips_model is not None:
                    import torch
                    # LPIPS expects RGB tensors in [-1,1], N,C,H,W.
                    a = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
                    b = torch.from_numpy(pr.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
                    with torch.no_grad():
                        lv = float(lpips_model(a.to(device), b.to(device)).item())
                    vals["lpips"] = lv
                    lpips_values.append(lv)
                per_frame.append(vals)
        row = {
            "trajectory_id": rec["traj"],
            "interaction_index": rec["interaction"],
            "start_frame_index": rec["start"],
            "file": rec["path"],
            "num_views": V,
            "num_frames_per_view": T,
            "num_frame_pairs": V * T,
        }
        for key in ("psnr", "ssim", "lpips"):
            x = [d[key] for d in per_frame if key in d and math.isfinite(d[key])]
            if x:
                row[key] = float(np.mean(x))
                row[key + "_std"] = float(np.std(x))
        rows.append(row)
        print("[PAIRED]", Path(rec["path"]).name,
              " ".join(f"{k}={row[k]:.5f}" for k in ("psnr", "ssim", "lpips") if k in row))
    return rows



def compute_fid(records, device):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    count = 0
    for r in records:
        # flatten views and frames to N,H,W,C then N,C,H,W uint8
        for arr, real in ((r["gt"], True), (r["pred"], False)):
            x = torch.from_numpy(arr.reshape(-1, *arr.shape[2:]).copy())
            x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.uint8)
            metric.update(x, real=real)
            if real:
                count += x.shape[0]
    if count < 2:
        raise RuntimeError("FID needs at least two real and two generated images.")
    return float(metric.compute().item()), count


def build_trajectory_videos(records, clip_len, stride, max_clips):
    """Join interaction chunks in temporal order, removing repeated index overlap."""
    grouped = {}
    for r in records:
        grouped.setdefault(r["traj"], []).append(r)
    real_clips, fake_clips, clip_sources = [], [], []
    for traj, items in grouped.items():
        items.sort(key=lambda x: (x["start"], x["interaction"]))
        # Keep a separate sequence per camera/view.
        nviews = min(x["gt"].shape[0] for x in items)
        for view in range(nviews):
            gt_seq, pr_seq, last_end = [], [], None
            for item in items:
                start = item["start"]
                gt, pr = item["gt"][view], item["pred"][view]
                # start_frame_index is treated as the absolute index of frame 0.
                offset = 0 if last_end is None else max(0, last_end - start)
                if offset >= len(gt):
                    continue
                gt_part, pr_part = gt[offset:], pr[offset:]
                gt_seq.append(gt_part)
                pr_seq.append(pr_part)
                last_end = max(last_end or start, start + len(gt))
            if not gt_seq:
                continue
            g = np.concatenate(gt_seq, axis=0)
            p = np.concatenate(pr_seq, axis=0)
            # Use non-overlapping clips for a conservative, reproducible sample.
            made = 0
            for s in range(0, min(len(g), len(p)) - clip_len + 1, stride):
                real_clips.append(g[s:s + clip_len])
                fake_clips.append(p[s:s + clip_len])
                clip_sources.append(f"{traj}:view{view}:frame{s}")
                made += 1
                if made >= max_clips:
                    break
    return real_clips, fake_clips, clip_sources



def compute_fvd(real_clips, fake_clips, device):
    """
    Compute FVD using cd-fvd with the I3D backbone.
    Input clips: (B, T, H, W, C), uint8.
    """
    from cdfvd import fvd

    real_array = np.ascontiguousarray(
        np.stack(real_clips), dtype=np.uint8
    )
    fake_array = np.ascontiguousarray(
        np.stack(fake_clips), dtype=np.uint8
    )

    if real_array.shape != fake_array.shape:
        raise ValueError(
            f"Real/fake shapes differ: "
            f"{real_array.shape} vs {fake_array.shape}"
        )

    evaluator = fvd.cdfvd(
        model="i3d",
        n_real="full",
        n_fake=len(fake_array),
        device=device,
        half_precision=False,
    )

    score = evaluator.compute_fvd(
        real_videos=real_array,
        fake_videos=fake_array,
    )

    return float(score)



def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    requested = set(args.metrics)
    data_dir = Path(args.data_dir)
    records = load_records(data_dir)
    run_dir = make_run_dir(Path(args.output_root), args.run_name)
    print(f"[INFO] Loaded {len(records)} interaction files")
    print(f"[INFO] Output: {run_dir}")

    frame_rows = paired_metrics(records, requested, args.device, args.lpips_net,
                                args.ssim_data_range)
    write_csv(run_dir / "per_interaction_metrics.csv", frame_rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "num_npz_files": len(records),
        "num_trajectories": len(set(r["traj"] for r in records)),
        "num_interactions": len(records),
        "metrics_requested": sorted(requested),
        "protocol": {
            "paired_metrics": "mean over paired RGB frames, then report pooled mean/std",
            "fid": "Inception feature distance over all real/generated frames",
            "fvd": "I3D feature distance over reconstructed trajectory clips",
            "fvd_clip_len": args.fvd_clip_len,
            "fvd_stride": args.fvd_stride,
            "overlap_handling": "sort by start_frame_index and trim repeated frame indices",
        },
        "paired_metrics": {},
        "distribution_metrics": {},
    }

    for key in ("psnr", "ssim", "lpips"):
        if key not in requested:
            continue

        total_n = 0
        total_sum = 0.0
        total_sum_sq = 0.0

        for row in frame_rows:
            if key not in row:
                continue

            n = int(row["num_frame_pairs"])
            mean = float(row[key])
            std = float(row[key + "_std"])

            total_n += n
            total_sum += n * mean
            total_sum_sq += n * (std**2 + mean**2)

        if total_n:
            overall_mean = total_sum / total_n
            overall_variance = max(
                0.0,
                total_sum_sq / total_n - overall_mean**2
            )

            summary["paired_metrics"][key] = {
                "mean": overall_mean,
                "std": overall_variance**0.5,
                "n_frame_pairs": total_n
            }

    if "fid" in requested:
        fid, nframes = compute_fid(records, args.device)
        summary["distribution_metrics"]["fid"] = {
            "value": fid, "num_real_and_generated_frames_each": nframes
        }
        print(f"[RESULT] FID={fid:.5f} (frames per distribution={nframes})")

    if "fvd" in requested:
        real, fake, sources = build_trajectory_videos(
            records, args.fvd_clip_len, args.fvd_stride,
            args.max_fvd_clips_per_trajectory)
        if len(real) < 2:
            raise RuntimeError(
                f"Only {len(real)} FVD clips were assembled. Your saved NPZ chunks "
                "may be too short or the start_frame_index metadata may not represent "
                "the first frame in each chunk. Inspect the saved arrays/metadata, or "
                "collect longer trajectories before reporting FVD."
            )
        fvd = compute_fvd(real, fake, args.device)
        summary["distribution_metrics"]["fvd"] = {
            "value": fvd, "num_video_clips_each": len(real),
            "clip_length_frames": args.fvd_clip_len,
            "clip_sources": sources,
        }
        print(f"[RESULT] FVD={fvd:.5f} (clips per distribution={len(real)})")

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print("\n=== SUMMARY ===")
    for section in ("paired_metrics", "distribution_metrics"):
        for name, values in summary[section].items():
            print(name.upper(), values)
    print(f"\n[OK] Results saved in: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
