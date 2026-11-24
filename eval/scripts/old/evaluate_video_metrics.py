#!/usr/bin/env python3
"""Memory-aware, batched evaluation for Ctrl-World NPZ rollout outputs.

Metrics and protocol retained from the repository implementation:
  * PSNR / SSIM / LPIPS are paired per-frame metrics.
  * FID uses TorchMetrics Inception features over all real/generated frames.
  * FVD uses cd-fvd I3D features over reconstructed trajectory clips.
  * Interaction chunks are sorted by start_frame_index and repeated overlap
    frames are trimmed before FVD clip extraction.

Main improvements:
  * NPZ files are loaded one at a time (not all video arrays retained in RAM).
  * LPIPS is evaluated in GPU batches rather than one frame at a time.
  * FID feature extraction is batched.
  * PSNR is vectorized; SSIM can use CPU worker threads.
  * Per-stage wall time and peak CUDA memory are recorded.

FVD's cdfvd API accepts complete clip arrays, so its final clip tensors remain
in host memory. This is deliberately not silently changed to a different FVD
implementation or feature protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_root", default="evaluation_results")
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips", "fid", "fvd"],
                   choices=["psnr", "ssim", "lpips", "fid", "fvd"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--fvd_clip_len", type=int, default=16)
    p.add_argument("--fvd_stride", type=int, default=16)
    p.add_argument("--max_fvd_clips_per_trajectory", type=int, default=64)
    p.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    p.add_argument("--lpips_batch_size", type=int, default=32)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--ssim_workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--ssim_data_range", type=float, default=255.0)
    p.add_argument("--run_name", default=None)
    p.add_argument("--profile", action="store_true", help="Record stage times and peak CUDA memory")
    return p.parse_args()


def make_run_dir(root: Path, run_name=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        out = root / run_name
        out.mkdir(parents=False, exist_ok=False)
        return out
    nums = [int(m.group(1)) for x in root.glob("run_*")
            if x.is_dir() and (m := re.fullmatch(r"run_(\d+)", x.name))]
    out = root / f"run_{max(nums, default=0) + 1:03d}"
    out.mkdir()
    return out


def list_npz_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")
    return files


def as_views_frames(arr: np.ndarray) -> np.ndarray:
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
            finite = a[np.isfinite(a)]
            if finite.size and finite.max() <= 1.0:
                a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def load_record(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if "ground_truth" not in z or "prediction" not in z:
            return None
        gt = as_views_frames(z["ground_truth"])
        pred = as_views_frames(z["prediction"])
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch in {path}: GT {gt.shape}, pred {pred.shape}")
        meta = {}
        for key in ("trajectory_id", "interaction_index", "start_frame_index", "fps", "instruction"):
            if key in z:
                value = z[key]
                meta[key] = value.item() if np.asarray(value).ndim == 0 else str(value)
    traj = str(meta.get("trajectory_id", path.parent.name))
    interaction = int(meta.get("interaction_index", 0))
    start = int(meta.get("start_frame_index", interaction * gt.shape[1]))
    return {"path": str(path), "gt": gt, "pred": pred, "traj": traj,
            "interaction": interaction, "start": start, "meta": meta}


def read_metadata(path: Path) -> dict | None:
    """Read only scalar metadata; used to group FVD chunks without loading videos."""
    with np.load(path, allow_pickle=False) as z:
        if "ground_truth" not in z or "prediction" not in z:
            return None
        # Shape metadata is enough to establish fallback start positions.
        shape = z["ground_truth"].shape
        if len(shape) == 4:
            frames = shape[0]
        elif len(shape) == 5:
            frames = shape[1]
        else:
            raise ValueError(f"Unexpected video shape in {path}: {shape}")
        meta = {}
        for key in ("trajectory_id", "interaction_index", "start_frame_index"):
            if key in z:
                value = z[key]
                meta[key] = value.item() if np.asarray(value).ndim == 0 else str(value)
    traj = str(meta.get("trajectory_id", path.parent.name))
    interaction = int(meta.get("interaction_index", 0))
    start = int(meta.get("start_frame_index", interaction * frames))
    return {"path": path, "traj": traj, "interaction": interaction, "start": start}


def psnr_batch(gt: np.ndarray, pred: np.ndarray, data_range: float) -> np.ndarray:
    # Equivalent to per-image peak_signal_noise_ratio for RGB uint8 frames.
    a = gt.astype(np.float32)
    b = pred.astype(np.float32)
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3), dtype=np.float64)
    scores = np.full(mse.shape, np.inf, dtype=np.float64)
    nonzero = mse > 0
    scores[nonzero] = 10.0 * np.log10((data_range ** 2) / mse[nonzero])
    return scores


def paired_metrics(files, requested, args, timings):
    import torch

    lpips_model = None
    if "lpips" in requested:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net=args.lpips_net).to(args.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "LPIPS initialization failed. Check lpips and PyTorch/torchvision. "
                f"Original error: {exc}"
            ) from exc

    rows = []
    ssim_pool = ThreadPoolExecutor(max_workers=max(1, args.ssim_workers)) if "ssim" in requested else None
    try:
        for file_index, path in enumerate(files, 1):
            t0 = time.perf_counter()
            rec = load_record(path)
            timings["npz_load_seconds"] += time.perf_counter() - t0
            if rec is None:
                print(f"[SKIP] {path}: missing ground_truth/prediction")
                continue

            gt = rec["gt"].reshape(-1, *rec["gt"].shape[2:])
            pred = rec["pred"].reshape(-1, *rec["pred"].shape[2:])
            row = {
                "trajectory_id": rec["traj"],
                "interaction_index": rec["interaction"],
                "start_frame_index": rec["start"],
                "file": rec["path"],
                "num_views": rec["gt"].shape[0],
                "num_frames_per_view": rec["gt"].shape[1],
                "num_frame_pairs": len(gt),
            }

            if "psnr" in requested:
                t0 = time.perf_counter()
                values = psnr_batch(gt, pred, args.ssim_data_range)
                timings["psnr_seconds"] += time.perf_counter() - t0
                finite = values[np.isfinite(values)]
                if finite.size:
                    row["psnr"] = float(finite.mean())
                    row["psnr_std"] = float(finite.std())

            if "ssim" in requested:
                t0 = time.perf_counter()
                # Threaded CPU work; preserves skimage's original SSIM implementation.
                values = list(ssim_pool.map(
                    lambda pair: float(structural_similarity(
                        pair[0], pair[1], channel_axis=-1,
                        data_range=args.ssim_data_range)),
                    zip(gt, pred),
                ))
                timings["ssim_seconds"] += time.perf_counter() - t0
                values = np.asarray(values, dtype=np.float64)
                finite = values[np.isfinite(values)]
                if finite.size:
                    row["ssim"] = float(finite.mean())
                    row["ssim_std"] = float(finite.std())

            if lpips_model is not None:
                t0 = time.perf_counter()
                values = []
                for start in range(0, len(gt), args.lpips_batch_size):
                    end = min(start + args.lpips_batch_size, len(gt))
                    a = torch.from_numpy(gt[start:end].copy()).permute(0, 3, 1, 2).float()
                    b = torch.from_numpy(pred[start:end].copy()).permute(0, 3, 1, 2).float()
                    a = a.to(args.device, non_blocking=True).div_(127.5).sub_(1.0)
                    b = b.to(args.device, non_blocking=True).div_(127.5).sub_(1.0)
                    with torch.inference_mode():
                        score = lpips_model(a, b)
                    values.extend(score.reshape(-1).float().cpu().tolist())
                    del a, b, score
                timings["lpips_seconds"] += time.perf_counter() - t0
                values = np.asarray(values, dtype=np.float64)
                finite = values[np.isfinite(values)]
                if finite.size:
                    row["lpips"] = float(finite.mean())
                    row["lpips_std"] = float(finite.std())

            rows.append(row)
            print(f"[PAIRED {file_index}/{len(files)}] {path.name} " + " ".join(
                f"{key}={row[key]:.5f}" for key in ("psnr", "ssim", "lpips") if key in row
            ))
    finally:
        if ssim_pool is not None:
            ssim_pool.shutdown(wait=True)
    return rows


def compute_fid(files, device, batch_size, timings):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    t0 = time.perf_counter()
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    count = 0
    with torch.inference_mode():
        for path in files:
            rec = load_record(path)
            if rec is None:
                continue
            for arr, real in ((rec["gt"], True), (rec["pred"], False)):
                images = arr.reshape(-1, *arr.shape[2:])
                for start in range(0, len(images), batch_size):
                    batch = images[start:start + batch_size]
                    x = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2)
                    x = x.to(device=device, dtype=torch.uint8, non_blocking=True)
                    metric.update(x, real=real)
                    if real:
                        count += len(batch)
            del rec
    if count < 2:
        raise RuntimeError("FID needs at least two real and two generated images.")
    value = float(metric.compute().item())
    timings["fid_seconds"] += time.perf_counter() - t0
    del metric
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return value, count


def build_trajectory_videos(files, clip_len, stride, max_clips):
    """Reconstruct per-trajectory/per-view sequences and extract FVD clips.

    Preserves the previous script's per-view max_clips behavior and overlap rule.
    Only one trajectory's NPZ video arrays are resident at a time.
    """
    metadata = []
    for path in files:
        item = read_metadata(path)
        if item is not None:
            metadata.append(item)
    grouped = defaultdict(list)
    for item in metadata:
        grouped[item["traj"]].append(item)

    real_clips, fake_clips, clip_sources = [], [], []
    for traj, items in grouped.items():
        items.sort(key=lambda x: (x["start"], x["interaction"]))
        gt_sequences = None
        pred_sequences = None
        last_ends = None
        for item in items:
            rec = load_record(item["path"])
            if rec is None:
                continue
            nviews = rec["gt"].shape[0]
            if gt_sequences is None:
                gt_sequences = [[] for _ in range(nviews)]
                pred_sequences = [[] for _ in range(nviews)]
                last_ends = [None] * nviews
            nviews = min(nviews, len(gt_sequences))
            for view in range(nviews):
                start = item["start"]
                gt, pred = rec["gt"][view], rec["pred"][view]
                last_end = last_ends[view]
                offset = 0 if last_end is None else max(0, last_end - start)
                if offset < len(gt):
                    gt_sequences[view].append(gt[offset:])
                    pred_sequences[view].append(pred[offset:])
                    last_ends[view] = max(last_end or start, start + len(gt))
            del rec

        if gt_sequences is None:
            continue
        for view, (gt_parts, pred_parts) in enumerate(zip(gt_sequences, pred_sequences)):
            if not gt_parts:
                continue
            gt_seq = np.concatenate(gt_parts, axis=0)
            pred_seq = np.concatenate(pred_parts, axis=0)
            made = 0
            for start in range(0, min(len(gt_seq), len(pred_seq)) - clip_len + 1, stride):
                real_clips.append(gt_seq[start:start + clip_len])
                fake_clips.append(pred_seq[start:start + clip_len])
                clip_sources.append(f"{traj}:view{view}:frame{start}")
                made += 1
                if made >= max_clips:
                    break
            del gt_seq, pred_seq
        del gt_sequences, pred_sequences
    return real_clips, fake_clips, clip_sources


def compute_fvd(real_clips, fake_clips, device, timings):
    from cdfvd import fvd
    t0 = time.perf_counter()
    real_array = np.ascontiguousarray(np.stack(real_clips), dtype=np.uint8)
    fake_array = np.ascontiguousarray(np.stack(fake_clips), dtype=np.uint8)
    if real_array.shape != fake_array.shape:
        raise ValueError(f"Real/fake shapes differ: {real_array.shape} vs {fake_array.shape}")
    evaluator = fvd.cdfvd(
        model="i3d", n_real="full", n_fake=len(fake_array),
        device=device, half_precision=False,
    )
    score = evaluator.compute_fvd(real_videos=real_array, fake_videos=fake_array)
    timings["fvd_seconds"] += time.perf_counter() - t0
    return float(score)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_paired(frame_rows, requested):
    result = {}
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
            total_sum_sq += n * (std * std + mean * mean)
        if total_n:
            mean = total_sum / total_n
            variance = max(0.0, total_sum_sq / total_n - mean * mean)
            result[key] = {"mean": mean, "std": math.sqrt(variance), "n_frame_pairs": total_n}
    return result


def main():
    args = parse_args()
    if args.lpips_batch_size < 1 or args.fid_batch_size < 1 or args.ssim_workers < 1:
        raise ValueError("Batch sizes and worker count must be positive.")
    if args.fvd_clip_len < 1 or args.fvd_stride < 1 or args.max_fvd_clips_per_trajectory < 1:
        raise ValueError("FVD clip length, stride and max clips must be positive.")

    requested = set(args.metrics)
    data_dir = Path(args.data_dir)
    files = list_npz_files(data_dir)
    run_dir = make_run_dir(Path(args.output_root), args.run_name)
    timings = defaultdict(float)

    import torch
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print(f"[INFO] Found {len(files)} NPZ files")
    print(f"[INFO] Output: {run_dir}")
    print(f"[INFO] Device={args.device}; LPIPS batch={args.lpips_batch_size}; "
          f"FID batch={args.fid_batch_size}; SSIM workers={args.ssim_workers}")

    t0 = time.perf_counter()
    frame_rows = paired_metrics(files, requested, args, timings)
    write_csv(run_dir / "per_interaction_metrics.csv", frame_rows)
    timings["paired_total_seconds"] = time.perf_counter() - t0

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "num_npz_files_found": len(files),
        "num_valid_interactions": len(frame_rows),
        "num_trajectories": len({row["trajectory_id"] for row in frame_rows}),
        "metrics_requested": sorted(requested),
        "protocol": {
            "paired_metrics": "mean over paired RGB frames; per-interaction mean/std then frame-count-weighted global mean/std",
            "psnr": "vectorized per-frame MSE formula; data_range from --ssim_data_range",
            "ssim": "skimage structural_similarity, channel_axis=-1",
            "lpips": f"LPIPS {args.lpips_net}, RGB [-1,1], batched inference",
            "fid": "TorchMetrics FrechetInceptionDistance(feature=2048, normalize=False), batched updates",
            "fvd": "cd-fvd I3D; reconstructed per-trajectory/per-view clips",
            "fvd_clip_len": args.fvd_clip_len,
            "fvd_stride": args.fvd_stride,
            "max_fvd_clips_per_trajectory_per_view": args.max_fvd_clips_per_trajectory,
            "overlap_handling": "sort by start_frame_index and trim repeated frame indices",
        },
        "paired_metrics": summarize_paired(frame_rows, requested),
        "distribution_metrics": {},
    }

    if "fid" in requested:
        fid, nframes = compute_fid(files, args.device, args.fid_batch_size, timings)
        summary["distribution_metrics"]["fid"] = {
            "value": fid, "num_real_and_generated_frames_each": nframes,
        }
        print(f"[RESULT] FID={fid:.5f} (frames per distribution={nframes})")

    if "fvd" in requested:
        t0 = time.perf_counter()
        real, fake, sources = build_trajectory_videos(
            files, args.fvd_clip_len, args.fvd_stride,
            args.max_fvd_clips_per_trajectory,
        )
        timings["fvd_clip_build_seconds"] = time.perf_counter() - t0
        if len(real) < 2:
            raise RuntimeError(
                f"Only {len(real)} FVD clips assembled. Check chunk lengths and start_frame_index metadata."
            )
        fvd_score = compute_fvd(real, fake, args.device, timings)
        summary["distribution_metrics"]["fvd"] = {
            "value": fvd_score, "num_video_clips_each": len(real),
            "clip_length_frames": args.fvd_clip_len,
            "clip_sources": sources,
        }
        print(f"[RESULT] FVD={fvd_score:.5f} (clips per distribution={len(real)})")

    if args.profile:
        cuda_peak = None
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            cuda_peak = {
                "allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            }
        timings["total_seconds"] = sum(
            value for key, value in timings.items() if key.endswith("_seconds")
        )
        summary["performance"] = {
            "stage_seconds": dict(timings),
            "peak_cuda_memory": cuda_peak,
            "cpu_count": os.cpu_count(),
        }
        print("[PROFILE]", json.dumps(summary["performance"], indent=2))

    with (run_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (run_dir / "run_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    print("\n=== SUMMARY ===")
    for section in ("paired_metrics", "distribution_metrics"):
        for name, values in summary[section].items():
            print(name.upper(), values if name != "fvd" else {k: v for k, v in values.items() if k != "clip_sources"})
    print(f"\n[OK] Results saved in: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
