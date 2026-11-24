"""Dataset-level image FID and source-index-aligned video FVD."""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..data_loader import iter_samples

FID_WEIGHTS_FILENAME = "weights-inception-2015-12-05-6726825d.pth"
FVD_BATCH_SIZE = 16


def resolve_fid_weights(weights_path: str | None) -> tuple[Path | None, str | None]:
    """Find explicit or cached Inception weights without triggering downloads."""
    if weights_path is not None:
        weights = Path(weights_path).expanduser()
        if weights.is_file():
            return weights, None
        return None, f"FID weights file does not exist: {weights}"

    try:
        import torch
    except ImportError as exc:
        return None, f"PyTorch is unavailable for FID: {exc}"
    cached = Path(torch.hub.get_dir()) / "checkpoints" / FID_WEIGHTS_FILENAME
    if cached.is_file():
        return cached, None
    return (
        None,
        "No local Inception-v3 FID weights were found. Supply --fid_weights; "
        "automatic weight downloads are disabled.",
    )


def load_fid_metric(weights_path: Path, device: str) -> Any:
    """Initialize TorchMetrics FID with an explicit local weight file."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(
        feature=2048,
        normalize=False,
        feature_extractor_weights_path=str(weights_path),
        antialias=False,
    ).to(device)
    metric.set_dtype(torch.float64)
    return metric


def update_fid_metric(
    metric: Any, ground_truth: np.ndarray, prediction: np.ndarray, batch_size: int, device: str
) -> int:
    """Update dataset-level FID statistics with all camera/frame RGB images."""
    import torch

    frame_count = ground_truth.shape[0] * ground_truth.shape[1]
    real_images = ground_truth.reshape(frame_count, *ground_truth.shape[2:])
    generated_images = prediction.reshape(frame_count, *prediction.shape[2:])
    with torch.inference_mode():
        for images, is_real in ((real_images, True), (generated_images, False)):
            for offset in range(0, frame_count, batch_size):
                batch = torch.from_numpy(images[offset : offset + batch_size].copy())
                batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.uint8)
                metric.update(batch, real=is_real)
    return frame_count


def _read_record_metadata(
    path: Path, frame_stride: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Read only NPZ metadata needed to order FVD chunks."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "ground_truth" not in archive or "prediction" not in archive:
                return None, "missing ground_truth or prediction"

            trajectory_value = (
                np.asarray(archive["trajectory_id"]).reshape(()).item()
                if "trajectory_id" in archive
                else ""
            )
            if isinstance(trajectory_value, bytes):
                trajectory_value = trajectory_value.decode("utf-8")
            trajectory_id = str(trajectory_value)
            if not trajectory_id:
                trajectory_id = path.parent.name.removeprefix("trajectory_")

            interaction_value = (
                np.asarray(archive["interaction_index"]).reshape(()).item()
                if "interaction_index" in archive
                else 0
            )
            interaction_index = int(interaction_value)
            start_value = (
                np.asarray(archive["start_frame_index"]).reshape(()).item()
                if "start_frame_index" in archive
                else None
            )
            start_frame_index = int(start_value) if start_value is not None else None

            if "frame_indices" in archive:
                frame_indices = np.asarray(archive["frame_indices"]).reshape(-1)
                if not np.issubdtype(frame_indices.dtype, np.number):
                    return None, "frame_indices must be numeric"
                selected_indices = frame_indices[::frame_stride]
            else:
                selected_indices = np.empty(0, dtype=np.int64)
    except (OSError, ValueError, EOFError, TypeError, OverflowError) as exc:
        return None, str(exc)

    first_index = (
        int(selected_indices[0])
        if len(selected_indices)
        else start_frame_index
        if start_frame_index is not None
        else 2**63 - 1
    )
    return (
        {
            "path": path,
            "trajectory_id": trajectory_id,
            "interaction_index": interaction_index,
            "first_index": first_index,
        },
        None,
    )


def _iter_segments(
    frame_map: dict[int, tuple[np.ndarray, np.ndarray]],
) -> Iterator[list[tuple[np.ndarray, np.ndarray]]]:
    """Yield source-contiguous, fixed-resolution runs for one camera."""
    segment: list[tuple[np.ndarray, np.ndarray]] = []
    previous_index: int | None = None
    previous_shape: tuple[int, ...] | None = None
    for source_index, pair in sorted(frame_map.items()):
        shape = pair[0].shape
        if segment and (
            previous_index is None
            or source_index != previous_index + 1
            or shape != previous_shape
        ):
            yield segment
            segment = []
        segment.append(pair)
        previous_index = source_index
        previous_shape = shape
    if segment:
        yield segment


def _iter_video_clip_batches(
    files: list[Path],
    frame_stride: int,
    clip_length: int,
    clip_stride: int,
    max_clips_per_trajectory_camera: int,
    stats: dict[str, int],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Reconstruct clips by source index, deduplicating overlaps and splitting gaps."""
    records = []
    for path in files:
        record, _ = _read_record_metadata(path, frame_stride)
        if record is None:
            stats["metadata_read_failures"] += 1
            continue
        records.append(record)

    records.sort(
        key=lambda item: (
            item["trajectory_id"],
            item["first_index"],
            item["interaction_index"],
            str(item["path"]),
        )
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["trajectory_id"]].append(record)

    pending_real: list[np.ndarray] = []
    pending_generated: list[np.ndarray] = []
    pending_shape: tuple[int, ...] | None = None

    def flush() -> tuple[np.ndarray, np.ndarray] | None:
        nonlocal pending_shape
        if not pending_real:
            return None
        batch = (
            np.ascontiguousarray(np.stack(pending_real), dtype=np.uint8),
            np.ascontiguousarray(np.stack(pending_generated), dtype=np.uint8),
        )
        pending_real.clear()
        pending_generated.clear()
        pending_shape = None
        return batch

    for trajectory_records in grouped.values():
        frames_by_camera: dict[
            int, dict[int, tuple[np.ndarray, np.ndarray]]
        ] = defaultdict(dict)
        for record in trajectory_records:
            _, sample, _ = next(iter_samples([record["path"]], frame_stride=frame_stride))
            if sample is None:
                stats["sample_load_failures"] += 1
                continue
            if sample.frame_indices is None:
                stats["samples_without_source_indices"] += 1
                continue
            for camera in range(sample.ground_truth.shape[0]):
                camera_frames = frames_by_camera[camera]
                for position, source_index_value in enumerate(sample.frame_indices):
                    source_index = int(source_index_value)
                    if source_index < 0:
                        stats["negative_source_indices_skipped"] += 1
                        continue
                    if source_index in camera_frames:
                        stats["duplicate_frame_indices_dropped"] += 1
                        continue
                    camera_frames[source_index] = (
                        sample.ground_truth[camera, position],
                        sample.prediction[camera, position],
                    )

        for camera_frames in frames_by_camera.values():
            camera_clip_count = 0
            for segment in _iter_segments(camera_frames):
                if len(segment) < clip_length:
                    stats["segments_too_short"] += 1
                    continue
                for start in range(0, len(segment) - clip_length + 1, clip_stride):
                    real_clip = np.stack(
                        [pair[0] for pair in segment[start : start + clip_length]]
                    )
                    generated_clip = np.stack(
                        [pair[1] for pair in segment[start : start + clip_length]]
                    )
                    if pending_shape is not None and real_clip.shape != pending_shape:
                        batch = flush()
                        if batch is not None:
                            yield batch
                    if pending_shape is None:
                        pending_shape = real_clip.shape
                    pending_real.append(real_clip)
                    pending_generated.append(generated_clip)
                    camera_clip_count += 1
                    stats["clip_count"] += 1
                    if len(pending_real) >= FVD_BATCH_SIZE:
                        batch = flush()
                        if batch is not None:
                            yield batch
                    if camera_clip_count >= max_clips_per_trajectory_camera:
                        break
                if camera_clip_count >= max_clips_per_trajectory_camera:
                    break

    batch = flush()
    if batch is not None:
        yield batch


def compute_video_fvd(
    files: list[Path],
    frame_stride: int,
    clip_length: int,
    clip_stride: int,
    max_clips_per_trajectory_camera: int,
    device: str,
    weights_path: str | None,
) -> dict[str, Any]:
    """Compute I3D FVD from reconstructed local clips without fetching weights."""
    stats = {
        "clip_count": 0,
        "metadata_read_failures": 0,
        "sample_load_failures": 0,
        "samples_without_source_indices": 0,
        "negative_source_indices_skipped": 0,
        "duplicate_frame_indices_dropped": 0,
        "segments_too_short": 0,
    }
    weights = None
    if weights_path is not None:
        weights = Path(weights_path).expanduser()
        if not weights.is_file():
            reason = f"FVD I3D weights file does not exist: {weights}"
    else:
        package = importlib.util.find_spec("cdfvd")
        if package is None or package.submodule_search_locations is None:
            reason = "The cd-fvd package is unavailable; install the declared dependency to enable video FVD"
        else:
            package_root = Path(next(iter(package.submodule_search_locations)))
            bundled_weights = (
                package_root / "third_party" / "i3d" / "i3d_pretrained_400.pt"
            )
            if bundled_weights.is_file():
                weights = bundled_weights
                reason = ""
            else:
                reason = (
                    "No local cd-fvd I3D weights were found. Supply --fvd_weights; "
                    "automatic weight downloads are disabled."
                )
    if weights is None:
        return {
            "metric": "fvd",
            "status": "unavailable",
            "value": None,
            "reason": reason,
            "mode": "video_i3d",
            "frame_stride": frame_stride,
            "clip_length": clip_length,
            "clip_stride": clip_stride,
            "max_clips_per_trajectory_camera": max_clips_per_trajectory_camera,
            **stats,
        }

    try:
        from cdfvd import fvd

        evaluator = fvd.cdfvd(
            model="i3d",
            n_real="full",
            n_fake="full",
            device=device,
            half_precision=False,
            ckpt_path=str(weights),
        )
        for real_batch, generated_batch in _iter_video_clip_batches(
            files,
            frame_stride,
            clip_length,
            clip_stride,
            max_clips_per_trajectory_camera,
            stats,
        ):
            evaluator.add_real_stats(real_batch)
            evaluator.add_fake_stats(generated_batch)
        if stats["clip_count"] < 2:
            reason = (
                f"FVD requires at least two clips per distribution; "
                f"only {stats['clip_count']} eligible clips were reconstructed"
            )
            return {
                "metric": "fvd",
                "status": "unavailable",
                "value": None,
                "reason": reason,
                "mode": "video_i3d",
                "feature_extractor": "cd-fvd I3D-400",
                "weights_path": str(weights),
                "frame_stride": frame_stride,
                "real_clip_count": stats["clip_count"],
                "generated_clip_count": stats["clip_count"],
                "clip_length": clip_length,
                "clip_stride": clip_stride,
                "max_clips_per_trajectory_camera": max_clips_per_trajectory_camera,
                **stats,
            }
        value = float(evaluator.compute_fvd_from_stats())
        if not np.isfinite(value):
            raise ValueError("I3D FVD computation returned a non-finite value")
    except (ImportError, OSError, EOFError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return {
            "metric": "fvd",
            "status": "unavailable",
            "value": None,
            "reason": f"Video FVD extraction failed: {exc}",
            "mode": "video_i3d",
            "feature_extractor": "cd-fvd I3D-400",
            "weights_path": str(weights),
            "frame_stride": frame_stride,
            "real_clip_count": stats["clip_count"],
            "generated_clip_count": stats["clip_count"],
            "clip_length": clip_length,
            "clip_stride": clip_stride,
            "max_clips_per_trajectory_camera": max_clips_per_trajectory_camera,
            **stats,
        }
    return {
        "metric": "fvd",
        "status": "available",
        "value": value,
        "reason": None,
        "mode": "video_i3d",
        "feature_extractor": "cd-fvd I3D-400",
        "weights_path": str(weights),
        "frame_stride": frame_stride,
        "real_clip_count": stats["clip_count"],
        "generated_clip_count": stats["clip_count"],
        "clip_length": clip_length,
        "clip_stride": clip_stride,
        "max_clips_per_trajectory_camera": max_clips_per_trajectory_camera,
        **stats,
    }
