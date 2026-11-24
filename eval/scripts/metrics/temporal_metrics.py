"""Frame-transition consistency and Farneback optical-flow comparison."""

from __future__ import annotations

from typing import Any

import numpy as np


def _gray(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _flow(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(
        _gray(first),
        _gray(second),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def compute_temporal_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray, sample_id: str
) -> list[dict[str, Any]]:
    """Compute temporal difference error and estimated flow-field difference.

    `optical_flow_difference` is the mean Euclidean difference between two
    independently estimated Farneback vector fields in pixels per sampled
    frame interval. It is not endpoint error against annotated optical flow.
    """
    rows = []
    cameras, frame_count = ground_truth.shape[:2]
    if frame_count < 2:
        return rows
    for camera in range(cameras):
        for frame in range(1, frame_count):
            gt_delta = (
                ground_truth[camera, frame].astype(np.float32)
                - ground_truth[camera, frame - 1].astype(np.float32)
            )
            pred_delta = (
                prediction[camera, frame].astype(np.float32)
                - prediction[camera, frame - 1].astype(np.float32)
            )
            temporal_error = float(np.mean(np.abs(gt_delta - pred_delta)))
            gt_flow = _flow(ground_truth[camera, frame - 1], ground_truth[camera, frame])
            pred_flow = _flow(prediction[camera, frame - 1], prediction[camera, frame])
            flow_error = float(
                np.mean(np.linalg.norm(gt_flow - pred_flow, axis=-1), dtype=np.float64)
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "camera_index": camera,
                    "frame_position": frame,
                    "temporal_difference_error": temporal_error,
                    "optical_flow_difference": flow_error,
                    "optical_flow_method": "OpenCV Farneback, default fixed parameters",
                }
            )
    return rows


def mean_flow_magnitude(video: np.ndarray) -> np.ndarray:
    """Return each transition's mean Farneback flow magnitude in pixels."""
    if len(video) < 2:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [
            np.linalg.norm(_flow(video[index - 1], video[index]), axis=-1).mean()
            for index in range(1, len(video))
        ],
        dtype=np.float64,
    )


def horizon_rows(
    frame_rows: list[dict[str, Any]], metric_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Summarize available metrics by zero-based prediction-frame position."""
    grouped: dict[int, dict[str, list[float]]] = {}
    observations: dict[int, dict[str, set[tuple[str, int]]]] = {}
    for row in frame_rows:
        if not any(
            row.get(name) is not None and np.isfinite(row[name])
            for name in metric_names
        ):
            continue
        horizon = int(row["frame_position"])
        bucket = grouped.setdefault(horizon, {name: [] for name in metric_names})
        metric_observations = observations.setdefault(
            horizon, {name: set() for name in metric_names}
        )
        sample_camera = (str(row["sample_id"]), int(row.get("camera_index", 0)))
        for name in metric_names:
            if name in ("temporal_difference_error", "optical_flow_difference") and horizon == 0:
                continue
            metric_observations[name].add(sample_camera)
            value = row.get(name)
            if value is not None and np.isfinite(value):
                bucket[name].append(float(value))
    result = []
    for horizon, metrics in sorted(grouped.items()):
        record: dict[str, Any] = {"horizon": horizon}
        for name, values in metrics.items():
            record[name] = float(np.mean(values)) if values else None
            record[f"{name}_median"] = float(np.median(values)) if values else None
            record[f"{name}_std"] = float(np.std(values)) if values else None
            record[f"{name}_min"] = float(np.min(values)) if values else None
            record[f"{name}_max"] = float(np.max(values)) if values else None
            record[f"{name}_valid_count"] = len(values)
            record[f"{name}_invalid_count"] = (
                len(observations[horizon][name]) - len(values)
            )
        result.append(record)
    return result
