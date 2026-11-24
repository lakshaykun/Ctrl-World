"""DROID annotation retrieval with explicit source-frame alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import AnnotationData, EvaluationSample


def _resolve_annotation(root: Path, trajectory_id: str) -> Path | None:
    annotation_dir = root / "annotation" / "val"
    exact = annotation_dir / f"{trajectory_id}.json"
    if exact.is_file():
        return exact
    if not trajectory_id.isdigit() or not annotation_dir.is_dir():
        return None
    numeric_matches = [
        path
        for path in annotation_dir.glob("*.json")
        if path.stem.isdigit() and int(path.stem) == int(trajectory_id)
    ]
    return numeric_matches[0] if len(numeric_matches) == 1 else None


def _numeric_sequence(annotation: dict[str, Any], key: str) -> np.ndarray:
    if key not in annotation:
        raise ValueError(f"Required annotation field {key!r} is missing")
    values = np.asarray(annotation[key])
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Annotation field {key!r} must be a numeric 2D array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Annotation field {key!r} contains non-finite values")
    return values


def load_annotation(sample: EvaluationSample, annotation_root: str | Path) -> AnnotationData:
    """Load states/joints aligned to explicit NPZ frame indices.

    For older NPZ files without `frame_indices`, alignment is inferred only from
    a valid start_frame_index and contiguous stored frames that fit the annotation.
    """
    root = Path(annotation_root)
    annotation_path = _resolve_annotation(root, sample.trajectory_id)
    if annotation_path is None:
        return AnnotationData(
            "unavailable",
            f"No unique annotation JSON found for trajectory {sample.trajectory_id!r} "
            f"under {root / 'annotation' / 'val'}",
            None,
            sample.trajectory_id,
        )
    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        if not isinstance(annotation, dict):
            raise ValueError("Annotation root must be a JSON object")
        raw_video_length = annotation["video_length"]
        video_length = int(raw_video_length)
        if isinstance(raw_video_length, bool) or float(raw_video_length) != video_length:
            raise ValueError("video_length must be an integer")
        if video_length < 1:
            raise ValueError("video_length must be positive")
        states = _numeric_sequence(annotation, "states")
        joints = None
        joints_reason = None
        try:
            joints = _numeric_sequence(annotation, "joints")
        except ValueError as exc:
            joints_reason = str(exc)
        if states.shape[0] < video_length:
            raise ValueError(
                f"states length {states.shape[0]} is shorter than video_length {video_length}"
            )
        if joints is not None and joints.shape[0] < video_length:
            joints_reason = (
                f"joints length {joints.shape[0]} is shorter than video_length {video_length}"
            )
            joints = None
        if states.shape[1] != 7:
            raise ValueError(f"Expected 7D Cartesian states, got dimension {states.shape[1]}")
        if joints is not None and joints.shape[1] != 8:
            joints_reason = f"Expected 8D joint/gripper positions, got dimension {joints.shape[1]}"
            joints = None
        texts_value = annotation.get("texts", [])
        if not isinstance(texts_value, list) or not all(isinstance(text, str) for text in texts_value):
            raise ValueError("texts must be a list of strings")

        if sample.frame_indices is not None:
            indices = sample.frame_indices.astype(np.int64, copy=False)
        elif sample.start_frame_index is not None:
            indices = sample.start_frame_index + np.arange(sample.ground_truth.shape[1])
        else:
            raise ValueError(
                "NPZ has no frame_indices and no start_frame_index; safe annotation alignment "
                "cannot be established"
            )
        if len(indices) != sample.ground_truth.shape[1]:
            raise ValueError(
                f"Aligned index count {len(indices)} does not match stored frames "
                f"{sample.ground_truth.shape[1]}"
            )
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= video_length):
            raise ValueError(
                f"Frame indices must be within [0, {video_length - 1}], got "
                f"{indices.tolist()}"
            )
        return AnnotationData(
            "available",
            None,
            str(annotation_path),
            sample.trajectory_id,
            indices,
            states[indices],
            joints[indices] if joints is not None else None,
            joints_reason,
            texts_value,
            video_length,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return AnnotationData(
            "unavailable",
            f"Could not safely align annotation {annotation_path}: {exc}",
            str(annotation_path),
            sample.trajectory_id,
        )
