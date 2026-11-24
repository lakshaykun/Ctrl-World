"""Shared data types and serialization helpers for offline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EvaluationSample:
    """Canonical representation of one saved interaction NPZ."""

    path: Path
    ground_truth: np.ndarray
    prediction: np.ndarray
    trajectory_id: str
    interaction_index: int | None = None
    start_frame_index: int | None = None
    frame_indices: np.ndarray | None = None
    instruction: str | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationData:
    """Annotation fields aligned to a sample's source-frame indices."""

    status: str
    reason: str | None
    annotation_path: str | None
    trajectory_id: str
    frame_indices: np.ndarray | None = None
    states: np.ndarray | None = None
    joints: np.ndarray | None = None
    joints_reason: str | None = None
    texts: list[str] | None = None
    video_length: int | None = None


def json_safe(value: Any) -> Any:
    """Convert NumPy and pathlib values recursively into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def metric_status(name: str, status: str, value: Any = None, reason: str | None = None) -> dict[str, Any]:
    """Create the common explicit metric-status representation."""
    return {"metric": name, "status": status, "value": json_safe(value), "reason": reason}


def dataclass_dict(value: Any) -> dict[str, Any]:
    """Serialize a dataclass while preserving JSON-safe scalar values."""
    return json_safe(asdict(value))
