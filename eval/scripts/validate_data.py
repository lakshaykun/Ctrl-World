"""Structured validation for rollout arrays and optional aligned annotations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import AnnotationData, EvaluationSample


def validate_sample(
    sample: EvaluationSample, annotation: AnnotationData | None = None
) -> dict[str, Any]:
    """Return validation checks without discarding malformed or unavailable data."""
    gt = sample.ground_truth
    pred = sample.prediction
    checks = {
        "required_fields": {"status": "valid", "fields": ["ground_truth", "prediction"]},
        "shape": {
            "status": "valid" if gt.shape == pred.shape else "invalid",
            "ground_truth": list(gt.shape),
            "prediction": list(pred.shape),
            "layout": "camera,time,height,width,channel",
        },
        "camera_count": {"status": "valid", "count": int(gt.shape[0])},
        "frame_count": {"status": "valid", "count": int(gt.shape[1])},
        "image": {
            "status": "valid",
            "height": int(gt.shape[2]),
            "width": int(gt.shape[3]),
            "channels": int(gt.shape[4]),
            "dtype": str(gt.dtype),
            "finite": bool(np.isfinite(gt).all() and np.isfinite(pred).all()),
        },
        "frame_indices": {"status": "available" if sample.frame_indices is not None else "missing"},
        "fps": {
            "status": (
                "missing"
                if sample.fps is None
                else "available"
                if np.isfinite(sample.fps) and sample.fps > 0
                else "invalid"
            ),
            "value": sample.fps,
        },
        "annotation": (
            {"status": annotation.status, "reason": annotation.reason}
            if annotation is not None
            else {"status": "not_requested"}
        ),
    }
    if sample.frame_indices is not None:
        checks["frame_indices"].update(
            {
                "count": int(len(sample.frame_indices)),
                "valid_integer_values": bool(np.issubdtype(sample.frame_indices.dtype, np.integer)),
                "strictly_increasing": bool(
                    len(sample.frame_indices) < 2
                    or np.all(np.diff(sample.frame_indices) > 0)
                ),
                "nonnegative": bool(np.all(sample.frame_indices >= 0)),
            }
        )
        if np.any(sample.frame_indices < 0):
            checks["frame_indices"]["status"] = "invalid"
    if annotation is not None and annotation.status == "available":
        checks["annotation"].update(
            {
                "path": annotation.annotation_path,
                "aligned_frame_count": int(len(annotation.frame_indices)),
                "state_shape": list(annotation.states.shape),
                "joint_shape": (
                    list(annotation.joints.shape) if annotation.joints is not None else None
                ),
                "joint_status": "available" if annotation.joints is not None else "unavailable",
                "joint_reason": annotation.joints_reason,
            }
        )
    invalid = any(item.get("status") == "invalid" for item in checks.values())
    return {
        "sample": str(sample.path),
        "trajectory_id": sample.trajectory_id,
        "interaction_index": sample.interaction_index,
        "status": "invalid" if invalid else "valid",
        "checks": checks,
    }


def validation_failure(path: str, reason: str) -> dict[str, Any]:
    """Represent a failed load while retaining the discovered source path."""
    return {
        "sample": path,
        "trajectory_id": None,
        "interaction_index": None,
        "status": "invalid",
        "checks": {"load": {"status": "invalid", "reason": reason}},
    }
