"""Conservative image-space physical-violation candidate diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .object_metrics import intersection_over_union


def detect_overlap_candidates(
    tracks_by_frame: list[list[dict[str, Any]]],
    sample_id: str,
    camera_index: int,
    overlap_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Report simultaneous box overlaps as 2D candidates, not 3D penetration."""
    events = []
    for frame_index, tracks in enumerate(tracks_by_frame):
        for left_index, left in enumerate(tracks):
            for right in tracks[left_index + 1 :]:
                overlap = intersection_over_union(
                    np.asarray(left["box_xyxy"]), np.asarray(right["box_xyxy"])
                )
                if overlap >= overlap_threshold:
                    events.append(
                        {
                            "sample_id": sample_id,
                            "frame_position": frame_index,
                            "camera_index": camera_index,
                            "event": "image_space_box_overlap_candidate",
                            "object_id_a": left.get("track_id"),
                            "object_id_b": right.get("track_id"),
                            "box_xyxy_a": list(left["box_xyxy"]),
                            "box_xyxy_b": list(right["box_xyxy"]),
                            "evidence": f"2D bounding-box IoU={overlap:.6f}",
                            "severity": float(overlap),
                        }
                    )
    return events


def summarize_physical_candidates(
    events: list[dict[str, Any]],
    frame_count: int,
    generated_detection_count: int,
    sample_id: str,
) -> dict[str, Any]:
    """Summarize observed overlap candidates with an explicit proxy denominator."""
    candidate_frames = len({event["frame_position"] for event in events})
    observations = max(1, frame_count)
    rate = (
        candidate_frames / observations
        if generated_detection_count > 0
        else None
    )
    return {
        "sample_id": sample_id,
        "physical_violation_rate": rate,
        "physical_violation_rate_status": (
            "available" if rate is not None else "unavailable"
        ),
        "physical_violation_rate_reason": (
            None
            if rate is not None
            else "No generated object detections were available for the overlap check"
        ),
        "physical_violation_candidate_count": len(events),
        "physical_violation_rate_denominator": observations,
        "physical_violation_rate_interpretation": (
            "Fraction of observed frames with at least one 2D bounding-box overlap candidate; "
            "not a verified 3D penetration or physics violation"
        ),
        "floating_object_candidates": None,
        "floating_object_candidates_status": "unavailable",
        "floating_object_candidates_reason": (
            "No support-surface labels, depth, or 3D geometry are configured"
        ),
        "contact_consistency_status": "unavailable",
        "contact_consistency_reason": (
            "No contact labels, calibrated robot-camera transform, or 3D geometry are available"
        ),
        "spatial_relationship_consistency_status": "unavailable",
        "spatial_relationship_consistency_reason": (
            "No task-specific object relation labels are configured"
        ),
        "physical_frame_count": frame_count,
    }
