"""Annotation-based robot-state summaries and image-space motion diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .object_metrics import _box, _match_frame
from ..utils import AnnotationData, metric_status


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else None


def compute_action_diagnostics(
    annotation: AnnotationData | None,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    reference_tracks: list[list[dict[str, Any]]] | None = None,
    object_tracks: list[list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize actual 7D states and their association with image-space motion.

    The first six state dimensions are treated as Cartesian pose coordinates;
    their units are annotation-native and are not assumed to be metric. The
    seventh dimension is retained as the gripper/state scalar and excluded from
    translational displacement magnitude.
    """
    if annotation is None or annotation.status != "available" or annotation.states is None:
        reason = annotation.reason if annotation is not None else "Annotation loading was not requested"
        return {
            "action_data_status": "unavailable",
            "action_data_reason": reason,
            "action_state_dimension": None,
            "joint_state_dimension": None,
            "action_mean_translation_step": None,
            "action_mean_rotation_step": None,
            "joint_mean_step": None,
            "action_image_motion_correlation": None,
            "action_motion_interpretation": "Unavailable: no valid aligned annotation states",
        }, []

    states = np.asarray(annotation.states, dtype=np.float64)
    joints = (
        np.asarray(annotation.joints, dtype=np.float64)
        if annotation.joints is not None
        else None
    )
    state_delta = np.diff(states, axis=0)
    translation_magnitude = (
        np.linalg.norm(state_delta[:, :3], axis=1)
        if len(state_delta)
        else np.empty(0, dtype=np.float64)
    )
    rotation_magnitude = (
        np.linalg.norm(state_delta[:, 3:6], axis=1)
        if len(state_delta)
        else np.empty(0, dtype=np.float64)
    )
    joint_magnitude = (
        np.linalg.norm(np.diff(joints, axis=0), axis=1)
        if joints is not None and len(joints) > 1
        else np.empty(0, dtype=np.float64)
    )
    image_motion = np.empty(0, dtype=np.float64)
    flow_reason = None
    frame_rows = []
    if ground_truth.shape[1] > 1:
        try:
            from .temporal_metrics import mean_flow_magnitude

            per_camera = [
                mean_flow_magnitude(prediction[camera])
                for camera in range(prediction.shape[0])
            ]
            image_motion = (
                np.mean(np.stack(per_camera), axis=0)
                if per_camera and len(per_camera[0])
                else np.empty(0, dtype=np.float64)
            )
        except ImportError as exc:
            flow_reason = f"Image-space motion correlation unavailable: {exc}"
    aligned_action = translation_magnitude
    correlation = _safe_correlation(aligned_action, image_motion)
    values = {
        "action_data_status": "available",
        "action_data_reason": None,
        "action_state_dimension": int(states.shape[1]),
        "joint_state_dimension": int(joints.shape[1]) if joints is not None else None,
        "action_mean_translation_step": (
            float(np.mean(translation_magnitude)) if len(translation_magnitude) else None
        ),
        "action_mean_rotation_step": (
            float(np.mean(rotation_magnitude)) if len(rotation_magnitude) else None
        ),
        "joint_mean_step": float(np.mean(joint_magnitude)) if len(joint_magnitude) else None,
        "joint_data_status": "available" if joints is not None else "unavailable",
        "joint_data_reason": annotation.joints_reason,
        "action_image_motion_correlation": correlation,
        "action_image_motion_correlation_reason": flow_reason,
        "action_motion_interpretation": (
            "Correlation is descriptive only; image flow is camera-dependent and does not "
            "establish causality or correct physical response"
        ),
    }
    for index in range(len(translation_magnitude)):
        frame_rows.append(
            {
                "sample_id": sample_id,
                "frame_position": index + 1,
                "end_effector_translation_step": float(translation_magnitude[index]),
                "end_effector_rotation_step": float(rotation_magnitude[index]),
                "joint_position_step": (
                    float(joint_magnitude[index]) if index < len(joint_magnitude) else None
                ),
                "recorded_state_frame_index": int(annotation.frame_indices[index + 1]),
            }
        )
    if (
        reference_tracks is not None
        and object_tracks is not None
        and len(reference_tracks) == len(states)
        and len(object_tracks) == len(states)
    ):
        displacements = []
        center_pairs = []
        for reference_frame, generated_frame in zip(reference_tracks, object_tracks):
            matches, _, _ = _match_frame(reference_frame, generated_frame, 0.5)
            paired_centers = {}
            for reference_index, generated_index in matches:
                reference_track = reference_frame[reference_index]
                generated_track = generated_frame[generated_index]
                reference_box = _box(reference_track)
                generated_box = _box(generated_track)
                pair_id = (
                    reference_track.get("track_id", reference_index),
                    generated_track.get("track_id", generated_index),
                )
                paired_centers[pair_id] = (
                    (reference_box[:2] + reference_box[2:]) / 2,
                    (generated_box[:2] + generated_box[2:]) / 2,
                )
            center_pairs.append(paired_centers)
        diagonal = float(np.hypot(ground_truth.shape[2], ground_truth.shape[3]))
        for frame_index in range(1, len(center_pairs)):
            previous = center_pairs[frame_index - 1]
            current = center_pairs[frame_index]
            for pair_id in previous.keys() & current.keys():
                reference_displacement = current[pair_id][0] - previous[pair_id][0]
                generated_displacement = current[pair_id][1] - previous[pair_id][1]
                error = float(
                    np.linalg.norm(generated_displacement - reference_displacement)
                    / diagonal
                )
                generated_motion = float(np.linalg.norm(generated_displacement))
                displacements.append(
                    (frame_index, error, generated_motion)
                )
        values["tracked_object_count_mean"] = float(
            np.mean([len(frame) for frame in object_tracks])
        )
        values["action_object_displacement_error"] = (
            float(np.mean([error for _, error, _ in displacements]))
            if displacements
            else None
        )
        values["action_object_displacement_correlation"] = (
            _safe_correlation(
                np.asarray(
                    [translation_magnitude[index - 1] for index, _, _ in displacements]
                ),
                np.asarray([distance for _, _, distance in displacements]),
            )
            if displacements
            else None
        )
        values["action_object_displacement_interpretation"] = (
            "Image-space track displacement correlation is camera-dependent and descriptive only"
        )
        values["object_displacement_analysis_status"] = "available" if displacements else "unavailable"
        if not displacements:
            values["object_displacement_analysis_reason"] = (
                "No matched consecutive reference/generated object tracks were available"
            )
    else:
        values["object_displacement_analysis_status"] = "unavailable"
        values["object_displacement_analysis_reason"] = (
            "Valid reference and generated object tracks aligned to the annotation frames are required"
        )
    unavailable = metric_status(
        "contact_timing_error",
        "unavailable",
        reason="No independently recorded contact timestamps or contact labels are available",
    )
    values["contact_timing_error"] = unavailable
    return values, frame_rows


def unavailable_action_metrics(reason: str) -> dict[str, Any]:
    """Build explicit unavailability information for the action metric group."""
    return metric_status("action_consistency", "unavailable", reason=reason)
