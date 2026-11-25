"""Annotation-based robot-state summaries and image-space motion diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .object_metrics import _box, _match_frame
from ..utils import AnnotationData, metric_status

RAW_SAMPLES_PER_VIDEO_FRAME = 3
GRIPPER_TRANSITION_THRESHOLD = 0.1
RECORDED_ACTION_METRICS = (
    "cartesian_translation_command_observation_mae",
    "cartesian_rotation_command_observation_mae",
    "joint_position_command_observation_mae",
    "gripper_command_observation_mae",
    "gripper_transition_agreement",
    "joint_velocity_consistency_correlation",
    "joint_velocity_consistency_mae",
)


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


def _raw_field_reason(annotation: AnnotationData, key: str) -> str | None:
    if key in annotation.raw_fields:
        return None
    return annotation.raw_field_reasons.get(
        key, f"Raw annotation field {key!r} is unavailable"
    )


def _metric_status_fields(
    metric: str, values: list[float], expected_count: int, reason: str | None
) -> dict[str, Any]:
    valid_count = len(values)
    if not valid_count:
        status = "unavailable"
    elif valid_count < expected_count:
        status = "partially_available"
    else:
        status = "available"
    return {
        metric: float(np.mean(values)) if values else None,
        f"{metric}_status": status,
        f"{metric}_reason": reason if status != "available" else None,
        f"{metric}_valid_count": valid_count,
        f"{metric}_invalid_count": max(0, expected_count - valid_count),
    }


def compute_recorded_action_metrics(
    annotation: AnnotationData | None,
    frame_indices: np.ndarray | None,
    sample_id: str,
    trajectory_id: str,
    interaction_index: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Report conservative recorded-command diagnostics at aligned raw rows."""
    metric_fields: dict[str, Any] = {}
    frame_rows: list[dict[str, Any]] = []
    annotation_reason = None
    if annotation is None:
        annotation_reason = "No annotation root was supplied"
    elif annotation.status != "available":
        annotation_reason = annotation.reason or "The annotation could not be aligned"
    elif annotation.raw_length is None:
        annotation_reason = "The annotation has no valid raw_length"
    elif annotation.video_length is None:
        annotation_reason = "The annotation has no valid video_length"
    elif annotation.raw_length != (
        RAW_SAMPLES_PER_VIDEO_FRAME * annotation.video_length - 2
    ):
        annotation_reason = (
            "The verified raw/video alignment requires raw_length == "
            f"{RAW_SAMPLES_PER_VIDEO_FRAME} * video_length - 2; got "
            f"{annotation.raw_length} and {annotation.video_length}"
        )
    elif frame_indices is None or len(frame_indices) == 0:
        annotation_reason = "No source-frame indices are available for raw-row alignment"

    unique_frames: list[tuple[int, int]] = []
    seen_sources: set[int] = set()
    if frame_indices is not None:
        for frame_position, raw_source_index in enumerate(frame_indices):
            source_index = int(raw_source_index)
            if source_index < 0:
                continue
            if (
                annotation is not None
                and annotation.status == "available"
                and annotation.video_length is not None
                and source_index >= annotation.video_length
            ):
                continue
            if source_index not in seen_sources:
                unique_frames.append((frame_position, source_index))
                seen_sources.add(source_index)

    if not unique_frames and annotation_reason is None:
        annotation_reason = "No in-range, unique source-frame indices are available"
    raw_mapping_valid = annotation_reason is None

    frame_by_source: dict[int, dict[str, Any]] = {}
    raw_arrays = annotation.raw_fields if annotation is not None else {}
    command_metrics = (
        "cartesian_translation_command_observation_mae",
        "cartesian_rotation_command_observation_mae",
        "joint_position_command_observation_mae",
        "gripper_command_observation_mae",
    )
    command_pairing_reason = (
        annotation_reason
        or (
            "The annotation rows align with sampled video frames, but the repository does not "
            "document whether action values are current commands or targets for the next "
            "observation; command/observation MAEs are therefore not computed"
        )
    )
    velocity_reason = annotation_reason
    if velocity_reason is None and annotation is not None:
        missing_velocity_reason = _raw_field_reason(
            annotation, "action.joint_velocity"
        )
        velocity_reason = (
            f"{missing_velocity_reason}; raw timestamps/control period and velocity units "
            "are also absent"
            if missing_velocity_reason
            else "Raw timestamps/control period and velocity units are absent; the 3:1 video "
            "sampling ratio does not establish a physical velocity interval"
        )
    velocity_reason = velocity_reason or (
        "Raw timestamps/control period and velocity units are absent; the 3:1 video "
        "sampling ratio does not establish a physical velocity interval"
    )

    for frame_position, source_index in unique_frames:
        raw_index = (
            RAW_SAMPLES_PER_VIDEO_FRAME * source_index
            if raw_mapping_valid
            else None
        )
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "trajectory_id": trajectory_id,
            "interaction_index": interaction_index,
            "camera_index": None,
            "frame_position": frame_position,
            "source_frame_index": source_index,
            "raw_index": raw_index,
            "raw_index_mapping_status": (
                "available" if raw_mapping_valid else "unavailable"
            ),
            "raw_index_mapping_reason": annotation_reason,
        }
        for metric in command_metrics:
            row[metric] = None
            row[f"{metric}_status"] = "unavailable"
            row[f"{metric}_reason"] = command_pairing_reason
        row["joint_velocity_consistency_correlation"] = None
        row["joint_velocity_consistency_correlation_status"] = "unavailable"
        row["joint_velocity_consistency_correlation_reason"] = velocity_reason
        row["joint_velocity_consistency_mae"] = None
        row["joint_velocity_consistency_mae_status"] = "unavailable"
        row["joint_velocity_consistency_mae_reason"] = (
            row["joint_velocity_consistency_correlation_reason"]
        )
        frame_rows.append(row)
        frame_by_source[source_index] = row

    expected_frame_count = len(unique_frames)
    for metric in command_metrics:
        metric_fields.update(
            _metric_status_fields(
                metric,
                [],
                expected_frame_count,
                command_pairing_reason,
            )
        )

    for metric in (
        "joint_velocity_consistency_correlation",
        "joint_velocity_consistency_mae",
    ):
        metric_fields.update(
            _metric_status_fields(metric, [], expected_frame_count, velocity_reason)
        )

    transition_values: list[float] = []
    transition_error = None
    adjacent_source_pairs = [
        (source_index, next_source_index)
        for (_, source_index), (_, next_source_index) in zip(
            unique_frames, unique_frames[1:]
        )
        if next_source_index == source_index + 1
    ]
    eligible_intervals = len(adjacent_source_pairs)
    excluded_no_transition_count = 0
    invalid_interval_count = 0
    if annotation_reason is not None:
        transition_error = annotation_reason
    elif annotation is not None:
        required_fields = ("action.gripper_position", "observation.state.gripper_position")
        transition_error = next(
            (
                _raw_field_reason(annotation, key)
                for key in required_fields
                if _raw_field_reason(annotation, key) is not None
            ),
            None,
        )
        if transition_error is not None:
            invalid_interval_count = eligible_intervals
            for row in frame_rows:
                row["gripper_transition_agreement"] = None
                row["gripper_transition_agreement_status"] = "unavailable"
                row["gripper_transition_agreement_reason"] = transition_error
        else:
            for source_index, next_source_index in adjacent_source_pairs:
                raw_start = RAW_SAMPLES_PER_VIDEO_FRAME * source_index
                raw_end = RAW_SAMPLES_PER_VIDEO_FRAME * next_source_index
                if (
                    annotation.raw_length is None
                    or raw_end >= annotation.raw_length
                    or raw_end >= len(raw_arrays["action.gripper_position"])
                    or raw_end >= len(raw_arrays["observation.state.gripper_position"])
                ):
                    transition_error = (
                        f"Gripper transition raw interval {raw_start}:{raw_end} "
                        "is outside the validated annotation length"
                    )
                    invalid_interval_count += 1
                    row = frame_by_source.get(next_source_index)
                    if row is not None:
                        row["gripper_transition_agreement"] = None
                        row["gripper_transition_agreement_status"] = "unavailable"
                        row["gripper_transition_agreement_reason"] = transition_error
                    continue
                command_delta = float(
                    raw_arrays["action.gripper_position"][raw_end, 0]
                    - raw_arrays["action.gripper_position"][raw_start, 0]
                )
                observed_delta = float(
                    raw_arrays["observation.state.gripper_position"][raw_end, 0]
                    - raw_arrays["observation.state.gripper_position"][raw_start, 0]
                )
                command_transition = (
                    1
                    if command_delta >= GRIPPER_TRANSITION_THRESHOLD
                    else -1
                    if command_delta <= -GRIPPER_TRANSITION_THRESHOLD
                    else 0
                )
                observed_transition = (
                    1
                    if observed_delta >= GRIPPER_TRANSITION_THRESHOLD
                    else -1
                    if observed_delta <= -GRIPPER_TRANSITION_THRESHOLD
                    else 0
                )
                if command_transition == 0 and observed_transition == 0:
                    excluded_no_transition_count += 1
                    continue
                agreement = float(command_transition == observed_transition)
                transition_values.append(agreement)
                row = frame_by_source.get(next_source_index)
                if row is not None:
                    row["gripper_transition_agreement"] = agreement
                    row["gripper_transition_agreement_status"] = "available"
                    row["gripper_transition_agreement_reason"] = None
                    row["gripper_transition_raw_start_index"] = raw_start
                    row["gripper_transition_raw_end_index"] = raw_end

    if transition_values:
        metric_fields.update(
            {
                "gripper_transition_agreement": float(np.mean(transition_values)),
                "gripper_transition_agreement_status": (
                    "partially_available" if invalid_interval_count else "available"
                ),
                "gripper_transition_agreement_reason": (
                    transition_error if invalid_interval_count else None
                ),
                "gripper_transition_agreement_valid_count": len(transition_values),
                "gripper_transition_agreement_invalid_count": invalid_interval_count,
                "gripper_transition_agreement_eligible_interval_count": eligible_intervals,
                "gripper_transition_agreement_excluded_no_transition_count": (
                    excluded_no_transition_count
                ),
            }
        )
        for row in frame_rows:
            if (
                "gripper_transition_agreement" not in row
                and row.get("gripper_transition_agreement_status") != "unavailable"
            ):
                row["gripper_transition_agreement"] = None
                row["gripper_transition_agreement_status"] = "not_applicable"
                row["gripper_transition_agreement_reason"] = (
                    "Neither sequence changed beyond the transition threshold in this interval"
                )
    else:
        if transition_error is None:
            transition_error = (
                "No command or observed gripper transition exceeded the "
                f"{GRIPPER_TRANSITION_THRESHOLD} native-unit threshold"
                if eligible_intervals
                else "No adjacent source-frame intervals were available for transition comparison"
            )
        metric_fields.update(
            {
                "gripper_transition_agreement": None,
                "gripper_transition_agreement_status": "unavailable",
                "gripper_transition_agreement_reason": transition_error,
                "gripper_transition_agreement_valid_count": 0,
                "gripper_transition_agreement_invalid_count": invalid_interval_count,
                "gripper_transition_agreement_eligible_interval_count": eligible_intervals,
                "gripper_transition_agreement_excluded_no_transition_count": (
                    excluded_no_transition_count
                ),
            }
        )
        for row in frame_rows:
            row["gripper_transition_agreement"] = None
            row["gripper_transition_agreement_status"] = "unavailable"
            row["gripper_transition_agreement_reason"] = transition_error

    metric_fields["recorded_action_alignment_status"] = (
        "available" if annotation_reason is None else "unavailable"
    )
    metric_fields["recorded_action_alignment_reason"] = annotation_reason
    metric_fields["raw_annotation_field_status"] = {
        key: (
            "available"
            if annotation is not None and key in annotation.raw_fields
            else "unavailable"
        )
        for key in (
            "observation.state.cartesian_position",
            "action.cartesian_position",
            "observation.state.joint_position",
            "action.joint_position",
            "observation.state.gripper_position",
            "action.gripper_position",
            "action.joint_velocity",
        )
    }
    metric_fields["raw_annotation_field_reasons"] = {
        key: (
            None
            if annotation is not None and key in annotation.raw_fields
            else annotation.reason
            if annotation is not None and annotation.status != "available"
            else annotation.raw_field_reasons.get(key, "Raw annotation fields were not loaded")
            if annotation is not None
            else annotation_reason
        )
        for key in (
            "observation.state.cartesian_position",
            "action.cartesian_position",
            "observation.state.joint_position",
            "action.joint_position",
            "observation.state.gripper_position",
            "action.gripper_position",
            "action.joint_velocity",
        )
    }
    metric_fields["command_observation_temporal_alignment_status"] = (
        "unavailable"
    )
    metric_fields["command_observation_temporal_alignment_reason"] = (
        command_pairing_reason
    )
    metric_fields["recorded_action_raw_frame_mapping"] = (
        f"raw_index = {RAW_SAMPLES_PER_VIDEO_FRAME} * source_frame_index"
    )
    metric_fields["recorded_action_deduplicated_source_frame_count"] = expected_frame_count
    metric_fields["cartesian_rotation_error_interpretation"] = (
        "Not computed: temporal pairing, rotation parameterization, and angle convention are undocumented"
    )
    metric_fields["joint_position_error_interpretation"] = (
        "Not computed: command/observation temporal pairing is undocumented"
    )
    metric_fields["command_observation_interpretation"] = (
        "Not computed: the repository does not document the temporal meaning of action "
        "rows relative to observation rows"
    )
    return metric_fields, frame_rows


def unavailable_action_metrics(reason: str) -> dict[str, Any]:
    """Build explicit unavailability information for the action metric group."""
    return metric_status("action_consistency", "unavailable", reason=reason)
