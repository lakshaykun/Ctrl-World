"""Non-zero-filling aggregation at interaction, trajectory, camera, and horizon levels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

IDENTIFIER_FIELDS = {
    "sample_id",
    "trajectory_id",
    "interaction_index",
    "camera_index",
    "frame_position",
    "horizon",
    "sample",
    "file",
    "instruction",
    "fps",
    "start_frame_index",
}


def _metric_fields(rows: list[dict[str, Any]]) -> list[str]:
    candidates = {key for row in rows for key in row}
    fields = []
    for key in sorted(candidates - IDENTIFIER_FIELDS):
        if (
            key.endswith(("_count", "_index", "_dimension", "_status", "_reason", "_path"))
            or "interpretation" in key
            or key.startswith("annotation_")
        ):
            continue
        if any(
            isinstance(row.get(key), (int, float, np.number))
            and row.get(key) is not None
            and np.isfinite(row[key])
            for row in rows
        ):
            fields.append(key)
    return fields


def aggregate_rows(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Aggregate numeric columns, excluding missing/non-finite values.

    Each input row is one equally weighted observation at the requested level.
    Higher-level reports aggregate interaction means, so long interactions do
    not silently dominate trajectory or dataset results.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_keys)
        grouped[key].append(row)
    metrics = _metric_fields(rows)
    output = []
    for group, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = {field: value for field, value in zip(group_keys, group)}
        result["interaction_count" if not group_keys else "sample_count"] = len(values)
        for metric in metrics:
            numbers = np.asarray(
                [
                    float(row[metric])
                    for row in values
                    if row.get(metric) is not None
                    and isinstance(row.get(metric), (int, float, np.number))
                    and np.isfinite(row[metric])
                ],
                dtype=np.float64,
            )
            result[metric] = float(numbers.mean()) if len(numbers) else None
            result[f"{metric}_median"] = float(np.median(numbers)) if len(numbers) else None
            result[f"{metric}_std"] = float(numbers.std()) if len(numbers) else None
            result[f"{metric}_min"] = float(numbers.min()) if len(numbers) else None
            result[f"{metric}_max"] = float(numbers.max()) if len(numbers) else None
            result[f"{metric}_valid_count"] = int(len(numbers))
            result[f"{metric}_invalid_count"] = len(values) - int(len(numbers))
        output.append(result)
    return output


def dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize interaction-level metrics with interaction-equal weighting."""
    result: dict[str, Any] = {}
    for metric in _metric_fields(rows):
        values = np.asarray(
            [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None and np.isfinite(row[metric])
            ],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "standard_deviation": float(values.std()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "valid_count": int(len(values)),
            "invalid_count": len(rows) - int(len(values)),
        }
    return result


def aggregate_unique_source_frame_metrics(
    rows: list[dict[str, Any]], metric_names: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Aggregate frame metrics once per trajectory/source-frame across overlaps."""
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("trajectory_id"))
            if row.get("trajectory_id") is not None
            else f"~{row.get('sample_id')}",
            int(row.get("source_frame_index", -1)),
            int(row.get("interaction_index") or 0),
            str(row.get("sample_id")),
        ),
    )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in ordered:
        if row.get("source_frame_index") is None:
            continue
        trajectory_key = (
            str(row.get("trajectory_id"))
            if row.get("trajectory_id") is not None
            else f"sample:{row.get('sample_id')}"
        )
        key = (trajectory_key, int(row["source_frame_index"]))
        selected = unique.setdefault(
            key,
            {
                "trajectory_id": row.get("trajectory_id"),
                "sample_id": row.get("sample_id"),
                "source_frame_index": int(row["source_frame_index"]),
            },
        )
        for metric in metric_names:
            if metric in row and metric not in selected:
                if (
                    metric == "gripper_transition_agreement"
                    and row.get(f"{metric}_status") == "not_applicable"
                ):
                    continue
                selected[metric] = row.get(metric)
                selected[f"{metric}_status"] = row.get(f"{metric}_status")
                selected[f"{metric}_reason"] = row.get(f"{metric}_reason")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (trajectory_key, _), row in unique.items():
        grouped[trajectory_key].append(row)

    trajectory_rows = []
    for trajectory_key, source_rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "trajectory_id": source_rows[0].get("trajectory_id"),
            "source_frame_count": len(source_rows),
        }
        for metric in metric_names:
            metric_rows = [row for row in source_rows if metric in row]
            values = np.asarray(
                [
                    float(row[metric])
                    for row in metric_rows
                    if row.get(metric) is not None
                    and isinstance(row.get(metric), (int, float, np.number))
                    and np.isfinite(row[metric])
                ],
                dtype=np.float64,
            )
            valid_count = int(len(values))
            invalid_count = len(metric_rows) - valid_count
            partial_count = sum(
                row.get(f"{metric}_status") == "partially_available"
                for row in metric_rows
            )
            record[metric] = float(values.mean()) if valid_count else None
            record[f"{metric}_median"] = float(np.median(values)) if valid_count else None
            record[f"{metric}_std"] = float(values.std()) if valid_count else None
            record[f"{metric}_min"] = float(values.min()) if valid_count else None
            record[f"{metric}_max"] = float(values.max()) if valid_count else None
            record[f"{metric}_valid_count"] = valid_count
            record[f"{metric}_invalid_count"] = invalid_count
            record[f"{metric}_status"] = (
                "available"
                if valid_count and not invalid_count and not partial_count
                else "partially_available"
                if valid_count
                else "unavailable"
            )
            if invalid_count or partial_count or not metric_rows:
                record[f"{metric}_reason"] = next(
                    (
                        row.get(f"{metric}_reason")
                        for row in metric_rows
                        if row.get(f"{metric}_reason")
                    ),
                    "No aligned source-frame rows were available for this metric"
                    if not metric_rows
                    else "No valid value for one or more unique source frames",
                )
        trajectory_rows.append(record)

    dataset_metrics: dict[str, dict[str, Any]] = {}
    all_source_rows = list(unique.values())
    for metric in metric_names:
        metric_rows = [row for row in all_source_rows if metric in row]
        values = np.asarray(
            [
                float(row[metric])
                for row in metric_rows
                if row.get(metric) is not None
                and isinstance(row.get(metric), (int, float, np.number))
                and np.isfinite(row[metric])
            ],
            dtype=np.float64,
        )
        valid_count = int(len(values))
        invalid_count = len(metric_rows) - valid_count
        partial_count = sum(
            row.get(f"{metric}_status") == "partially_available"
            for row in metric_rows
        )
        result = {
            "mean": float(values.mean()) if valid_count else None,
            "median": float(np.median(values)) if valid_count else None,
            "standard_deviation": float(values.std()) if valid_count else None,
            "minimum": float(values.min()) if valid_count else None,
            "maximum": float(values.max()) if valid_count else None,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "status": (
                "available"
                if valid_count and not invalid_count and not partial_count
                else "partially_available"
                if valid_count
                else "unavailable"
            ),
        }
        if invalid_count or partial_count or not metric_rows:
            result["reason"] = next(
                (
                    row.get(f"{metric}_reason")
                    for row in metric_rows
                    if row.get(f"{metric}_reason")
                ),
                "No aligned source-frame rows were available for this metric"
                if not metric_rows
                else "No valid value for one or more unique source frames",
            )
        dataset_metrics[metric] = result
    return trajectory_rows, dataset_metrics


def aggregate_object_tracks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each generated track with one object-level observation."""
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("sample_id"), row.get("camera_index"), row.get("track_id"))
        grouped[key].append(row)
    result = []
    for (sample_id, camera_index, track_id), observations in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        observations.sort(key=lambda row: int(row["frame_position"]))
        centers = [
            (
                (float(row["x1"]) + float(row["x2"])) / 2,
                (float(row["y1"]) + float(row["y2"])) / 2,
            )
            for row in observations
        ]
        path_length = sum(
            float(np.hypot(right[0] - left[0], right[1] - left[1]))
            for index, (left, right) in enumerate(zip(centers, centers[1:]))
            if int(observations[index + 1]["frame_position"])
            == int(observations[index]["frame_position"]) + 1
        )
        result.append(
            {
                "sample_id": sample_id,
                "trajectory_id": observations[0].get("trajectory_id"),
                "interaction_index": observations[0].get("interaction_index"),
                "camera_index": camera_index,
                "track_id": track_id,
                "label": observations[0].get("label"),
                "observation_count": len(observations),
                "mean_confidence": float(
                    np.mean([float(row["confidence"]) for row in observations])
                ),
                "mean_center_x": float(np.mean([center[0] for center in centers])),
                "mean_center_y": float(np.mean([center[1] for center in centers])),
                "consecutive_path_length_pixels": path_length,
                "first_frame_position": int(observations[0]["frame_position"]),
                "last_frame_position": int(observations[-1]["frame_position"]),
            }
        )
    return result
