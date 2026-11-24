"""Camera-wise aggregation and honest availability of geometric cross-view metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_cameras(
    frame_rows: list[dict[str, Any]], metric_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Aggregate per-interaction camera means without mixing views."""
    camera_ids = sorted(
        {
            int(row["camera_index"])
            for row in frame_rows
            if row.get("camera_index") is not None
            and any(
                row.get(metric) is not None
                and isinstance(row.get(metric), (int, float, np.number))
                and np.isfinite(row[metric])
                for metric in metric_names
            )
        }
    )
    output = []
    for camera in camera_ids:
        rows = [row for row in frame_rows if int(row["camera_index"]) == camera]
        sample_ids = sorted({str(row["sample_id"]) for row in rows})
        per_interaction: dict[str, dict[str, list[float]]] = {
            sample_id: {metric: [] for metric in metric_names} for sample_id in sample_ids
        }
        for row in rows:
            sample_id = str(row["sample_id"])
            for metric in metric_names:
                value = row.get(metric)
                if (
                    value is not None
                    and isinstance(value, (int, float, np.number))
                    and np.isfinite(value)
                ):
                    per_interaction[sample_id][metric].append(float(value))
        record: dict[str, Any] = {
            "camera_index": camera,
            "frame_pair_count": len(
                {(str(row["sample_id"]), row["frame_position"]) for row in rows}
            ),
            "interaction_count": len(sample_ids),
            "cross_view_geometry_status": "unavailable",
            "cross_view_geometry_reason": (
                "No camera calibration or valid cross-view correspondence metadata provided"
            ),
        }
        for metric in metric_names:
            values = [
                float(np.mean(sample_values[metric]))
                for sample_values in per_interaction.values()
                if sample_values[metric]
            ]
            record[metric] = float(np.mean(values)) if values else None
            record[f"{metric}_median"] = float(np.median(values)) if values else None
            record[f"{metric}_std"] = float(np.std(values)) if values else None
            record[f"{metric}_min"] = float(np.min(values)) if values else None
            record[f"{metric}_max"] = float(np.max(values)) if values else None
            record[f"{metric}_valid_count"] = len(values)
            record[f"{metric}_invalid_count"] = len(sample_ids) - len(values)
        output.append(record)
    return output
