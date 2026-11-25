"""Camera-wise summaries and cross-view agreement/feature-consistency proxies."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from ..utils import EvaluationSample

CVOA_METRICS = (
    "cross_view_object_agreement_reference",
    "cross_view_object_agreement_generated",
    "cross_view_object_agreement_difference",
)
CVFC_METRICS = (
    "cross_view_feature_similarity_reference",
    "cross_view_feature_similarity_generated",
    "cross_view_feature_similarity_difference",
    "cross_view_feature_consistency_error",
)


def load_cvfc_model(device: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load fixed ImageNet ResNet-50 weights and their official preprocessing."""
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except (ImportError, OSError, RuntimeError) as exc:
        return None, f"CVFC requires PyTorch and torchvision: {exc}"
    try:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        return {
            "model": model,
            "preprocess": weights.transforms(),
            "device": device,
            "weights": "ResNet50_Weights.IMAGENET1K_V2",
        }, None
    except (AssertionError, OSError, EOFError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"Could not load CVFC ResNet-50 ImageNet weights: {exc}"


def _unique_frame_positions(
    sample: EvaluationSample,
) -> tuple[list[tuple[int, int]], str | None]:
    camera_count, frame_count = sample.ground_truth.shape[:2]
    if sample.frame_indices is None or len(sample.frame_indices) != frame_count:
        return [], "CV metrics require one source-frame index for every stored frame"
    positions = []
    seen = set()
    for position, source_index in enumerate(sample.frame_indices):
        try:
            numeric_index = float(source_index)
        except (TypeError, ValueError, OverflowError):
            return [], "Source-frame indices must be finite non-negative integers"
        if not np.isfinite(numeric_index) or numeric_index < 0 or not numeric_index.is_integer():
            return [], "Source-frame indices must be finite non-negative integers"
        source = int(numeric_index)
        if source not in seen:
            positions.append((position, source))
            seen.add(source)
    if not positions:
        return [], "No unique synchronized source-frame indices are available"
    if camera_count < 2:
        return positions, "At least two synchronized camera views are required"
    return positions, None


def _sample_frame_metadata(
    sample: EvaluationSample, sample_id: str, position: int, source_index: int
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "trajectory_id": sample.trajectory_id,
        "interaction_index": sample.interaction_index,
        "camera_index": None,
        "frame_position": position,
        "source_frame_index": source_index,
    }


def _detection_label(detection: Any) -> str:
    if isinstance(detection, dict):
        value = detection.get("label", "object")
    else:
        value = getattr(detection, "label", "object")
    return str(value or "object").strip().casefold() or "object"


def _category_agreement(left: list[Any], right: list[Any]) -> float | None:
    left_categories = {_detection_label(item) for item in left}
    right_categories = {_detection_label(item) for item in right}
    if not left_categories and not right_categories:
        return None
    union = left_categories | right_categories
    return len(left_categories & right_categories) / len(union) if union else None


def _metric_summary(
    rows: list[dict[str, Any]],
    metric: str,
    reason: str | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    values = [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None
        and isinstance(row.get(metric), (int, float, np.number))
        and np.isfinite(row[metric])
    ]
    expected = len(rows) if expected_count is None else expected_count
    valid = len(values)
    status = (
        "unavailable"
        if not valid
        else "partially_available"
        if valid < expected
        else "available"
    )
    if status != "available" and not reason:
        reason = "No valid comparison for one or more synchronized source frames"
    return {
        metric: float(np.mean(values)) if values else None,
        f"{metric}_status": status,
        f"{metric}_reason": reason if status != "available" else None,
        f"{metric}_valid_count": valid,
        f"{metric}_invalid_count": expected - valid,
    }


def compute_cvoa(
    sample: EvaluationSample,
    sample_id: str,
    reference_detections: list[list[list[Any]]] | None,
    generated_detections: list[list[list[Any]]] | None,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute pairwise category-set Jaccard agreement for synchronized views."""
    positions, alignment_reason = _unique_frame_positions(sample)
    camera_count, frame_count = sample.ground_truth.shape[:2]
    reason = unavailable_reason or alignment_reason
    frame_rows = []
    pair_rows = []
    per_metric_rows: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in CVOA_METRICS
    }
    if reason is None and (
        reference_detections is None
        or generated_detections is None
        or len(reference_detections) != camera_count
        or len(generated_detections) != camera_count
        or any(len(camera) != frame_count for camera in reference_detections)
        or any(len(camera) != frame_count for camera in generated_detections)
    ):
        reason = "Detector results are unavailable or do not align with sample frames"

    pairs = (
        list(combinations(range(camera_count), 2))
        if camera_count >= 2
        else []
    )
    pair_values: dict[tuple[int, int], dict[str, list[float]]] = {
        pair: {metric: [] for metric in CVOA_METRICS} for pair in pairs
    }
    for position, source_index in positions:
        frame_row = _sample_frame_metadata(sample, sample_id, position, source_index)
        for metric in CVOA_METRICS:
            frame_row[metric] = None
            frame_row[f"{metric}_status"] = "unavailable"
            frame_row[f"{metric}_reason"] = reason
        reference_scores = []
        generated_scores = []
        pair_differences = []
        if reason is None:
            for camera_a, camera_b in pairs:
                reference_score = _category_agreement(
                    reference_detections[camera_a][position],
                    reference_detections[camera_b][position],
                )
                generated_score = _category_agreement(
                    generated_detections[camera_a][position],
                    generated_detections[camera_b][position],
                )
                difference = (
                    generated_score - reference_score
                    if reference_score is not None and generated_score is not None
                    else None
                )
                pair_row = {
                    **frame_row,
                    "camera_pair": f"{camera_a}-{camera_b}",
                    "camera_a": camera_a,
                    "camera_b": camera_b,
                    CVOA_METRICS[0]: reference_score,
                    CVOA_METRICS[1]: generated_score,
                    CVOA_METRICS[2]: difference,
                    "cross_view_object_agreement_reference_status": (
                        "available" if reference_score is not None else "unavailable"
                    ),
                    "cross_view_object_agreement_generated_status": (
                        "available" if generated_score is not None else "unavailable"
                    ),
                    "cross_view_object_agreement_difference_status": (
                        "available" if difference is not None else "unavailable"
                    ),
                    "cross_view_object_agreement_reason": (
                        None
                        if reference_score is not None and generated_score is not None
                        else "Both-empty category sets are excluded from agreement"
                    ),
                }
                for metric, score in zip(
                    CVOA_METRICS,
                    (reference_score, generated_score, difference),
                ):
                    pair_row[f"{metric}_reason"] = (
                        None
                        if score is not None
                        else "Both-empty category comparisons are excluded from agreement"
                    )
                pair_rows.append(pair_row)
                if reference_score is not None:
                    pair_values[(camera_a, camera_b)][CVOA_METRICS[0]].append(
                        reference_score
                    )
                    per_metric_rows[CVOA_METRICS[0]].append(
                        {CVOA_METRICS[0]: reference_score}
                    )
                    reference_scores.append(reference_score)
                if generated_score is not None:
                    pair_values[(camera_a, camera_b)][CVOA_METRICS[1]].append(
                        generated_score
                    )
                    per_metric_rows[CVOA_METRICS[1]].append(
                        {CVOA_METRICS[1]: generated_score}
                    )
                    generated_scores.append(generated_score)
                if difference is not None:
                    pair_values[(camera_a, camera_b)][CVOA_METRICS[2]].append(
                        difference
                    )
                    per_metric_rows[CVOA_METRICS[2]].append(
                        {CVOA_METRICS[2]: difference}
                    )
                    pair_differences.append(difference)
        else:
            for camera_a, camera_b in pairs:
                pair_rows.append(
                    {
                        **frame_row,
                        "camera_pair": f"{camera_a}-{camera_b}",
                        "camera_a": camera_a,
                        "camera_b": camera_b,
                        **{metric: None for metric in CVOA_METRICS},
                        **{
                            f"{metric}_status": "unavailable"
                            for metric in CVOA_METRICS
                        },
                        **{
                            f"{metric}_reason": reason
                            for metric in CVOA_METRICS
                        },
                        "cross_view_object_agreement_reason": reason,
                    }
                )
        frame_values = {
            CVOA_METRICS[0]: reference_scores,
            CVOA_METRICS[1]: generated_scores,
            CVOA_METRICS[2]: pair_differences,
        }
        for metric, values in frame_values.items():
            value = float(np.mean(values)) if values else None
            frame_row[metric] = value
            frame_row[f"{metric}_status"] = (
                "unavailable"
                if value is None
                else "partially_available"
                if len(values) < len(pairs)
                else "available"
            )
            frame_row[f"{metric}_reason"] = (
                reason or "No valid pairwise category-set comparison for this source frame"
                if value is None
                else "One or more camera pairs had no score"
                if len(values) < len(pairs)
                else None
            )
        frame_row["cross_view_object_reference_valid_pair_count"] = len(
            reference_scores
        )
        frame_row["cross_view_object_generated_valid_pair_count"] = len(
            generated_scores
        )
        frame_row["cross_view_object_difference_valid_pair_count"] = len(
            pair_differences
        )
        frame_rows.append(frame_row)

    values: dict[str, Any] = {
        "cross_view_object_camera_pair_count": len(pairs),
        "cross_view_object_source_frame_count": len(positions),
        "cross_view_object_visibility_limitations": (
            "Set agreement ignores object identity, count multiplicity, occlusion, and "
            "cross-camera visibility; empty-empty sets are not scored"
        ),
    }
    values["cross_view_object_reference_valid_pair_count"] = sum(
        len(metrics[CVOA_METRICS[0]]) for metrics in pair_values.values()
    )
    values["cross_view_object_generated_valid_pair_count"] = sum(
        len(metrics[CVOA_METRICS[1]]) for metrics in pair_values.values()
    )
    values["cross_view_object_difference_valid_pair_count"] = sum(
        len(metrics[CVOA_METRICS[2]]) for metrics in pair_values.values()
    )
    for metric in CVOA_METRICS:
        values.update(
            _metric_summary(
                per_metric_rows[metric],
                metric,
                reason,
                expected_count=len(frame_rows) * len(pairs),
            )
        )
    for pair, metrics in pair_values.items():
        for metric, scores in metrics.items():
            if scores:
                continue
            values[f"{metric}_camera_pair_{pair[0]}_{pair[1]}_status"] = "unavailable"
            values[f"{metric}_camera_pair_{pair[0]}_{pair[1]}_reason"] = (
                reason or "Both-empty category comparisons are excluded from agreement"
            )
    return values, frame_rows, pair_rows


def _encode_cvfc(
    model_info: dict[str, Any],
    images: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import torch
    from PIL import Image

    model = model_info["model"]
    preprocess = model_info["preprocess"]
    device = model_info["device"]
    transformed = [
        preprocess(Image.fromarray(image.astype(np.uint8, copy=False)))
        for image in images
    ]
    features = []
    with torch.inference_mode():
        for start in range(0, len(transformed), batch_size):
            batch = torch.stack(transformed[start : start + batch_size]).to(device)
            output = model(batch)
            features.append(output.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_cvfc(
    sample: EvaluationSample,
    sample_id: str,
    model_info: dict[str, Any] | None,
    batch_size: int,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare mean pairwise cosine similarity of fixed ResNet-50 features."""
    positions, alignment_reason = _unique_frame_positions(sample)
    camera_count = sample.ground_truth.shape[0]
    reason = unavailable_reason or alignment_reason
    frame_rows = []
    pair_rows = []
    per_metric_rows: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in CVFC_METRICS
    }
    if reason is None and model_info is None:
        reason = "The CVFC feature extractor is unavailable"
    pairs = (
        list(combinations(range(camera_count), 2))
        if camera_count >= 2
        else []
    )
    pair_values: dict[tuple[int, int], dict[str, list[float]]] = {
        pair: {metric: [] for metric in CVFC_METRICS} for pair in pairs
    }
    selected_positions = [position for position, _ in positions]
    reference_features = generated_features = None
    if reason is None and model_info is not None:
        try:
            real_images = sample.ground_truth[:, selected_positions].reshape(
                -1, *sample.ground_truth.shape[2:]
            )
            generated_images = sample.prediction[:, selected_positions].reshape(
                -1, *sample.prediction.shape[2:]
            )
            reference_features = _encode_cvfc(model_info, real_images, batch_size)
            generated_features = _encode_cvfc(model_info, generated_images, batch_size)
            if (
                reference_features.ndim != 2
                or generated_features.shape != reference_features.shape
                or not np.isfinite(reference_features).all()
                or not np.isfinite(generated_features).all()
            ):
                raise ValueError("CVFC extractor returned invalid feature matrices")
            reference_features = reference_features.reshape(
                camera_count, len(positions), -1
            )
            generated_features = generated_features.reshape(
                camera_count, len(positions), -1
            )
        except (
            AssertionError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            reason = f"Could not extract CVFC features: {exc}"

    for selected_index, (position, source_index) in enumerate(positions):
        frame_row = _sample_frame_metadata(sample, sample_id, position, source_index)
        for metric in CVFC_METRICS:
            frame_row[metric] = None
            frame_row[f"{metric}_status"] = "unavailable"
            frame_row[f"{metric}_reason"] = reason
        reference_scores = []
        generated_scores = []
        differences = []
        errors = []
        if reason is None and reference_features is not None and generated_features is not None:
            for camera_a, camera_b in pairs:
                ref_a = reference_features[camera_a, selected_index]
                ref_b = reference_features[camera_b, selected_index]
                gen_a = generated_features[camera_a, selected_index]
                gen_b = generated_features[camera_b, selected_index]
                ref_norm = float(np.linalg.norm(ref_a) * np.linalg.norm(ref_b))
                gen_norm = float(np.linalg.norm(gen_a) * np.linalg.norm(gen_b))
                if ref_norm <= 1e-12 or gen_norm <= 1e-12:
                    pair_reason = "A cross-view feature vector has zero norm"
                    ref_similarity = generated_similarity = difference = error = None
                else:
                    ref_similarity = float(np.clip(np.dot(ref_a, ref_b) / ref_norm, -1, 1))
                    generated_similarity = float(
                        np.clip(np.dot(gen_a, gen_b) / gen_norm, -1, 1)
                    )
                    difference = generated_similarity - ref_similarity
                    error = abs(difference)
                    pair_reason = None
                    reference_scores.append(ref_similarity)
                    generated_scores.append(generated_similarity)
                    differences.append(difference)
                    errors.append(error)
                pair_row = {
                    **frame_row,
                    "camera_pair": f"{camera_a}-{camera_b}",
                    "camera_a": camera_a,
                    "camera_b": camera_b,
                    CVFC_METRICS[0]: ref_similarity,
                    CVFC_METRICS[1]: generated_similarity,
                    CVFC_METRICS[2]: difference,
                    CVFC_METRICS[3]: error,
                    "cross_view_feature_consistency_status": (
                        "available" if error is not None else "unavailable"
                    ),
                    "cross_view_feature_consistency_reason": pair_reason,
                }
                for metric, value in zip(
                    CVFC_METRICS,
                    (ref_similarity, generated_similarity, difference, error),
                ):
                    pair_row[f"{metric}_status"] = (
                        "available" if value is not None else "unavailable"
                    )
                    pair_row[f"{metric}_reason"] = pair_reason
                pair_rows.append(pair_row)
                for metric, value in zip(
                    CVFC_METRICS,
                    (ref_similarity, generated_similarity, difference, error),
                ):
                    if value is not None:
                        pair_values[(camera_a, camera_b)][metric].append(value)
                        per_metric_rows[metric].append({metric: value})
        else:
            for camera_a, camera_b in pairs:
                pair_rows.append(
                    {
                        **frame_row,
                        "camera_pair": f"{camera_a}-{camera_b}",
                        "camera_a": camera_a,
                        "camera_b": camera_b,
                        **{metric: None for metric in CVFC_METRICS},
                        **{
                            f"{metric}_status": "unavailable"
                            for metric in CVFC_METRICS
                        },
                        **{
                            f"{metric}_reason": (
                                reason or "CVFC features are unavailable"
                            )
                            for metric in CVFC_METRICS
                        },
                        "cross_view_feature_consistency_status": "unavailable",
                        "cross_view_feature_consistency_reason": (
                            reason or "CVFC features are unavailable"
                        ),
                    }
                )
        frame_values = {
            CVFC_METRICS[0]: reference_scores,
            CVFC_METRICS[1]: generated_scores,
            CVFC_METRICS[2]: differences,
            CVFC_METRICS[3]: errors,
        }
        for metric, scores in frame_values.items():
            value = float(np.mean(scores)) if scores else None
            frame_row[metric] = value
            frame_row[f"{metric}_status"] = (
                "unavailable"
                if value is None
                else "partially_available"
                if len(scores) < len(pairs)
                else "available"
            )
            frame_row[f"{metric}_reason"] = (
                reason or "No valid camera-pair features"
                if value is None
                else "One or more camera pairs had no valid feature comparison"
                if len(scores) < len(pairs)
                else None
            )
        frame_row["cross_view_feature_reference_valid_pair_count"] = len(
            reference_scores
        )
        frame_row["cross_view_feature_generated_valid_pair_count"] = len(
            generated_scores
        )
        frame_row["cross_view_feature_difference_valid_pair_count"] = len(
            differences
        )
        frame_row["cross_view_feature_error_valid_pair_count"] = len(errors)
        frame_rows.append(frame_row)

    values: dict[str, Any] = {
        "cross_view_feature_camera_pair_count": len(pairs),
        "cross_view_feature_source_frame_count": len(positions),
        "cross_view_feature_extractor": (
            model_info.get("weights") if model_info is not None else None
        ),
        "cross_view_feature_preprocessing": (
            "Torchvision ResNet-50 IMAGENET1K_V2 weights.transforms(); 2048-D pooled "
            "features; cosine similarity; same weights and preprocessing for reference/generated"
        ),
        "cross_view_feature_limitations": (
            "ImageNet semantic feature similarity is not geometric correspondence, object "
            "identity, or calibrated 3D consistency"
        ),
    }
    for metric, count_metric in zip(
        CVFC_METRICS,
        (
            "cross_view_feature_reference_valid_pair_count",
            "cross_view_feature_generated_valid_pair_count",
            "cross_view_feature_difference_valid_pair_count",
            "cross_view_feature_error_valid_pair_count",
        ),
    ):
        values[count_metric] = sum(
            len(pair_metric_values[metric])
            for pair_metric_values in pair_values.values()
        )
    for metric in CVFC_METRICS:
        values.update(
            _metric_summary(
                per_metric_rows[metric],
                metric,
                reason,
                expected_count=len(frame_rows) * len(pairs),
            )
        )
    return values, frame_rows, pair_rows


def summarize_camera_pairs(
    frame_rows: list[dict[str, Any]], metric_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Aggregate pairwise scores once per trajectory/source-frame, not per chunk."""
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    ordered = sorted(
        frame_rows,
        key=lambda row: (
            str(row.get("camera_pair")),
            str(row.get("trajectory_id")),
            int(row.get("source_frame_index", -1)),
            int(row.get("interaction_index") or 0),
            str(row.get("sample_id")),
        ),
    )
    for row in ordered:
        if row.get("camera_pair") is None or row.get("source_frame_index") is None:
            continue
        key = (
            str(row["camera_pair"]),
            str(row.get("trajectory_id")),
            int(row["source_frame_index"]),
        )
        target = selected.setdefault(
            key,
            {
                "camera_index": None,
                "camera_pair": key[0],
                "camera_a": row.get("camera_a"),
                "camera_b": row.get("camera_b"),
                "source_frame_count": 0,
            },
        )
        target["source_frame_count"] = 1
        target.setdefault("trajectory_id", key[1])
        target.setdefault("interaction_index", row.get("interaction_index"))
        target.setdefault("sample_id", row.get("sample_id"))
        target.setdefault("unique_source_frame_index", key[2])
        for metric in metric_names:
            if metric not in target and metric in row:
                target[metric] = row.get(metric)
                target[f"{metric}_status"] = row.get(f"{metric}_status")
                target[f"{metric}_reason"] = row.get(
                    f"{metric}_reason",
                    row.get("cross_view_object_agreement_reason")
                    or row.get("cross_view_feature_consistency_reason"),
                )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected.values():
        grouped.setdefault(str(row["camera_pair"]), []).append(row)
    output = []
    for pair, rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "camera_index": None,
            "camera_pair": pair,
            "camera_a": rows[0].get("camera_a"),
            "camera_b": rows[0].get("camera_b"),
            "source_frame_count": len(rows),
            "trajectory_count": len({row.get("trajectory_id") for row in rows}),
        }
        for metric in metric_names:
            scores = [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None
                and isinstance(row.get(metric), (int, float, np.number))
                and np.isfinite(row[metric])
            ]
            record[metric] = float(np.mean(scores)) if scores else None
            record[f"{metric}_median"] = float(np.median(scores)) if scores else None
            record[f"{metric}_std"] = float(np.std(scores)) if scores else None
            record[f"{metric}_min"] = float(np.min(scores)) if scores else None
            record[f"{metric}_max"] = float(np.max(scores)) if scores else None
            record[f"{metric}_valid_count"] = len(scores)
            record[f"{metric}_invalid_count"] = len(rows) - len(scores)
            record[f"{metric}_status"] = (
                "available"
                if scores and len(scores) == len(rows)
                else "partially_available"
                if scores
                else "unavailable"
            )
            if not scores:
                record[f"{metric}_reason"] = next(
                    (
                        row.get(f"{metric}_reason")
                        for row in rows
                        if row.get(f"{metric}_reason")
                    ),
                    "No valid cross-view score for this camera pair",
                )
        output.append(record)
    return output


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
