"""Run offline evaluation of saved Ctrl-World predictions; never initializes the model."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import (
    aggregate_object_tracks,
    aggregate_rows,
    aggregate_unique_source_frame_metrics,
    dataset_summary,
)
from .annotation_loader import load_annotation
from .data_loader import discover_npz_files, iter_samples
from .metrics.action_consistency import (
    RECORDED_ACTION_METRICS,
    compute_action_diagnostics,
    compute_recorded_action_metrics,
)
from .metrics.multiview_metrics import (
    CVOA_METRICS,
    CVFC_METRICS,
    compute_cvoa,
    compute_cvfc,
    load_cvfc_model,
    summarize_camera_pairs,
    summarize_cameras,
)
from .metrics.object_metrics import (
    Detector,
    IoUTracker,
    compute_object_metrics,
    detect_video,
    track_detections,
)
from .metrics.perceptual_metrics import (
    compute_lpips,
    frechet_distance_from_features,
    load_local_lpips,
    unavailable_fvd,
)
from .metrics.physical_metrics import detect_overlap_candidates, summarize_physical_candidates
from .metrics.pixel_metrics import compute_pixel_metrics
from .metrics.temporal_metrics import compute_temporal_metrics, horizon_rows, mean_flow_magnitude
from .metrics.video_distribution_metrics import (
    compute_video_fvd,
    load_fid_metric,
    resolve_fid_weights,
    update_fid_metric,
)
from .reporting import create_run_directory, write_reports
from .utils import EvaluationSample, json_safe
from .validate_data import validate_sample, validation_failure

ALL_METRICS = {
    "visual",
    "lpips",
    "temporal",
    "action",
    "objects",
    "physical",
    "multiview",
    "fid",
    "fvd",
}
VALID_METRICS = ALL_METRICS | {"lpips"}
FRAME_METRICS = (
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "temporal_difference_error",
    "optical_flow_difference",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse documented command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Root containing evaluation NPZ files")
    parser.add_argument("--output_dir", default="eval/results")
    parser.add_argument("--annotation_root", default=None, help="Dataset root containing annotation/val")
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated groups: all, visual, temporal, lpips, action, objects, physical, multiview, fid, fvd",
    )
    parser.add_argument("--device", default="cpu", help="Device for optional neural metrics (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for LPIPS")
    parser.add_argument(
        "--cvfc_batch_size", type=int, default=32, help="Independent batch size for CVFC"
    )
    parser.add_argument("--fid_batch_size", type=int, default=64, help="Independent batch size for FID")
    parser.add_argument("--frame_stride", type=int, default=1, help="Evaluate every Nth stored frame")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_comparison_videos", action="store_true")
    parser.add_argument(
        "--detector",
        default="none",
        help="none or a local Python factory as module:callable returning an object with detect(frame)",
    )
    parser.add_argument("--tracker", choices=("iou", "none"), default="iou")
    parser.add_argument("--lpips_weights", default=None, help="Path to local calibrated LPIPS weights")
    parser.add_argument("--lpips_network", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument(
        "--fid_weights",
        default=None,
        help="Path to local Torch-Fidelity Inception-v3 weights; cached weights are used if present",
    )
    parser.add_argument(
        "--fvd_features",
        default=None,
        help="Optional NPZ with compatible real_features and generated_features clip matrices",
    )
    parser.add_argument(
        "--fvd_weights",
        default=None,
        help="Path to local cd-fvd I3D-400 weights; automatic downloads are disabled",
    )
    parser.add_argument("--fvd_clip_len", type=int, default=16)
    parser.add_argument("--fvd_stride", type=int, default=16)
    parser.add_argument("--max_fvd_clips_per_trajectory", type=int, default=64)
    parser.add_argument(
        "--fvd_extractor_name",
        default=None,
        help="Provenance label for the local feature extractor used to create --fvd_features",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.cvfc_batch_size < 1:
        parser.error("--cvfc_batch_size must be positive")
    if args.fid_batch_size < 1:
        parser.error("--fid_batch_size must be positive")
    if args.fvd_clip_len < 1 or args.fvd_stride < 1:
        parser.error("--fvd_clip_len and --fvd_stride must be positive")
    if args.max_fvd_clips_per_trajectory < 1:
        parser.error("--max_fvd_clips_per_trajectory must be positive")
    if args.frame_stride < 1:
        parser.error("--frame_stride must be positive")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or a CUDA device such as cuda:0")
    requested = {part.strip().lower() for part in args.metrics.split(",") if part.strip()}
    if requested == {"all"}:
        requested = set(ALL_METRICS)
    elif "all" in requested or not requested:
        parser.error("'all' must be used alone and at least one metric group is required")
    unknown = requested - VALID_METRICS
    if unknown:
        parser.error(f"Unsupported metric group(s): {', '.join(sorted(unknown))}")
    args.requested_metrics = requested
    return args


def _plugin_factory(specification: str, device: str) -> Detector:
    """Load a user-selected local detector factory; no weights are downloaded here."""
    if ":" not in specification:
        raise ValueError("--detector must be 'none' or a local module:factory")
    module_name, factory_name = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    detector = factory(device=device)
    if not callable(getattr(detector, "detect", None)):
        raise TypeError("Detector factory must return an object implementing detect(frame)")
    return detector


def _sample_id(sample: EvaluationSample) -> str:
    return str(sample.path.resolve())


def _merge_frame_rows(*collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for collection in collections:
        for row in collection:
            key = (
                str(row["sample_id"]),
                int(row.get("camera_index", 0)),
                int(row["frame_position"]),
            )
            merged.setdefault(key, {}).update(row)
    return list(merged.values())


def _mean(values: list[dict[str, Any]], key: str) -> float | None:
    data = [
        float(row[key])
        for row in values
        if row.get(key) is not None
        and isinstance(row.get(key), (int, float, np.number))
        and np.isfinite(row[key])
    ]
    return float(np.mean(data)) if data else None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def _load_fvd_features(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if path is None:
        return None, None, "No compatible precomputed clip feature file was supplied"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "real_features" not in archive or "generated_features" not in archive:
                raise ValueError("NPZ must include real_features and generated_features")
            return (
                np.asarray(archive["real_features"]),
                np.asarray(archive["generated_features"]),
                None,
            )
    except (OSError, ValueError, EOFError) as exc:
        return None, None, f"Could not load FVD features from {path}: {exc}"


def _unavailable_group(name: str, reason: str) -> dict[str, Any]:
    return {
        "metric_group": name,
        "metric": name,
        "status": "unavailable",
        "value": None,
        "reason": reason,
    }


def run_evaluation(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    """Execute selected offline metrics and write all meaningful reports."""
    files = discover_npz_files(args.input_dir)
    run_dir = create_run_directory(args.output_dir)
    requested = args.requested_metrics
    run_id = run_dir.name

    unavailable: list[dict[str, Any]] = []
    metric_configuration: dict[str, Any] = {
        "pixel_preprocessing": "channel-last RGB; uint8 intensity [0,255]; no resizing",
        "ssim": "scikit-image; channel_axis=-1; data_range=255; adaptive odd window up to 7",
        "optical_flow": "OpenCV Farneback; estimated-field difference, not ground-truth endpoint error",
        "object_metrics": (
            "Detector plugin outputs xyxy boxes; greedy IoU tracker; custom image-space metrics"
        ),
        "action_states": "annotation states are 7D; first 3 dimensions used for translation step",
        "recorded_action_metrics": {
            "raw_to_video_alignment": "raw_index = 3 * source_frame_index",
            "command_observation_alignment": (
                "MAEs unavailable because command/observation temporal semantics are undocumented"
            ),
            "gripper_transition_threshold_native_units": 0.1,
            "velocity_metrics": (
                "unavailable without raw timestamps/control period and declared velocity units"
            ),
        },
        "cross_view_object_agreement": (
            "pairwise category-set Jaccard; synchronized source frames; both-empty pairs excluded"
        ),
        "cross_view_feature_consistency": {
            "extractor": "Torchvision ResNet-50 IMAGENET1K_V2",
            "preprocessing": "associated weights.transforms()",
            "batch_size": args.cvfc_batch_size,
            "metric": "mean pairwise cosine similarity across synchronized views",
        },
        "fvd": {
            "mode": (
                "precomputed_clip_features"
                if args.fvd_features
                else "video_i3d"
            ),
            "clip_length": args.fvd_clip_len,
            "clip_stride": args.fvd_stride,
            "max_clips_per_trajectory_camera": args.max_fvd_clips_per_trajectory,
            "frame_alignment": (
                "exact frame_indices or contiguous start_frame_index fallback; first sorted "
                "trajectory chunk wins overlaps; duplicate indices removed; gaps split clips"
            ),
        },
        "fid": {
            "feature_extractor": "TorchMetrics Inception-v3, 2048 features",
            "preprocessing": "uint8 RGB; 299x299 resize; antialias=False",
            "batch_size": args.fid_batch_size,
            "weights": "explicit local file or existing Torch-Fidelity cache; no downloads",
        },
    }

    fid_metric: Any | None = None
    fid_reason: str | None = None
    fid_weights_path: Path | None = None
    fid_real_frame_count = 0
    fid_generated_frame_count = 0
    if "fid" in requested:
        fid_weights_path, fid_reason = resolve_fid_weights(args.fid_weights)
        if fid_weights_path is not None:
            try:
                fid_metric = load_fid_metric(fid_weights_path, args.device)
            except (ImportError, OSError, EOFError, TypeError, ValueError, RuntimeError) as exc:
                fid_reason = f"Could not initialize FID from local weights: {exc}"

    lpips_model = None
    if "lpips" in requested:
        lpips_model, lpips_reason = load_local_lpips(
            args.lpips_network, args.lpips_weights, args.device
        )
        if lpips_reason:
            unavailable.append(_unavailable_group("lpips", lpips_reason))

    detector = None
    detector_reason = None
    tracking_reason = None
    tracking_requested = "objects" in requested or "physical" in requested
    if tracking_requested and args.tracker == "none":
        tracking_reason = "Object and physical metrics require tracking; --tracker none was selected"
    detector_needed = "multiview" in requested or (
        tracking_requested and args.tracker != "none"
    )
    if detector_needed:
        if args.detector == "none":
            detector_reason = "No local object detector was configured"
        else:
            try:
                detector = _plugin_factory(args.detector, args.device)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                detector_reason = f"Could not initialize the configured local detector: {exc}"
        if detector_reason:
            if tracking_requested:
                if "objects" in requested:
                    unavailable.append(_unavailable_group("objects", detector_reason))
                if "physical" in requested:
                    unavailable.append(_unavailable_group("physical", detector_reason))
            if "multiview" in requested:
                unavailable.append(_unavailable_group("cross_view_object_agreement", detector_reason))
    if tracking_reason:
        if "objects" in requested:
            unavailable.append(_unavailable_group("objects", tracking_reason))
        if "physical" in requested:
            unavailable.append(_unavailable_group("physical", tracking_reason))

    cvfc_model = None
    cvfc_reason = None
    if "multiview" in requested:
        cvfc_model, cvfc_reason = load_cvfc_model(args.device)
        if cvfc_reason:
            unavailable.append(_unavailable_group("cross_view_feature_consistency", cvfc_reason))

    frame_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    object_camera_rows_all: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    physical_camera_rows_all: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    action_frame_rows: list[dict[str, Any]] = []
    recorded_action_frame_rows: list[dict[str, Any]] = []
    multiview_frame_rows: list[dict[str, Any]] = []
    camera_pair_frame_rows: list[dict[str, Any]] = []
    camera_ids_seen: set[int] = set()
    visual_dir = run_dir / "visualizations"
    sample_count = valid_count = 0

    for path, sample, load_error in iter_samples(files, frame_stride=args.frame_stride):
        sample_count += 1
        if sample is None:
            validation.append(validation_failure(str(path), load_error or "Unknown load error"))
            interaction_rows.append(
                {
                    "sample_id": str(path.resolve()),
                    "file": str(path.resolve()),
                    "status": "invalid",
                    "validation_reason": load_error,
                }
            )
            print(f"[INVALID] {path}: {load_error}")
            continue

        sample_id = _sample_id(sample)
        annotation = (
            load_annotation(
                sample,
                args.annotation_root,
                include_raw_fields="action" in requested,
            )
            if args.annotation_root is not None
            else None
        )
        if "action" in requested and annotation is None:
            unavailable.append(
                _unavailable_group(
                    "action",
                    "No --annotation_root supplied; aligned state and raw command metrics are unavailable",
                )
            )
        elif (
            "action" in requested
            and annotation is not None
            and annotation.status != "available"
        ):
            unavailable.append(
                _unavailable_group(
                    f"action:{sample.trajectory_id}:{sample.interaction_index}",
                    annotation.reason or "Annotation alignment is unavailable",
                )
            )
        validation_item = validate_sample(sample, annotation)
        validation.append(validation_item)
        sample_is_valid = validation_item["status"] == "valid"
        valid_count += int(sample_is_valid)
        camera_ids_seen.update(range(sample.ground_truth.shape[0]))
        if fid_metric is not None:
            try:
                added_frames = update_fid_metric(
                    fid_metric,
                    sample.ground_truth,
                    sample.prediction,
                    args.fid_batch_size,
                    args.device,
                )
                fid_real_frame_count += added_frames
                fid_generated_frame_count += added_frames
            except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
                fid_reason = f"FID feature extraction failed for {sample.path}: {exc}"
                fid_metric = None
        sample_metric_rows: list[dict[str, Any]] = []
        pixel_rows: list[dict[str, Any]] = []
        temporal_rows: list[dict[str, Any]] = []
        lpips_rows: list[dict[str, Any]] = []

        if "visual" in requested:
            try:
                pixel_rows = compute_pixel_metrics(
                    sample.ground_truth, sample.prediction, sample_id
                )
                sample_metric_rows.extend(pixel_rows)
                if any(row.get("ssim_status") == "unavailable" for row in pixel_rows):
                    unavailable.append(
                        _unavailable_group(
                            f"ssim:{sample_id}",
                            "Images are smaller than the minimum SSIM window size",
                        )
                    )
            except (ImportError, ValueError, RuntimeError) as exc:
                unavailable.append(_unavailable_group(f"visual:{sample_id}", str(exc)))
        if "temporal" in requested and sample.ground_truth.shape[1] > 1:
            try:
                temporal_rows = compute_temporal_metrics(
                    sample.ground_truth, sample.prediction, sample_id
                )
                sample_metric_rows.extend(temporal_rows)
            except (ImportError, ValueError, RuntimeError) as exc:
                unavailable.append(_unavailable_group(f"temporal:{sample_id}", str(exc)))
        if lpips_model is not None:
            try:
                lpips_rows = compute_lpips(
                    lpips_model,
                    sample.ground_truth,
                    sample.prediction,
                    sample_id,
                    batch_size=args.batch_size,
                )
                sample_metric_rows.extend(lpips_rows)
            except (ImportError, ValueError, RuntimeError) as exc:
                unavailable.append(_unavailable_group(f"lpips:{sample_id}", str(exc)))

        # Detection results are shared by tracking metrics and category agreement.
        tracks_by_camera: list[list[list[dict[str, Any]]]] = []
        reference_tracks_by_camera: list[list[list[dict[str, Any]]]] = []
        reference_detections_by_camera: list[list[list[Any]]] = []
        generated_detections_by_camera: list[list[list[Any]]] = []
        object_camera_rows = []
        sample_physical_events = []
        physical_camera_rows = []
        sample_detector_error = None
        if detector is not None:
            try:
                for camera in range(sample.prediction.shape[0]):
                    reference_detections = detect_video(sample.ground_truth[camera], detector)
                    generated_detections = detect_video(sample.prediction[camera], detector)
                    reference_detections_by_camera.append(reference_detections)
                    generated_detections_by_camera.append(generated_detections)
                    if args.tracker != "none":
                        reference_tracks = track_detections(
                            reference_detections, IoUTracker()
                        )
                        generated_tracks = track_detections(
                            generated_detections, IoUTracker()
                        )
                        reference_tracks_by_camera.append(reference_tracks)
                        tracks_by_camera.append(generated_tracks)
                    else:
                        reference_tracks = []
                        generated_tracks = []
                    if "objects" in requested and args.tracker != "none":
                        object_values, current_objects = compute_object_metrics(
                            reference_tracks,
                            generated_tracks,
                            sample.prediction.shape[2],
                            sample.prediction.shape[3],
                            sample_id,
                            camera,
                        )
                        for object_row in current_objects:
                            source_position = int(object_row["frame_position"])
                            object_row["trajectory_id"] = sample.trajectory_id
                            object_row["interaction_index"] = sample.interaction_index
                            object_row["source_frame_index"] = (
                                int(sample.frame_indices[source_position])
                                if sample.frame_indices is not None
                                else None
                            )
                        object_rows.extend(current_objects)
                        object_camera_rows.append(
                            {"sample_id": sample_id, "camera_index": camera, **object_values}
                        )
                    if "physical" in requested and args.tracker != "none":
                        events = detect_overlap_candidates(
                            generated_tracks, sample_id, camera
                        )
                        for event in events:
                            source_position = int(event["frame_position"])
                            event["trajectory_id"] = sample.trajectory_id
                            event["interaction_index"] = sample.interaction_index
                            event["source_frame_index"] = (
                                int(sample.frame_indices[source_position])
                                if sample.frame_indices is not None
                                else None
                            )
                        sample_physical_events.extend(events)
                        physical_rows.extend(events)
                        generated_detection_count = sum(len(frame) for frame in generated_tracks)
                        physical_camera_rows.append(
                            {
                                "sample_id": sample_id,
                                "camera_index": camera,
                                **summarize_physical_candidates(
                                    events,
                                    sample.ground_truth.shape[1],
                                    generated_detection_count,
                                    sample_id,
                                ),
                            }
                        )
            except (ImportError, TypeError, ValueError, RuntimeError) as exc:
                sample_detector_error = (
                    f"Configured detector/tracking failed for {sample_id}: {exc}"
                )
                
                if "objects" in requested:
                    unavailable.append(
                        _unavailable_group(f"objects:{sample_id}", sample_detector_error)
                    )
                if "physical" in requested:
                    unavailable.append(
                        _unavailable_group(f"physical:{sample_id}", sample_detector_error)
                    )
                if "multiview" in requested:
                    unavailable.append(
                        _unavailable_group(
                            f"cross_view_object_agreement:{sample_id}", sample_detector_error
                        )
                    )
                tracks_by_camera = []
                reference_tracks_by_camera = []
                reference_detections_by_camera = []
                generated_detections_by_camera = []
        object_camera_rows_all.extend(object_camera_rows)
        physical_camera_rows_all.extend(physical_camera_rows)

        action_values: dict[str, Any] = {}
        recorded_action_values: dict[str, Any] = {}
        if "action" in requested:
            try:
                action_values, current_action_rows = compute_action_diagnostics(
                    annotation,
                    sample.ground_truth,
                    sample.prediction,
                    sample_id,
                    reference_tracks_by_camera[0] if reference_tracks_by_camera else None,
                    tracks_by_camera[0] if tracks_by_camera else None,
                )
                if sample.prediction.shape[1] > 1:
                    try:
                        flow = np.mean(
                            np.stack(
                                [
                                    mean_flow_magnitude(sample.prediction[camera])
                                    for camera in range(sample.prediction.shape[0])
                                ]
                            ),
                            axis=0,
                        )
                        for index, row in enumerate(current_action_rows):
                            if index < len(flow):
                                row["image_motion_magnitude"] = float(flow[index])
                    except ImportError as exc:
                        action_values["action_image_motion_correlation_reason"] = (
                            f"Image-space motion correlation unavailable: {exc}"
                        )
                action_frame_rows.extend(current_action_rows)
                for row in current_action_rows:
                    row["camera_index"] = 0
                    sample_metric_rows.append(row)
                action_values["annotation_path"] = (
                    annotation.annotation_path if annotation is not None else None
                )
                action_values["annotation_status"] = (
                    annotation.status if annotation is not None else "not_requested"
                )
                action_values["annotation_reason"] = (
                    annotation.reason if annotation is not None else None
                )
            except (ImportError, TypeError, ValueError, RuntimeError) as exc:
                action_values = {
                    "action_data_status": "unavailable",
                    "action_data_reason": f"Action diagnostics failed: {exc}",
                }
                unavailable.append(_unavailable_group(f"action:{sample_id}", str(exc)))
            try:
                recorded_action_values, current_recorded_rows = (
                    compute_recorded_action_metrics(
                        annotation,
                        sample.frame_indices,
                        sample_id,
                        sample.trajectory_id,
                        sample.interaction_index,
                    )
                )
                for row in current_recorded_rows:
                    row["start_frame_index"] = sample.start_frame_index
                    row["fps"] = sample.fps
                recorded_action_frame_rows.extend(current_recorded_rows)
                for metric in RECORDED_ACTION_METRICS:
                    if recorded_action_values.get(f"{metric}_status") == "unavailable":
                        unavailable.append(
                            _unavailable_group(
                                f"{metric}:{sample_id}",
                                recorded_action_values.get(f"{metric}_reason")
                                or "Recorded-action metric is unavailable",
                            )
                        )
            except (TypeError, ValueError, RuntimeError) as exc:
                recorded_action_values = {
                    metric: None for metric in RECORDED_ACTION_METRICS
                }
                for metric in RECORDED_ACTION_METRICS:
                    recorded_action_values[f"{metric}_status"] = "unavailable"
                    recorded_action_values[f"{metric}_reason"] = (
                        f"Recorded-action metrics failed: {exc}"
                    )
                unavailable.append(
                    _unavailable_group(
                        f"recorded_action:{sample_id}",
                        f"Recorded-action metrics failed: {exc}",
                    )
                )

        multiview_values: dict[str, Any] = {}
        if "multiview" in requested:
            cvoa_reason = (
                sample_detector_error
                or detector_reason
                or ("No local object detector was configured" if detector is None else None)
            )
            cvoa_values, cvoa_rows, cvoa_pair_rows = compute_cvoa(
                sample,
                sample_id,
                reference_detections_by_camera if detector is not None else None,
                generated_detections_by_camera if detector is not None else None,
                unavailable_reason=cvoa_reason,
            )
            cvfc_values, cvfc_rows, cvfc_pair_rows = compute_cvfc(
                sample,
                sample_id,
                cvfc_model,
                args.cvfc_batch_size,
                unavailable_reason=cvfc_reason,
            )
            multiview_values.update(cvoa_values)
            multiview_values.update(cvfc_values)
            multiview_values.update(
                {
                    "cross_view_geometry_status": "unavailable",
                    "cross_view_geometry_reason": (
                        "No camera calibration or valid cross-view correspondence metadata provided"
                    ),
                }
            )
            for row in cvoa_rows + cvfc_rows:
                row["start_frame_index"] = sample.start_frame_index
                row["fps"] = sample.fps
            multiview_frame_rows.extend(cvoa_rows)
            multiview_frame_rows.extend(cvfc_rows)
            camera_pair_frame_rows.extend(cvoa_pair_rows)
            camera_pair_frame_rows.extend(cvfc_pair_rows)

        sample_metric_rows = _merge_frame_rows(sample_metric_rows)
        for row in sample_metric_rows:
            position = int(row["frame_position"])
            row["trajectory_id"] = sample.trajectory_id
            row["interaction_index"] = sample.interaction_index
            row["start_frame_index"] = sample.start_frame_index
            row["fps"] = sample.fps
            row["source_frame_index"] = (
                int(sample.frame_indices[position])
                if sample.frame_indices is not None
                else None
            )
        frame_rows.extend(sample_metric_rows)
        metric_names = tuple(
            key
            for key, selected in (
                ("mse", "visual" in requested),
                ("psnr", "visual" in requested),
                ("ssim", "visual" in requested),
                ("lpips", lpips_model is not None),
                ("temporal_difference_error", "temporal" in requested),
                ("optical_flow_difference", "temporal" in requested),
            )
            if selected
        )
        interaction = {
            "sample_id": sample_id,
            "file": str(sample.path.resolve()),
            "trajectory_id": sample.trajectory_id,
            "interaction_index": sample.interaction_index,
            "start_frame_index": sample.start_frame_index,
            "instruction": sample.instruction,
            "fps": sample.fps,
            "camera_count": int(sample.ground_truth.shape[0]),
            "frame_count": int(sample.ground_truth.shape[1]),
            "status": validation_item["status"],
        }
        for metric in metric_names:
            metric_values = [
                float(row[metric])
                for row in sample_metric_rows
                if row.get(metric) is not None and np.isfinite(row[metric])
            ]
            interaction[metric] = (
                float(np.mean(metric_values)) if metric_values else None
            )
            interaction[f"{metric}_median"] = (
                float(np.median(metric_values)) if metric_values else None
            )
            interaction[f"{metric}_std"] = (
                float(np.std(metric_values)) if metric_values else None
            )
            interaction[f"{metric}_min"] = (
                float(np.min(metric_values)) if metric_values else None
            )
            interaction[f"{metric}_max"] = (
                float(np.max(metric_values)) if metric_values else None
            )
            interaction[f"{metric}_valid_count"] = len(metric_values)
            expected_count = (
                sample.ground_truth.shape[0]
                * max(0, sample.ground_truth.shape[1] - 1)
                if metric in ("temporal_difference_error", "optical_flow_difference")
                else sample.ground_truth.shape[0] * sample.ground_truth.shape[1]
            )
            interaction[f"{metric}_invalid_count"] = (
                expected_count - len(metric_values)
            )
        interaction.update(action_values)
        interaction.update(recorded_action_values)
        interaction.update(multiview_values)
        if "objects" in requested:
            if object_camera_rows:
                camera_object_statuses = [
                    item.get("object_metrics_status") for item in object_camera_rows
                ]
                for metric in object_camera_rows[0]:
                    if metric not in (
                        "sample_id",
                        "camera_index",
                        "object_metrics_status",
                        "object_metrics_reason",
                    ):
                        interaction[metric] = _mean(object_camera_rows, metric)
                valid_camera_count = sum(
                    status == "available" for status in camera_object_statuses
                )
                if (
                    valid_camera_count == sample.ground_truth.shape[0]
                    and sample_detector_error is None
                ):
                    interaction["object_metrics_status"] = "available"
                elif valid_camera_count:
                    interaction["object_metrics_status"] = "partially_available"
                    interaction["object_metrics_reason"] = sample_detector_error or (
                        "One or more cameras had no usable detector outputs"
                    )
                else:
                    interaction["object_metrics_status"] = "unavailable"
                    interaction["object_metrics_reason"] = (
                        "The configured detector returned no detections in either video"
                    )
                    unavailable.append(
                        _unavailable_group(
                            f"objects:{sample_id}",
                            interaction["object_metrics_reason"],
                        )
                    )
            elif tracking_reason or detector_reason:
                interaction["object_metrics_status"] = "unavailable"
                interaction["object_metrics_reason"] = tracking_reason or detector_reason
        if "physical" in requested and detector is not None and args.tracker != "none":
            numeric_physical = [
                row["physical_violation_rate"]
                for row in physical_camera_rows
                if row.get("physical_violation_rate") is not None
            ]
            physical_partial = len(numeric_physical) < sample.ground_truth.shape[0]
            interaction.update(
                {
                    "physical_violation_rate": (
                        float(np.mean(numeric_physical)) if numeric_physical else None
                    ),
                    "physical_violation_rate_status": (
                        "unavailable"
                        if not numeric_physical
                        else "partially_available"
                        if physical_partial
                        else "available"
                    ),
                    "physical_violation_rate_reason": (
                        "One or more cameras had no generated object detections"
                        if physical_partial and numeric_physical
                        else "No generated object detections were available"
                        if not numeric_physical
                        else None
                    ),
                    "physical_violation_candidate_count": len(sample_physical_events),
                    "physical_violation_rate_interpretation": (
                        "Unweighted mean of per-camera fractions of observed frames with at least "
                        "one 2D box overlap candidate; not verified 3D penetration"
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
                    "physical_frame_count": sample.ground_truth.shape[1],
                }
            )
            if not numeric_physical or physical_partial:
                unavailable.append(
                    _unavailable_group(
                        f"physical:{sample_id}",
                        interaction["physical_violation_rate_reason"],
                    )
                )
        elif "physical" in requested:
            physical_reason = tracking_reason or detector_reason or "No local object detector was configured"
            interaction.update(
                {
                    "physical_violation_rate": None,
                    "physical_violation_rate_status": "unavailable",
                    "physical_violation_rate_reason": physical_reason,
                    "physical_violation_candidate_count": 0,
                    "physical_frame_count": sample.ground_truth.shape[1],
                }
            )
        interaction_rows.append(interaction)

        if args.save_visualizations:
            try:
                from .visualization import save_frame_comparisons, save_object_diagnostics

                save_frame_comparisons(
                    sample,
                    visual_dir
                    / _safe_name(
                        f"{sample.trajectory_id}_{sample.interaction_index}_{sample.path.stem}"
                    ),
                )
                if tracks_by_camera:
                    save_object_diagnostics(
                        sample,
                        tracks_by_camera,
                        sample_physical_events,
                        visual_dir
                        / _safe_name(
                            f"{sample.trajectory_id}_{sample.interaction_index}_{sample.path.stem}"
                        ),
                    )
            except (ImportError, OSError, ValueError) as exc:
                unavailable.append(_unavailable_group(f"visualization:{sample_id}", str(exc)))
        if args.save_comparison_videos:
            try:
                from .visualization import save_comparison_videos

                save_comparison_videos(
                    sample, visual_dir / "comparison_videos" / _safe_name(sample.trajectory_id)
                )
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                unavailable.append(_unavailable_group(f"comparison_video:{sample_id}", str(exc)))
        print(f"[EVALUATED {sample_count}/{len(files)}] {sample.path.name}")

    trajectory_rows = aggregate_rows(interaction_rows, ("trajectory_id",))
    source_frame_metric_names = tuple(
        (RECORDED_ACTION_METRICS if "action" in requested else ())
        + ((CVOA_METRICS + CVFC_METRICS) if "multiview" in requested else ())
    )
    source_trajectory_rows, source_dataset_metrics = aggregate_unique_source_frame_metrics(
        recorded_action_frame_rows + multiview_frame_rows,
        source_frame_metric_names,
    )
    source_trajectory_by_id = {
        str(row.get("trajectory_id")): row for row in source_trajectory_rows
    }
    for row in trajectory_rows:
        source_row = source_trajectory_by_id.get(str(row.get("trajectory_id")))
        if source_row is not None:
            row.update(source_row)
    camera_rows = summarize_cameras(frame_rows, FRAME_METRICS)
    camera_ids = {row["camera_index"] for row in camera_rows}
    for camera in sorted(
        (
            {row["camera_index"] for row in object_camera_rows_all}
            | {row["camera_index"] for row in physical_camera_rows_all}
            | camera_ids_seen
        )
        - camera_ids
    ):
        camera_rows.append(
            {
                "camera_index": camera,
                "frame_pair_count": 0,
                "cross_view_geometry_status": "unavailable",
                "cross_view_geometry_reason": (
                    "No camera calibration or valid cross-view correspondence metadata provided"
                ),
            }
        )
    # Include detector-derived camera metrics alongside reference-based metrics.
    for row in camera_rows:
        matching = [
            item
            for item in object_camera_rows_all
            if item.get("camera_index") == row.get("camera_index")
        ] if detector is not None else []
        if matching:
            for metric in matching[0]:
                if metric not in (
                    "sample_id",
                    "camera_index",
                    "object_metrics_status",
                    "object_metrics_reason",
                ):
                    row[metric] = _mean(matching, metric)
            status_count = sum(
                item.get("object_metrics_status") == "available" for item in matching
            )
            row["object_metrics_status"] = (
                "available"
                if status_count == len(matching)
                else "partially_available"
                if status_count
                else "unavailable"
            )
            if status_count != len(matching):
                row["object_metrics_reason"] = (
                    "One or more interaction-camera detector outputs were unavailable"
                )
    for row in camera_rows:
        physical_matches = [
            item
            for item in physical_camera_rows_all
            if item.get("camera_index") == row.get("camera_index")
        ]
        if physical_matches:
            valid_physical_count = sum(
                item.get("physical_violation_rate") is not None
                for item in physical_matches
            )
            row["physical_violation_candidate_count"] = sum(
                item["physical_violation_candidate_count"] for item in physical_matches
            )
            row["physical_violation_rate"] = _mean(physical_matches, "physical_violation_rate")
            row["physical_violation_rate_valid_count"] = valid_physical_count
            row["physical_violation_rate_invalid_count"] = (
                len(physical_matches) - valid_physical_count
            )
            row["physical_violation_rate_status"] = (
                "unavailable"
                if valid_physical_count == 0
                else "partially_available"
                if valid_physical_count < len(physical_matches)
                else "available"
            )
    horizon_metric_names = tuple(
        key
        for key, selected in (
            ("mse", "visual" in requested),
            ("psnr", "visual" in requested),
            ("ssim", "visual" in requested),
            ("lpips", lpips_model is not None),
            ("temporal_difference_error", "temporal" in requested),
            ("optical_flow_difference", "temporal" in requested),
        )
        if selected
    )
    horizon_summary = horizon_rows(frame_rows, horizon_metric_names)
    frame_rows.extend(recorded_action_frame_rows)
    frame_rows.extend(multiview_frame_rows)
    if "fid" in requested:
        if fid_reason is not None:
            fid_status = {
                "metric": "fid",
                "status": "unavailable",
                "value": None,
                "reason": fid_reason,
                "mode": "dataset_frame_distribution",
                "feature_extractor": "TorchMetrics Inception-v3, 2048 features",
                "weights_path": str(fid_weights_path) if fid_weights_path else None,
                "real_frame_count": fid_real_frame_count,
                "generated_frame_count": fid_generated_frame_count,
                "batch_size": args.fid_batch_size,
            }
        elif fid_metric is None:
            fid_status = {
                "metric": "fid",
                "status": "unavailable",
                "value": None,
                "reason": "FID metric initialization was unavailable",
                "mode": "dataset_frame_distribution",
                "feature_extractor": "TorchMetrics Inception-v3, 2048 features",
                "weights_path": str(fid_weights_path) if fid_weights_path else None,
                "real_frame_count": fid_real_frame_count,
                "generated_frame_count": fid_generated_frame_count,
                "batch_size": args.fid_batch_size,
            }
        elif fid_real_frame_count < 2 or fid_generated_frame_count < 2:
            fid_status = {
                "metric": "fid",
                "status": "unavailable",
                "value": None,
                "reason": "FID requires at least two real and two generated frames",
                "mode": "dataset_frame_distribution",
                "feature_extractor": "TorchMetrics Inception-v3, 2048 features",
                "weights_path": str(fid_weights_path) if fid_weights_path else None,
                "real_frame_count": fid_real_frame_count,
                "generated_frame_count": fid_generated_frame_count,
                "batch_size": args.fid_batch_size,
            }
        else:
            try:
                fid_value = float(fid_metric.compute().item())
                fid_status = {
                    "metric": "fid",
                    "status": "available",
                    "value": fid_value,
                    "reason": None,
                    "mode": "dataset_frame_distribution",
                    "feature_extractor": "TorchMetrics Inception-v3, 2048 features",
                    "weights_path": str(fid_weights_path) if fid_weights_path else None,
                    "real_frame_count": fid_real_frame_count,
                    "generated_frame_count": fid_generated_frame_count,
                    "batch_size": args.fid_batch_size,
                }
                if not np.isfinite(fid_value):
                    fid_status.update(
                        {
                            "status": "unavailable",
                            "value": None,
                            "reason": "FID computation returned a non-finite value",
                        }
                    )
            except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
                fid_status = {
                    "metric": "fid",
                    "status": "unavailable",
                    "value": None,
                    "reason": f"FID computation failed: {exc}",
                    "mode": "dataset_frame_distribution",
                    "feature_extractor": "TorchMetrics Inception-v3, 2048 features",
                    "weights_path": str(fid_weights_path) if fid_weights_path else None,
                    "real_frame_count": fid_real_frame_count,
                    "generated_frame_count": fid_generated_frame_count,
                    "batch_size": args.fid_batch_size,
                }
        if fid_status["status"] == "unavailable":
            unavailable.append(fid_status)
    else:
        fid_status = {"metric": "fid", "status": "not_requested", "value": None, "reason": None}

    if "fvd" in requested:
        if args.fvd_features:
            real_features, generated_features, fvd_reason = _load_fvd_features(args.fvd_features)
            if fvd_reason is None and not args.fvd_extractor_name:
                fvd_reason = "Supply --fvd_extractor_name to record feature-extractor provenance"
            if fvd_reason is None:
                try:
                    fvd_value = frechet_distance_from_features(real_features, generated_features)
                    fvd_status = {
                        "metric": "fvd",
                        "status": "available",
                        "value": fvd_value,
                        "reason": None,
                        "mode": "precomputed_clip_features",
                        "feature_extractor": args.fvd_extractor_name,
                        "real_clip_count": int(len(real_features)),
                        "generated_clip_count": int(len(generated_features)),
                    }
                except (ValueError, np.linalg.LinAlgError) as exc:
                    fvd_status = unavailable_fvd(str(exc))
                    fvd_status["mode"] = "precomputed_clip_features"
            else:
                fvd_status = unavailable_fvd(fvd_reason)
                fvd_status["mode"] = "precomputed_clip_features"
        else:
            fvd_status = compute_video_fvd(
                files,
                args.frame_stride,
                args.fvd_clip_len,
                args.fvd_stride,
                args.max_fvd_clips_per_trajectory,
                args.device,
                args.fvd_weights,
            )
        if fvd_status["status"] == "unavailable":
            unavailable.append(fvd_status)
    else:
        fvd_status = {"metric": "fvd", "status": "not_requested", "value": None, "reason": None}

    if "objects" in requested and detector is None and not any(
        item["metric_group"] == "objects" for item in unavailable
    ):
        unavailable.append(_unavailable_group("objects", detector_reason or "No detector configured"))
    if "physical" in requested and detector is None and not any(
        item["metric_group"] == "physical" for item in unavailable
    ):
        unavailable.append(_unavailable_group("physical", detector_reason or "No detector configured"))
    if "temporal" in requested and not any(
        row.get("optical_flow_difference") is not None for row in frame_rows
    ):
        unavailable.append(
            _unavailable_group("temporal", "No valid frame pairs or optical-flow computation failed")
        )
    if "multiview" in requested:
        unavailable.append(
            _unavailable_group(
                "cross_view_geometry",
                "Per-camera metrics are available; no calibration/correspondence was provided",
            )
        )

    config = vars(args).copy()
    config["requested_metrics"] = sorted(requested)
    config["input_paths"] = [str(path.resolve()) for path in files]
    config["run_id"] = run_id
    cross_view_camera_pair_rows = summarize_camera_pairs(
        camera_pair_frame_rows, CVOA_METRICS + CVFC_METRICS
    )
    aggregated_metrics = dataset_summary(interaction_rows)
    aggregated_metrics.update(source_dataset_metrics)
    summary = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "input_dir": str(Path(args.input_dir).resolve()),
        "annotation_root": str(Path(args.annotation_root).resolve())
        if args.annotation_root
        else None,
        "number_of_discovered_samples": len(files),
        "number_of_valid_samples": valid_count,
        "number_of_invalid_samples": sample_count - valid_count,
        "available_metrics": {
            "pixel_and_ssim": any(row.get("mse") is not None for row in frame_rows),
            "lpips": any(row.get("lpips") is not None for row in frame_rows),
            "temporal": any(row.get("temporal_difference_error") is not None for row in frame_rows),
            "action": any(row.get("action_data_status") == "available" for row in interaction_rows),
            "recorded_action": any(
                row.get(f"{metric}_status") in ("available", "partially_available")
                for row in interaction_rows
                for metric in RECORDED_ACTION_METRICS
            ),
            "objects": any(
                row.get("object_metrics_status") in ("available", "partially_available")
                for row in interaction_rows
            ),
            "physical_candidates": any(
                row.get("physical_violation_rate") is not None for row in interaction_rows
            ),
            "fid": fid_status["status"] == "available",
            "fvd": fvd_status["status"] == "available",
            "per_camera": True,
            "cross_view_geometry": False,
            "cross_view_object_agreement": any(
                row.get(f"{CVOA_METRICS[0]}_status")
                in ("available", "partially_available")
                for row in interaction_rows
            ),
            "cross_view_feature_consistency": any(
                row.get(f"{CVFC_METRICS[3]}_status")
                in ("available", "partially_available")
                for row in interaction_rows
            ),
        },
        "unavailable_metrics": unavailable,
        "aggregated_metrics": aggregated_metrics,
        "camera_wise_results": camera_rows,
        "cross_view_camera_pair_results": cross_view_camera_pair_rows,
        "horizon_wise_results": horizon_summary,
        "fid": fid_status,
        "fvd": fvd_status,
        "distribution_metrics": {"fid": fid_status, "fvd": fvd_status},
        "metric_configuration": metric_configuration,
        "detector_configuration": {
            "detector": args.detector,
            "tracker": args.tracker,
            "status": (
                "unavailable"
                if detector is None
                else "partially_unavailable"
                if detector_reason
                else "available"
            ),
            "reason": detector_reason,
            "tracking_reason": tracking_reason,
        },
        "preprocessing": {
            "canonical_layout": "camera,time,height,width,channel",
            "color": "RGB",
            "pixel_range": "[0,255]",
            "resizing": "none",
            "frame_stride": args.frame_stride,
            "action_alignment": "explicit frame_indices; safe contiguous fallback from start_frame_index",
            "raw_action_alignment": (
                "raw_index = 3 * source_frame_index; same-row command/observation timing "
                "semantics are not inferred"
            ),
        },
        "warnings": [
            "Pixel fidelity is not evidence of physical understanding.",
            "Image-space overlap and flow metrics are not verified 3D physical measurements.",
            "Action-motion correlation is descriptive and is not causal evidence.",
            "Recorded command/observation consistency is not generated-video action prediction accuracy.",
            "Cross-view category agreement is limited by detector labels and per-view visibility; it is not object identity or 3D agreement.",
            "ResNet-50 cross-view feature similarity is a semantic proxy, not geometric correspondence.",
        ],
        "validation_report": validation,
        "invalid_sample_count": sum(item["status"] == "invalid" for item in validation),
    }
    tables = {
        "frames": frame_rows,
        "interactions": interaction_rows,
        "trajectories": trajectory_rows,
        "cameras": camera_rows + cross_view_camera_pair_rows,
        "horizons": horizon_summary,
        "object_summaries": aggregate_object_tracks(object_rows),
        "objects": object_rows,
        "physical_violations": physical_rows,
    }
    if args.save_visualizations and frame_rows:
        try:
            from .visualization import save_metric_plots

            plots = save_metric_plots(
                horizon_summary,
                camera_rows,
                action_frame_rows,
                run_dir / "plots",
            )
            summary["visualization_files"] = plots
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            unavailable.append(_unavailable_group("plots", str(exc)))
    written = write_reports(run_dir, json_safe(config), json_safe(summary), tables)
    exit_code = 1 if sample_count != valid_count else 0
    summary["written_csv_files"] = written
    return run_dir, summary, exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns 2 for fatal configuration/input errors."""
    args = parse_args(argv)
    try:
        run_dir, summary, exit_code = run_evaluation(args)
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        f"[SUMMARY] discovered={summary['number_of_discovered_samples']} "
        f"valid={summary['number_of_valid_samples']} "
        f"invalid={summary['number_of_invalid_samples']}"
    )
    print(f"[OUTPUT] {run_dir.resolve()}")
    for name, result in summary["aggregated_metrics"].items():
        if result["valid_count"]:
            print(f"[METRIC] {name}: mean={result['mean']:.6g} n={result['valid_count']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
