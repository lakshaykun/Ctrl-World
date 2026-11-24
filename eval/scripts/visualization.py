"""Optional frame/video and quantitative visualization output."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from .utils import EvaluationSample


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def save_frame_comparisons(
    sample: EvaluationSample, output_dir: str | Path, max_frames: int = 8
) -> list[str]:
    """Save paired GT/prediction RGB panels without resizing source frames."""
    from PIL import Image, ImageDraw

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = []
    frame_ids = np.linspace(
        0, sample.ground_truth.shape[1] - 1, min(max_frames, sample.ground_truth.shape[1]), dtype=int
    )
    for camera in range(sample.ground_truth.shape[0]):
        for frame in frame_ids:
            gt = Image.fromarray(sample.ground_truth[camera, frame])
            pred = Image.fromarray(sample.prediction[camera, frame])
            panel = Image.new("RGB", (gt.width + pred.width, gt.height + 24), "white")
            panel.paste(gt, (0, 24))
            panel.paste(pred, (gt.width, 24))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 5), "Ground truth", fill="black")
            draw.text((gt.width + 8, 5), "Prediction", fill="black")
            path = target / (
                f"{_safe_component(sample.trajectory_id)}_interaction_{sample.interaction_index}_"
                f"{_safe_component(sample.path.stem)}_"
                f"camera_{camera}_frame_{frame}.png"
            )
            panel.save(path)
            files.append(str(path))
    return files


def save_comparison_videos(sample: EvaluationSample, output_dir: str | Path) -> list[str]:
    """Encode side-by-side RGB comparison MP4 files, one per camera."""
    import cv2

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    fps = sample.fps if sample.fps and sample.fps > 0 else 5.0
    outputs = []
    for camera in range(sample.ground_truth.shape[0]):
        height, width = sample.ground_truth.shape[2:4]
        path = target / (
            f"{_safe_component(sample.trajectory_id)}_interaction_{sample.interaction_index}_"
            f"{_safe_component(sample.path.stem)}_camera_{camera}.mp4"
        )
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
        )
        if not writer.isOpened():
            raise OSError(f"Could not open comparison video writer: {path}")
        try:
            for frame in range(sample.ground_truth.shape[1]):
                panel = np.concatenate(
                    [sample.ground_truth[camera, frame], sample.prediction[camera, frame]], axis=1
                )
                writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        outputs.append(str(path))
    return outputs


def save_object_diagnostics(
    sample: EvaluationSample,
    tracks_by_camera: list[list[list[dict[str, Any]]]],
    physical_events: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[str]:
    """Save generated-frame track overlays and frames with overlap candidates."""
    from PIL import Image, ImageDraw

    root = Path(output_dir)
    tracking_dir = root / "object_tracking"
    physical_dir = root / "physical_candidates"
    outputs = []
    stem = _safe_component(
        f"{sample.trajectory_id}_{sample.interaction_index}_{sample.path.stem}"
    )
    for camera_index, frames in enumerate(tracks_by_camera):
        for frame_index, tracks in enumerate(frames):
            image = Image.fromarray(sample.prediction[camera_index, frame_index])
            draw = ImageDraw.Draw(image)
            for track in tracks:
                box = tuple(float(value) for value in track["box_xyxy"])
                draw.rectangle(box, outline="lime", width=2)
                draw.text((box[0], box[1]), str(track.get("track_id", "?")), fill="lime")
            tracking_dir.mkdir(parents=True, exist_ok=True)
            path = tracking_dir / f"{stem}_camera_{camera_index}_frame_{frame_index}.png"
            image.save(path)
            outputs.append(str(path))

    grouped_events: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for event in physical_events:
        key = (int(event["camera_index"]), int(event["frame_position"]))
        grouped_events.setdefault(key, []).append(event)
    for (camera, frame), events in grouped_events.items():
        image = Image.fromarray(sample.prediction[camera, frame])
        draw = ImageDraw.Draw(image)
        for event in events:
            for key in ("box_xyxy_a", "box_xyxy_b"):
                draw.rectangle(
                    tuple(float(value) for value in event[key]), outline="red", width=3
                )
        physical_dir.mkdir(parents=True, exist_ok=True)
        path = physical_dir / (
            f"{stem}_camera_{camera}_frame_{frame}_overlap.png"
        )
        image.save(path)
        outputs.append(str(path))
    return outputs


def save_metric_plots(
    horizon_rows: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[str]:
    """Create only plots for metrics that have finite observed values."""
    import matplotlib.pyplot as plt

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    plot_specs = (
        ("mse", "Mean squared error", "Squared RGB intensity"),
        ("psnr", "Peak signal-to-noise ratio", "dB"),
        ("ssim", "Structural similarity", "SSIM"),
        ("temporal_difference_error", "Temporal difference error", "Mean absolute RGB difference"),
        ("optical_flow_difference", "Optical-flow difference", "Pixels per sampled interval"),
        ("lpips", "LPIPS", "Perceptual distance"),
    )
    for metric, title, ylabel in plot_specs:
        points = [
            row
            for row in horizon_rows
            if row.get(metric) is not None and np.isfinite(row[metric])
        ]
        if not points:
            continue
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(
            [row["horizon"] for row in points],
            [row[metric] for row in points],
            marker="o",
            label=metric,
        )
        axis.set_title(title)
        axis.set_xlabel("Prediction frame position")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        path = target / f"{metric}_by_horizon.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs.append(str(path))

    for metric, title, ylabel in plot_specs:
        valid = [row for row in camera_rows if row.get(metric) is not None]
        if not valid:
            continue
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(
            [row["camera_index"] for row in valid],
            [row[metric] for row in valid],
            marker="o",
        )
        axis.set_title(f"{title} by camera")
        axis.set_xlabel("Camera index")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        path = target / f"{metric}_per_camera.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs.append(str(path))

    action_points = [
        row
        for row in action_rows
        if row.get("end_effector_translation_step") is not None
        and row.get("image_motion_magnitude") is not None
    ]
    if action_points:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(
            [row["end_effector_translation_step"] for row in action_points],
            [row["image_motion_magnitude"] for row in action_points],
        )
        axis.set_title("Recorded end-effector movement vs predicted image motion")
        axis.set_xlabel("End-effector translation step (annotation units)")
        axis.set_ylabel("Mean optical-flow magnitude (pixels)")
        figure.tight_layout()
        path = target / "action_vs_image_motion.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs.append(str(path))
    return outputs
