"""Discovery and safe canonical loading of saved rollout NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .utils import EvaluationSample

REQUIRED_FIELDS = ("ground_truth", "prediction")
METADATA_FIELDS = (
    "trajectory_id",
    "interaction_index",
    "start_frame_index",
    "instruction",
    "fps",
    "frame_indices",
)


def discover_npz_files(input_dir: str | Path) -> list[Path]:
    """Return all NPZ files below an input directory in deterministic order."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation input directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No evaluation NPZ files found under: {root}")
    return files


def _scalar(value: np.ndarray, field: str) -> object:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Metadata field {field!r} must contain one scalar, got {array.shape}")
    value = array.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _integer_metadata(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Metadata field {field!r} must be an integer")
    try:
        integer = int(value)
        if float(value) != integer:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Metadata field {field!r} must be an integer, got {value!r}") from exc
    return integer


def _canonical_video(array: np.ndarray, field: str) -> np.ndarray:
    """Normalize known layouts to (camera, time, height, width, channel).

    The replay scripts save camera × time × H × W × RGB arrays. A single-camera
    time × H × W × RGB array is also accepted. Ambiguous layouts are rejected.
    """
    video = np.asarray(array)
    if video.ndim == 4:
        if video.shape[-1] not in (1, 3, 4):
            raise ValueError(f"{field} must use channel-last RGB(A), got shape {video.shape}")
        video = video[np.newaxis, ...]
    elif video.ndim == 5:
        if video.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"{field} must use the saved camera,time,height,width,channel layout; "
                f"got shape {video.shape}"
            )
    else:
        raise ValueError(f"{field} must be 4D or 5D, got shape {video.shape}")
    if video.shape[0] < 1 or video.shape[1] < 1 or video.shape[2] < 1 or video.shape[3] < 1:
        raise ValueError(f"{field} has an empty camera, frame, or image dimension: {video.shape}")
    if video.shape[-1] == 4:
        video = video[..., :3]
    elif video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    if not np.issubdtype(video.dtype, np.number):
        raise ValueError(f"{field} must have a numeric dtype, got {video.dtype}")
    if not np.isfinite(video).all():
        raise ValueError(f"{field} contains NaN or infinite pixel values")
    if np.issubdtype(video.dtype, np.floating):
        finite_min = float(video.min())
        finite_max = float(video.max())
        if finite_min < 0:
            raise ValueError(f"{field} floating pixels must be non-negative, got minimum {finite_min}")
        if finite_max <= 1.0:
            video = video * 255.0
    if float(video.min()) < 0 or float(video.max()) > 255:
        raise ValueError(f"{field} pixel values must fit [0,255]")
    return video.astype(np.uint8, copy=False)


def load_sample(path: str | Path, frame_stride: int = 1) -> EvaluationSample:
    """Load one NPZ, validate core arrays, and preserve its metadata."""
    source = Path(path)
    if frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = [field for field in REQUIRED_FIELDS if field not in archive]
            if missing:
                raise ValueError(f"Missing required NPZ fields: {', '.join(missing)}")
            ground_truth = _canonical_video(archive["ground_truth"], "ground_truth")
            prediction = _canonical_video(archive["prediction"], "prediction")
            metadata = {
                key: _scalar(archive[key], key)
                for key in METADATA_FIELDS
                if key in archive and key != "frame_indices"
            }
            frame_indices = (
                np.asarray(archive["frame_indices"]).reshape(-1)
                if "frame_indices" in archive
                else None
            )
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Could not load evaluation sample {source}: {exc}") from exc

    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground-truth/prediction shape mismatch in {source}: "
            f"{ground_truth.shape} != {prediction.shape}"
        )
    if frame_indices is not None:
        if not np.issubdtype(frame_indices.dtype, np.integer):
            if not np.issubdtype(frame_indices.dtype, np.number) or not np.equal(
                frame_indices, np.floor(frame_indices)
            ).all():
                raise ValueError(f"frame_indices must contain integers in {source}")
        frame_indices = frame_indices.astype(np.int64, copy=False)
        if len(frame_indices) != ground_truth.shape[1]:
            raise ValueError(
                f"frame_indices length {len(frame_indices)} does not match frame count "
                f"{ground_truth.shape[1]} in {source}"
            )

    selection = np.arange(0, ground_truth.shape[1], frame_stride)
    if frame_indices is None and metadata.get("start_frame_index") is not None:
        frame_indices = _integer_metadata(metadata["start_frame_index"], "start_frame_index") + selection
    elif frame_indices is not None:
        frame_indices = frame_indices[selection]
    ground_truth = ground_truth[:, selection]
    prediction = prediction[:, selection]

    trajectory_id = str(metadata.get("trajectory_id", ""))
    if not trajectory_id:
        trajectory_id = source.parent.name.removeprefix("trajectory_")
    interaction_index = metadata.get("interaction_index")
    start_frame_index = metadata.get("start_frame_index")
    return EvaluationSample(
        path=source,
        ground_truth=ground_truth,
        prediction=prediction,
        trajectory_id=trajectory_id,
        interaction_index=(
            _integer_metadata(interaction_index, "interaction_index")
            if interaction_index is not None
            else None
        ),
        start_frame_index=(
            _integer_metadata(start_frame_index, "start_frame_index")
            if start_frame_index is not None
            else None
        ),
        frame_indices=frame_indices,
        instruction=(
            str(metadata["instruction"]) if metadata.get("instruction") is not None else None
        ),
        fps=float(metadata["fps"]) if metadata.get("fps") is not None else None,
        metadata=metadata,
    )


def iter_samples(
    files: list[Path], frame_stride: int = 1
) -> Iterator[tuple[Path, EvaluationSample | None, str | None]]:
    """Yield every discovered path with either its loaded sample or a failure reason."""
    for path in files:
        try:
            yield path, load_sample(path, frame_stride=frame_stride), None
        except (OSError, ValueError, TypeError) as exc:
            yield path, None, str(exc)
