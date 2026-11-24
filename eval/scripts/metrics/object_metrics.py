"""Object detector/tracker contracts and image-space track consistency metrics.

Custom definitions: persistence is the fraction of reference detections matched
at IoU >= 0.5; hallucination rate is unmatched generated detections divided by
generated detections (a detector disagreement proxy, not proof of hallucination);
identity switches count changes in the generated track matched to a reference
track; trajectory error is the matched center distance normalized by image
diagonal; PAR is the fraction of generated track steps longer than 0.25 image
diagonals; TCS is the fraction of generated track steps no longer than 0.10
diagonals; HSS is unmatched generated box area / full image area per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A detector result in pixel-space xyxy coordinates."""

    box_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0
    label: str = "object"


class Detector(Protocol):
    """Local detector interface; implementations must not download weights."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class Tracker(Protocol):
    """Tracker interface for framewise detections."""

    def update(self, detections: list[Detection]) -> list[dict[str, Any]]: ...


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate intersection-over-union for xyxy boxes."""
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """Deterministic greedy IoU association for detections from a real detector."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 2):
        if not 0 <= iou_threshold <= 1 or max_age < 0:
            raise ValueError("Invalid IoU tracker threshold or max_age")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, dict[str, Any]] = {}

    def update(self, detections: list[Detection]) -> list[dict[str, Any]]:
        """Associate current detections and return only currently observed tracks."""
        for track in self._tracks.values():
            track["age"] += 1
        candidates = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = intersection_over_union(
                    np.asarray(track["box_xyxy"]), np.asarray(detection.box_xyxy)
                )
                if overlap >= self.iou_threshold and track["label"] == detection.label:
                    candidates.append((overlap, track_id, detection_index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            detection = detections[detection_index]
            self._tracks[track_id].update(
                {"box_xyxy": detection.box_xyxy, "age": 0, "confidence": detection.confidence}
            )
            used_tracks.add(track_id)
            used_detections.add(detection_index)
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = {
                "box_xyxy": detection.box_xyxy,
                "label": detection.label,
                "confidence": detection.confidence,
                "age": 0,
            }
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track["age"] <= self.max_age
        }
        output = []
        for track_id, track in self._tracks.items():
            if track["age"] == 0:
                output.append(
                    {
                        "track_id": track_id,
                        "box_xyxy": tuple(float(value) for value in track["box_xyxy"]),
                        "confidence": float(track["confidence"]),
                        "label": track["label"],
                    }
                )
        return output


def _coerce_detection(value: Detection | dict[str, Any]) -> Detection:
    if isinstance(value, Detection):
        return value
    box = value.get("box_xyxy")
    if box is None or len(box) != 4:
        raise ValueError("Detector output must provide a four-value box_xyxy")
    return Detection(
        tuple(float(item) for item in box),
        float(value.get("confidence", 1.0)),
        str(value.get("label", "object")),
    )


def track_video(video: np.ndarray, detector: Detector, tracker: Tracker) -> list[list[dict[str, Any]]]:
    """Run one detector and tracker once over a video, reusing results downstream."""
    all_frames = []
    for frame in video:
        detections = [_coerce_detection(item) for item in detector.detect(frame)]
        all_frames.append(tracker.update(detections))
    return all_frames


def _box(track: dict[str, Any]) -> np.ndarray:
    return np.asarray(track["box_xyxy"], dtype=np.float64)


def _match_frame(
    reference: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    candidates = []
    for ref_index, ref in enumerate(reference):
        for gen_index, gen in enumerate(generated):
            if ref.get("label", "object") != gen.get("label", "object"):
                continue
            overlap = intersection_over_union(_box(ref), _box(gen))
            if overlap >= threshold:
                candidates.append((overlap, ref_index, gen_index))
    matches = []
    used_ref: set[int] = set()
    used_gen: set[int] = set()
    for _, ref_index, gen_index in sorted(candidates, reverse=True):
        if ref_index not in used_ref and gen_index not in used_gen:
            matches.append((ref_index, gen_index))
            used_ref.add(ref_index)
            used_gen.add(gen_index)
    return matches, used_ref, used_gen


def compute_object_metrics(
    reference_tracks: list[list[dict[str, Any]]],
    generated_tracks: list[list[dict[str, Any]]],
    image_height: int,
    image_width: int,
    sample_id: str,
    camera_index: int,
    iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare detected tracks and return sample metrics plus generated track rows."""
    if len(reference_tracks) != len(generated_tracks):
        raise ValueError("Reference/generated track frame counts do not match")
    diagonal = float(np.hypot(image_height, image_width))
    total_reference = total_generated = matched_total = 0
    unmatched_area = 0.0
    frame_matches: list[tuple[int, int, int]] = []
    positions: dict[
        tuple[Any, Any], list[tuple[int, np.ndarray, np.ndarray]]
    ] = {}
    for frame_index, (reference, generated) in enumerate(zip(reference_tracks, generated_tracks)):
        matches, _, matched_gen = _match_frame(reference, generated, iou_threshold)
        total_reference += len(reference)
        total_generated += len(generated)
        matched_total += len(matches)
        frame_matches.extend((frame_index, ref_index, gen_index) for ref_index, gen_index in matches)
        for ref_index, gen_index in matches:
            ref, gen = reference[ref_index], generated[gen_index]
            ref_id = ref.get("track_id", ref_index)
            gen_id = gen.get("track_id", gen_index)
            ref_box = _box(ref)
            gen_box = _box(gen)
            ref_center = (ref_box[:2] + ref_box[2:]) / 2
            gen_center = (gen_box[:2] + gen_box[2:]) / 2
            positions.setdefault((ref_id, gen_id), []).append(
                (frame_index, ref_center, gen_center)
            )
        for index, gen in enumerate(generated):
            unmatched = index not in matched_gen
            if unmatched:
                box = _box(gen)
                x1, y1 = np.clip(box[:2], [0, 0], [image_width, image_height])
                x2, y2 = np.clip(box[2:], [0, 0], [image_width, image_height])
                unmatched_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)

    observed_match: dict[Any, list[tuple[int, Any]]] = {}
    for frame_index, ref_index, gen_index in frame_matches:
        ref = reference_tracks[frame_index][ref_index]
        gen = generated_tracks[frame_index][gen_index]
        observed_match.setdefault(ref.get("track_id", ref_index), []).append(
            (frame_index, gen.get("track_id", gen_index))
        )
    identity_switches = 0
    for matches in observed_match.values():
        ids = [track_id for _, track_id in sorted(matches)]
        identity_switches += sum(left != right for left, right in zip(ids, ids[1:]))

    trajectory_errors = []
    for paired_positions in positions.values():
        for _, reference_center, generated_center in paired_positions:
            trajectory_errors.append(
                float(np.linalg.norm(generated_center - reference_center) / diagonal)
            )

    generated_positions: dict[Any, list[tuple[int, np.ndarray]]] = {}
    for frame_index, frame in enumerate(generated_tracks):
        for index, track in enumerate(frame):
            box = _box(track)
            center = (box[:2] + box[2:]) / 2
            generated_positions.setdefault(track.get("track_id", index), []).append(
                (frame_index, center)
            )
    step_lengths = []
    for track_positions in generated_positions.values():
        track_positions.sort()
        for (frame_a, center_a), (frame_b, center_b) in zip(
            track_positions, track_positions[1:]
        ):
            if frame_b == frame_a + 1:
                step = float(np.linalg.norm(center_b - center_a) / diagonal)
                step_lengths.append(step)

    frame_count = max(1, len(reference_tracks))
    image_area = float(image_height * image_width)
    values = {
        "object_persistence_score": matched_total / total_reference if total_reference else None,
        "object_hallucination_rate": (
            (total_generated - matched_total) / total_generated if total_generated else None
        ),
        "identity_switch_count": identity_switches if total_reference else None,
        "object_trajectory_error": float(np.mean(trajectory_errors)) if trajectory_errors else None,
        "position_anomaly_rate": (
            sum(step > 0.25 for step in step_lengths) / len(step_lengths)
            if step_lengths
            else None
        ),
        "temporal_consistency_score": (
            sum(step <= 0.10 for step in step_lengths) / len(step_lengths)
            if step_lengths
            else None
        ),
        "hallucination_severity_score": (
            unmatched_area / (frame_count * image_area) if total_generated else None
        ),
        "object_reference_detections": total_reference,
        "object_generated_detections": total_generated,
        "object_matched_detections": matched_total,
        "object_metrics_status": (
            "available" if total_reference or total_generated else "unavailable"
        ),
        "object_metrics_reason": (
            None
            if total_reference or total_generated
            else "Detector returned no reference or generated detections in this sample"
        ),
    }
    object_rows = []
    for frame_index, frame in enumerate(generated_tracks):
        for index, track in enumerate(frame):
            box = _box(track)
            object_rows.append(
                {
                    "sample_id": sample_id,
                    "camera_index": camera_index,
                    "frame_position": frame_index,
                    "track_id": track.get("track_id", index),
                    "label": track.get("label", "object"),
                    "confidence": track.get("confidence", 1.0),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )
    return values, object_rows
