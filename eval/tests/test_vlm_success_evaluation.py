from __future__ import annotations

import json
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from eval.scripts.data_loader import load_sample
from eval.scripts.vlm_success_evaluation import (
    JUDGING_PROMPT,
    TrajectoryUnavailableError,
    _build_parser,
    build_cache_identity,
    build_vlm_messages,
    compose_prediction_camera_grid,
    discover_trajectory_groups,
    find_cached_prediction,
    inspect_trajectory,
    judge_with_retries,
    load_droid_annotation,
    parse_vlm_json,
    prepare_sampled_images,
    reconstruct_prediction_trajectory,
    sample_frame_positions,
    summarize_predictions,
    run_evaluation,
    write_evaluation_outputs,
)


def _write_interaction(
    directory: Path,
    trajectory_id: str,
    interaction_index: int,
    frame_indices: list[int],
    pixel_base: int | None = None,
    fps: int = 7,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if pixel_base is None:
        pixel_base = interaction_index * 10
    prediction = np.empty((3, len(frame_indices), 4, 5, 3), dtype=np.uint8)
    for camera in range(3):
        for frame_position in range(len(frame_indices)):
            prediction[camera, frame_position].fill(
                min(255, pixel_base + frame_position + camera)
            )
    path = directory / f"interaction_{interaction_index:03d}.npz"
    np.savez_compressed(
        path,
        ground_truth=np.full_like(prediction, 251),
        prediction=prediction,
        trajectory_id=trajectory_id,
        interaction_index=interaction_index,
        start_frame_index=frame_indices[0],
        instruction="untrusted NPZ instruction",
        fps=fps,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
    )
    return path


def _write_annotation(root: Path, trajectory_id: str, value: dict) -> Path:
    path = root / "annotation" / "val" / f"{trajectory_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class ReplayDiscoveryAndReconstructionTests(unittest.TestCase):
    def test_npz_discovery_groups_by_trajectory_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation_data"
            first = _write_interaction(root / "trajectory_12", "12", 0, [0, 1, 2])
            second = _write_interaction(root / "trajectory_12", "12", 1, [2, 3, 4])
            _write_interaction(root / "trajectory_34", "34", 0, [0, 1, 2])
            grouped = discover_trajectory_groups(root)
            self.assertEqual(set(grouped), {"12", "34"})
            self.assertEqual(grouped["12"], [first, second])

    def test_interactions_are_ordered_and_boundaries_match_replay_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory_5"
            third = _write_interaction(directory, "5", 2, [4, 5, 6], 30)
            first = _write_interaction(directory, "5", 0, [0, 1, 2], 10)
            second = _write_interaction(directory, "5", 1, [2, 3, 4], 20)
            plan = inspect_trajectory([third, first, second], "5", video_length=7)
            reconstructed = reconstruct_prediction_trajectory(plan)
            self.assertEqual(reconstructed.source_frame_indices.tolist(), list(range(7)))
            self.assertEqual(
                reconstructed.frames[:, 0, 0, 0, 0].tolist(),
                [10, 11, 20, 21, 30, 31, 32],
            )
            self.assertFalse(np.any(reconstructed.frames == 251))

    def test_missing_interaction_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory_5"
            first = _write_interaction(directory, "5", 0, [0, 1, 2])
            third = _write_interaction(directory, "5", 2, [2, 3, 4])
            with self.assertRaisesRegex(TrajectoryUnavailableError, "consecutive"):
                inspect_trajectory([first, third], "5", video_length=5)

    def test_source_gap_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory_5"
            first = _write_interaction(directory, "5", 0, [0, 1, 2])
            second = _write_interaction(directory, "5", 1, [2, 4, 5])
            third = _write_interaction(directory, "5", 2, [5, 6, 7])
            with self.assertRaisesRegex(TrajectoryUnavailableError, "gap"):
                inspect_trajectory([first, second, third], "5", video_length=8)

    def test_incomplete_trajectory_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory_5"
            first = _write_interaction(directory, "5", 0, [0, 1, 2])
            second = _write_interaction(directory, "5", 1, [2, 3, 4])
            with self.assertRaisesRegex(TrajectoryUnavailableError, "final source frame"):
                inspect_trajectory([first, second], "5", video_length=6)

    def test_out_of_range_integer_pixels_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory_5" / "interaction_000.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(
                path,
                prediction=np.full((3, 1, 4, 5, 3), -1, dtype=np.int16),
                trajectory_id="5",
                interaction_index=0,
                start_frame_index=0,
                fps=7,
                frame_indices=np.asarray([0]),
            )
            with self.assertRaisesRegex(TrajectoryUnavailableError, r"\[0,255\]"):
                inspect_trajectory([path], "5", video_length=1)

    def test_nonfinite_source_indices_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory_5" / "interaction_000.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(
                path,
                prediction=np.zeros((3, 1, 4, 5, 3), dtype=np.uint8),
                trajectory_id="5",
                interaction_index=0,
                start_frame_index=0,
                fps=7,
                frame_indices=np.asarray([np.nan]),
            )
            with self.assertRaisesRegex(TrajectoryUnavailableError, "contain integers"):
                inspect_trajectory([path], "5", video_length=1)

    def test_clamped_terminal_source_frames_remain_in_generated_chronology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory_5"
            first = _write_interaction(directory, "5", 0, [0, 1, 2])
            second = _write_interaction(directory, "5", 1, [2, 3, 3])
            plan = inspect_trajectory([first, second], "5", video_length=4)
            reconstructed = reconstruct_prediction_trajectory(plan)
            self.assertEqual(reconstructed.source_frame_indices.tolist(), [0, 1, 2, 3, 3])
            self.assertEqual(reconstructed.frames.shape[0], 5)

    def test_sampled_positions_are_uniform_bounded_and_include_final_frame(self) -> None:
        positions = sample_frame_positions(101, fps=10, sampling_fps=5, max_frames=20)
        self.assertEqual(len(positions), 20)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 100)
        self.assertEqual(len(set(positions)), len(positions))

    def test_camera_grid_preserves_all_synchronized_views_under_pixel_budget(self) -> None:
        frames = np.stack(
            [
                np.full((12, 16, 3), (255, 0, 0), dtype=np.uint8),
                np.full((12, 16, 3), (0, 255, 0), dtype=np.uint8),
                np.full((12, 16, 3), (0, 0, 255), dtype=np.uint8),
            ]
        )
        image = compose_prediction_camera_grid(frames, max_pixels=300)
        self.assertLessEqual(image.width * image.height, 300)
        self.assertEqual(image.mode, "RGB")

    def test_sampled_images_use_reconstructed_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory_5"
            first = _write_interaction(directory, "5", 0, [0, 1, 2], 10)
            second = _write_interaction(directory, "5", 1, [2, 3, 4], 20)
            plan = inspect_trajectory([first, second], "5", video_length=5)
            trajectory = reconstruct_prediction_trajectory(plan)
            images, source_indices = prepare_sampled_images(trajectory, 5, 50, 100_000)
            self.assertEqual(source_indices[0], 0)
            self.assertEqual(source_indices[-1], 4)
            self.assertTrue(all(isinstance(image, Image.Image) for image in images))
            self.assertNotIn(251, np.asarray(images[0]))


class AnnotationAndStructuredOutputTests(unittest.TestCase):
    def test_instruction_and_success_come_from_droid_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_annotation(
                root,
                "12",
                {"texts": ["DROID task"], "success": 1, "video_length": 7},
            )
            annotation = load_droid_annotation(root, "12")
            self.assertEqual(annotation.status, "valid")
            self.assertEqual(annotation.instruction, "DROID task")
            self.assertEqual(annotation.ground_truth_success, 1)
            self.assertEqual(annotation.ground_truth_label, "success")

    def test_zero_success_label_maps_to_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_annotation(
                root,
                "12",
                {"texts": ["DROID task"], "success": 0, "video_length": 7},
            )
            self.assertEqual(load_droid_annotation(root, "12").ground_truth_label, "failure")

    def test_empty_instruction_is_invalid_but_valid_label_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_annotation(
                root,
                "12",
                {"texts": [""], "success": 0, "video_length": 7},
            )
            annotation = load_droid_annotation(root, "12")
            self.assertEqual(annotation.status, "invalid")
            self.assertIsNone(annotation.instruction)
            self.assertEqual(annotation.ground_truth_success, 0)

    def test_zero_padded_annotation_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_annotation(
                root,
                "000012",
                {"texts": ["DROID task"], "success": True, "video_length": 7},
            )
            annotation = load_droid_annotation(root, "12")
            self.assertEqual(annotation.status, "valid")
            self.assertEqual(annotation.ground_truth_success, 1)
            self.assertTrue(annotation.annotation_path.endswith("000012.json"))

    def test_missing_annotation_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            annotation = load_droid_annotation(Path(temporary), "404")
            self.assertEqual(annotation.status, "missing")
            self.assertIsNone(annotation.ground_truth_success)
            self.assertIn("404.json", annotation.annotation_path or "")

    def test_invalid_success_values_are_not_coerced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_annotation(
                root,
                "12",
                {"texts": ["DROID task"], "success": "1", "video_length": 7},
            )
            annotation = load_droid_annotation(root, "12")
            self.assertEqual(annotation.status, "invalid")
            self.assertIsNone(annotation.ground_truth_success)
            self.assertIn("success must", annotation.reason or "")

    def test_vlm_json_validation_accepts_uncertain_and_rejects_bad_fields(self) -> None:
        result = parse_vlm_json(
            '{"prediction":"uncertain","confidence":0.4,"reasoning":"Outcome is occluded."}'
        )
        self.assertEqual(result["predicted_label"], "uncertain")
        for text in (
            "not json",
            '{"prediction":"maybe","confidence":0.4,"reasoning":"x"}',
            '{"prediction":"success","confidence":1.1,"reasoning":"x"}',
            '{"prediction":"success","confidence":true,"reasoning":"x"}',
            '{"prediction":"failure","confidence":0.4,"reasoning":" "}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_vlm_json(text)

    def test_vlm_retries_malformed_output_with_a_bounded_count(self) -> None:
        image = Image.new("RGB", (4, 4), "red")
        responses = iter(
            [
                '{"prediction":"success","confidence":2,"reasoning":"invalid"}',
                '{"prediction":"failure","confidence":0.8,"reasoning":"The object stayed in place."}',
            ]
        )
        seen_messages: list[list[dict]] = []

        def generate(messages: list[dict]) -> str:
            seen_messages.append(messages)
            return next(responses)

        result = judge_with_retries("Move the object.", [image], generate, max_retries=1)
        self.assertEqual(result["predicted_label"], "failure")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(len(seen_messages), 2)
        with self.assertRaisesRegex(ValueError, "after 1 attempts"):
            judge_with_retries("Move the object.", [image], lambda _: "invalid", max_retries=0)

    def test_vlm_messages_contain_instruction_and_prediction_images_only(self) -> None:
        image = Image.new("RGB", (4, 4), "blue")
        messages = build_vlm_messages("Place the cup.", [image])
        content = messages[1]["content"]
        self.assertIn("Place the cup.", content[0]["text"])
        self.assertEqual(content[1], {"type": "image", "image": image})
        self.assertNotIn("hidden-ground-truth-label", repr(messages))
        self.assertTrue(JUDGING_PROMPT)


class MetricsCacheAndResumeTests(unittest.TestCase):
    @staticmethod
    def _record(
        trajectory_id: str,
        ground_truth: int | None,
        prediction: str | None,
        status: str,
        annotation_status: str = "valid",
    ) -> dict:
        return {
            "trajectory_id": trajectory_id,
            "ground_truth_success": ground_truth,
            "predicted_label": prediction,
            "inference_status": status,
            "annotation_status": annotation_status,
            "reconstruction_status": "valid",
        }

    def test_cache_identity_changes_for_content_instruction_model_or_sampling(self) -> None:
        arguments = {
            "trajectory_id": "12",
            "source_hashes": [("chunk.npz", "source-a")],
            "annotation_sha256": "annotation-a",
            "instruction": "Move the cup.",
            "ground_truth_success": 1,
            "model_name": "Qwen/Qwen3-VL-4B-Instruct",
            "model_revision": "revision-a",
            "sampling_configuration": {"fps": 5},
            "processor_configuration": {"max_pixels": 1000},
            "max_new_tokens": 64,
        }
        original = build_cache_identity(**arguments)
        for field, value in (
            ("source_hashes", [("chunk.npz", "source-b")]),
            ("instruction", "Move the plate."),
            ("model_revision", "revision-b"),
            ("sampling_configuration", {"fps": 4}),
            ("processor_configuration", {"max_pixels": 2000}),
            ("max_parse_retries", 1),
        ):
            changed_arguments = dict(arguments)
            changed_arguments[field] = value
            if field == "max_parse_retries":
                changed = build_cache_identity(**arguments, max_parse_retries=value)
            else:
                changed = build_cache_identity(**changed_arguments)
            self.assertNotEqual(original, changed, field)

    def test_resume_reuses_only_completed_valid_predictions(self) -> None:
        completed = {"cache_identity": "same", "inference_status": "completed", "predicted_label": "uncertain"}
        failed = {"cache_identity": "same", "inference_status": "failed", "predicted_label": None}
        self.assertEqual(find_cached_prediction([failed, completed], "same"), completed)
        self.assertIsNone(find_cached_prediction([failed], "same"))

    def test_binary_metrics_and_abstention_matrix_count_every_outcome(self) -> None:
        records = [
            self._record("1", 1, "success", "completed"),
            self._record("2", 0, "failure", "completed"),
            self._record("3", 1, "uncertain", "completed"),
            self._record("4", 0, "success", "completed"),
            self._record("5", None, None, "not_run", "missing"),
            self._record("6", 1, None, "failed"),
        ]
        summary, confusion = summarize_predictions(records)
        metrics = summary["binary_classification_metrics"]
        self.assertEqual(metrics["eligible_trajectories"], 3)
        self.assertEqual(metrics["correctly_classified"], 2)
        self.assertEqual(metrics["incorrectly_classified"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["uncertain_predictions_excluded"], 1)
        matrix = confusion["three_outcome_abstention_matrix"]
        self.assertEqual(matrix["rows_actual_columns_predicted"], [[1, 1, 0], [0, 1, 1]])
        self.assertIn("no ground-truth uncertain class", matrix["convention"])
        self.assertEqual(summary["missing_annotations"], 1)
        self.assertEqual(summary["failed_inferences"], 1)
        self.assertEqual(summary["excluded_from_binary_metrics"], 3)
        self.assertEqual(summary["ground_truth_labels_available"], 5)
        self.assertEqual(summary["success_rate_comparison"]["original_droid"]["denominator"], 5)
        self.assertEqual(summary["success_rate_comparison"]["vlm_predictions"]["denominator"], 4)

    def test_required_output_files_and_validation_report_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = [
                self._record("1", None, None, "not_run", "missing"),
                self._record("2", 1, "uncertain", "completed"),
            ]
            write_evaluation_outputs(run_dir, {"model": "test"}, records)
            for name in (
                "config.json",
                "summary.json",
                "predictions.json",
                "predictions.csv",
                "confusion_matrix.json",
                "validation_report.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            validation = json.loads((run_dir / "validation_report.json").read_text())
            self.assertEqual(validation[0]["trajectory_id"], "1")


class ExistingEvaluatorRegressionTests(unittest.TestCase):
    def test_invalid_cli_sampling_settings_fail_before_input_discovery(self) -> None:
        args = _build_parser().parse_args(["--rollout-dir", "unused", "--max-pixels", "0"])
        with self.assertRaisesRegex(ValueError, "--max-pixels must be positive"):
            run_evaluation(args)

    def test_existing_npz_loader_still_reads_replay_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "interaction.npz"
            video = np.zeros((3, 2, 4, 5, 3), dtype=np.uint8)
            np.savez_compressed(
                path,
                ground_truth=video,
                prediction=video,
                trajectory_id="12",
                interaction_index=0,
                start_frame_index=0,
                fps=7,
                frame_indices=np.asarray([0, 1]),
            )
            sample = load_sample(path)
            self.assertEqual(sample.prediction.shape, (3, 2, 4, 5, 3))
            self.assertEqual(sample.frame_indices.tolist(), [0, 1])

    def test_run_pipeline_with_local_fakes_builds_reports_without_model_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rollout = root / "Rollouts_replay_droid_subset"
            trajectory_dir = rollout / "evaluation_data" / "trajectory_1"
            _write_interaction(trajectory_dir, "1", 0, [0, 1, 2])
            _write_interaction(trajectory_dir, "1", 1, [2, 3, 4])
            dataset_dir = root / "dataset"
            _write_annotation(
                dataset_dir,
                "1",
                {"texts": ["Move the cup."], "success": 1, "video_length": 5},
            )
            config_module = ModuleType("config")

            def fake_wm_args(
                task_type: str,
                dataset_root_path: str | None = None,
                dataset_names: str | None = None,
            ) -> SimpleNamespace:
                self.assertEqual(task_type, "replay")
                return SimpleNamespace(
                    dataset_root_path=str(dataset_root_path or dataset_dir.parent),
                    dataset_names=dataset_names or "droid_subset",
                    val_dataset_dir=str(dataset_dir),
                )

            config_module.wm_args = fake_wm_args
            args = _build_parser().parse_args(
                [
                    "--rollout-dir",
                    str(rollout),
                    "--output-dir",
                    str(root / "reports"),
                    "--trajectory-id",
                    "1",
                ]
            )
            with (
                patch.dict(sys.modules, {"config": config_module}),
                patch(
                    "eval.scripts.vlm_success_evaluation.resolve_model_revision",
                    return_value="resolved-revision",
                ),
                patch(
                    "eval.scripts.vlm_success_evaluation.load_vlm",
                    return_value=(object(), object(), "resolved-revision"),
                ),
                patch(
                    "eval.scripts.vlm_success_evaluation.judge_prediction",
                    return_value={
                        "predicted_label": "success",
                        "confidence": 0.9,
                        "reasoning": "The object reaches the requested position.",
                        "attempt_count": 1,
                    },
                ),
            ):
                run_dir, summary = run_evaluation(args)

            self.assertEqual(summary["eligible_trajectories"], 1)
            records = json.loads((run_dir / "predictions.json").read_text())
            self.assertEqual(records[0]["sampled_source_frame_indices"][0], 0)
            self.assertEqual(records[0]["sampled_source_frame_indices"][-1], 4)
            self.assertEqual(records[0]["inference_status"], "completed")
            self.assertTrue((run_dir / "validation_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
