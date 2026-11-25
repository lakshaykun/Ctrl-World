from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval.scripts import run_evaluation as runner
from eval.scripts.checkpointing import (
    build_sample_cache_identity,
    load_sample_checkpoint,
    sha256_file,
    write_sample_checkpoint,
)
from eval.scripts.reporting import write_reports


def _write_sample(path: Path, prediction_value: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth = np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)
    prediction = np.full_like(ground_truth, prediction_value)
    np.savez_compressed(
        path,
        ground_truth=ground_truth,
        prediction=prediction,
        trajectory_id=path.parent.name.removeprefix("trajectory_"),
        interaction_index=0,
        start_frame_index=0,
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        fps=5,
    )


def _write_annotation(root: Path, trajectory_id: str = "7", value: float = 1.0) -> Path:
    path = root / "annotation" / "val" / f"{trajectory_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video_length": 2,
                "states": [[value] * 7, [value + 1] * 7],
                "texts": ["test task"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _args(
    input_dir: Path,
    output_dir: Path,
    *extra: str,
) -> runner.argparse.Namespace:
    return runner.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--metrics",
            "visual",
            *extra,
        ]
    )


class ResumableEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "inputs"
        self.output_dir = self.root / "outputs"
        self.sample_path = self.input_dir / "trajectory_7" / "interaction_000.npz"
        _write_sample(self.sample_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, *extra: str):
        return runner.run_evaluation(_args(self.input_dir, self.output_dir, *extra))

    def test_existing_cli_defaults_and_new_options(self) -> None:
        args = _args(self.input_dir, self.output_dir)
        self.assertIsNone(args.resume_run)
        self.assertEqual(args.checkpoint_every, 1)
        self.assertFalse(args.force)
        resumed = _args(
            self.input_dir,
            self.output_dir,
            "--resume-run",
            str(self.output_dir / "run"),
            "--checkpoint-every",
            "3",
            "--force",
        )
        self.assertEqual(resumed.checkpoint_every, 3)
        self.assertTrue(resumed.force)

    def test_new_run_writes_atomic_per_sample_checkpoint_and_metadata(self) -> None:
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            run_dir, summary, exit_code = self._run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["number_of_discovered_samples"], 1)
        self.assertTrue((run_dir / "run_metadata.json").is_file())
        self.assertTrue((run_dir / "config.json").is_file())
        checkpoints = list((run_dir / "sample_checkpoints").glob("*.json"))
        self.assertEqual(len(checkpoints), 1)
        checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["status"], "completed")
        self.assertEqual(checkpoint["results"]["validation"][0]["status"], "valid")
        self.assertEqual(compute.call_count, 1)

    def test_checkpoint_every_batches_checkpoint_flushes(self) -> None:
        _write_sample(self.input_dir / "trajectory_8" / "interaction_000.npz", 40)
        _write_sample(self.input_dir / "trajectory_9" / "interaction_000.npz", 60)
        with patch(
            "eval.scripts.run_evaluation._write_checkpoint_queue",
            wraps=runner._write_checkpoint_queue,
        ) as flush:
            run_dir, _, _ = runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--output_dir",
                        str(self.output_dir),
                        "--metrics",
                        "visual",
                        "--checkpoint-every",
                        "2",
                    ]
                )
            )
        self.assertEqual(flush.call_count, 2)
        self.assertEqual(len(list((run_dir / "sample_checkpoints").glob("*.json"))), 3)

    def test_resume_reuses_completed_sample_without_metric_recomputation(self) -> None:
        run_dir, _, _ = self._run()
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            side_effect=AssertionError("cached pixel metrics were recomputed"),
        ):
            resumed_dir, summary, exit_code = runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(resumed_dir, run_dir)
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["number_of_valid_samples"], 1)
        self.assertEqual(summary["aggregated_metrics"]["mse"]["valid_count"], 1)

    def test_pending_sample_is_evaluated_and_reports_have_no_duplicate_rows(self) -> None:
        run_dir, _, _ = self._run()
        _write_sample(self.input_dir / "trajectory_8" / "interaction_000.npz", 40)
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            _, summary, _ = runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(summary["number_of_discovered_samples"], 2)
        with (run_dir / "metrics_per_frame.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({(row["sample_id"], row["frame_position"]) for row in rows}), 4)

    def test_changed_npz_content_invalidates_cached_sample(self) -> None:
        run_dir, first_summary, _ = self._run()
        _write_sample(self.sample_path, 80)
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            _, second_summary, _ = runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)
        self.assertNotEqual(
            first_summary["aggregated_metrics"]["mse"]["mean"],
            second_summary["aggregated_metrics"]["mse"]["mean"],
        )

    def test_changed_annotation_content_invalidates_relevant_sample(self) -> None:
        annotation_root = self.root / "dataset"
        annotation = _write_annotation(annotation_root)
        first_args = _args(
            self.input_dir,
            self.output_dir,
            "--annotation_root",
            str(annotation_root),
        )
        run_dir, _, _ = runner.run_evaluation(first_args)
        first_hash = sha256_file(annotation)
        _write_annotation(annotation_root, value=9.0)
        self.assertNotEqual(sha256_file(annotation), first_hash)
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                        "--annotation_root",
                        str(annotation_root),
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)

    def test_changed_metric_configuration_invalidates_sample_checkpoint(self) -> None:
        run_dir, _, _ = self._run()
        with (
            patch(
                "eval.scripts.run_evaluation.compute_pixel_metrics",
                wraps=runner.compute_pixel_metrics,
            ) as pixel,
            patch(
                "eval.scripts.run_evaluation.compute_temporal_metrics",
                return_value=[
                    {
                        "sample_id": str(self.sample_path.resolve()),
                        "camera_index": 0,
                        "frame_position": 1,
                        "temporal_difference_error": 0.0,
                        "optical_flow_difference": 0.0,
                    }
                ],
            ) as temporal,
        ):
            runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual,temporal",
                    ]
                )
            )
        self.assertEqual(pixel.call_count, 1)
        self.assertEqual(temporal.call_count, 1)

    def test_global_metric_change_does_not_invalidate_sample_level_cache(self) -> None:
        run_dir, _, _ = self._run()
        with (
            patch(
                "eval.scripts.run_evaluation.resolve_fid_weights",
                return_value=(None, "no local test weights"),
            ),
            patch(
                "eval.scripts.run_evaluation.compute_pixel_metrics",
                side_effect=AssertionError("sample-level cache was invalidated"),
            ),
        ):
            _, summary, _ = runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual,fid",
                    ]
                )
            )
        self.assertEqual(summary["fid"]["status"], "unavailable")
        self.assertEqual(summary["aggregated_metrics"]["mse"]["valid_count"], 1)

    def test_force_recomputes_cached_sample(self) -> None:
        run_dir, _, _ = self._run()
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                        "--force",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)

    def test_corrupted_checkpoint_is_recomputed(self) -> None:
        run_dir, _, _ = self._run()
        checkpoint_path = next((run_dir / "sample_checkpoints").glob("*.json"))
        checkpoint_path.write_text("{", encoding="utf-8")
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(json.loads(checkpoint_path.read_text())["status"], "completed")

    def test_structurally_incomplete_checkpoint_is_recomputed(self) -> None:
        run_dir, _, _ = self._run()
        checkpoint_path = next((run_dir / "sample_checkpoints").glob("*.json"))
        envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        envelope["results"].pop("frame_rows")
        checkpoint_path.write_text(json.dumps(envelope), encoding="utf-8")
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(
            len(json.loads(checkpoint_path.read_text())["results"]["frame_rows"]),
            2,
        )

    def test_failed_sample_checkpoint_is_retried_after_input_is_repaired(self) -> None:
        self.sample_path.write_bytes(b"not an npz")
        run_dir, summary, exit_code = self._run()
        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["number_of_invalid_samples"], 1)
        checkpoint_path = next((run_dir / "sample_checkpoints").glob("*.json"))
        self.assertEqual(json.loads(checkpoint_path.read_text())["status"], "failed")

        _write_sample(self.sample_path, 30)
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            _, repaired_summary, repaired_exit_code = runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(repaired_exit_code, 0)
        self.assertEqual(repaired_summary["number_of_valid_samples"], 1)
        self.assertEqual(json.loads(checkpoint_path.read_text())["status"], "completed")

    def test_metric_failure_checkpoint_is_retried_on_resume(self) -> None:
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            side_effect=ValueError("temporary metric failure"),
        ):
            run_dir, _, _ = self._run()
        checkpoint_path = next((run_dir / "sample_checkpoints").glob("*.json"))
        self.assertEqual(json.loads(checkpoint_path.read_text())["status"], "failed")
        with patch(
            "eval.scripts.run_evaluation.compute_pixel_metrics",
            wraps=runner.compute_pixel_metrics,
        ) as compute:
            runner.run_evaluation(
                runner.parse_args(
                    [
                        "--input_dir",
                        str(self.input_dir),
                        "--resume-run",
                        str(run_dir),
                        "--metrics",
                        "visual",
                    ]
                )
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(json.loads(checkpoint_path.read_text())["status"], "completed")

    def test_resumed_and_clean_runs_have_equal_sample_aggregates(self) -> None:
        partial_dir, _, _ = self._run()
        second = self.input_dir / "trajectory_8" / "interaction_000.npz"
        _write_sample(second, 45)
        _, resumed_summary, _ = runner.run_evaluation(
            runner.parse_args(
                [
                    "--input_dir",
                    str(self.input_dir),
                    "--resume-run",
                    str(partial_dir),
                    "--metrics",
                    "visual",
                ]
            )
        )
        clean_output = self.root / "clean_output"
        clean_args = runner.parse_args(
            [
                "--input_dir",
                str(self.input_dir),
                "--output_dir",
                str(clean_output),
                "--metrics",
                "visual",
            ]
        )
        _, clean_summary, _ = runner.run_evaluation(clean_args)
        self.assertEqual(
            resumed_summary["aggregated_metrics"],
            clean_summary["aggregated_metrics"],
        )
        self.assertEqual(
            resumed_summary["camera_wise_results"],
            clean_summary["camera_wise_results"],
        )

    def test_fid_and_fvd_are_recomputed_over_all_samples_on_resume(self) -> None:
        _write_sample(self.input_dir / "trajectory_8" / "interaction_000.npz", 40)
        weights = self.root / "inception.pth"
        weights.write_bytes(b"local test weights")

        class FakeFid:
            def compute(self):
                return np.asarray(2.5)

        arguments = [
            "--input_dir",
            str(self.input_dir),
            "--output_dir",
            str(self.output_dir),
            "--metrics",
            "fid,fvd",
            "--fid_weights",
            str(weights),
        ]
        with (
            patch("eval.scripts.run_evaluation.resolve_fid_weights", return_value=(weights, None)),
            patch("eval.scripts.run_evaluation.load_fid_metric", return_value=FakeFid()),
            patch(
                "eval.scripts.run_evaluation.update_fid_metric",
                side_effect=lambda metric, ground_truth, prediction, batch, device: (
                    ground_truth.shape[0] * ground_truth.shape[1]
                ),
            ) as fid_update,
            patch(
                "eval.scripts.run_evaluation.compute_video_fvd",
                return_value={
                    "metric": "fvd",
                    "status": "available",
                    "value": 4.5,
                    "reason": None,
                },
            ) as fvd_compute,
        ):
            first_dir, first_summary, _ = runner.run_evaluation(
                runner.parse_args(arguments)
            )
            self.assertEqual(first_summary["fid"]["value"], 2.5)
            self.assertEqual(first_summary["fvd"]["value"], 4.5)
            self.assertEqual(fid_update.call_count, 2)
            fid_update.reset_mock()
            fvd_compute.reset_mock()
            resumed_args = runner.parse_args(
                [
                    "--input_dir",
                    str(self.input_dir),
                    "--resume-run",
                    str(first_dir),
                    "--metrics",
                    "fid,fvd",
                    "--fid_weights",
                    str(weights),
                ]
            )
            _, resumed_summary, _ = runner.run_evaluation(resumed_args)
        self.assertEqual(fid_update.call_count, 2)
        self.assertEqual(fvd_compute.call_count, 1)
        self.assertEqual(len(fvd_compute.call_args.args[0]), 2)
        self.assertEqual(resumed_summary["fid"]["real_frame_count"], 4)
        self.assertEqual(resumed_summary["fvd"]["value"], 4.5)

    def test_atomic_checkpoint_write_preserves_previous_file_on_replace_failure(self) -> None:
        sample_id = str(self.sample_path.resolve())
        identity = build_sample_cache_identity(self.sample_path, "same-hash", None, {})
        results = {
            "frame_rows": [],
            "interaction_rows": [{"sample_id": sample_id, "status": "valid"}],
            "object_rows": [],
            "object_camera_rows": [],
            "physical_rows": [],
            "physical_camera_rows": [],
            "validation": [{"status": "valid"}],
            "action_frame_rows": [],
            "recorded_action_frame_rows": [],
            "multiview_frame_rows": [],
            "camera_pair_frame_rows": [],
            "unavailable": [],
            "camera_indices": [0],
        }
        run_dir = self.root / "atomic_run"
        write_sample_checkpoint(run_dir, sample_id, identity, "completed", results)
        with patch("eval.scripts.checkpointing.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                write_sample_checkpoint(
                    run_dir,
                    sample_id,
                    identity,
                    "completed",
                    {**results, "interaction_rows": [{"sample_id": sample_id, "new": True}]},
                )
        loaded, reason = load_sample_checkpoint(run_dir, sample_id, identity)
        self.assertIsNone(reason)
        self.assertEqual(
            loaded["interaction_rows"][0],
            {"sample_id": sample_id, "status": "valid"},
        )
        self.assertEqual(list((run_dir / "sample_checkpoints").glob("*.tmp")), [])

    def test_report_writer_removes_csv_tables_that_are_empty_after_resume(self) -> None:
        run_dir = self.root / "report_run"
        write_reports(
            run_dir,
            {},
            {"validation_report": []},
            {"frames": [{"sample_id": "sample", "mse": 1.0}]},
        )
        report_path = run_dir / "metrics_per_frame.csv"
        self.assertTrue(report_path.is_file())
        write_reports(run_dir, {}, {"validation_report": []}, {"frames": []})
        self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
