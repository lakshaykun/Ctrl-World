"""Reproducible output directory and JSON/CSV report writing."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import json_safe


def create_run_directory(output_dir: str | Path) -> Path:
    """Create a unique UTC-named run directory below output_dir."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    candidate = root / run_id
    suffix = 1
    while candidate.exists():
        candidate = root / f"{run_id}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON with deterministic indentation and safe NumPy conversion."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, allow_nan=False)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> bool:
    """Write a CSV only when rows contain meaningful records."""
    if not rows:
        return False
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_safe(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else json_safe(value)
                    for key, value in row.items()
                }
            )
    return True


def write_reports(
    run_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Write required JSON reports and only non-empty CSV tables."""
    write_json(run_dir / "config.json", config)
    written = []
    names = {
        "frames": "metrics_per_frame.csv",
        "interactions": "metrics_per_interaction.csv",
        "trajectories": "metrics_per_trajectory.csv",
        "cameras": "metrics_per_camera.csv",
        "horizons": "metrics_per_horizon.csv",
        "object_summaries": "metrics_per_object.csv",
        "objects": "object_tracks.csv",
        "physical_violations": "physical_violations.csv",
    }
    for key, filename in names.items():
        report_path = run_dir / filename
        if write_csv(report_path, tables.get(key, [])):
            written.append(filename)
        elif report_path.is_file():
            report_path.unlink()
    summary_with_outputs = dict(summary)
    summary_with_outputs["written_csv_files"] = written
    write_json(run_dir / "summary.json", summary_with_outputs)
    write_json(run_dir / "validation_report.json", summary.get("validation_report", []))
    return written
