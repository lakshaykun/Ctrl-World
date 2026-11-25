"""Atomic JSON checkpoints and content identities for offline evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .utils import json_safe

CHECKPOINT_SCHEMA_VERSION = 1
SAMPLE_RESULT_LISTS = (
    "frame_rows",
    "interaction_rows",
    "object_rows",
    "object_camera_rows",
    "physical_rows",
    "physical_camera_rows",
    "validation",
    "action_frame_rows",
    "recorded_action_frame_rows",
    "multiview_frame_rows",
    "camera_pair_frame_rows",
    "unavailable",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return {
            "path": str(resolved),
            "exists": resolved.is_file(),
            "sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    except OSError as exc:
        return {
            "path": str(resolved),
            "exists": resolved.is_file(),
            "sha256": None,
            "hash_error": f"{type(exc).__name__}: {exc}",
        }


def stable_identity(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_sample_cache_identity(
    npz_path: str | Path,
    npz_sha256: str | None,
    annotation_identity: dict[str, Any] | None,
    sample_configuration: dict[str, Any],
) -> str:
    return stable_identity(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "absolute_npz_path": str(Path(npz_path).expanduser().resolve()),
            "npz_sha256": npz_sha256,
            "annotation": annotation_identity,
            "sample_configuration": sample_configuration,
        }
    )


def checkpoint_directory(run_dir: str | Path) -> Path:
    return Path(run_dir) / "sample_checkpoints"


def checkpoint_path(run_dir: str | Path, sample_id: str) -> Path:
    key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return checkpoint_directory(run_dir) / f"{key}.json"


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(
                json_safe(value),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def write_run_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(run_dir) / "run_metadata.json"
    _atomic_json_write(path, metadata)
    return path


def read_run_metadata(run_dir: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(run_dir) / "run_metadata.json"
    if not path.is_file():
        return None, "run_metadata.json is missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read run metadata: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "run_metadata.json must contain a JSON object"
    if value.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None, "Run metadata uses an unsupported checkpoint schema version"
    return value, None


def write_sample_checkpoint(
    run_dir: str | Path,
    sample_id: str,
    cache_identity: str,
    status: str,
    results: dict[str, Any],
    failure_reasons: list[str] | None = None,
) -> Path:
    if status not in ("completed", "failed"):
        raise ValueError(f"Unsupported sample checkpoint status: {status}")
    missing = [key for key in SAMPLE_RESULT_LISTS if not isinstance(results.get(key), list)]
    if missing:
        raise ValueError(f"Sample checkpoint is missing result lists: {', '.join(missing)}")
    if not isinstance(results.get("camera_indices"), list):
        raise ValueError("Sample checkpoint is missing camera_indices")
    path = checkpoint_path(run_dir, sample_id)
    _atomic_json_write(
        path,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "cache_identity": cache_identity,
            "status": status,
            "failure_reasons": failure_reasons or [],
            "results": results,
        },
    )
    return path


def load_sample_checkpoint(
    run_dir: str | Path,
    sample_id: str,
    cache_identity: str,
) -> tuple[dict[str, Any] | None, str | None]:
    path = checkpoint_path(run_dir, sample_id)
    if not path.is_file():
        return None, "checkpoint does not exist"
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"checkpoint is unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(envelope, dict):
        return None, "checkpoint root is not a JSON object"
    if envelope.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None, "checkpoint schema version is unsupported"
    if envelope.get("sample_id") != sample_id:
        return None, "checkpoint sample identity does not match"
    if envelope.get("cache_identity") != cache_identity:
        return None, "checkpoint cache identity does not match current inputs/configuration"
    if envelope.get("status") != "completed":
        reasons = envelope.get("failure_reasons", [])
        detail = "; ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
        return None, f"previous checkpoint is not completed{': ' + detail if detail else ''}"
    results = envelope.get("results")
    if not isinstance(results, dict):
        return None, "checkpoint results are missing or invalid"
    for key in SAMPLE_RESULT_LISTS:
        rows = results.get(key)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return None, f"checkpoint result collection {key!r} is invalid"
    if len(results["validation"]) != 1 or len(results["interaction_rows"]) != 1:
        return None, "completed checkpoint must have one validation and interaction result"
    if results["validation"][0].get("status") not in ("valid", "invalid"):
        return None, "checkpoint validation result has no recognized status"
    if not isinstance(results.get("camera_indices"), list) or any(
        type(camera) is not int or camera < 0 for camera in results["camera_indices"]
    ):
        return None, "checkpoint camera indices are invalid"
    interaction = results["interaction_rows"][0]
    if interaction.get("sample_id") != sample_id:
        return None, "checkpoint interaction row belongs to a different sample"
    if not isinstance(interaction.get("status"), str):
        return None, "checkpoint interaction row has no status"
    for key in SAMPLE_RESULT_LISTS:
        if key in ("validation", "unavailable"):
            continue
        if any(row.get("sample_id") != sample_id for row in results[key]):
            return None, f"checkpoint collection {key!r} contains a different sample"
    return results, None
