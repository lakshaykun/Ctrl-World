#!/usr/bin/env python3
"""Build a consolidated, replay-compatible Ctrl-World RGB validation benchmark.

Expected source layout:
  dataset_example/droid_subset/{annotation,videos}/val/...
  dataset_example/droid_new_setup/{annotation,videos}/val/...
  dataset_example/droid_new_setup_full/<task>/{annotation,videos}/val/...

Output layout:
  droid_benchmark_v1/
    annotation/val/<global_trajectory_id>.json
    videos/val/<global_trajectory_id>/{0,1,2}.mp4
    manifest.json
    dataset_metadata.json

Identical MP4 contents are stored once and hard-linked into each trajectory folder
(when supported), so every trajectory still exposes its own 0.mp4/1.mp4/2.mp4 paths.
The original source datasets are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAMERA_NAMES = {"0": "camera_0", "1": "camera_1", "2": "camera_2"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def discover_sources(dataset_root: Path) -> list[dict[str, Any]]:
    """Discover annotation/video pairs from the three known source families."""
    sources: list[dict[str, Any]] = []
    specs = [
        ("droid_subset", dataset_root / "droid_subset", "droid_subset", None),
        ("droid_new_setup", dataset_root / "droid_new_setup", "droid_new_setup", None),
    ]
    for source_name, source_root, family, task in specs:
        ann_dir = source_root / "annotation" / "val"
        video_root = source_root / "videos" / "val"
        if not ann_dir.is_dir():
            print(f"[WARN] Missing annotation directory: {ann_dir}")
            continue
        for ann_path in sorted(ann_dir.glob("*.json")):
            episode_id = ann_path.stem
            sources.append({
                "source_dataset": source_name,
                "source_family": family,
                "task": task or source_name,
                "source_root": source_root,
                "annotation_path": ann_path,
                "video_root": video_root,
                "episode_id": episode_id,
            })

    full_root = dataset_root / "droid_new_setup_full"
    if full_root.is_dir():
        for task_dir in sorted(p for p in full_root.iterdir() if p.is_dir()):
            ann_dir = task_dir / "annotation" / "val"
            video_root = task_dir / "videos" / "val"
            if not ann_dir.is_dir():
                continue
            for ann_path in sorted(ann_dir.glob("*.json")):
                sources.append({
                    "source_dataset": "droid_new_setup_full",
                    "source_family": "droid_new_setup_full",
                    "task": task_dir.name,
                    "source_root": task_dir,
                    "annotation_path": ann_path,
                    "video_root": video_root,
                    "episode_id": ann_path.stem,
                })
    return sources


def source_video_path(source: dict[str, Any], video_entry: dict[str, Any]) -> Path:
    rel = Path(video_entry["video_path"])
    # In the supplied annotations, video_path is relative to the dataset/task root.
    candidate = source["source_root"] / rel
    if candidate.is_file():
        return candidate
    # Fallback for annotations whose path is already relative to videos/val.
    candidate = source["video_root"] / rel
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot resolve video_path={video_entry.get('video_path')!r} "
        f"for annotation {source['annotation_path']}. Tried both dataset root "
        f"and videos/val root."
    )


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "already_exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_root}\n"
                "Choose another --output path or pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    sources = discover_sources(dataset_root)
    if not sources:
        raise SystemExit("No validation annotations found in the expected dataset layout.")

    # Deterministic global IDs. Include source family and task in manifest, not filename.
    sources.sort(key=lambda x: (
        x["source_dataset"], x["task"], x["episode_id"]
    ))

    ann_out = output_root / "annotation" / "val"
    video_out = output_root / "videos" / "val"
    dedup_store = output_root / "_video_store" / "sha256"
    ann_out.mkdir(parents=True, exist_ok=True)
    video_out.mkdir(parents=True, exist_ok=True)
    dedup_store.mkdir(parents=True, exist_ok=True)

    hash_to_store: dict[str, Path] = {}
    hash_occurrences: dict[str, list[str]] = defaultdict(list)
    manifest_trajectories: list[dict[str, Any]] = []
    video_count = 0
    hardlink_count = 0
    copy_count = 0
    warnings: list[str] = []

    for index, source in enumerate(sources, start=1):
        trajectory_id = f"trajectory_{index:06d}"
        src_ann_path: Path = source["annotation_path"]
        annotation = json.loads(src_ann_path.read_text())
        video_entries = annotation.get("videos")
        if not isinstance(video_entries, list) or len(video_entries) != 3:
            raise ValueError(
                f"Expected exactly 3 camera entries in {src_ann_path}; "
                f"found {len(video_entries) if isinstance(video_entries, list) else 'no videos list'}."
            )

        new_video_entries = []
        camera_records = []
        seen_camera_ids = set()
        for fallback_index, entry in enumerate(video_entries):
            if not isinstance(entry, dict) or "video_path" not in entry:
                raise ValueError(f"Malformed video entry in {src_ann_path}: {entry!r}")
            src_video = source_video_path(source, entry)
            camera_id = src_video.stem
            if camera_id not in {"0", "1", "2"}:
                camera_id = str(fallback_index)
            if camera_id in seen_camera_ids:
                raise ValueError(f"Duplicate camera ID {camera_id} in {src_ann_path}")
            seen_camera_ids.add(camera_id)

            digest = sha256_file(src_video)
            canonical = hash_to_store.get(digest)
            if canonical is None:
                canonical = dedup_store / digest[:2] / f"{digest}.mp4"
                canonical.parent.mkdir(parents=True, exist_ok=True)
                if not canonical.exists():
                    shutil.copy2(src_video, canonical)
                hash_to_store[digest] = canonical

            relative_video = Path("videos") / "val" / trajectory_id / f"{camera_id}.mp4"
            destination = output_root / relative_video
            method = link_or_copy(canonical, destination)
            hardlink_count += method == "hardlink"
            copy_count += method == "copy"
            video_count += 1
            hash_occurrences[digest].append(f"{trajectory_id}/{camera_id}.mp4")

            # Preserve any extra per-camera annotation metadata, replacing only video_path.
            new_entry = dict(entry)
            new_entry["video_path"] = relative_video.as_posix()
            new_video_entries.append(new_entry)
            camera_records.append({
                "camera_id": camera_id,
                "source_video": str(src_video.relative_to(dataset_root)),
                "benchmark_video": relative_video.as_posix(),
                "sha256": digest,
                "storage_method": method,
                "bytes": src_video.stat().st_size,
            })

        if seen_camera_ids != {"0", "1", "2"}:
            raise ValueError(
                f"Expected camera files 0.mp4, 1.mp4, 2.mp4 in {src_ann_path}; "
                f"found {sorted(seen_camera_ids)}"
            )

        annotation["videos"] = new_video_entries
        # This is an RGB replay benchmark. Existing replay code VAE-encodes RGB videos;
        # do not leave latent paths pointing to files that were not copied.
        annotation.pop("latent_videos", None)
        annotation["benchmark_trajectory_id"] = trajectory_id
        annotation["benchmark_source"] = {
            "dataset": source["source_dataset"],
            "task": source["task"],
            "original_episode_id": str(source["episode_id"]),
            "original_annotation": str(src_ann_path.relative_to(dataset_root)),
        }

        out_ann_path = ann_out / f"{trajectory_id}.json"
        write_json(out_ann_path, annotation)

        manifest_trajectories.append({
            "trajectory_id": trajectory_id,
            "source_dataset": source["source_dataset"],
            "task": source["task"],
            "source_episode_id": str(source["episode_id"]),
            "source_annotation": str(src_ann_path.relative_to(dataset_root)),
            "benchmark_annotation": str(out_ann_path.relative_to(output_root)),
            "instruction": (annotation.get("texts") or [None])[0],
            "success": annotation.get("success"),
            "video_length": annotation.get("video_length"),
            "num_camera_views": len(camera_records),
            "cameras": sorted(camera_records, key=lambda x: x["camera_id"]),
        })

    duplicate_hashes = {
        digest: locations
        for digest, locations in hash_occurrences.items()
        if len(locations) > 1
    }
    task_counts = Counter(x["task"] for x in manifest_trajectories)
    source_counts = Counter(x["source_dataset"] for x in manifest_trajectories)
    unique_video_count = len(hash_to_store)

    manifest = {
        "benchmark_name": "droid_benchmark_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_root": str(dataset_root),
        "split": "val",
        "trajectory_count": len(manifest_trajectories),
        "video_reference_count": video_count,
        "unique_video_content_count": unique_video_count,
        "duplicate_video_content_count": video_count - unique_video_count,
        "duplicate_sha256_groups": duplicate_hashes,
        "trajectories": manifest_trajectories,
    }
    metadata = {
        "benchmark_name": "droid_benchmark_v1",
        "purpose": "Consolidated RGB validation benchmark for Ctrl-World replay evaluation",
        "split": "validation only",
        "trajectory_count": len(manifest_trajectories),
        "video_reference_count": video_count,
        "unique_video_content_count": unique_video_count,
        "duplicate_video_reference_count": video_count - unique_video_count,
        "source_trajectory_counts": dict(source_counts),
        "task_trajectory_counts": dict(task_counts),
        "camera_views_per_trajectory": 3,
        "video_storage": "SHA-256 content-addressed store with hardlinks where supported; fallback copies otherwise",
        "annotation_notes": [
            "Original annotation fields are retained except latent_videos, which is omitted because this is an RGB replay benchmark and latent files are not consolidated.",
            "video_path values are rewritten relative to the benchmark root, as required by Agent.get_traj_info().",
            "Original success labels and instructions are preserved; no trajectory is filtered by success.",
        ],
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "hardlink_count": hardlink_count,
        "copy_fallback_count": copy_count,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "dataset_metadata.json", metadata)

    # Validate every output annotation and every referenced video path.
    validation_errors = []
    for trajectory in manifest_trajectories:
        ann_path = output_root / trajectory["benchmark_annotation"]
        ann = json.loads(ann_path.read_text())
        if len(ann.get("videos", [])) != 3:
            validation_errors.append(f"{ann_path}: expected 3 video references")
        for item in ann.get("videos", []):
            target = output_root / item["video_path"]
            if not target.is_file():
                validation_errors.append(f"Missing referenced video: {target}")

    report = {
        "passed": not validation_errors,
        "trajectory_count": len(manifest_trajectories),
        "video_reference_count": video_count,
        "unique_video_content_count": unique_video_count,
        "duplicate_video_reference_count": video_count - unique_video_count,
        "validation_errors": validation_errors,
    }
    write_json(output_root / "validation_report.json", report)

    print("\n========== BENCHMARK BUILD SUMMARY ==========")
    print(f"Output:                    {output_root}")
    print(f"Trajectories:              {len(manifest_trajectories)}")
    print(f"Video references:          {video_count}")
    print(f"Unique video contents:     {unique_video_count}")
    print(f"Duplicate references:      {video_count - unique_video_count}")
    print(f"Hardlinks created:         {hardlink_count}")
    print(f"Copy fallbacks:            {copy_count}")
    print("Source counts:")
    for name, count in sorted(source_counts.items()):
        print(f"  {name}: {count}")
    print("Task counts:")
    for name, count in sorted(task_counts.items()):
        print(f"  {name}: {count}")
    print(f"Validation:                {'PASSED' if report['passed'] else 'FAILED'}")
    if validation_errors:
        for error in validation_errors:
            print("  ERROR:", error)
        raise SystemExit(2)
    print("=============================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_example"))
    parser.add_argument("--output", type=Path, default=Path("droid_benchmark_v1"))
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output directory first")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
