import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def load_and_process_ann_file(
    data_root, ann_file, sequence_interval=1, start_interval=1, sequence_length=8
):
    samples = []
    try:
        with open(Path(data_root) / ann_file, "r") as f:
            ann = json.load(f)
        n_frames = int(ann["video_length"])
        episode_id = ann["episode_id"]
        states = np.asarray(ann["states"])
    except Exception as exc:
        return samples, f"{ann_file}: {type(exc).__name__}: {exc}"

    traj_len = int(sequence_length * sequence_interval)
    end_idx = max(1, n_frames - int(traj_len * 0.5))
    for start_frame in range(0, end_idx, start_interval):
        samples.append({
            "episode_id": episode_id,
            "frame_ids": [start_frame],
            "states": states[start_frame:start_frame + 1],
        })
    return samples, None


def init_anns(dataset_root, data_dir):
    final_path = Path(dataset_root) / data_dir
    if not final_path.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {final_path}")
    return [
        str(Path(data_dir) / name)
        for name in sorted(os.listdir(final_path))
        if name.endswith(".json")
    ]


def init_sequences(
    data_root, ann_files, sequence_interval, start_interval,
    sequence_length, workers=32,
):
    samples = []
    failures = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                load_and_process_ann_file, data_root, ann_file,
                sequence_interval, start_interval, sequence_length,
            ): ann_file
            for ann_file in ann_files
        }
        with tqdm(total=len(futures), desc="Reading annotations", unit="traj") as pbar:
            for future in as_completed(futures):
                ann_samples, error = future.result()
                samples.extend(ann_samples)
                if error:
                    failures.append(error)
                elapsed = time.time() - start
                done = pbar.n + 1
                rate = done / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "samples": len(samples),
                    "bad": len(failures),
                    "traj/min": f"{rate * 60:.1f}",
                    "ETA": time.strftime(
                        "%H:%M:%S",
                        time.gmtime((len(futures) - done) / rate if rate else 0),
                    ),
                })
                pbar.update(1)
    return samples, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_output_path", type=str, default="dataset_example/droid_subset")
    parser.add_argument("--dataset_name", type=str, default="droid_subset")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    sequence_length = 8
    sequence_interval = 1
    start_interval = 1
    splits = ["val", "train"] if args.split == "all" else [args.split]
    output_dir = Path("dataset_meta_info") / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    print(f"Dataset root: {args.droid_output_path}")
    print(f"Metadata output: {output_dir}")
    print(f"Worker threads: {args.workers} | Shuffle seed: {args.seed}")

    for data_type in splits:
        split_start = time.time()
        ann_dir = f"annotation/{data_type}"
        ann_files = init_anns(args.droid_output_path, ann_dir)
        if args.debug:
            ann_files = ann_files[:5]

        print(f"\n=== Building {data_type.upper()} metadata ===")
        print(f"Annotation files found: {len(ann_files):,}")

        samples, failures = init_sequences(
            args.droid_output_path, ann_files,
            sequence_interval, start_interval, sequence_length,
            workers=args.workers,
        )

        # Preserve the original metadata format: states are removed from samples.
        for sample in samples:
            sample.pop("states", None)

        random.Random(args.seed).shuffle(samples)
        output_path = output_dir / f"{data_type}_sample.json"
        temp_path = output_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(samples, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        elapsed = time.time() - split_start
        print(f"\n--- {data_type.upper()} SUMMARY ---")
        print(f"Trajectories discovered: {len(ann_files):,}")
        print(f"Samples generated: {len(samples):,}")
        print(f"Annotation failures: {len(failures):,}")
        print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        print(f"Average throughput: {len(ann_files) / elapsed * 60:.2f} trajectories/min" if elapsed else "Average throughput: n/a")
        print(f"Saved: {output_path}")

        if failures:
            failure_path = output_dir / f"{data_type}_metadata_errors.txt"
            failure_path.write_text("\n".join(failures) + "\n")
            print(f"Failure details: {failure_path}")

    total_elapsed = time.time() - overall_start
    print("\n=== METADATA GENERATION COMPLETE ===")
    print(f"Splits generated: {', '.join(splits)}")
    print(f"Total elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
