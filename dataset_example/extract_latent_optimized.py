#!/usr/bin/env python3
"""Memory- and throughput-optimized latent extraction for Ctrl-World.

Data-quality contract:
- Keeps the original episode split, frame sampling (frames[::rgb_skip]), resize
  settings, normalization order, VAE sampling (latent_dist.sample()), scaling,
  video FPS, annotation fields, and output paths.
- Does NOT enable mixed precision by default or change latent sampling.
- Completed episodes are detected by the original six-output + annotation check.
- If an episode was interrupted, already completed camera video/latent pairs are
  reused; incomplete temporary files are ignored and cleaned up.
"""
import argparse
import json
import os
import time
from pathlib import Path

import mediapy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from tqdm.auto import tqdm
from torch.utils.data import Dataset


STATE_ACTION_COLUMNS = [
    "observation.state.cartesian_position",
    "observation.state.joint_position",
    "observation.state.gripper_position",
    "action.cartesian_position",
    "action.joint_position",
    "action.gripper_position",
    "action.joint_velocity",
]
CAMERA_NAMES = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]



def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def directory_size(path):
    total = 0
    path = Path(path)
    if not path.exists():
        return total
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def atomic_save_json(data, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class EncodeLatentDataset(Dataset):
    def __init__(self, old_path, new_path, svd_path, device,
                 size=(192, 320), rgb_skip=3, split="all", vae_batch_size=32,
                 resize_batch_size=64, profile=False):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.device = device
        self.vae_batch_size = vae_batch_size
        self.resize_batch_size = resize_batch_size
        self.profile = profile

        print(f"Loading SVD VAE from: {svd_path}", flush=True)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path, subfolder="vae"
        ).to(device)
        self.vae.eval()

        meta_file = Path(old_path) / "meta" / "episodes.jsonl"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Missing DROID episode metadata: {meta_file}")
        with open(meta_file, "r") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        if split not in ("all", "train", "val"):
            raise ValueError(f"Invalid split: {split}")
        if split == "val":
            self.data = [x for x in self.data if int(x["episode_index"]) % 100 == 99]
        elif split == "train":
            self.data = [x for x in self.data if int(x["episode_index"]) % 100 != 99]
        self.split = split
        print(f"Selected {len(self.data):,} episodes for split={split}", flush=True)

    def __len__(self):
        return len(self.data)

    def _episode_outputs_exist(self, traj_id, data_type):
        root = Path(self.new_path)
        paths = [root / "annotation" / data_type / f"{traj_id}.json"]
        for camera_id in range(3):
            paths.extend([
                root / "videos" / data_type / str(traj_id) / f"{camera_id}.mp4",
                root / "latent_videos" / data_type / str(traj_id) / f"{camera_id}.pt",
            ])
        return all(p.is_file() and p.stat().st_size > 0 for p in paths)

    @staticmethod
    def _camera_outputs_exist(video_path, latent_path):
        return (video_path.is_file() and video_path.stat().st_size > 0 and
                latent_path.is_file() and latent_path.stat().st_size > 0)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        traj_id = int(traj_data["episode_index"])
        data_type = "val" if traj_id % 100 == 99 else "train"
        if self._episode_outputs_exist(traj_id, data_type):
            return {"status": "skipped", "episode_id": traj_id}

        try:
            instruction = traj_data["tasks"][0]
            chunk_id = traj_id // 1000
            parquet_path = (Path(self.old_path) / "data" / f"chunk-{chunk_id:03d}" /
                           f"episode_{traj_id:06d}.parquet")
            # Read only the columns actually used; keeps the DataFrame smaller.
            read_columns = STATE_ACTION_COLUMNS + ["is_episode_successful"]
            df = pd.read_parquet(parquet_path, columns=read_columns)
            length = len(df[STATE_ACTION_COLUMNS[0]])
            # Keep source values/types as close to the original path as possible.
            traj_info = {
                key: [v.tolist() if hasattr(v, "tolist") else v for v in df[key]]
                for key in STATE_ACTION_COLUMNS
            }
            traj_info["success"] = df["is_episode_successful"].iloc[0]
            del df

            video_paths = [
                Path(self.old_path) / "videos" / f"chunk-{chunk_id:03d}" /
                camera / f"episode_{traj_id:06d}.mp4"
                for camera in CAMERA_NAMES
            ]
            timing = self.process_traj(
                video_paths, traj_info, instruction, self.new_path, traj_id,
                data_type, self.size, self.skip, self.device,
            )
            return {"status": "processed", "episode_id": traj_id,
                    "raw_frames": length, "timing": timing}
        except Exception as exc:
            return {"status": "failed", "episode_id": traj_id, "error": repr(exc)}

    def process_traj(self, video_paths, traj_info, instruction, save_root, traj_id=0,
                     data_type="val", size=(192, 320), rgb_skip=3, device="cuda"):
        last_frame_count = None
        timings = {"decode": 0.0, "preprocess_and_encode": 0.0, "video_write": 0.0,
                   "latent_save": 0.0, "camera_skipped": 0}
        root = Path(save_root)
        video_dir = root / "videos" / data_type / str(traj_id)
        latent_dir = root / "latent_videos" / data_type / str(traj_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        latent_dir.mkdir(parents=True, exist_ok=True)

        for camera_id, video_path in enumerate(video_paths):
            final_video = video_dir / f"{camera_id}.mp4"
            final_latent = latent_dir / f"{camera_id}.pt"
            temp_video = video_dir / f".{camera_id}.tmp.mp4"
            temp_latent = latent_dir / f".{camera_id}.tmp.pt"
            # Per-camera checkpointing: completed camera outputs survive a restart.
            if self._camera_outputs_exist(final_video, final_latent):
                timings["camera_skipped"] += 1
                # We still need frame count for state/video alignment. Read metadata
                # from the saved latent tensor (frame dimension) instead of decoding.
                latent_shape = torch.load(final_latent, map_location="cpu", weights_only=True).shape
                last_frame_count = int(latent_shape[0])
                continue

            for tmp in (temp_video, temp_latent):
                if tmp.exists():
                    tmp.unlink()

            t0 = time.perf_counter()
            # Keep the decoder behavior of the original script to avoid changing
            # decoded pixels/quality. The decoded uint8 array is the only full-video
            # resident representation; avoid full-video float32 copies.
            video_u8 = mediapy.read_video(str(video_path))
            timings["decode"] += time.perf_counter() - t0
            if len(video_u8) == 0:
                raise ValueError(f"Empty video: {video_path}")

            sampled_u8 = video_u8[::rgb_skip]
            last_frame_count = len(sampled_u8)
            if last_frame_count == 0:
                raise ValueError(f"No frames after rgb_skip={rgb_skip}: {video_path}")

            t0 = time.perf_counter()
            resized_chunks = []
            latent_tensor = None
            latent_cursor = 0
            with torch.inference_mode():
                for start in range(0, last_frame_count, self.resize_batch_size):
                    chunk_u8 = sampled_u8[start:start + self.resize_batch_size]
                    # Match original conversion and interpolation sequence.
                    frames = torch.from_numpy(np.ascontiguousarray(chunk_u8)).permute(0, 3, 1, 2).float() / 255.0
                    frames = frames * 2.0 - 1.0
                    resized = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
                    # Preserve the exact original uint8 video conversion formula.
                    video_chunk = (((resized / 2.0 + 0.5).clamp(0, 1) * 255)
                                  .permute(0, 2, 3, 1).numpy().astype(np.uint8))
                    resized_chunks.append(video_chunk)

                    # Transfer only this VAE batch/chunk to GPU, never the whole episode.
                    for bstart in range(0, len(resized), self.vae_batch_size):
                        batch_cpu = resized[bstart:bstart + self.vae_batch_size]
                        batch_gpu = batch_cpu.to(device, non_blocking=False)
                        latent = (self.vae.encode(batch_gpu).latent_dist.sample()
                                  .mul_(self.vae.config.scaling_factor).cpu())
                        if latent_tensor is None:
                            latent_tensor = torch.empty(
                                (last_frame_count, *latent.shape[1:]), dtype=latent.dtype,
                                device="cpu")
                        latent_tensor[latent_cursor:latent_cursor + len(latent)] = latent
                        latent_cursor += len(latent)
                        del batch_gpu, latent, batch_cpu
                    del frames, resized, video_chunk
            timings["preprocess_and_encode"] += time.perf_counter() - t0
            del video_u8, sampled_u8

            if latent_tensor is None or latent_cursor != last_frame_count:
                raise RuntimeError(f"Latent frame count mismatch for {video_path}: "
                                   f"{latent_cursor}/{last_frame_count}")

            t0 = time.perf_counter()
            # Preserve original mediapy writer and fps; output frames are same uint8 values.
            resized_video = np.concatenate(resized_chunks, axis=0)
            mediapy.write_video(str(temp_video), resized_video, fps=5)
            timings["video_write"] += time.perf_counter() - t0
            del resized_chunks, resized_video

            t0 = time.perf_counter()
            torch.save(latent_tensor, temp_latent)
            timings["latent_save"] += time.perf_counter() - t0
            del latent_tensor

            for temp_file in (temp_video, temp_latent):
                if not temp_file.is_file() or temp_file.stat().st_size == 0:
                    raise IOError(f"Empty temporary output: {temp_file}")
            os.replace(temp_video, final_video)
            os.replace(temp_latent, final_latent)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # States and joints: same construction, float32 conversion, and sampling.
        pose = np.asarray(traj_info["observation.state.cartesian_position"], dtype=np.float32)
        gripper = np.asarray(traj_info["observation.state.gripper_position"], dtype=np.float32).reshape(-1, 1)
        states = np.concatenate((pose, gripper), axis=-1)[::rgb_skip]
        joint_position = np.asarray(traj_info["observation.state.joint_position"], dtype=np.float32)
        joints = np.concatenate((joint_position, gripper), axis=-1)[::rgb_skip]

        if len(states) != last_frame_count:
            raise ValueError(f"State/video length mismatch for episode {traj_id}: states={len(states)}, video={last_frame_count}")
        if len(joints) != last_frame_count:
            raise ValueError(f"Joint/video length mismatch for episode {traj_id}: joints={len(joints)}, video={last_frame_count}")
        if joints.shape[1] != 8:
            raise ValueError(f"Expected 8D joints for episode {traj_id}, got shape {joints.shape}")

        info = {
            "texts": [instruction], "episode_id": traj_id,
            "success": int(traj_info["success"]), "video_length": last_frame_count,
            "state_length": len(states),
            "raw_length": len(traj_info["observation.state.cartesian_position"]),
            "videos": [{"video_path": f"videos/{data_type}/{traj_id}/{i}.mp4"} for i in range(3)],
            "latent_videos": [{"latent_video_path": f"latent_videos/{data_type}/{traj_id}/{i}.pt"} for i in range(3)],
            "states": states.tolist(), "joints": joints.tolist(),
        }
        for key in STATE_ACTION_COLUMNS:
            info[key] = traj_info[key]
        annotation_path = root / "annotation" / data_type / f"{traj_id}.json"
        atomic_save_json(info, annotation_path)
        return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--droid_hf_path", type=str, default="/cephfs/shared/droid_hf/droid_1.0.1")
    parser.add_argument("--droid_output_path", type=str, default="dataset_example/droid_subset")
    parser.add_argument("--svd_path", type=str, default="/cephfs/shared/llm/stable-video-diffusion-img2vid")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--vae_batch_size", type=int, default=96, help="VAE batch size (original script used 96); tune only if needed")
    parser.add_argument("--resize_batch_size", type=int, default=96, help="CPU resize/preprocess chunk size")
    parser.add_argument("--profile", action="store_true", help="Print per-episode stage timings")
    args = parser.parse_args()

    if args.vae_batch_size < 1 or args.resize_batch_size < 1:
        parser.error("Batch sizes must be >= 1")
    accelerator = Accelerator()
    if accelerator.num_processes > 1:
        raise RuntimeError("Single-process extraction is required because outputs are shared. "
                          "Use --num_processes 1.")

    dataset = EncodeLatentDataset(
        args.droid_hf_path, args.droid_output_path, args.svd_path, accelerator.device,
        size=(192, 320), rgb_skip=3, split=args.split,
        vae_batch_size=args.vae_batch_size, resize_batch_size=args.resize_batch_size,
        profile=args.profile,
    )
    total = min(len(dataset), 5) if args.debug else len(dataset)
    output_root = Path(args.droid_output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    error_log = output_root / f"latent_extraction_errors_{args.split}.jsonl"
    timing_log = output_root / f"latent_extraction_timing_{args.split}.jsonl"
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    start_time = time.time()
    start_bytes = directory_size(output_root)

    print(f"Device: {accelerator.device}", flush=True)
    print(f"VAE batch size: {args.vae_batch_size}; resize chunk: {args.resize_batch_size}", flush=True)
    print("Precision: unchanged FP32 path (no autocast)", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())} | "
              f"CUDA allocated: {format_bytes(torch.cuda.memory_allocated())}", flush=True)
    print(f"Existing output size: {format_bytes(start_bytes)}", flush=True)
    print(f"Failure log: {error_log}", flush=True)

    progress = tqdm(range(total), total=total, desc="Extracting latents", unit="episode")
    with open(error_log, "a", buffering=1) as errors, open(timing_log, "a", buffering=1) as timings:
        for idx in progress:
            result = dataset[idx]
            status = result["status"]
            counts[status] += 1
            if status == "failed":
                errors.write(json.dumps(result) + "\n")
            elif status == "processed" and args.profile:
                timings.write(json.dumps(result) + "\n")

            completed = idx + 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed) / rate if rate > 0 else 0.0
            progress.set_postfix({"ok": counts["processed"], "skip": counts["skipped"],
                                 "fail": counts["failed"], "ep/min": f"{rate * 60:.2f}",
                                 "ETA": time.strftime("%H:%M:%S", time.gmtime(remaining))})
            if args.log_every > 0 and completed % args.log_every == 0:
                tqdm.write(f"[{completed}/{total}] processed={counts['processed']} "
                           f"skipped={counts['skipped']} failed={counts['failed']} "
                           f"elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    elapsed = time.time() - start_time
    final_bytes = directory_size(output_root)
    print("\n=== LATENT EXTRACTION SUMMARY ===")
    print(f"Split: {args.split}; attempted: {total:,}")
    print(f"Newly processed: {counts['processed']:,}; already complete: {counts['skipped']:,}; failed: {counts['failed']:,}")
    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Average speed: {total / elapsed * 60:.2f} episodes/min" if elapsed else "Average speed: n/a")
    print(f"Output directory size: {format_bytes(final_bytes)}")
    print(f"New output written: {format_bytes(max(0, final_bytes - start_bytes))}")
    print(f"Failure details: {error_log}")
    if args.profile:
        print(f"Timing details: {timing_log}")


if __name__ == "__main__":
    main()
