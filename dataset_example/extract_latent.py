import argparse
import json
import os
import time
from pathlib import Path

import mediapy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from tqdm.auto import tqdm
from torch.utils.data import Dataset


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


class EncodeLatentDataset(Dataset):
    def __init__(
        self, old_path, new_path, svd_path, device,
        size=(192, 320), rgb_skip=3, split="all",
    ):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.device = device

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
    def _atomic_save_json(data, destination):
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

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        traj_id = int(traj_data["episode_index"])
        data_type = "val" if traj_id % 100 == 99 else "train"

        if self._episode_outputs_exist(traj_id, data_type):
            return {"status": "skipped", "episode_id": traj_id}

        try:
            instruction = traj_data["tasks"][0]
            chunk_id = traj_id // 1000
            parquet_path = (
                Path(self.old_path) / "data" / f"chunk-{chunk_id:03d}"
                / f"episode_{traj_id:06d}.parquet"
            )
            df = pd.read_parquet(parquet_path)
            length = len(df["observation.state.cartesian_position"])

            columns = [
                "observation.state.cartesian_position",
                "observation.state.joint_position",
                "observation.state.gripper_position",
                "action.cartesian_position",
                "action.joint_position",
                "action.gripper_position",
                "action.joint_velocity",
            ]
            traj_info = {key: [v.tolist() if hasattr(v, "tolist") else v for v in df[key]] for key in columns}
            traj_info["success"] = df["is_episode_successful"].iloc[0]

            camera_names = [
                "observation.images.exterior_1_left",
                "observation.images.exterior_2_left",
                "observation.images.wrist_left",
            ]
            video_paths = [
                Path(self.old_path) / "videos" / f"chunk-{chunk_id:03d}"
                / camera / f"episode_{traj_id:06d}.mp4"
                for camera in camera_names
            ]

            self.process_traj(
                video_paths, traj_info, instruction, self.new_path, traj_id,
                data_type, self.size, self.skip, self.device,
            )
            return {"status": "processed", "episode_id": traj_id, "raw_frames": length}
        except Exception as exc:
            return {"status": "failed", "episode_id": traj_id, "error": repr(exc)}

    def process_traj(
        self, video_paths, traj_info, instruction, save_root, traj_id=0,
        data_type="val", size=(192, 320), rgb_skip=3, device="cuda",
    ):
        last_frame_count = None
        for camera_id, video_path in enumerate(video_paths):
            video = mediapy.read_video(str(video_path))
            frames = torch.as_tensor(video).permute(0, 3, 1, 2).float() / 255.0
            frames = frames * 2.0 - 1.0
            frames = frames[::rgb_skip]
            if len(frames) == 0:
                raise ValueError(f"No frames after rgb_skip={rgb_skip}: {video_path}")

            x = torch.nn.functional.interpolate(
                frames, size=size, mode="bilinear", align_corners=False
            )
            last_frame_count = len(x)
            resized_video = (
                ((x / 2.0 + 0.5).clamp(0, 1) * 255)
                .permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            )

            video_dir = Path(save_root) / "videos" / data_type / str(traj_id)
            latent_dir = Path(save_root) / "latent_videos" / data_type / str(traj_id)
            video_dir.mkdir(parents=True, exist_ok=True)
            latent_dir.mkdir(parents=True, exist_ok=True)

            final_video = video_dir / f"{camera_id}.mp4"
            final_latent = latent_dir / f"{camera_id}.pt"
            temp_video = video_dir / f".{camera_id}.tmp.mp4"
            temp_latent = latent_dir / f".{camera_id}.tmp.pt"

            try:
                mediapy.write_video(str(temp_video), resized_video, fps=5)
                x = x.to(device)
                latent_batches = []
                with torch.inference_mode():
                    for start in range(0, len(x), 128):
                        latent = (
                            self.vae.encode(x[start:start + 128])
                            .latent_dist.sample()
                            .mul_(self.vae.config.scaling_factor)
                            .cpu()
                        )
                        latent_batches.append(latent)

                if not latent_batches:
                    raise RuntimeError(f"No latent batches generated: {video_path}")
                torch.save(torch.cat(latent_batches, dim=0), temp_latent)

                for temp_file in (temp_video, temp_latent):
                    if not temp_file.is_file() or temp_file.stat().st_size == 0:
                        raise IOError(f"Empty temporary output: {temp_file}")

                os.replace(temp_video, final_video)
                os.replace(temp_latent, final_latent)
            finally:
                for temp_file in (temp_video, temp_latent):
                    if temp_file.exists():
                        temp_file.unlink()

        pose = np.asarray(traj_info["observation.state.cartesian_position"])
        gripper = np.asarray(traj_info["observation.state.gripper_position"])[:, None]
        states = np.concatenate((pose, gripper), axis=-1)[::rgb_skip].tolist()

        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info["success"]),
            "video_length": last_frame_count,
            "state_length": len(states),
            "raw_length": len(traj_info["observation.state.cartesian_position"]),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/{i}.mp4"}
                for i in range(3)
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/{i}.pt"}
                for i in range(3)
            ],
            "states": states,
        }
        for key in (
            "observation.state.cartesian_position",
            "observation.state.joint_position",
            "observation.state.gripper_position",
            "action.cartesian_position",
            "action.joint_position",
            "action.gripper_position",
            "action.joint_velocity",
        ):
            info[key] = traj_info[key]

        annotation_path = (
            Path(save_root) / "annotation" / data_type / f"{traj_id}.json"
        )
        self._atomic_save_json(info, annotation_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_hf_path", type=str, default="/cephfs/shared/droid_hf/droid_1.0.1")
    parser.add_argument("--droid_output_path", type=str, default="dataset_example/droid_subset")
    parser.add_argument("--svd_path", type=str, default="/cephfs/shared/llm/stable-video-diffusion-img2vid")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--log_every", type=int, default=25, help="Update progress display every N episodes")
    args = parser.parse_args()

    accelerator = Accelerator()
    if accelerator.num_processes > 1:
        raise RuntimeError(
            "This script currently expects one process because each episode writes to shared output paths. "
            "Run with: accelerate launch --num_processes 1 ..."
        )

    dataset = EncodeLatentDataset(
        args.droid_hf_path, args.droid_output_path, args.svd_path,
        accelerator.device, size=(192, 320), rgb_skip=3, split=args.split,
    )
    total = min(len(dataset), 5) if args.debug else len(dataset)
    output_root = Path(args.droid_output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    error_log = output_root / f"latent_extraction_errors_{args.split}.jsonl"

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    start_time = time.time()
    start_bytes = directory_size(output_root)

    print(f"Device: {accelerator.device}", flush=True)
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())} | "
            f"CUDA memory allocated: {format_bytes(torch.cuda.memory_allocated())}",
            flush=True,
        )
    print(f"Existing output size: {format_bytes(start_bytes)}", flush=True)
    print(f"Failure log: {error_log}", flush=True)

    progress = tqdm(range(total), total=total, desc="Extracting latents", unit="episode")
    with open(error_log, "a", buffering=1) as errors:
        for idx in progress:
            result = dataset[idx]
            status = result["status"]
            counts[status] += 1
            if status == "failed":
                errors.write(json.dumps(result) + "\n")

            completed = idx + 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed) / rate if rate > 0 else 0.0
            progress.set_postfix({
                "ok": counts["processed"],
                "skip": counts["skipped"],
                "fail": counts["failed"],
                "ep/min": f"{rate * 60:.2f}",
                "ETA": time.strftime("%H:%M:%S", time.gmtime(remaining)),
            })
            if args.log_every > 0 and completed % args.log_every == 0:
                tqdm.write(
                    f"[{completed}/{total}] processed={counts['processed']} "
                    f"skipped={counts['skipped']} failed={counts['failed']} "
                    f"elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
                )

    elapsed = time.time() - start_time
    final_bytes = directory_size(output_root)
    print("\n=== LATENT EXTRACTION SUMMARY ===")
    print(f"Split: {args.split}")
    print(f"Total attempted: {total:,}")
    print(f"Newly processed: {counts['processed']:,}")
    print(f"Already complete (skipped): {counts['skipped']:,}")
    print(f"Failed: {counts['failed']:,}")
    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Average speed: {total / elapsed * 60:.2f} episodes/min" if elapsed else "Average speed: n/a")
    print(f"Output directory size: {format_bytes(final_bytes)}")
    print(f"New output written this run: {format_bytes(max(0, final_bytes - start_bytes))}")
    print(f"Failure details: {error_log}")


if __name__ == "__main__":
    main()
