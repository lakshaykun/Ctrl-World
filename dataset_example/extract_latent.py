import json
import os
import tempfile
from pathlib import Path

import mediapy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset


class EncodeLatentDataset(Dataset):
    def __init__(
        self,
        old_path,
        new_path,
        svd_path,
        device,
        size=(192, 320),
        rgb_skip=3,
        split="all",
    ):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.device = device

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path, subfolder="vae"
        ).to(device)
        self.vae.eval()

        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            return data

        self.data = load_json_file(f"{old_path}/meta/episodes.jsonl")

        if split != "all":
            if split not in ("train", "val"):
                raise ValueError(f"Invalid split: {split}")

            if split == "val":
                self.data = [
                    traj for traj in self.data
                    if traj["episode_index"] % 100 == 99
                ]
            else:
                self.data = [
                    traj for traj in self.data
                    if traj["episode_index"] % 100 != 99
                ]

        print(f"Selected {len(self.data)} episodes for split={split}")

    def __len__(self):
        return len(self.data)

    def _episode_outputs_exist(self, traj_id, data_type):
        """An episode is complete only when every expected output is present."""
        annotation_path = (
            Path(self.new_path) / "annotation" / data_type / f"{traj_id}.json"
        )
        output_paths = [annotation_path]

        for video_id in range(3):
            output_paths.append(
                Path(self.new_path) / "videos" / data_type
                / str(traj_id) / f"{video_id}.mp4"
            )
            output_paths.append(
                Path(self.new_path) / "latent_videos" / data_type
                / str(traj_id) / f"{video_id}.pt"
            )

        return all(path.is_file() and path.stat().st_size > 0 for path in output_paths)

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

        # Skip only if the full set of outputs exists. If a previous run stopped
        # halfway through an episode, that episode will be processed again.
        if self._episode_outputs_exist(traj_id, data_type):
            return 0

        try:
            instruction = traj_data["tasks"][0]
            chunk_id = int(traj_id / 1000)

            file_path = (
                f"{self.old_path}/data/chunk-{chunk_id:03d}/"
                f"episode_{traj_id:06d}.parquet"
            )
            df = pd.read_parquet(file_path)
            length = len(df["observation.state.cartesian_position"])

            obs_car = []
            obs_joint = []
            obs_gripper = []
            action_car = []
            action_joint = []
            action_gripper = []
            action_joint_vel = []

            for i in range(length):
                obs_car.append(df["observation.state.cartesian_position"][i].tolist())
                obs_joint.append(df["observation.state.joint_position"][i].tolist())
                obs_gripper.append(df["observation.state.gripper_position"][i].tolist())
                action_car.append(df["action.cartesian_position"][i].tolist())
                action_joint.append(df["action.joint_position"][i].tolist())
                action_gripper.append(df["action.gripper_position"][i].tolist())
                action_joint_vel.append(df["action.joint_velocity"][i].tolist())

            success = df["is_episode_successful"][0]

            video_paths = [
                (
                    f"{self.old_path}/videos/chunk-{chunk_id:03d}/"
                    f"observation.images.exterior_1_left/episode_{traj_id:06d}.mp4"
                ),
                (
                    f"{self.old_path}/videos/chunk-{chunk_id:03d}/"
                    f"observation.images.exterior_2_left/episode_{traj_id:06d}.mp4"
                ),
                (
                    f"{self.old_path}/videos/chunk-{chunk_id:03d}/"
                    f"observation.images.wrist_left/episode_{traj_id:06d}.mp4"
                ),
            ]

            traj_info = {
                "success": success,
                "observation.state.cartesian_position": obs_car,
                "observation.state.joint_position": obs_joint,
                "observation.state.gripper_position": obs_gripper,
                "action.cartesian_position": action_car,
                "action.joint_position": action_joint,
                "action.gripper_position": action_gripper,
                "action.joint_velocity": action_joint_vel,
            }

            self.process_traj(
                video_paths=video_paths,
                traj_info=traj_info,
                instruction=instruction,
                save_root=self.new_path,
                traj_id=traj_id,
                data_type=data_type,
                size=self.size,
                rgb_skip=self.skip,
                device=self.vae.device,
            )

        except Exception as exc:
            print(f"Error processing trajectory {traj_id}: {exc}")
            return 0

        return 0

    def process_traj(
        self,
        video_paths,
        traj_info,
        instruction,
        save_root,
        traj_id=0,
        data_type="val",
        size=(192, 320),
        rgb_skip=3,
        device="cuda",
    ):
        last_frame_count = None

        for video_id, video_path in enumerate(video_paths):
            # Read, downsample and resize the source video.
            video = mediapy.read_video(video_path)
            frames = torch.as_tensor(video).permute(0, 3, 1, 2).float() / 255.0
            frames = frames * 2.0 - 1.0
            frames = frames[::rgb_skip]

            if len(frames) == 0:
                raise ValueError(
                    f"Video {video_path} contains no frames after rgb_skip={rgb_skip}"
                )

            x = torch.nn.functional.interpolate(
                frames, size=size, mode="bilinear", align_corners=False
            )
            last_frame_count = len(x)

            resized_video = (
                ((x / 2.0 + 0.5).clamp(0, 1) * 255)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            video_dir = Path(save_root) / "videos" / data_type / str(traj_id)
            latent_dir = (
                Path(save_root) / "latent_videos" / data_type / str(traj_id)
            )
            video_dir.mkdir(parents=True, exist_ok=True)
            latent_dir.mkdir(parents=True, exist_ok=True)

            final_video_path = video_dir / f"{video_id}.mp4"
            final_latent_path = latent_dir / f"{video_id}.pt"

            # Keep the expected extension at the end so mediapy/torch handle
            # the temporary files with the correct format.
            temp_video_path = video_dir / f".{video_id}.tmp.mp4"
            temp_latent_path = latent_dir / f".{video_id}.tmp.pt"

            try:
                mediapy.write_video(str(temp_video_path), resized_video, fps=5)

                x = x.to(device)
                latent_batches = []
                with torch.no_grad():
                    batch_size = 64
                    for start in range(0, len(x), batch_size):
                        batch = x[start : start + batch_size]
                        latent = (
                            self.vae.encode(batch)
                            .latent_dist.sample()
                            .mul_(self.vae.config.scaling_factor)
                            .cpu()
                        )
                        latent_batches.append(latent)

                if not latent_batches:
                    raise RuntimeError(f"No latents generated for {video_path}")

                latents = torch.cat(latent_batches, dim=0)
                torch.save(latents, temp_latent_path)

                if (
                    not temp_video_path.is_file()
                    or temp_video_path.stat().st_size == 0
                    or not temp_latent_path.is_file()
                    or temp_latent_path.stat().st_size == 0
                ):
                    raise IOError(f"Temporary output is empty for camera {video_id}")

                # Each file becomes visible at its final path only after it has
                # been completely written.
                os.replace(temp_video_path, final_video_path)
                os.replace(temp_latent_path, final_latent_path)

            finally:
                if temp_video_path.exists():
                    temp_video_path.unlink()
                if temp_latent_path.exists():
                    temp_latent_path.unlink()

        # Align Cartesian pose and gripper state with the downsampled video.
        cartesian_pose = np.asarray(
            traj_info["observation.state.cartesian_position"]
        )
        cartesian_gripper = np.asarray(
            traj_info["observation.state.gripper_position"]
        )[:, None]
        cartesian_states = np.concatenate(
            (cartesian_pose, cartesian_gripper), axis=-1
        )[::rgb_skip].tolist()

        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info["success"]),
            "video_length": last_frame_count,
            "state_length": len(cartesian_states),
            "raw_length": len(
                traj_info["observation.state.cartesian_position"]
            ),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/2.mp4"},
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/2.pt"},
            ],
            "states": cartesian_states,
            "observation.state.cartesian_position": traj_info[
                "observation.state.cartesian_position"
            ],
            "observation.state.joint_position": traj_info[
                "observation.state.joint_position"
            ],
            "observation.state.gripper_position": traj_info[
                "observation.state.gripper_position"
            ],
            "action.cartesian_position": traj_info["action.cartesian_position"],
            "action.joint_position": traj_info["action.joint_position"],
            "action.gripper_position": traj_info["action.gripper_position"],
            "action.joint_velocity": traj_info["action.joint_velocity"],
        }

        annotation_path = (
            Path(save_root) / "annotation" / data_type / f"{traj_id}.json"
        )
        # Write the annotation last. Its presence, together with all six media
        # artifacts, is the completion marker used by _episode_outputs_exist().
        self._atomic_save_json(info, annotation_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--droid_hf_path",
        type=str,
        default="/cephfs/shared/droid_hf/droid_1.0.1",
    )
    parser.add_argument(
        "--droid_output_path",
        type=str,
        default="dataset_example/droid_subset",
    )
    parser.add_argument(
        "--svd_path",
        type=str,
        default="/cephfs/shared/llm/stable-video-diffusion-img2vid",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Which episodes to process",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        old_path=args.droid_hf_path,
        new_path=args.droid_output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=(192, 320),
        rgb_skip=3,
        split=args.split,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    data_loader = accelerator.prepare_data_loader(data_loader)

    for idx, _ in enumerate(data_loader):
        if args.debug and idx >= 5:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Processed {idx}/{len(dataset)} samples")


# Example:
# accelerate launch dataset_example/extract_latent.py \
#   --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 \
#   --droid_output_path dataset_example/droid_subset \
#   --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid \
#   --split val
