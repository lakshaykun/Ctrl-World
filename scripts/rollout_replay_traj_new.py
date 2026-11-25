import os
import sys
import json
import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import einops
import mediapy
from accelerate import Accelerator
from decord import VideoReader, cpu

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.dynamic_memory import select_dynamic_memories


def _decode_one_camera(video_path, length, frames_ids):
    """Decode only the required frames from one camera stream."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)

    try:
        frames = vr.get_batch(frames_ids.tolist()).asnumpy()
    except AttributeError:
        frames = vr.get_batch(frames_ids.tolist()).numpy()

    return frames


def _atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_npz_save(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _compose_video_strip(ground_truth, prediction):
    # Keep the original layout: GT beside prediction for each camera,
    # then stack the three camera strips vertically.
    camera_strips = np.concatenate([ground_truth, prediction], axis=-3)
    return np.concatenate([camera for camera in camera_strips], axis=-2).astype(np.uint8)


def _rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class agent:
    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        self.model = CrtlWorld(args)
        print("Loading Ctrl-World checkpoint using memory mapping...")
        state_dict = torch.load(
            args.val_model_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict

        print("Moving model to GPU and converting dtype...")
        self.model.to(device=self.accelerator.device, dtype=self.dtype)
        self.model.eval()
        print("load world model success")

        print("\n========== MEMORY CHECK ==========")
        print("GPU allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
        print("GPU reserved:", round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")
        print("Model dtype:", next(self.model.parameters()).dtype)
        print("Model device:", next(self.model.parameters()).device)
        print("=================================\n")

        with open(args.data_stat_path, "r") as f:
            data_stat = json.load(f)
        self.state_p01 = np.array(data_stat["state_01"])[None, :]
        self.state_p99 = np.array(data_stat["state_99"])[None, :]

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, trajectory_id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{trajectory_id}.json"

        with open(annotation_path) as f:
            anno = json.load(f)
        length = int(anno["video_length"])

        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        frames_ids = np.minimum(frames_ids, length - 1).astype(int)
        print("Ground truth frames ids", frames_ids)

        instruction = anno["texts"][0]
        car_action = np.array(anno["states"])[frames_ids]
        joint_pos = np.array(anno["joints"])[frames_ids]

        # Decode independent camera streams concurrently, preserving their
        # original annotation order. VAE encoding remains sequential and
        # unchanged to preserve the baseline inference path.
        video_paths = [
            f"{val_dataset_dir}/{video_info['video_path']}"
            for video_info in anno["videos"]
        ]
        workers = min(len(video_paths), max(1, min(3, os.cpu_count() or 1)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            video_dict = list(
                pool.map(
                    lambda path: _decode_one_camera(path, length, frames_ids),
                    video_paths,
                )
            )

        video_latent = []
        device = self.device
        vae = self.model.pipeline.vae
        for true_video in video_dict:
            true_video_tensor = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            with torch.no_grad():
                batch_size = 32
                latents = []
                for j in range(0, len(x), batch_size):
                    batch = x[j : j + batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(
                        vae.config.scaling_factor
                    )
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            video_latent.append(x)

        return car_action, joint_pos, video_dict, video_latent, instruction, frames_ids

    def forward_wm(
        self,
        action_cond,
        video_latent_true,
        video_latent_cond,
        his_cond=None,
        text=None,
    ):
        args = self.args
        image_cond = video_latent_cond

        action_cond = self.normalize_bound(
            action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1
        )
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames + args.num_history, args.action_dim)

        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(
                    action_cond, text, self.model.tokenizer, self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
            )

        latents = einops.rearrange(
            latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1
        )

        # Decode ground-truth video exactly as in the original replay script.
        true_video = torch.stack(video_latent_true, dim=0)
        decoded_video = []
        bsz, frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0, 1)
        decode_kwargs = {}
        for j in range(0, true_video.shape[0], args.decode_chunk_size):
            chunk = true_video[j : j + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video, dim=0)
        true_video = true_video.reshape(bsz, frame_num, *true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1) * 255)
        true_video = (
            true_video.detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            .transpose(0, 1, 3, 4, 2)
            .astype(np.uint8)
        )

        # Decode predicted video exactly as in the original replay script.
        decoded_video = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        for j in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[j : j + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = (
            videos.detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            .transpose(0, 1, 3, 4, 2)
            .astype(np.uint8)
        )

        videos_cat = np.concatenate([true_video, videos], axis=-3)
        videos_cat = np.concatenate([video for video in videos_cat], axis=-2).astype(np.uint8)
        return videos_cat, true_video, videos, latents


if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="replay")

    # Skip MP4 generation to reduce CPU and disk overhead
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip final MP4 video reconstruction and saving",
    )

    parser.add_argument(
        "--dynamic_memory",
        action="store_true",
        help="Use rule-based dynamic memory selection instead of fixed history indices",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom directory to save evaluation outputs, resume checkpoints, and videos",
    )

    args_new = parser.parse_args()

    args = wm_args(
        task_type=args_new.task_type,
        dataset_names=args_new.dataset_names or "droid_subset",
        dataset_root_path=args_new.dataset_root_path or "dataset_example",
    )

    def merge_args(args_obj, new_args):
        for key, value in new_args.__dict__.items():
            if value is not None:
                args_obj.__dict__[key] = value
        return args_obj

    args = merge_args(args, args_new)

    if args.output_dir is not None:
        args.save_dir = args.output_dir

    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f"rollout with {args.task_type}")
    print(f"Output directory: {args.save_dir}")

    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):
        eval_dir = os.path.join(
            args.save_dir, args.task_name, "evaluation_data", f"trajectory_{val_id_i}"
        )
        os.makedirs(eval_dir, exist_ok=True)

        eval_paths = [
            os.path.join(
                eval_dir,
                f"interaction_{interaction_i:03d}_start_{start_idx_i + interaction_i * (pred_step - 1)}.npz",
            )
            for interaction_i in range(interact_num)
        ]
        resume_dir = os.path.join(args.save_dir, args.task_name, "resume_states")
        resume_path = os.path.join(resume_dir, f"trajectory_{val_id_i}.pt")

        signature = {
            "trajectory_id": str(val_id_i),
            "start_idx": int(start_idx_i),
            "interact_num": int(interact_num),
            "pred_step": int(pred_step),
            "num_inference_steps": int(args.num_inference_steps),
            "num_frames": int(args.num_frames),
            "num_history": int(args.num_history),
            "width": int(args.width),
            "height": int(args.height),
            "ckpt_path": os.path.abspath(args.ckpt_path),
        }

        # If all interaction outputs already exist, no model inference is needed.
        if all(os.path.isfile(path) for path in eval_paths):
            print(f"Trajectory {val_id_i}: all interaction files already exist; rebuilding video.")
            next_interaction = interact_num
            resume_state = None
        else:
            next_interaction = 0
            resume_state = None
            if os.path.isfile(resume_path):
                try:
                    loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
                    if loaded.get("signature") != signature:
                        raise ValueError("Resume settings do not match this run.")
                    next_interaction = int(loaded["next_interaction"])
                    resume_state = loaded
                    print(
                        f"Resuming trajectory {val_id_i} from interaction "
                        f"{next_interaction}/{interact_num}"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not safely resume trajectory {val_id_i} from {resume_path}: {exc}"
                    ) from exc

        eef_gt, joint_pos_gt, video_dict, video_latents, instruction, trajectory_frame_ids = (
            Agent.get_traj_info(
                val_id_i,
                start_idx=start_idx_i,
                steps=int(pred_step * interact_num + 8),
            )
        )
        text_i = instruction
        print("text_i:", instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)
        assert first_latent.shape == (1, 4, 72, 40), (
            f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        )

        his_cond = [first_latent for _ in range(Agent.args.num_history * 4)]
        his_joint = [joint_pos_gt[0:1] for _ in range(Agent.args.num_history * 4)]
        his_eef = [eef_gt[0:1] for _ in range(Agent.args.num_history * 4)]

        # Restore histories and RNG after loading/encoding the trajectory. This
        # preserves the random stream for the next not-yet-completed interaction.
        if resume_state is not None:
            his_cond = [tensor.to(device=Agent.device, dtype=Agent.dtype) for tensor in resume_state["his_cond"]]
            his_joint = resume_state["his_joint"]
            his_eef = resume_state["his_eef"]
            _restore_rng_state(resume_state["rng_state"])

        video_to_save = None if args.skip_video else []

        # Reconstruct completed video sections only when video output is enabled.
        if not args.skip_video:
            for interaction_i in range(next_interaction):
                with np.load(eval_paths[interaction_i], allow_pickle=False) as saved:
                    gt = saved["ground_truth"]
                    pred = saved["prediction"]

                strip = _compose_video_strip(gt, pred)

                if interaction_i == interact_num - 1:
                    video_to_save.append(strip)
                else:
                    video_to_save.append(strip[: pred_step - 1])

        for i in range(next_interaction, interact_num):
            start_id = int(i * (pred_step - 1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]

            assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
            assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"

            print("################ policy forward ####################")
            cartesian_pose = eef_gt[start_id:end_id]

            print("################ world model forward ################")
            print(f"traj_id:{val_id_i}, interact step: {i}/{interact_num}")

            if args.dynamic_memory:
                his_cond_input, his_pose, selected_memory_indices = select_dynamic_memories(
                    history_latents=his_cond,
                    history_poses=his_eef,
                    current_pose=state_first,
                    future_actions=cartesian_pose,
                    num_memories=num_history,
                )

                print("Dynamic memory indices:", selected_memory_indices.tolist())

            else:
                # Original Ctrl-World fixed-memory baseline
                history_idx = [0, 0, -8, -6, -4, -2]

                his_pose = np.concatenate(
                    [his_eef[idx] for idx in history_idx],
                    axis=0,
                )

                his_cond_input = torch.cat(
                    [his_cond[idx] for idx in history_idx],
                    dim=0,
                ).unsqueeze(0)

            action_cond = np.concatenate(
                [his_pose, cartesian_pose],
                axis=0,
            )
            current_latent = his_cond[-1]
            assert current_latent.shape == (1, 4, 72, 40)
            assert action_cond.shape == (int(num_history + num_frames), 7)
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40)

            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(
                action_cond,
                video_latent_true,
                current_latent,
                his_cond=his_cond_input,
                text=text_i if Agent.args.text_cond else None,
            )

            eval_path = eval_paths[i]
            _atomic_npz_save(
                eval_path,
                ground_truth=true_videos,
                prediction=video_dict_pred,
                trajectory_id=str(val_id_i),
                interaction_index=int(i),
                start_frame_index=int(start_idx_i + start_id),
                instruction=str(instruction),
                fps=int(args.fps),
                frame_indices=trajectory_frame_ids[start_id:end_id],
            )
            print(f"Saved evaluation frames to {eval_path}")

            # Preserve the original autoregressive history updates.
            his_eef.append(cartesian_pose[pred_step - 1 : pred_step])
            his_cond.append(
                torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0)
            )
            if not args.skip_video:
                if i == interact_num - 1:
                    video_to_save.append(videos_cat)
                else:
                    video_to_save.append(videos_cat[: pred_step - 1])

            # Commit resume state only after output and history have both been saved.
            state_to_save = {
                "signature": signature,
                "next_interaction": i + 1,
                "his_cond": [tensor.detach().cpu() for tensor in his_cond],
                "his_joint": his_joint,
                "his_eef": his_eef,
                "rng_state": _rng_state(),
            }
            _atomic_torch_save(state_to_save, resume_path)
            print(f"Resume checkpoint saved: next interaction {i + 1}/{interact_num}")

        if not args.skip_video:
            video = np.concatenate(video_to_save, axis=0)

            text_id = (
                text_i.replace(" ", "_")
                .replace(",", "")
                .replace(".", "")
                .replace("'", "")
                .replace('"', "")
            )[:30]

            uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            filename_video = (
                f"{args.save_dir}/{args.task_name}/video/"
                f"time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"
            )

            os.makedirs(os.path.dirname(filename_video), exist_ok=True)
            mediapy.write_video(filename_video, video, fps=4)

            print(f"Saving video to {filename_video}")

        else:
            print(f"Skipping MP4 generation for trajectory {val_id_i}")

        print("##########################################################################")