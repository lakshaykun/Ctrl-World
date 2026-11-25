"""Rule-based dynamic memory selection for Ctrl-World trajectory replay.

This module does not train or modify model weights. It selects existing history
latent/pose pairs using current-pose relevance, upcoming-action relevance,
recency, and temporal diversity.
"""
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
import torch


def _pose_array(poses: Sequence) -> np.ndarray:
    """Convert list of (1, D), (D,), or tensor pose entries to (N, D)."""
    rows = []
    for pose in poses:
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        rows.append(arr)
    if not rows:
        raise ValueError("history_poses must not be empty")
    dims = {row.shape[0] for row in rows}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent pose dimensions in history: {dims}")
    return np.stack(rows, axis=0)


def _normalized_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-row Euclidean distance normalized by a robust scale."""
    diff = a - b
    # Normalize dimensions by their candidate-pool spread to avoid one coordinate
    # dominating merely because it has a larger numerical scale.
    scale = np.std(a, axis=0)
    scale = np.maximum(scale, eps)
    d = np.sqrt(np.mean((diff / scale) ** 2, axis=-1))
    return d / (np.max(d) + eps)


def select_dynamic_memories(
    history_latents: Sequence[torch.Tensor],
    history_poses: Sequence,
    current_pose,
    future_actions,
    num_memories: int = 6,
    pose_weight: float = 0.45,
    action_weight: float = 0.40,
    recency_weight: float = 0.15,
    diversity_threshold: float = 0.08,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Select memory pairs and return (latents, poses, selected_indices).

    Latents are stacked as (1, num_memories, C, H, W), matching the caller's
    existing history tensor convention. Poses are returned as (num_memories, D).
    Selected indices are sorted chronologically before return.
    """
    if len(history_latents) != len(history_poses):
        raise ValueError(
            f"Latent/pose history lengths differ: {len(history_latents)} vs "
            f"{len(history_poses)}"
        )
    n = len(history_poses)
    if n == 0:
        raise ValueError("No history candidates available")
    k = min(int(num_memories), n)
    if k <= 0:
        raise ValueError("num_memories must be positive")

    poses = _pose_array(history_poses)
    current = np.asarray(current_pose, dtype=np.float32).reshape(1, -1)
    actions = np.asarray(future_actions, dtype=np.float32).reshape(-1, poses.shape[1])
    if current.shape[1] != poses.shape[1]:
        raise ValueError("current_pose dimension does not match history pose dimension")

    # Pose relevance: closer to current pose is preferred.
    pose_dist = _normalized_distance(poses, np.repeat(current, n, axis=0))
    pose_score = 1.0 - pose_dist

    # Action relevance: candidate memories near the upcoming action trajectory
    # receive a higher score. This is a simple geometric proxy, not semantic
    # action understanding.
    if len(actions):
        action_dists = np.stack(
            [_normalized_distance(poses, np.repeat(a[None, :], n, axis=0)) for a in actions],
            axis=1,
        )
        action_dist = np.min(action_dists, axis=1)
        action_score = 1.0 - action_dist
    else:
        action_score = np.zeros(n, dtype=np.float32)

    # Recent memories score higher. Normalize to [0, 1].
    recency_score = np.linspace(0.0, 1.0, n, dtype=np.float32)

    total_weight = pose_weight + action_weight + recency_weight
    if total_weight <= 0:
        raise ValueError("At least one scoring weight must be positive")
    score = (
        pose_weight * pose_score
        + action_weight * action_score
        + recency_weight * recency_score
    ) / total_weight

    # Greedy relevance selection with pose-space diversity. The newest memory is
    # mandatory to preserve the current state context.
    selected = [n - 1]
    remaining = set(range(n - 1))
    while len(selected) < k and remaining:
        best_idx = None
        best_value = -float("inf")
        for idx in remaining:
            # Reject candidates too close to any selected pose where possible.
            # Pairwise normalized pose distance to already selected memories.
            scale = np.maximum(np.std(poses, axis=0), 1e-8)
            pairwise = np.sqrt(np.mean(((poses[selected] - poses[idx]) / scale) ** 2, axis=1))
            min_dist = float(pairwise.min())
            diversity_ratio = min(min_dist / max(diversity_threshold, 1e-8), 1.0)
            duplicate_penalty = 0.10 * (1.0 - diversity_ratio)
            value = float(score[idx]) - duplicate_penalty
            if value > best_value:
                best_value, best_idx = value, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    # Chronological order is important: memory frames should retain their
    # temporal ordering when fed into the world-model conditioning sequence.
    selected = sorted(selected)
    selected_latents = torch.cat([history_latents[i] for i in selected], dim=0)
    selected_poses = poses[selected]
    return selected_latents.unsqueeze(0), selected_poses, np.asarray(selected, dtype=np.int64)
