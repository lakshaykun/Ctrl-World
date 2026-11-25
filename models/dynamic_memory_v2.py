"""Dynamic Memory V2: action-first retrieval with controlled temporal diversity.

This is a heuristic selector (no training/weight updates). It retains the newest
memory, reserves slots for action-relevant memories, then fills remaining slots
using temporal bins and pose diversity. Returns chronological memory order.
"""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import torch


def _poses_to_array(history_poses: Sequence) -> np.ndarray:
    rows = []
    for pose in history_poses:
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        rows.append(np.asarray(pose, dtype=np.float32).reshape(-1))
    if not rows:
        raise ValueError("history_poses must not be empty")
    dims = {r.size for r in rows}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent pose dimensions: {dims}")
    return np.stack(rows)


def _distance_to_targets(candidates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Scale-normalized RMS distance; robust to different pose dimensions."""
    scale = np.std(candidates, axis=0)
    scale = np.maximum(scale, 1e-6)
    if targets.ndim == 1:
        targets = targets[None, :]
    d = np.sqrt(np.mean(((candidates[:, None, :] - targets[None, :, :]) / scale) ** 2, axis=-1))
    # Normalize distances so the action score has a stable [0,1] range.
    dmax = float(np.max(d))
    return d / (dmax + 1e-8)


def select_dynamic_memories_v2(
    history_latents: Sequence[torch.Tensor],
    history_poses: Sequence,
    current_pose,
    future_actions,
    num_memories: int = 6,
    action_weight: float = 0.70,
    recency_weight: float = 0.15,
    current_pose_weight: float = 0.15,
    action_slots: int = 2,
    temporal_bins: int = 4,
    diversity_weight: float = 0.20,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Select action-relevant memories with controlled temporal coverage.


    Selection policy:
      1. Always retain the latest memory (current-state anchor).
      2. Reserve up to ``action_slots`` additional slots for nearest upcoming-action
         states, using a soft score with action relevance and recency.
      3. Fill remaining slots by temporal-bin coverage plus pose-space diversity.
      4. Return chronological order for model conditioning.
    """
    if len(history_latents) != len(history_poses):
        raise ValueError("history_latents and history_poses must have equal lengths")
    n = len(history_poses)
    k = min(int(num_memories), n)
    if n == 0 or k <= 0:
        raise ValueError("History must be non-empty and num_memories positive")

    poses = _poses_to_array(history_poses)
    current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    actions = np.asarray(future_actions, dtype=np.float32).reshape(-1, poses.shape[1])
    if current.size != poses.shape[1]:
        raise ValueError("current_pose dimension differs from history pose dimension")

    # Normalize recency across the full history; index n-1 is newest.
    recency = np.linspace(0.0, 1.0, n, dtype=np.float32)
    current_dist = _distance_to_targets(poses, current).reshape(-1)
    current_score = 1.0 - current_dist

    if actions.size:
        action_dist = _distance_to_targets(poses, actions)
        # A memory is relevant if it is near any upcoming action pose.
        action_score = 1.0 - np.min(action_dist, axis=1)
    else:
        action_score = np.zeros(n, dtype=np.float32)

    wsum = max(action_weight + recency_weight + current_pose_weight, 1e-8)
    relevance = (action_weight * action_score + recency_weight * recency + current_pose_weight * current_score) / wsum

    selected = [n - 1]  # preserve newest/current state anchor
    remaining = set(range(n)) - set(selected)

    # Action-first reserved slots: high action score, with recency tie-break.
    reserved = min(max(0, int(action_slots)), max(0, k - 1))
    for _ in range(reserved):
        if not remaining:
            break
        idx = max(remaining, key=lambda j: (action_score[j], relevance[j], recency[j]))
        selected.append(idx)
        remaining.remove(idx)

    # Controlled temporal coverage: divide remaining history into bins and
    # preferentially select one candidate per bin before unrestricted filling.
    bins = max(1, min(int(temporal_bins), n))
    bin_ids = np.minimum((np.arange(n) * bins) // n, bins - 1)
    covered = set(int(bin_ids[i]) for i in selected)
    scale = np.maximum(np.std(poses, axis=0), 1e-6)

    while len(selected) < k and remaining:
        uncovered = [b for b in range(bins) if b not in covered]
        pool = remaining
        if uncovered:
            bin_pool = {j for j in remaining if int(bin_ids[j]) in uncovered}
            if bin_pool:
                pool = bin_pool
        def score_idx(j):
            pair_dist = np.sqrt(np.mean(((poses[selected] - poses[j]) / scale) ** 2, axis=1))
            diversity = min(float(pair_dist.min()), 1.0)
            return float(relevance[j]) + diversity_weight * diversity
        idx = max(pool, key=score_idx)
        selected.append(idx)
        remaining.remove(idx)
        covered.add(int(bin_ids[idx]))

    selected = sorted(selected)
    selected_latents = torch.cat([history_latents[i] for i in selected], dim=0).unsqueeze(0)
    selected_poses = poses[selected]
    return selected_latents, selected_poses, np.asarray(selected, dtype=np.int64)
