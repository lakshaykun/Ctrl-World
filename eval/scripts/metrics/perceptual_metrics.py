"""Optional LPIPS and reusable Fréchet video feature-distance calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import sqrtm

from ..utils import metric_status


def load_local_lpips(
    network: str, weights_path: str | Path | None, device: str
) -> tuple[Any | None, str | None]:
    """Load LPIPS only when both calibrated weights and cached backbone weights exist.

    The local-cache check prevents torchvision from downloading a backbone.
    """
    if weights_path is None:
        return None, "LPIPS requires an explicit local --lpips_weights file"
    weights = Path(weights_path)
    if not weights.is_file():
        return None, f"LPIPS weights file does not exist: {weights}"
    try:
        import torch
        import lpips
    except ImportError as exc:
        return None, f"LPIPS dependencies are unavailable: {exc}"

    cache_names = {
        "alex": "alexnet-owt-7be5be79.pth",
        "vgg": "vgg16-397923af.pth",
        "squeeze": "squeezenet1_1-b8a52dc0.pth",
    }
    if network not in cache_names:
        return None, f"Unsupported LPIPS backbone: {network}"
    backbone = Path(torch.hub.get_dir()) / "checkpoints" / cache_names[network]
    if not backbone.is_file():
        return None, f"Cached {network} backbone weights are missing: {backbone}"
    try:
        model = lpips.LPIPS(net=network, model_path=str(weights), pnet_rand=False)
        model = model.to(device).eval()
        return model, None
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"Could not initialize LPIPS from local weights: {exc}"


def compute_lpips(
    model: Any,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Compute calibrated LPIPS using RGB batches normalized to [-1, 1]."""
    import torch

    rows = []
    frames = ground_truth.shape[1]
    gt = ground_truth.reshape(-1, *ground_truth.shape[2:])
    pred = prediction.reshape(-1, *prediction.shape[2:])
    with torch.inference_mode():
        for offset in range(0, len(gt), batch_size):
            end = min(offset + batch_size, len(gt))
            a = torch.from_numpy(gt[offset:end].copy()).permute(0, 3, 1, 2).float()
            b = torch.from_numpy(pred[offset:end].copy()).permute(0, 3, 1, 2).float()
            device = next(model.parameters()).device
            a = a.to(device).div_(127.5).sub_(1.0)
            b = b.to(device).div_(127.5).sub_(1.0)
            values = model(a, b).reshape(-1).float().cpu().numpy()
            for local_index, score in enumerate(values):
                index = offset + local_index
                rows.append(
                    {
                        "sample_id": sample_id,
                        "camera_index": index // frames,
                        "frame_position": index % frames,
                        "lpips": float(score),
                    }
                )
    return rows


def frechet_distance_from_features(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """Compute FVD from compatible clip-level I3D feature vectors.

    This is the standard Gaussian Fréchet distance over independent clip
    features. Both feature sets must use the same frozen pretrained extractor
    and preprocessing. At least two clips per distribution are required.
    """
    real = np.asarray(real_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    if (
        real.ndim != 2
        or generated.ndim != 2
        or real.shape[1] != generated.shape[1]
        or real.shape[1] < 2
    ):
        raise ValueError("FVD features must be 2D with matching feature dimensions")
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("FVD requires at least two real and generated clip features")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("FVD feature matrices contain non-finite values")
    mean_real = real.mean(axis=0)
    mean_generated = generated.mean(axis=0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_generated = np.cov(generated, rowvar=False)
    covariance_mean = sqrtm(covariance_real @ covariance_generated)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_real.shape[0]) * 1e-6
        covariance_mean = sqrtm((covariance_real + offset) @ (covariance_generated + offset))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(covariance_mean.imag, 0, atol=1e-3):
            raise ValueError("FVD covariance square root has a material imaginary component")
        covariance_mean = covariance_mean.real
    delta = mean_real - mean_generated
    return float(delta @ delta + np.trace(covariance_real + covariance_generated - 2 * covariance_mean))


def unavailable_fvd(reason: str) -> dict[str, Any]:
    """Report that FVD cannot be computed without compatible clip features."""
    return metric_status("fvd", "unavailable", reason=reason)
