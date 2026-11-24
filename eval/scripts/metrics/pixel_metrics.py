"""Reference-based per-frame pixel metrics (MSE, PSNR, and SSIM)."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def compute_pixel_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray, sample_id: str
) -> list[dict[str, Any]]:
    """Compute RGB MSE, PSNR, and SSIM for each camera/frame pair.

    Inputs use uint8 RGB values in [0, 255]. MSE is in squared intensity units,
    PSNR uses a fixed data range of 255, and SSIM uses scikit-image's channel-last
    implementation. Exact matches have infinite PSNR.
    """
    rows = []
    camera_count, frame_count = ground_truth.shape[:2]
    for camera in range(camera_count):
        for frame in range(frame_count):
            gt = ground_truth[camera, frame].astype(np.float32)
            pred = prediction[camera, frame].astype(np.float32)
            mse = float(np.mean(np.square(gt - pred), dtype=np.float64))
            psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0**2) / mse))
            smallest_side = min(gt.shape[0], gt.shape[1])
            if smallest_side < 3:
                ssim = None
                ssim_status = "unavailable"
                ssim_reason = "SSIM requires images at least 3 pixels wide and high"
            else:
                window = min(7, smallest_side if smallest_side % 2 else smallest_side - 1)
                ssim = float(
                    structural_similarity(
                        gt.astype(np.uint8),
                        pred.astype(np.uint8),
                        channel_axis=-1,
                        data_range=255,
                        win_size=window,
                    )
                )
                ssim_status = "available"
                ssim_reason = None
            rows.append(
                {
                    "sample_id": sample_id,
                    "camera_index": camera,
                    "frame_position": frame,
                    "mse": mse,
                    "psnr": psnr,
                    "ssim": ssim,
                    "ssim_status": ssim_status,
                    "ssim_reason": ssim_reason,
                }
            )
    return rows
