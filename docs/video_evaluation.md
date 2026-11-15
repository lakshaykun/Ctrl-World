# Offline evaluation of Ctrl-World replay rollouts

The replay rollout saves aligned ground-truth and predicted RGB arrays as NPZ files while generating the normal side-by-side MP4. This avoids rerunning the diffusion model or extracting frames from a compressed visualization video.

## Generate rollout data

Run the existing `scripts/rollout_replay_traj.py` command. Evaluation arrays are saved under:

```text
synthetic_traj/Rollouts_replay/evaluation_data/trajectory_<id>/
  interaction_000_start_<frame>.npz
```

Each NPZ contains `ground_truth` and `prediction` arrays with shape
`(views, frames, height, width, 3)`, plus trajectory ID, interaction index,
start frame, instruction and FPS.

## Dependencies

PSNR is NumPy-only. SSIM requires:

```bash
pip install scikit-image
```

Optional LPIPS:

```bash
pip install lpips
```

## Evaluate

```bash
python scripts/evaluate_video_metrics.py \
  --data_dir synthetic_traj/Rollouts_replay/evaluation_data \
  --output_dir evaluation_results \
  --metrics psnr ssim
```

Add LPIPS with `--metrics psnr ssim lpips --device cuda`.

Outputs:
- `per_interaction_metrics.csv`
- `summary.json`

## Metric scope and limitations

PSNR, SSIM and LPIPS are paired-frame metrics computed against corresponding
ground-truth frames, across all saved views and frames. Adjacent interaction
windows may include a shared boundary frame, following the rollout's own
prediction windows; disclose this sampling protocol in research reports.

FID and FVD are intentionally not included in this first evaluator. They are
distribution-level metrics requiring a sufficiently large, consistently sampled
evaluation set and a specified feature-extractor protocol. Do not treat scores
from a single trajectory as meaningful dataset-level FID/FVD. The small example
subset is suitable for a smoke test, not a paper-quality benchmark.
