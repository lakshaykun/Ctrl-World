# Ctrl-World Offline World-Model Evaluation

This directory evaluates already-generated Ctrl-World predictions against
their recorded observations. It does not initialize Ctrl-World, run inference,
train a model, or measure robot-policy success or task completion.

The evaluator reports visual similarity, temporal consistency, optional object
and action-conditioned diagnostics, conservative image-space physical
candidates, and camera/horizon summaries. Pixel similarity alone is not
evidence that a model learned physical dynamics.

## Files

```text
eval/
├── __init__.py
├── readme.md
├── scripts/
│   ├── run_evaluation.py
│   ├── data_loader.py
│   ├── annotation_loader.py
│   ├── validate_data.py
│   ├── aggregation.py
│   ├── visualization.py
│   ├── reporting.py
│   ├── utils.py
│   └── metrics/
│       ├── pixel_metrics.py
│       ├── perceptual_metrics.py
│       ├── temporal_metrics.py
│       ├── object_metrics.py
│       ├── action_consistency.py
│       ├── physical_metrics.py
│       └── multiview_metrics.py
└── results/
    └── .gitkeep
```

## Input data and array layout

Pass the directory containing the rollout evaluation files. Discovery is
recursive. Each `.npz` must contain `ground_truth` and `prediction`. The
rollout scripts save both as camera × time × height × width × RGB arrays. A
single-camera time × height × width × RGB array is also accepted. Images are
converted to uint8 RGB `[0,255]`; no spatial resizing is performed. RGB(A)
channel-last layouts are required because that is the layout saved by the
existing rollout implementation. Unsupported or mismatched shapes are
recorded as invalid samples rather than silently reshaped.

The loader preserves these optional metadata fields:

- `trajectory_id`
- `interaction_index`
- `start_frame_index`
- `frame_indices`
- `instruction`
- `fps`

`--frame_stride N` selects every Nth stored frame and applies the same
selection to its source frame indices. If `frame_indices` is absent, source
alignment can only be inferred when `start_frame_index` exists; the evaluator
then uses contiguous source indices from that start. Without either field,
annotation-dependent metrics are unavailable.

## Annotation and robot-state alignment

Supply a dataset root with `--annotation_root`. For a trajectory ID such as
`1799`, the expected file is:

```text
<annotation_root>/annotation/val/1799.json
```

The loader first tries the exact trajectory filename. For numeric IDs, it
accepts a uniquely matching zero-padded filename (for example `1799.json` or
`001799.json`); ambiguous or missing matches are not guessed.

The annotation fields are read as follows:

| JSON field | Meaning in this evaluator |
|---|---|
| `video_length` | Valid source-frame bound |
| `states` | Seven-dimensional Cartesian end-effector state/action-conditioning sequence |
| `joints` | Eight-dimensional joint/gripper position sequence, when valid and present |
| `texts` | Annotation instruction strings |
| `videos` | Not required for evaluation; rollout video paths are not decoded |

When `frame_indices` is present, those exact indices select rows from `states`
and `joints`. Bounds, row counts, numeric dimensions, JSON validity, and
sequence lengths are checked. The action result records annotation status and
the aligned source frame index for every state-step row. If `frame_indices`
is absent, contiguous indices beginning at `start_frame_index` are used only
when the resulting alignment is in bounds. Missing or invalid joint values
make joint summaries unavailable without discarding valid end-effector states.
No states, joints, or actions are fabricated.

The rollout code normalizes states component-wise using the repository's
`state_01`/`state_99` statistics and clips to `[-1,1]` before model
conditioning. This evaluator reports raw annotation-space changes, not those
normalized values. Translation diagnostics use the first three state
components; the following three are treated as orientation components for a
descriptive step magnitude. Their units and angle convention are not inferred.

## Metrics

### Visual fidelity

Calculated independently for each camera and frame:

- **MSE**: `mean((prediction - reference)^2)` over RGB pixels; lower is better,
  units are squared 8-bit intensity.
- **PSNR**: `10 log10(255^2 / MSE)`; higher is better, in dB. Exact matches
  have infinite PSNR and are excluded from finite-value aggregate statistics.
- **SSIM**: scikit-image structural similarity with channel-last RGB,
  `data_range=255`, and an odd window up to 7 pixels; higher is better.
- **LPIPS**: optional learned perceptual distance; lower is better. It is
  computed only when local calibrated LPIPS weights are supplied and the
  selected backbone weights are already in the local Torch cache. No weights
  are downloaded. Otherwise LPIPS is explicitly unavailable.

### Temporal consistency

- **Temporal Difference Error**: for each consecutive stored frame pair,
  `mean(abs((P_t - P_(t-1)) - (G_t - G_(t-1))))`; lower is better, in
  8-bit intensity units.
- **Optical Flow Difference**: mean vector distance between independently
  estimated prediction and reference flow fields. The estimator is OpenCV
  Farneback with fixed parameters in `temporal_metrics.py`; units are pixels
  per sampled frame interval, lower is better. This is **not** endpoint error
  against annotated ground-truth flow.
- **Prediction horizon**: applicable frame metrics are grouped by zero-based
  frame position within each stored interaction. Horizon summaries retain
  valid counts and do not fill missing values with zero.

At least two frames are required for transition metrics. Flow comparison uses
the samples after frame stride selection, so its interval is the sampled
interval.

### Object consistency (optional local detector)

The evaluator does not ship a detector and never downloads detector weights.
When a local detector plugin is configured, its `detect(frame)` method must
return a list of `Detection` instances or dictionaries with `box_xyxy`
(`x1,y1,x2,y2`), and optional `confidence` and `label`. A greedy IoU tracker
associates same-label boxes with IoU ≥ 0.3 and retains unmatched IDs for up to
two frames. The same extracted tracks are reused for metrics, action
diagnostics, and physical candidates.

The custom definitions are:

- **Object Persistence Score**: reference detections matched at IoU ≥ 0.5
  divided by all reference detections; range `[0,1]`, higher is better.
- **Object Hallucination Rate (OHR)**: generated detections unmatched to a
  reference detection divided by generated detections; range `[0,1]`, lower is
  better. This is detector disagreement, not proof of hallucination.
- **Identity Switch Count**: number of changes in generated track ID matched
  to the same reference track across observed frames; lower is better.
- **Object Trajectory Error**: mean matched reference/generated box-center
  distance divided by the image diagonal; dimensionless, lower is better.
- **Position Anomaly Rate (PAR)**: fraction of consecutive generated track
  steps longer than 0.25 image diagonals; lower is better.
- **Temporal Consistency Score (TCS)**: fraction of consecutive generated
  track steps no longer than 0.10 image diagonals; higher is better.
- **Hallucination Severity Score (HSS)**: unmatched generated box area divided
  by image area and frame count; lower is better.

These are detector- and image-space-dependent custom metrics, not exact
reproductions of an external benchmark. Independent detector errors, occlusion,
and viewpoint changes affect results. If a detector or tracker is unavailable,
the object metrics are reported as unavailable, not zero.

### Recorded-action diagnostics

Action metrics use aligned seven-dimensional `states` values from the original
annotation. They report the mean translation-step magnitude, mean
orientation-component step magnitude, optional joint-position step magnitude,
and Pearson correlation between translation step and mean predicted
image-space Farneback flow magnitude. When object tracks are configured, an
additional correlation with image-space tracked-object displacement is
reported, as is the mean difference between generated and reference
object-displacement vectors (normalized by image diagonal) for matched
consecutive tracks. These object-dependent values are unavailable when a
compatible detector/tracker or consecutive matched tracks are unavailable.

Correlations are descriptive only. They do not prove causal action response
or physical correctness; image-space motion depends on camera viewpoint and
may include camera or background movement. Contact timing, action-conditioned
object state error, and contact correctness are unavailable without
independently labeled contact times, object state ground truth, or calibrated
geometry. Annotation-dependent metrics are explicitly unavailable when
annotations are missing or cannot be aligned.

### Physical-consistency candidates

- **Physical Violation Rate**: fraction of observed frames containing at least
  one pair of detected boxes with 2D IoU ≥ 0.05; lower is preferable, but this
  is only an image-space overlap candidate rate.
- Candidate CSV rows retain sample, frame, camera, object IDs, overlap IoU,
  and evidence.
- Floating-object candidates, contact consistency, and task-specific spatial
  relation consistency are unavailable without support-surface labels, depth,
  camera-to-robot calibration, 3D geometry, or task-specific relation labels.

Bounding-box overlap does not establish 3D penetration or collision.

### Multi-camera summaries

Applicable metrics are summarized separately per camera. No pixels or tracks
are compared across cameras as if they were geometrically aligned. Cross-view
geometry is explicitly unavailable without calibration or validated
correspondences.

### FVD (optional)

FVD is the Fréchet distance between the means and covariances of compatible
real and generated **clip-level video feature** vectors. It is dataset-level,
not per-frame. The evaluator computes the Fréchet formula from an NPZ provided
with `--fvd_features`; that file must contain two 2D arrays named
`real_features` and `generated_features`, with matching feature dimensions and
at least two clips per distribution. The vectors must come from the same
frozen, compatible pretrained video feature extractor and identical clip
preprocessing (for example an I3D FVD protocol). Give its provenance using
`--fvd_extractor_name`. Features are not extracted by this evaluator. Without
compatible local features, FVD is unavailable; no extractor weights are
downloaded and no score is fabricated.

## Running

Run commands from the repository root. Visual and temporal evaluation does
not require annotation files:

```bash
python -m eval.scripts.run_evaluation \
  --input_dir synthetic_traj/Rollouts_replay_droid_subset/evaluation_data \
  --output_dir eval/results \
  --metrics visual,temporal \
  --device cpu
```

To include the aligned action/joint diagnostics:

```bash
python -m eval.scripts.run_evaluation \
  --input_dir synthetic_traj/Rollouts_replay_droid_subset/evaluation_data \
  --annotation_root dataset_example/droid_subset \
  --output_dir eval/results \
  --metrics all \
  --device cpu \
  --save_visualizations
```

Optional local LPIPS weights:

```bash
python -m eval.scripts.run_evaluation \
  --input_dir <evaluation-data-directory> \
  --output_dir eval/results \
  --metrics visual,lpips \
  --lpips_weights <local-lpips-linear-weights.pth> \
  --lpips_network alex \
  --device cuda:0 \
  --batch_size 16
```

Optional object detection uses a local Python factory and does not prescribe a
detector package:

```text
--detector my_local_detector:build_detector --tracker iou
```

The factory is called with `device=<selected device>` and must return the
detector described above. The caller is responsible for making its weights
available locally. `--detector none` is the default; `--tracker` supports
`iou` or `none`.

Optional FVD from precomputed features:

```bash
python -m eval.scripts.run_evaluation \
  --input_dir <evaluation-data-directory> \
  --output_dir eval/results \
  --metrics fvd \
  --fvd_features <compatible-clip-features.npz> \
  --fvd_extractor_name <extractor-and-preprocessing-version>
```

Supported metrics are `all`, `visual`, `temporal`, `lpips`, `action`,
`objects`, `physical`, `multiview`, and `fvd`, supplied as a comma-separated
list (or `all` alone). Other useful options are:

- `--frame_stride`: positive integer, default `1`.
- `--device`: `cpu` by default, or a requested CUDA device for LPIPS/detector.
- `--batch_size`: positive LPIPS batch size, default `16`.
- `--save_visualizations`: write paired frames and metric plots.
- `--save_comparison_videos`: write side-by-side MP4 videos.
- `--annotation_root`: optional dataset root containing `annotation/val`.

Exit code `0` indicates all discovered samples were valid, `1` indicates one
or more invalid samples were recorded, and `2` indicates a fatal input or
configuration error.

## Outputs and interpretation

Each invocation creates `eval/results/run_<UTC timestamp>/` with
`config.json`, `summary.json`, and `validation_report.json`. CSV files are
created only when the corresponding data exists:

- `metrics_per_frame.csv`
- `metrics_per_interaction.csv`
- `metrics_per_trajectory.csv`
- `metrics_per_camera.csv`
- `metrics_per_horizon.csv`
- `metrics_per_object.csv`
- `object_tracks.csv`
- `physical_violations.csv`

Requested visualizations are saved in `plots/` and `visualizations/`. Reports
retain original sample paths, trajectory and interaction IDs, camera indices,
frame positions, validity counts, annotation status, preprocessing, metric
configuration, and unavailable-metric reasons. Aggregates use each
interaction-level mean as one observation at trajectory/dataset level, so long
interactions do not implicitly receive greater weight. Missing and non-finite
values are excluded rather than treated as zero.

Interaction, trajectory, camera, horizon, and dataset summaries include mean,
median, standard deviation, minimum, maximum, valid count, and invalid count
where applicable. `metrics_per_object.csv` summarizes each generated track's
observation count, mean confidence and center, and consecutive-frame image-space
path length; it does not claim ground-truth object identity.

Use the metric-specific count and validity fields when comparing runs.
Unavailable metrics are not evidence of a zero error. Interpret visual,
temporal, object, action, and physical diagnostics together, and keep their
camera-, detector-, and data-dependent limitations in mind.

## Dependencies and reproducibility

The implementation reuses packages already listed in the repository's
`requirements.txt`: NumPy, SciPy, scikit-image, OpenCV, PyTorch, LPIPS, Pillow,
and Matplotlib. No dependency was added. Optional packages and local weights
are needed only for the corresponding metric. The evaluator does not install
packages, download weights, or initialize Ctrl-World.

Reproducibility details (input paths, chosen metrics, device, sampling,
detector/tracker settings, annotation root, and preprocessing assumptions)
are written to `config.json` and `summary.json`. Preserve the exact input NPZ
files, local model weights, detector implementation, and any FVD feature
extractor protocol alongside those reports for repeatable comparisons.
