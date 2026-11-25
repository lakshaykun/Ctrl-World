# Ctrl-World Offline World-Model Evaluation

This directory contains offline evaluators for already-generated Ctrl-World
predictions. The metric pipeline compares predictions with recorded
observations without initializing Ctrl-World or training a model. A separate,
optional local VLM path classifies complete replay predictions; it does not run
the Ctrl-World world model or measure real robot-policy success.

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
│   ├── vlm_success_evaluation.py
│   └── metrics/
│       ├── pixel_metrics.py
│       ├── perceptual_metrics.py
│       ├── temporal_metrics.py
│       ├── object_metrics.py
│       ├── action_consistency.py
│       ├── physical_metrics.py
│       ├── multiview_metrics.py
│       └── video_distribution_metrics.py
└── results/
    └── .gitkeep
```

The optional VLM path has focused tests in `eval/tests/` and its separate
Transformers constraint in `eval/requirements_vlm.txt`.

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
  to the same reference track across observed frames; matches use same-label
  greedy IoU ≥ 0.5; lower is better. Its existing definition is unchanged.
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
may include camera or background movement. Annotation-dependent metrics are
explicitly unavailable when annotations are missing or cannot be aligned.

When `action` is selected, the evaluator also validates the raw gripper action
and observed-position arrays from the same annotation JSON. Both are checked
for finite numeric values, one scalar per row, and `raw_length`. Source frame
`s` maps to raw row `3*s`; only these two gripper arrays are loaded for this
diagnostic, so their absence does not invalidate the aligned `states` metrics.

**Gripper transition agreement** is reported per interaction and at trajectory
and dataset levels, with per-frame rows when source-frame indices are
available. It compares command and observed gripper changes between adjacent
stored source frames (raw indices `3*s` and `3*(s+1)`). A transition is
positive or negative at a native-unit magnitude of at least `0.1`; intervals
with no transition in either sequence are excluded. The result is the fraction
of scored intervals whose transition direction agrees. It is a synchronized-
series diagnostic, not causal evidence that a command produced an observation.

### Physical-consistency candidates

- **Physical Violation Rate**: fraction of observed frames containing at least
  one pair of detected boxes with 2D IoU ≥ 0.05; lower is preferable, but this
  is only an image-space overlap candidate rate.
- Candidate CSV rows retain sample, frame, camera, object IDs, overlap IoU,
  and evidence.

Bounding-box overlap does not establish 3D penetration or collision.

### Multi-camera summaries

Applicable metrics are summarized separately per camera. No pixels or tracks
are compared across cameras as if they were geometrically aligned. Two
additional non-geometric proxies are reported when `multiview` is selected:

- **Cross-View Object Agreement (CVOA)** compares the detected category sets
  between every camera pair at the same source frame using Jaccard similarity
  (`|A ∩ B| / |A ∪ B|`). It reports reference and generated agreement and
  generated-minus-reference difference, along with valid pair counts. Two
  empty category sets are not scored. This ignores multiplicity, object
  identity, occlusion, and whether an object should be visible in both views.
  It requires the configured local detector, but reuses the same detections as
  object tracking when both metric groups are requested.
- **Cross-View Feature Consistency (CVFC)** uses the fixed Torchvision
  `ResNet50_Weights.IMAGENET1K_V2` model and its associated preprocessing.
  For each synchronized frame and camera pair, it computes cosine similarity
  between the 2048-dimensional pooled features, separately for reference and
  generated images, plus the signed difference and absolute consistency error.
  It is an ImageNet semantic-feature proxy, not a measure of object identity
  or 3D spatial agreement.

Both metrics require at least two views and valid source-frame indices. Pair
scores are written to `metrics_per_camera.csv` with camera-pair IDs. Their
per-frame rows are also written to `metrics_per_frame.csv`. Interaction
summaries are kept separate; trajectory and dataset summaries deduplicate
overlapping interaction windows by trajectory and source-frame index. The
detector is initialized for `multiview` when CVOA is requested, even if
`--tracker none` is selected. CVFC loads once per evaluation run and uses
`--cvfc_batch_size` (default `32`).

### Dataset-level FID and FVD (optional)

FID and FVD are distribution metrics reported only in `summary.json` (also
under `distribution_metrics`); they are not added to per-frame or
per-interaction CSVs.

- **FID** compares the dataset-level distributions of all successfully loaded real and
  generated RGB frames using TorchMetrics Inception-v3 2048-dimensional
  features. Updates are batched independently of LPIPS. The default batch size
  is 64; images are resized to the Inception input size by the metric. At least
  two real and two generated frames are required.
- **Video FVD** uses `cd-fvd` I3D-400 features and computes a dataset-level
  Fréchet distance over reconstructed video clips. Samples are grouped by
  trajectory, then deterministically ordered by first source frame index,
  interaction index, and path. When chunks overlap, the earliest ordered chunk
  supplies each camera/source-frame pair. Duplicate source-frame indices
  (including repeated terminal frames) are kept once. Source-frame gaps and
  resolution changes split the sequence; clips never cross a segment boundary
  or camera. Because gaps are respected, `--frame_stride` greater than 1 can
  reduce the number of eligible clips or leave FVD unavailable. At least two
  clips are required.
- **Precomputed FVD** remains supported. An NPZ containing matching 2D
  `real_features` and `generated_features` matrices can be supplied instead of
  video clips; at least two compatible clip features per distribution and
  `--fvd_extractor_name` provenance are required.

FID uses `--fid_weights` or an
already-cached Torch-Fidelity file named
`weights-inception-2015-12-05-6726825d.pth`. Video FVD uses `--fvd_weights` or
the I3D checkpoint already bundled in the local `cd-fvd` package. If weights,
packages, source indices, or enough frames/clips are unavailable, the metric is
reported as unavailable with a reason; no score is fabricated. CVFC uses the
standard Torchvision ImageNet weight cache and may download the fixed ResNet-50
weights when they are not cached.

## Running

### Resuming offline metric evaluations

Start a new run as usual; completed samples are checkpointed after each sample
by default:

```powershell
python -m eval.scripts.run_evaluation `
  --input_dir PATH_TO_NPZ_FILES `
  --output_dir eval/results `
  --metrics all
```

Resume into the exact existing run directory, or force recomputation while
continuing to use that directory:

```powershell
python -m eval.scripts.run_evaluation `
  --input_dir PATH_TO_NPZ_FILES `
  --resume-run eval/results/<existing_run_folder> `
  --metrics all

python -m eval.scripts.run_evaluation `
  --input_dir PATH_TO_NPZ_FILES `
  --resume-run eval/results/<existing_run_folder> `
  --metrics all `
  --force
```

Sample checkpoints are JSON files under `sample_checkpoints/`. They retain
per-frame and per-interaction rows, object and physical rows, action and
recorded-action rows, multiview and camera-pair rows, validation results,
camera indices, and sample-scoped unavailable reasons. Writes use a temporary
file and atomic replacement. A checkpoint is reused only when its schema,
completed status, contents, absolute NPZ path and SHA-256, relevant annotation
path and SHA-256, requested sample metric groups, frame stride, device, and
applicable detector/tracker, LPIPS, batching, and visualization settings
match. Corrupt, incomplete, failed, changed-input, or configuration-mismatched
checkpoints are recomputed. Failed samples are recorded but are deliberately
retried on a later resume. `run_metadata.json` records the run configuration
and cache identities. `--checkpoint-every N` changes the write frequency from
the default of one newly evaluated sample; setting a larger value trades
checkpoint I/O for the possibility of repeating up to `N-1` samples after an
interruption.

`--force` ignores all existing sample checkpoints. Changed global-only
settings such as FID/FVD weights do not invalidate compatible sample-level
metrics, but those global metrics are never restored from a checkpoint:
**FID is recomputed using all currently discovered samples, and FVD is
recomputed using the complete current input set or the current precomputed
feature file on every run and resume.** This preserves the existing dataset-
level feature statistics, source-frame alignment, overlap deduplication, and
clip reconstruction behavior. Consequently, global FID/FVD extraction may
still take substantial time during resume. Detector plugin code is fingerprinted
when importable, but external detector weight files are not introspected unless
their identity is represented by the plugin source/configuration.

### Optional Qwen3-VL replay success classification

This standalone path evaluates a complete saved replay trajectory against the
original DROID `success` annotation. It reads the task instruction, binary
label, and video length from the configured validation annotations. It
reconstructs chronology from each replay NPZ's `prediction` array and source
frame metadata, trimming the shared final frame from every non-final
interaction to match the replay writer. Missing interactions, inconsistent
boundaries, gaps, or incomplete coverage are reported as unavailable rather
than scored.

The VLM receives only the annotation instruction and a chronological sequence
of images. Each image is a labeled grid of synchronized prediction camera
views. Ground-truth frames, annotation labels, NPZ instructions, and comparison
videos are not provided to the model. Inference is deterministic. Malformed
structured outputs are retried up to the configured limit and then reported as
failed.

Install the optional Transformers dependency without changing the root
`requirements.txt` pin. PyTorch must be installed separately with a host-
compatible CUDA wheel; inference requires CUDA. The evaluator does not download
model weights when `--local-files-only` is specified, and the documentation or
unit tests do not require model loading.

```powershell
python -m pip install -r eval/requirements_vlm.txt
python -m eval.scripts.vlm_success_evaluation \
  --rollout-dir synthetic_traj/Rollouts_replay_droid_subset \
  --output-dir eval/results/vlm_success \
  --trajectory-id 18599
```

By default the dataset directory is resolved through `config.wm_args` using
the replay dataset name inferred from the rollout directory. Override it with
`--dataset-root-path` and `--dataset-name` if needed. The default model is
`Qwen/Qwen3-VL-4B-Instruct`; use `--model-revision` to pin a branch, tag, or
commit, or `--local-files-only` to require cached model/config files.

Each timestamped run writes `config.json`, `predictions.json`,
`predictions.csv`, `summary.json`, `confusion_matrix.json`, and
`validation_report.json`. Completed predictions can be resumed with
`--resume-run <run-directory>`; cache identity includes source NPZ and
annotation hashes, instruction, model revision, prompt, and sampling settings.
Use `--force` to rerun cached predictions. Useful controls include
`--sampling-fps`, `--max-frames`, `--max-pixels`, `--max-new-tokens`, and
`--max-parse-retries`.

The original DROID labels are binary (`0` failure, `1` success). `uncertain` is
an explicit prediction-side abstention, never a fabricated ground-truth
category. Binary accuracy, precision, recall, and F1 use only completed,
unambiguous predictions; the separate 2-by-3 outcome matrix retains uncertain
predictions and reports prediction coverage. Original annotation success rate
and VLM-predicted success rate are shown separately and are not interchangeable.

Do not run full-dataset or GPU inference as part of unit-test validation. A
single-trajectory run is available for a deliberate local smoke test.

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

Video-based FVD from locally available I3D weights:

```bash
python -m eval.scripts.run_evaluation \
  --input_dir <evaluation-data-directory> \
  --output_dir eval/results \
  --metrics fvd \
  --fvd_weights <local-i3d_pretrained_400.pt> \
  --fvd_clip_len 16 \
  --fvd_stride 16 \
  --max_fvd_clips_per_trajectory 64
```

Dataset-level FID with local Inception weights:

```bash
python -m eval.scripts.run_evaluation \
  --input_dir <evaluation-data-directory> \
  --output_dir eval/results \
  --metrics fid \
  --fid_weights <local-inception-weights.pth> \
  --fid_batch_size 64
```

Supported metrics are `all`, `visual`, `temporal`, `lpips`, `action`,
`objects`, `physical`, `multiview`, `fid`, and `fvd`, supplied as a comma-separated
list (or `all` alone). Other useful options are:

- `--frame_stride`: positive integer, default `1`.
- `--device`: `cpu` by default, or a requested CUDA device for LPIPS/detector.
- `--batch_size`: positive LPIPS batch size, default `16`.
- `--cvfc_batch_size`: positive CVFC feature-extraction batch size, default
  `32`; used only when `multiview` is requested.
- `--fid_batch_size`: positive, independent FID batch size, default `64`.
- `--fid_weights`: local Torch-Fidelity Inception-v3 weights; otherwise the
  evaluator checks its standard local Torch cache.
- `--fvd_weights`: local `cd-fvd` I3D-400 weights; otherwise the evaluator
  checks the checkpoint in the installed package.
- `--fvd_clip_len`, `--fvd_stride`: FVD clip length and start stride, defaults
  `16` and `16`.
- `--max_fvd_clips_per_trajectory`: per-camera trajectory cap, default `64`.
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

The seven raw robot diagnostics, CVOA, and CVFC use per-source-frame rows.
Trajectory and dataset values for these metrics deduplicate overlapping
interaction windows by trajectory/source-frame (and camera pair where
applicable), while per-interaction values remain independent.

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

The implementation uses packages listed in the repository's `requirements.txt`,
including TorchMetrics `1.8.2` for FID and `cd-fvd` for I3D FVD. TorchMetrics
`1.8.2` supports PyTorch `>=2.0,<2.9`, compatible with the repository's pinned
PyTorch `2.7.1`. Optional packages and local weights are needed only for the
corresponding metric. CVFC uses the already-declared `torchvision` dependency
and its fixed ImageNet weights. The evaluator does not install packages or
initialize Ctrl-World.

Reproducibility details (input paths, chosen metrics, device, sampling,
detector/tracker settings, annotation root, and preprocessing assumptions)
are written to `config.json` and `summary.json`. Preserve the exact input NPZ
files, local model weights, detector implementation, and any FVD feature
extractor protocol alongside those reports for repeatable comparisons.
