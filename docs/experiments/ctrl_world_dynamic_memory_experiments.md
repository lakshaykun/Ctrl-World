# Ctrl-World Dynamic Memory Experiments

## Research Goal

Investigate whether replacing Ctrl-World's fixed historical-memory selection with a dynamic memory-selection strategy can improve controllability, temporal consistency, visual quality, and object persistence.

The baseline uses a fixed set of historical memory indices. The experiments progressively introduce dynamic selection:

1. **Baseline:** Original fixed-memory selection.
2. **V1 — Dynamic Memory Selection:** Select memories dynamically using current pose, upcoming actions, recency, and pose diversity.
3. **V2 — Action-First Selection with Temporal Diversity:** Prioritize action-relevant memories while reserving memory slots for temporal coverage and pose diversity.

> **Evaluation status:** The results below are preliminary. Experiments were evaluated on four trajectories (1799, 18599, 199, and 899), with 12 samples per trajectory for the V1 evaluation. Do not interpret these results as statistically significant or generalizable. Some metrics have sparse/invalid measurements, and comparisons may not be fully paired.

---

# 1. Baseline — Original Fixed Historical Memory

## Method

The original Ctrl-World rollout uses a fixed historical-memory index pattern rather than adapting memory selection to the current scene or upcoming actions.

Example selection pattern:

```python
history_idx = [0, 0, -8, -6, -4, -2]
```

This provides a predetermined temporal sampling of historical latent memories.

## Aggregate Results

| Metric | Baseline |
|---|---:|
| PSNR ↑ | 22.9574 |
| SSIM ↑ | 0.83391 |
| LPIPS ↓ | 0.10807 |
| Temporal consistency ↑ | 0.97656 |
| Action-image-motion correlation ↑ | 0.39254 |
| Action-object displacement correlation | -0.07994* |
| Action-object displacement error ↓ | 0.0040* |
| Object persistence ↑ | 0.3940 |
| Object hallucination rate ↓ | 0.49864 |
| Hallucination severity ↓ | 0.10933 |
| Physical violation rate ↓ | 0.13264 |
| Position anomaly rate ↓ | 0.00000 |
| Cross-view feature consistency error ↓ | 0.0372 |
| Cross-view object agreement difference | -0.0053* |

\* Action-object metrics have limited valid measurements; trajectory 199 had a NaN for the relevant correlation, so the comparison is not fully paired.

## Baseline Interpretation

- The fixed memory pattern provides the reference point for evaluating dynamic memory-selection methods.
- It has the highest aggregate action-image-motion correlation among the three methods in the current preliminary comparison.
- It also has lower hallucination rate, hallucination severity, and physical-violation rate than both V1 and V2 in this preliminary aggregate.
- Its temporal consistency and cross-view feature consistency are weaker than the dynamic-memory variants.

---

# 2. V1 — Dynamic Memory Selection

## Method

V1 replaces the fixed historical indices with a dynamic selection mechanism.

The selection considers:

- Current pose
- Upcoming actions
- Recency
- Pose diversity

The objective is to select a set of historical memories that is more relevant to the current state and future action context than a fixed temporal pattern.

## Aggregate Results

| Metric | Baseline | V1 | Change (V1 − Baseline) |
|---|---:|---:|---:|
| PSNR ↑ | 22.9574 | 23.2045 | +0.2471 |
| SSIM ↑ | 0.83391 | 0.83978 | +0.00587 |
| LPIPS ↓ | 0.10807 | 0.10080 | -0.00727 |
| Temporal consistency ↑ | 0.97656 | 0.99390 | +0.01734 |
| Action-image-motion correlation ↑ | 0.39254 | 0.33742 | -0.05512 |
| Action-object displacement correlation | -0.07994* | 0.07711* | — |
| Action-object displacement error ↓ | 0.0040* | 0.0048* | +0.0008 |
| Object persistence ↑ | 0.3940 | 0.4428 | +0.0488 |
| Hallucination rate ↓ | 0.49864 | 0.53264 | +0.03400 |
| Hallucination severity ↓ | 0.10933 | 0.11151 | +0.00218 |
| Physical violation rate ↓ | 0.13264 | 0.13889 | +0.00625 |
| Position anomaly rate ↓ | 0.00000 | 0.00000 | 0.00000 |
| Cross-view feature consistency error ↓ | 0.0372 | 0.0358 | -0.0014 |
| Cross-view object agreement difference | -0.0053* | 0.0133* | — |

\* Sparse or metric-dependent measurements; avoid treating these as definitive paired improvements.

## V1 Interpretation

### Improvements over the baseline

- **Visual quality:** PSNR and SSIM increased.
- **Perceptual quality:** LPIPS decreased.
- **Temporal consistency:** Increased substantially.
- **Object persistence:** Improved.
- **Cross-view feature consistency:** Slightly improved.

### Trade-offs / regressions

- Action-image-motion correlation decreased.
- Hallucination rate increased.
- Hallucination severity increased slightly.
- Physical-violation rate increased slightly.
- Action-object displacement error increased in the available aggregate.

### Preliminary conclusion

V1 is the strongest overall visual-quality candidate among the current methods. It improves PSNR, SSIM, LPIPS, temporal consistency, object persistence, and cross-view feature error relative to the baseline, but these gains are accompanied by trade-offs in action-motion correlation and some physical/hallucination metrics.

---

# 3. V2 — Action-First Memory Selection with Temporal Diversity

## Motivation

V1 showed that dynamic memory selection can improve visual and temporal metrics, but the action-motion correlation decreased. V2 was designed to prioritize action-relevant historical memories while preserving temporal coverage and diversity.

## Method

V2 uses a heuristic selection strategy with the following stages:

1. **Retain a current/recent anchor memory.**
2. **Reserve two memory slots for upcoming-action-relevant memories.**
3. **Fill remaining slots using temporal bins/coverage and pose diversity.**
4. **Sort selected memories chronologically.**

The method is heuristic and does not introduce additional training or change model weights.

Key configurable parameters in the V2 implementation include:

- `num_memories = 6`
- `action_weight = 0.70`
- `recency_weight = 0.15`
- `current_pose_weight = 0.15`
- `action_slots = 2`
- `temporal_bins = 4`
- `diversity_weight = 0.20`

## V2 Aggregate Results

Unweighted mean across the four trajectories. NaN values are ignored for action-object displacement correlation.

| Metric | V2 Aggregate |
|---|---:|
| PSNR ↑ | 23.121232 |
| SSIM ↑ | 0.836737 |
| LPIPS ↓ | 0.105776 |
| Temporal consistency ↑ | 0.996528 |
| Action-image-motion correlation ↑ | 0.340672 |
| Action-object displacement correlation | 0.125894* |
| Action-object displacement error ↓ | 0.003206* |
| Object persistence ↑ | 0.356876 |
| Hallucination rate ↓ | 0.582936 |
| Hallucination severity ↓ | 0.117129 |
| Physical violation rate ↓ | 0.194444 |
| Cross-view feature consistency error ↓ | 0.034116 |
| Cross-view object agreement difference | -0.007964 |

\* Limited valid measurements; interpret cautiously.

## V2 Interpretation

### Improvements / strengths

- Highest temporal consistency among the three methods in the current aggregate.
- Lowest cross-view feature consistency error among the three.
- Visual metrics are modestly better than baseline, although not as strong as V1.
- The action-first design can produce strong action-motion correlation on some trajectories (for example, trajectory 18599), but performance is highly trajectory-dependent.

### Trade-offs / weaknesses

- V2's aggregate PSNR, SSIM, and LPIPS are worse than V1.
- Object persistence is lower than V1.
- Hallucination rate and severity are higher than baseline and V1.
- Physical-violation rate is highest in the current preliminary comparison.
- Action-motion correlation remains below baseline in the aggregate.
- Trajectory-level results vary substantially.

## Preliminary conclusion

V2 improves temporal consistency and cross-view feature consistency, but it does not provide a clear overall improvement over V1. Its action-first selection may help action relevance on some trajectories, while temporal diversity and action prioritization can trade off against object continuity and physical consistency.

---

# 4. Baseline vs. V1 vs. V2 — Aggregate Comparison

| Metric | Baseline | V1 | V2 | Best in Current Comparison |
|---|---:|---:|---:|---|
| PSNR ↑ | 22.9574 | **23.2045** | 23.1212 | V1 |
| SSIM ↑ | 0.83391 | **0.83978** | 0.83674 | V1 |
| LPIPS ↓ | 0.10807 | **0.10080** | 0.10578 | V1 |
| Temporal consistency ↑ | 0.97656 | 0.99390 | **0.99653** | V2 |
| Action-image-motion correlation ↑ | **0.39254** | 0.33742 | 0.34067 | Baseline |
| Object persistence ↑ | 0.3940 | **0.4428** | 0.3569 | V1 |
| Hallucination rate ↓ | **0.49864** | 0.53264 | 0.58294 | Baseline |
| Hallucination severity ↓ | **0.10933** | 0.11151 | 0.11713 | Baseline |
| Physical violation rate ↓ | **0.13264** | 0.13889 | 0.19444 | Baseline |
| Cross-view feature consistency error ↓ | 0.0372 | 0.0358 | **0.03412** | V2 |

## Overall Finding So Far

No single method dominates all metrics.

- **Baseline:** Strongest in action-image-motion correlation and the current hallucination/physical-violation metrics.
- **V1:** Strongest overall visual quality and object persistence; improves temporal consistency over baseline.
- **V2:** Strongest temporal consistency and cross-view feature consistency, but with weaker object persistence and worse hallucination/physical-violation metrics.

The current results suggest that memory selection involves competing objectives:

- Action relevance
- Temporal coverage
- Visual quality
- Object identity and persistence
- Physical consistency
- Cross-view consistency

A future memory-selection strategy may need to explicitly balance these objectives rather than optimizing action relevance or temporal diversity alone.

---

# 5. Limitations of the Current Evaluation

1. Only four trajectories have been evaluated in the current preliminary comparison.
2. The results are not sufficient to establish statistical significance.
3. Some action-object metrics have sparse or invalid measurements.
4. The aggregate comparison may not be fully paired across methods and metrics.
5. Metric averages can hide trajectory-specific behavior.
6. V2's selection heuristic has not yet been isolated through ablation studies.
7. Visual inspection of corresponding baseline/V1/V2 frames and videos is still needed.

---

# 6. Recommended Next Experiments

## Before designing V3

1. Compare baseline, V1, and V2 videos/frames using the same trajectories and interaction sequences.
2. Inspect the memory indices selected over time for each method.
3. Analyze per-interaction metrics, especially those associated with hallucinations and physical violations.
4. Determine whether V2's action-selected memories are actually relevant to the current action and whether they disrupt object continuity.
5. Run a larger and more consistently paired evaluation.

## Potential V3 direction

A refined method could explicitly balance:

- Action relevance
- Temporal coverage
- Current-state relevance
- Object continuity
- Pose diversity

However, V3 should be designed after the diagnostic analysis rather than immediately adding more heuristics.

---

# 7. Experiment Timeline

| Stage | Method | Main change | Current status |
|---|---|---|---|
| Baseline | Fixed historical memory | Predetermined history indices | Evaluated |
| V1 | Dynamic memory selection | Current pose, upcoming actions, recency, pose diversity | Implemented and evaluated |
| V2 | Action-first + temporal diversity | Action-relevant slots plus temporal coverage and pose diversity | Implemented and evaluated |
| V3 | Multi-objective refined selection | To be designed after diagnostics | Not started |

