# Task 5C — Frame-dependent residual source audit

`FRAME_EFFECT_SOURCE = D. MIXED`

## Scope and definitions

- FIT only: `001–018 + 025–036` (30 frames). Frame 027 is retained and explicitly marked.
- Validation `019–024, 037–040` was not loaded into truth/residual/model evaluation.
- Frozen model: Task 5A 29-frame Circular diagnostic model; no refit or correction.
- Residual: `e_lambda = lambda_truth - lambda_model`.
- Per-frame stripe coordinate is the dominant PCA direction of `(u,v)`, sign-fixed toward increasing v, zero-mean and normalized by maximum absolute projection.
- Sensor term: fixed 60px v-bin offset plus within-bin u slope. Overlap cells: 32px × 30px × 5mm.

## 1. Frame offset / tilt

- 027: bias=0.2979 mm, RMSE=0.3688 mm, P95=0.5769 mm, a=0.2937 mm, k=0.3753 mm/s, offset+tilt explained=97.0%.
- 027 RMSE rank: 1/30 (1 is largest). Median non-027 RMSE=0.0686 mm.
- Valid intersection losses total 239; per-frame counts are retained in provenance.

## 2. Pose and acquisition correlation

Both Pearson and Spearman are in `frame_pose_correlation.csv`; scatter plots expose leverage and nonlinearity. A correlation is called robust only when |Spearman rho|≥0.60, p<0.01, sign-consistent in all-30 and leave-027-out scopes.

- Robust pose correlations: 2.
- Robust acquisition/PnP/image-quality correlations: 0.
- Roll/pitch are explicitly normal-derived camera-axis tilt components, not an unobservable arbitrary in-plane board rotation.

Strongest leave-027-out associations (diagnostic, uncorrected for multiple comparisons):

| outcome | predictor | category | Spearman rho | p | Pearson r |
|---|---|---|---:|---:|---:|
| k_frame_mm_per_s | board_roll_deg | pose_orientation | -0.659 | 0.000103 | -0.634 |
| k_frame_mm_per_s | board_ny | pose_orientation | 0.659 | 0.000103 | 0.632 |
| a_frame_mm | acquisition_order | acquisition_order | -0.570 | 0.00125 | -0.589 |
| bias_mm | acquisition_order | acquisition_order | -0.568 | 0.00129 | -0.585 |
| rmse_mm | v_center_px | coverage | -0.480 | 0.00844 | -0.462 |
| rmse_mm | laser_dynamic_range_u8 | image_quality | 0.454 | 0.0134 | 0.389 |
| rmse_mm | chess_focus_laplacian | image_quality | -0.419 | 0.0236 | -0.337 |
| rmse_mm | nolaser_to_laser_s | acquisition_timing | -0.391 | 0.0359 | -0.417 |
| bias_mm | chess_focus_laplacian | image_quality | -0.386 | 0.0385 | -0.319 |
| a_frame_mm | chess_to_laser_s | acquisition_timing | -0.384 | 0.0399 | -0.231 |

## 3. Cross-frame overlap

- Supported overlap cells: 566, covering 30 frames.
- Weighted between-frame variance fraction within matched cells: 83.1%.
- Frame-median residual range per cell: median 0.0656 mm, P95 0.2072 mm.
- Leave-027-out sensitivity: 551 cells / 29 frames, between-frame fraction 65.5%, median/P95 range 0.0632/0.1648 mm.
- Cells are selected only by fixed coordinates; the comparison plot uses the four most-supported cells, not cells selected by residual magnitude.

## 4. Variance decomposition

| scope | sensor-only | frame-only | combined | allocated sensor | allocated frame | unexplained |
|---|---:|---:|---:|---:|---:|---:|
| all 30 | 24.9% | 65.9% | 98.0% | 28.5% | 69.5% | 2.0% |
| leave 027 out | 42.4% | 39.0% | 96.4% | 49.9% | 46.5% | 3.6% |

All-30 / leave-027 commonality interaction is -7.3% / -15.0%. A negative value denotes suppressor/synergy from non-orthogonal sensor and frame designs; it is not negative physical energy. Shapley allocation averages both entry orders. This is an in-sample diagnostic partition, not a deployable model.

## Conclusion and next step

`FRAME_EFFECT_SOURCE = D. MIXED`

Next: run repeated triplets at a small factorial set of board depth/tilt and sensor-edge locations. The same-pose repeats identify acquisition/frame variance, while crossed sensor locations identify the fixed component; keep validation untouched until a single hypothesis is frozen.

No validation, refit, frame deletion, complex surface, or correction was used.
