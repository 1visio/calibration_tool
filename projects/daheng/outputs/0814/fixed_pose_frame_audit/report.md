# Task 5D-1 — Fixed-pose repeated-triplet audit

`FIXED_POSE_FRAME_EFFECT = A. LOW`

## Scope and boundary

- Data: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0814`; five explicit FIT triplets `001–005`, fixed board pose, no re-placement.
- Historical Validation was not opened or used. No Cone was fitted, refit, or written back; no compensation was created.
- Each group independently runs PnP on `chess`, Steger laser-center extraction from `laser − nolaser`, ray/plane `lambda_truth`, and the frozen Circular Cone reconstruction.
- Frozen Circular provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal cone/config SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`.
- Residual convention: `e_lambda = lambda_truth - lambda_model`. `a_frame` and `k_frame` are the intercept and normalized-stripe slope of `e_lambda`; `k_frame` is not a time derivative.
- The ten between-repeat comparisons are the ten unordered pairs `C(5,2)=10`. Overlap is evaluated on a 1-pixel v-grid common to each pair.

## Per-group frozen-Cone residuals

| group | valid points | PnP RMSE (px) | bias (mm) | RMSE (mm) | P95 abs (mm) | a_frame (mm) | k_frame (mm / normalized stripe) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 900 | 0.1196 | 0.2907 | 0.2989 | 0.4176 | 0.2907 | 0.0088 |
| 002 | 900 | 0.1178 | 0.2818 | 0.2899 | 0.4022 | 0.2818 | 0.0168 |
| 003 | 900 | 0.1171 | 0.2658 | 0.2748 | 0.3839 | 0.2658 | 0.0281 |
| 004 | 900 | 0.1164 | 0.2732 | 0.2814 | 0.3944 | 0.2732 | 0.0124 |
| 005 | 900 | 0.1147 | 0.2643 | 0.2728 | 0.3842 | 0.2643 | 0.0110 |

## Repeatability across the ten pairs

- `a_frame` std/range: **0.0111 / 0.0264 mm**.
- `k_frame` std/range: **0.0077 / 0.0193 mm per normalized stripe**.
- Pointwise overlap residual delta RMSE: median **0.0214 mm**, P95 across pairs **0.0319 mm**, max **0.0327 mm**.
- Pointwise overlap residual delta P95: median **0.0383 mm**, P95 across pairs **0.0498 mm**.
- Median pairwise laser-center delta RMSE: **0.0627 px**; median pairwise PnP-truth delta RMSE: **0.0059 mm**.
- The median frozen-model group RMSE is **0.2814 mm**; pairwise residual repeatability / group RMSE = **7.6%**.

## chess→laser motion and board micro-motion

- chess→laser marker detection: **0/5** laser images.
- Because no laser image exposed a detectable chessboard/marker, chess→laser translation/rotation/homography is **not observable from these images**; this is reported as `not_detected`, not as zero motion.
- Independent chess→chess repeat registration across groups is available for 10 pairs: max P95 displacement **0.0395 px**, max center displacement **0.0243 px**, max absolute rotation **0.001749 deg**.
- These chess-only changes are subpixel and do not show material board micro-motion at the stated thresholds.

## Conclusion

`FIXED_POSE_FRAME_EFFECT = A. LOW`.
The repeated fixed-pose triplets show a small but measurable acquisition repeatability floor. It is not large enough to explain the dominant frozen-Cone residual bias; the evidence does not support calling the current frame effect strong or primarily caused by chessboard micro-motion.

Classification gates used in this report are descriptive and declared before the conclusion: LOW = pairwise overlap residual RMSE ≤10% of the median group RMSE, both a/k ranges ≤0.05 mm, and chess repeat P95 ≤0.10 px; MODERATE = corresponding ≤25%/≤0.10 mm; otherwise STRONG.
