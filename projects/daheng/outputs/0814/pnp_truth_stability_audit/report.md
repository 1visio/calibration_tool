# Task 6C — Multi-pose PnP truth stability audit

`PNP_TRUTH_STABILITY = C. WEAK`

## Scope and boundary

- FIT-only frames: `001–018`, `025–036` (30 frames, including retained sensitivity frame `027`). Only explicit FIT files were opened; Validation 019–024 and 037–040 were not read.
- Formal intrinsics/distortion and the existing 11×8, 20 mm PnP/Steger path were reused. No formal intrinsics were changed, no laser surface was fitted, no frame was deleted, and no correction was created.
- Frozen provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`. Cone is used only as the observed residual reference.

## Full-board baseline and subset definitions

- Full-board baseline is the current `solvePnP(ITERATIVE)` + `solvePnPRefineLM` result from the existing detector. Each subset uses the same detected corner coordinates and the same laser pixels as that frame's full baseline.
- Truth-only lambda statistics use all extracted centers with valid full-board ray-plane intersections; the frozen Cone z-range mask is not used to select truth pixels. Cone-valid points are reported separately for the residual reference.
- Subsets: `left_half`, `right_half`, `top_half`, `bottom_half` (center row/column retained in neighboring halves), and `central_subset` (x=40–160 mm, y=20–100 mm). All have rank-2 spatial support.
- Truth delta convention: `delta_lambda_truth = lambda_subset − lambda_full`. Frozen Cone reference convention here: `e_lambda = lambda_cone − lambda_truth_full`.

## Stability summary

- Maximum full-board PnP RMSE: **0.1836 px**.
- Maximum subset P95 |delta lambda_truth|: **0.2131 mm**; maximum subset RMSE: **0.1997 mm**.
- Solver diagnostic maximum P95 |delta lambda_truth|: **0.7047 mm**; maximum solver RMSE: **0.4348 mm**.
- Frozen Cone e_lambda RMSE across frames: median **0.0701 mm**, maximum **0.3688 mm**.
- Frame 027: subset truth P95 max **0.1217 mm**, solver P95 max **0.0193 mm**, Cone e_lambda RMSE **0.3688 mm**, a_frame **-0.2979 mm**, k_frame **-0.3746 mm/normalized stripe**.

| subset | median lambda RMSE / mm | max lambda RMSE / mm | median P95 / mm | max P95 / mm | max normal angle / deg | max plane distance / mm |
|---|---:|---:|---:|---:|---:|---:|
| left_half | 0.0286 | 0.1198 | 0.0440 | 0.1636 | 0.0420 | 0.1432 |
| right_half | 0.0338 | 0.0799 | 0.0460 | 0.1204 | 0.0655 | 0.1533 |
| top_half | 0.0575 | 0.1997 | 0.0852 | 0.2131 | 0.0904 | 0.4943 |
| bottom_half | 0.0454 | 0.1614 | 0.0713 | 0.1813 | 0.0782 | 0.2980 |
| central_subset | 0.0196 | 0.0567 | 0.0330 | 0.0885 | 0.0528 | 0.0964 |

## Planar solver comparison

| method | successful frames | median lambda RMSE / mm | max lambda RMSE / mm | max P95 / mm | max normal angle / deg |
|---|---:|---:|---:|---:|---:|
| ippe | 30 | 0.0139 | 0.4348 | 0.7047 | 0.4914 |
| homography | 30 | 0.0179 | 0.1174 | 0.1622 | 0.0453 |

## Reprojection residual field

- Mean left/right du difference: **0.00522 px**; mean top/bottom dv difference: **-0.00517 px**; radial magnitude Spearman rho: **0.19873** (p=0.0000).
- The complete per-corner du/dv field is in `corner_reprojection_field.csv`; vector-field and subset consistency figures are generated without fitting a correction.

## Truth instability versus observed Cone residual

- Maximum absolute frame-level correlation between truth-instability metrics and Cone RMSE/bias/a/k: **0.5043**; detailed Spearman values are in `frame_truth_uncertainty_summary.csv`.
- This comparison is quantitative only: it does not assign truth uncertainty to a correction term.

## Answers

1. Full-board PnP is subset-sensitive at the reported scale: **True**; see all subset rows and plane metrics.
2. IPPE/homography produce materially different lambda_truth: **True** under the 0.02 mm diagnostic margin.
3. PnP truth uncertainty is sufficient to explain the observed frame effect: **True**; observed Cone RMSE range is approximately 0.05–0.3 mm while the detailed truth deltas are in the CSV.
4. Pose-related reprojection pattern detected: **False** under declared descriptive margins.
5. Next step: **upgrade PnP truth construction**.

## Conclusion

`PNP_TRUTH_STABILITY = C. WEAK`.
Classification gates are declared descriptive gates: STRONG requires all subset/solver P95 deltas <0.02 mm and no meaningful frame correlation; MODERATE requires maxima <0.05 mm; WEAK means truth instability is at the observed residual scale or clearly correlated; otherwise INSUFFICIENT.

Generated figures: `p95_delta_lambda_truth_by_frame.png`, `truth_instability_vs_cone_residual_rmse.png`, `corner_reprojection_residual_vector_field.png`, and `plane_normal_distance_consistency_by_subset.png`.
