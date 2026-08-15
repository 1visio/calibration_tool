# Task 6A — Board-coordinate residual audit

`BOARD_COORDINATE_EFFECT = C. WEAK`

## Scope and boundary

- FIT-only frames: `001–018`, `025–036` (30 frames, 26663 valid points); frame `027` is retained and marked separately.
- Input roots: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane/fit` and `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane/fit_edge_extension/fit`. Inventory resolves only these explicit FIT filenames; no Validation image, Validation frames.csv, or Validation-derived points were opened.
- No Cone was fitted/refit, no correction/LUT was created, no frame was deleted, and the existing Steger configuration was used unchanged (uniform 900-point cap per frame).
- Frozen Circular provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`.
- Frozen provenance declares `validation_opened=false`, main FIT IDs excluding 027, and sensitivity case 027. Bootstrap: frame-resampling, B=300, seed=20260814.
- Coordinate convention: OpenCV PnP `P_cam = R P_board + t`; `P_board = R.T @ (P_cam - t)`. Board origin is the first detected inner corner, X/Y axes follow the 11×8 object-point order, and square size is 20 mm.
- Residual convention: `e_lambda = lambda_truth - lambda_model`; all relationship fits are one-dimensional linear or binned means only, never a high-order correction model.

## Board-plane/PnP consistency

- Across all frames, maximum PnP RMSE: **0.1836 px**; maximum per-frame board-Z RMSE: **0.000000 mm**.
- Frame 027: bias **0.2979 mm**, RMSE **0.3688 mm**, P95 **0.5769 mm**, stripe angle **83.928°**; bias z-score versus other FIT frames **7.980**.

## Frame-level residual geometry

| frame | 027 | bias mm | RMSE mm | P95 mm | a_frame mm | k_frame | stripe angle board ° | board-Z RMSE mm | PnP RMSE px |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 |  | 0.0144 | 0.0769 | 0.1378 | 0.0144 | -0.1324 | 98.690 | 0.000000 | 0.1633 |
| 002 |  | 0.0453 | 0.0652 | 0.0914 | 0.0453 | 0.0282 | 86.465 | 0.000000 | 0.1480 |
| 003 |  | 0.0616 | 0.0854 | 0.1333 | 0.0616 | 0.0643 | 170.915 | 0.000000 | 0.1508 |
| 004 |  | 0.0298 | 0.1207 | 0.2330 | 0.0298 | -0.1382 | 92.816 | 0.000000 | 0.1009 |
| 005 |  | 0.0392 | 0.0633 | 0.1278 | 0.0392 | -0.0015 | 3.273 | 0.000000 | 0.1030 |
| 006 |  | -0.0062 | 0.0655 | 0.1216 | -0.0062 | -0.0308 | 83.787 | 0.000000 | 0.0962 |
| 007 |  | 0.0516 | 0.0807 | 0.1574 | 0.0516 | 0.0193 | 91.090 | 0.000000 | 0.1162 |
| 008 |  | -0.0097 | 0.0676 | 0.1211 | -0.0097 | -0.0101 | 141.625 | 0.000000 | 0.0955 |
| 009 |  | 0.0400 | 0.0849 | 0.1551 | 0.0400 | 0.0025 | 89.349 | 0.000000 | 0.1097 |
| 010 |  | -0.0241 | 0.0686 | 0.1191 | -0.0241 | -0.0192 | 89.872 | 0.000000 | 0.1190 |
| 011 |  | -0.0336 | 0.0739 | 0.1461 | -0.0336 | 0.0471 | 87.375 | 0.000000 | 0.1176 |
| 012 |  | 0.0228 | 0.0494 | 0.0857 | 0.0228 | 0.0077 | 92.102 | 0.000000 | 0.0883 |
| 013 |  | 0.0076 | 0.0595 | 0.1159 | 0.0076 | 0.0481 | 1.290 | 0.000000 | 0.1072 |
| 014 |  | 0.0102 | 0.0758 | 0.1248 | 0.0102 | -0.0250 | 93.226 | 0.000000 | 0.0959 |
| 015 |  | -0.0293 | 0.0851 | 0.1688 | -0.0293 | -0.1091 | 100.481 | 0.000000 | 0.1107 |
| 016 |  | 0.0337 | 0.0607 | 0.1154 | 0.0337 | -0.0352 | 7.858 | 0.000000 | 0.1245 |
| 017 |  | -0.0242 | 0.0715 | 0.1295 | -0.0242 | -0.0398 | 101.428 | 0.000000 | 0.1183 |
| 018 |  | -0.0577 | 0.0932 | 0.1637 | -0.0577 | 0.0251 | 2.053 | 0.000000 | 0.1216 |
| 025 |  | 0.0358 | 0.0729 | 0.1170 | 0.0358 | 0.0457 | 172.726 | 0.000000 | 0.1562 |
| 026 |  | -0.0127 | 0.0634 | 0.1243 | -0.0127 | 0.0152 | 30.760 | 0.000000 | 0.1653 |
| 027 | yes | 0.2979 | 0.3688 | 0.5769 | 0.2979 | 0.3746 | 83.928 | 0.000000 | 0.1741 |
| 028 |  | -0.0494 | 0.0859 | 0.1611 | -0.0494 | -0.0025 | 2.693 | 0.000000 | 0.1836 |
| 029 |  | 0.0168 | 0.0575 | 0.1333 | 0.0168 | -0.0219 | 2.657 | 0.000000 | 0.1052 |
| 030 |  | -0.0050 | 0.0498 | 0.1080 | -0.0050 | -0.0082 | 5.626 | 0.000000 | 0.1310 |
| 031 |  | -0.0083 | 0.0473 | 0.1022 | -0.0083 | 0.0376 | 4.177 | 0.000000 | 0.1137 |
| 032 |  | -0.0386 | 0.0628 | 0.1323 | -0.0386 | 0.0241 | 4.007 | 0.000000 | 0.1294 |
| 033 |  | -0.0181 | 0.0496 | 0.1111 | -0.0181 | 0.0219 | 169.576 | 0.000000 | 0.1221 |
| 034 |  | 0.0246 | 0.0492 | 0.0938 | 0.0246 | 0.0242 | 0.817 | 0.000000 | 0.1003 |
| 035 |  | -0.0941 | 0.1107 | 0.2062 | -0.0941 | 0.0633 | 175.269 | 0.000000 | 0.1332 |
| 036 |  | -0.0509 | 0.0815 | 0.1659 | -0.0509 | 0.0314 | 172.207 | 0.000000 | 0.0908 |

## Spearman and binned explained variance

Top point-level Spearman magnitudes (all FIT points; LOO and frame bootstrap are in `board_vs_sensor_comparison.csv`):

| predictor | family | rho | p | binned EV | bootstrap EV 95% CI | LOO EV min/max |
|---|---|---:|---:|---:|---:|---:|
| stripe_angle_to_x_deg | stripe_angle | 0.14661 | 0.000 | 0.12966 | [0.03978, 0.39994] | [0.05376, 0.19609] |
| stripe_angle_to_y_deg | stripe_angle | -0.14661 | 0.000 | 0.12966 | [0.02473, 0.43651] | [0.05376, 0.19609] |
| distance_to_grid_intersection_mm | grid_boundary | 0.05774 | 0.000 | 0.00873 | [0.00355, 0.03068] | [0.00453, 0.01309] |
| distance_to_horizontal_grid_line_mm | grid_boundary | 0.05096 | 0.000 | 0.00355 | [0.00124, 0.02279] | [0.00215, 0.00724] |
| stripe_angle_board_deg | stripe_angle | -0.04532 | 0.000 | 0.14082 | [0.00232, 0.44452] | [0.00722, 0.30434] |
| board_Xb_mm | board | -0.04138 | 0.000 | 0.08593 | [0.05988, 0.21671] | [0.07530, 0.09776] |

Highest binned explained-variance predictors:

| predictor | family | binned EV | simple linear R² | bootstrap EV 95% CI | frame 027 EV |
|---|---|---:|---:|---:|---:|
| sensor_v_px | sensor | 0.20964 | 0.00006 | [0.15048, 0.44357] | 0.93959 |
| stripe_angle_board_deg | stripe_angle | 0.14082 | 0.00001 | [0.00232, 0.44452] | 0.00000 |
| stripe_angle_to_x_deg | stripe_angle | 0.12966 | 0.03548 | [0.03978, 0.39994] | 0.00000 |
| stripe_angle_to_y_deg | stripe_angle | 0.12966 | 0.03548 | [0.02473, 0.43651] | 0.00000 |
| board_Xb_mm | board | 0.08593 | 0.00150 | [0.05988, 0.21671] | 0.46542 |
| board_Yb_mm | board | 0.04936 | 0.00495 | [0.03402, 0.17211] | 0.95974 |

## Grid-boundary and stripe-angle checks

- Mean residual in the first 2 mm from a grid line: **0.0010 mm**; at 8–10 mm from a grid line: **0.0146 mm**. These are descriptive binned means, not a fitted correction.

| frame-level predictor → outcome | rho (leave 027 out) | bootstrap 95% CI | LOO sign consistency |
|---|---:|---:|---:|
| stripe_angle_board_deg → bias_mm | -0.03645 | [-0.48224, 0.33402] | 0.828 |
| stripe_angle_board_deg → a_frame_mm | -0.03645 | [-0.49199, 0.43567] | 0.828 |
| stripe_angle_board_deg → k_frame_mm_per_normalized_stripe | 0.00148 | [-0.45178, 0.41471] | 0.655 |

## Board versus sensor interpretation

- All FIT points: best sensor-coordinate binned EV **0.20964**; best board/grid predictor **board_Xb_mm = 0.08593**; board-minus-sensor = **-0.12371**.
- Excluding retained sensitivity frame 027: best sensor EV **0.37067**; best board/grid EV **0.07530** (`board_Xb_mm`).
- Best grid predictor: **grid_x_mod_20_mm = 0.02472**; stable grid effect = **False**; stable stripe-angle effect = **False**.
- Frame 027 follows the main frame-level bias pattern under the declared check: **False**.
- Current evidence classification: **sensor/model error**.

## Conclusion

`BOARD_COORDINATE_EFFECT = C. WEAK`.

1. The report does not treat a single Pearson coefficient as evidence: every predictor has Spearman, binned means, leave-one-frame-out ranges, and frame-bootstrap intervals in the CSV.
2. A stable board/grid/angle signature is **False** under the declared descriptive gates; a fixed checkerboard-coordinate explanation is therefore classified as **C. WEAK**.
3. The current source assessment is **sensor/model error**. Grid-boundary and angle results should be read together with the point-level extraction quality fields (intensity, contrast, FWHM, Steger response).
4. Task 6B horizontal in-plane rotation experiment: **NO / not yet**. With no stable board/grid/angle signature in this audit, it is not the next priority; it can be revisited specifically to diagnose the independent 027 anomaly.

Generated figures: `residual_vs_Xb_Yb.png`, `residual_vs_grid_phase.png`, `residual_vs_grid_distance.png`, `frame_bias_k_vs_stripe_angle.png`, and `board_vs_sensor_explained_variance.png`.
