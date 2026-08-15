# Task 6B — Cone-intrinsic residual audit

`INTRINSIC_SURFACE_STRUCTURE = C. WEAK`

## Scope and frozen-data boundary

- FIT-only frames: `001–018`, `025–036` (30 frames, 26663 valid points); frame `027` retained and marked.
- Only explicit FIT triplets under `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane/fit` and `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane/fit_edge_extension/fit` were opened. Validation 019–024 and 037–040 were not read.
- No Cone was fitted/refit, no Elliptical Cone was fitted, no correction/LUT was built, no frame was deleted, and the existing Steger/PnP path was unchanged.
- Frozen provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`; frame bootstrap B=300, seed=20260815.

## Cone-intrinsic convention

- `q = P_truth − apex`, `a = q·d`, `r = ||q − a d||`, `phi = atan2(q·e2,q·e1)` in degrees, and `alpha_truth = atan2(r,a)`.
- Frozen model half-angle: **88.87412°**; the exact per-point `alpha_model_deg` is in `cone_intrinsic_points.csv`.
- Fixed basis (camera coordinates): d=(0.960610, 0.008807, -0.277760), e1=(0.009168, -0.999958, 0.000000), e2=(-0.277748, -0.002546, -0.960651).
- `delta_alpha = alpha_truth − alpha_model`; `e_surface` uses the CircularConeModel signed normal-distance formula; `e_lambda = lambda_truth − lambda_model`.

## Plane/PnP sanity

- Maximum PnP RMSE: **0.1836 px**; maximum board-Z RMSE: **0.000000 mm**; geometry gate: **True**.
- Global alpha_truth: mean **88.87447°**, std **0.00392°**, range **0.04429°**; delta_alpha std **0.00392°**.

## Intrinsic residual relationships

| predictor → outcome | Spearman rho | binned EV | bootstrap rho 95% CI | bootstrap EV 95% CI |
|---|---:|---:|---:|---:|
| phi_deg → delta_alpha | 0.00222 | 0.03182 | [-0.11849, 0.10847] | [0.00656, 0.15553] |
| a_mm → delta_alpha | -0.08494 | 0.00690 | [-0.25932, 0.10307] | [0.00052, 0.05588] |

- e_lambda vs phi: rho **0.00112**, binned EV **0.03345**; e_lambda vs a: rho **-0.08647**, binned EV **0.00758**.
- Harmonic diagnostic delta_alpha(phi): R² **0.07776**, first-harmonic amplitude **28.47523°**, second-harmonic amplitude **7.21541°**; observed phi span **31.433°**, design condition **62535.3**, identifiable **False**. Because the sampled azimuth span is limited, this harmonic fit is diagnostic only and is not interpreted as an elliptical structure or correction.

## Cross-frame consistency

| coordinate system | scale | pair count | median matched delta RMSE / mm | median cross/within variance ratio |
|---|---|---:|---:|---:|
| cone_intrinsic | fine | 321 | 0.0689 | 0.9418 |
| cone_intrinsic | medium | 45 | 0.0668 | 1.4417 |
| sensor_v_overlap | 1px_v | 435 | 0.0761 | n/a |

- Frame 027 (retained separately): alpha_truth mean **88.88614°**, delta_alpha mean **0.01202°**, e_lambda bias **0.2979 mm**, e_lambda RMSE **0.3688 mm**.
- Best intrinsic scale: **medium**, median matched delta RMSE **0.0668 mm**; sensor-v overlap baseline **0.0761 mm**; relative improvement **12.2%**.
- Cross-frame sampling explanation gate: **False**. Intrinsic matching is considered an improvement only when the best scale reduces the sensor-v overlap delta by at least 20%; all reported scales remain visible in the CSV.

## Answers

1. Residual is more stable in Cone intrinsic coordinates than sensor-v: **False**.
2. Stable azimuth-dependent cone-angle deviation: **False** (rho=0.00222; frame-balanced bootstrap and binned checks are in `circular_symmetry_audit.csv`).
3. Frame effect is mainly different-frame sampling of distinct Cone surface regions: **False**.
4. Observed deviation is classified as **circular**; no Elliptical Cone was fitted.
5. Next step: **re-audit truth construction**.

## Conclusion

`INTRINSIC_SURFACE_STRUCTURE = C. WEAK`.
The classification is descriptive: it combines multiple intrinsic bin scales, point-level Spearman/binned statistics, frame-resampled bootstrap intervals, and pairwise cross-frame matching. It is not a correction model.

Generated figures: `delta_alpha_vs_phi.png`, `delta_alpha_vs_a.png`, `delta_alpha_heatmap_phi_a.png`, `e_lambda_heatmap_phi_a.png`, and `cross_frame_consistency_sensor_vs_intrinsic.png`.
