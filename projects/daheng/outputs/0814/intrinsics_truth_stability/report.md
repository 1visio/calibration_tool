# Task 6E — Camera intrinsics → ray-plane truth stability audit

`INTRINSICS_TRUTH_STABILITY = C. LOW`

## Scope and boundary

- Formal camera-calibration FIT: `chess 001.tif`–`chess 018.tif` (18 frames), all used by the current 0811 K/D fit. Only these FIT calibration images were opened.
- Laser diagnostic FIT: 001–018 and 025–036 (30 frames). No `laser_plane/validation` image was opened; no laser Validation data was read.
- Baseline intrinsics: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`; image size 4096×3000; pattern 11×8, square 20 mm.
- Formal K/D flags retained: `CALIB_FIX_K3` (k3=0); candidate PnP uses `SOLVEPNP_ITERATIVE` plus `solvePnPRefineLM` when available.
- Laser center UV, Steger settings, frozen Circular Cone, formal intrinsics, and frame membership were not changed. Cone is only used as an observed-residual reference.
- Frozen Cone provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`.

## Camera calibration coverage

- Board-center normalized span: u **0.335**, v **0.301**; center range u=1402.2–2773.1px, v=1016.5–1920.9px.
- Board tilt range: **2.53–24.77°**; formal per-image RMSE range: **0.0896–0.1621px**.
- The coverage is multi-pose and spans the sensor, but it is not a dense uniform calibration grid; edge leverage is represented by a subset of poses rather than every frame.

## Leave-one-calibration-frame-out

- Re-estimated K/D for **18** LOO candidates with all 88 corners per retained frame and the unchanged k3-fixed model.
- LOO max |Δfx|/fx = **0.0594%**, max |Δfy|/fy = **0.0600%**, max |Δcx| = **1.312px**, max |Δcy| = **2.336px**.

## Frame bootstrap

- Frame-level bootstrap: **500** successful / 500 requested replicates; samples whole calibration frames with replacement and never splits corners.

| parameter | formal | bootstrap median | 2.5% | 97.5% |
|---|---:|---:|---:|---:|
| fx | 7347.8651 | 7348.3467 | 7336.5856 | 7359.5316 |
| fy | 7348.8411 | 7349.2038 | 7337.4073 | 7360.8361 |
| cx | 2070.2165 | 2070.1855 | 2065.0696 | 2073.7935 |
| cy | 1512.4365 | 1512.1007 | 1504.4144 | 1519.6333 |
| k1 | -0.049137165 | -0.049447955 | -0.051046872 | -0.04773503 |
| k2 | 0.22727597 | 0.22796483 | 0.21203561 | 0.24277471 |
| p1 | -0.00051315423 | -0.00051376477 | -0.00076602007 | -0.00022791842 |
| p2 | -0.0004783779 | -0.00048832233 | -0.00075066626 | -0.00027640605 |
| k3 | 0 | 0 | 0 | 0 |

## Propagation to laser truth

- Candidate K/D were propagated to the same extracted laser UV points for all 30 FIT frames. Median candidate P95 |Δlambda|: **0.3523 mm**; median bootstrap 95%-candidate P95: **1.0944 mm**; worst frame tail: **1.1765 mm**.
- Intrinsics-induced / frozen-Cone RMSE ratio: median candidate **4.932**, median 95%-candidate tail **15.276**, worst **23.220**.
- Sensor-v edge/middle amplification: median **1.005×**, maximum **1.038×**.
- Across frames, bootstrap uncertainty vs board tilt Spearman rho = **-0.677** (p=0.000); this is a pose/coverage dependence in the frame bootstrap, not a sensor-v edge amplification.
- Signed direction check: median candidate Δlambda bias is **0.0278–0.0461 mm** across frames; versus Cone frame bias Spearman rho **0.253** (p=0.177), same sign in **0.500** of frames.

| frame | Cone RMSE mm | bootstrap P95-of-P95 Δlambda mm | uncertainty/Cone | v-edge/middle | 027 |
|---:|---:|---:|---:|---:|:---:|
| 001 | 0.07688 | 1.07011 | 13.920 | nan |  |
| 002 | 0.06524 | 1.07669 | 16.505 | nan |  |
| 003 | 0.08537 | 1.05140 | 12.315 | 1.013 |  |
| 004 | 0.12071 | 1.04663 | 8.671 | 0.976 |  |
| 005 | 0.06329 | 1.07436 | 16.975 | 1.016 |  |
| 006 | 0.06553 | 1.13876 | 17.377 | 1.002 |  |
| 007 | 0.08074 | 1.13334 | 14.037 | 0.979 |  |
| 008 | 0.06762 | 1.13927 | 16.848 | 1.004 |  |
| 009 | 0.08491 | 1.12527 | 13.252 | 0.983 |  |
| 010 | 0.06858 | 1.06489 | 15.528 | 0.991 |  |
| 011 | 0.07387 | 1.07588 | 14.564 | 0.969 |  |
| 012 | 0.04943 | 1.14091 | 23.080 | nan |  |
| 013 | 0.05948 | 1.16526 | 19.590 | 1.025 |  |
| 014 | 0.07585 | 1.13954 | 15.025 | 0.992 |  |
| 015 | 0.08509 | 1.07045 | 12.581 | 0.989 |  |
| 016 | 0.06074 | 1.06739 | 17.574 | 0.990 |  |
| 017 | 0.07154 | 1.06061 | 14.825 | 0.990 |  |
| 018 | 0.09322 | 1.15630 | 12.404 | 1.011 |  |
| 025 | 0.07287 | 1.03201 | 14.162 | 1.020 |  |
| 026 | 0.06337 | 1.07476 | 16.961 | 1.038 |  |
| 027 | 0.36880 | 1.04303 | 2.828 | 0.987 | yes |
| 028 | 0.08588 | 1.07732 | 12.545 | 1.014 |  |
| 029 | 0.05752 | 1.09116 | 18.969 | 1.020 |  |
| 030 | 0.04980 | 1.10465 | 22.180 | 1.001 |  |
| 031 | 0.04727 | 1.09763 | 23.220 | 1.005 |  |
| 032 | 0.06282 | 1.12285 | 17.874 | 1.024 |  |
| 033 | 0.04963 | 1.12002 | 22.565 | 1.020 |  |
| 034 | 0.04920 | 1.10574 | 22.472 | 1.031 |  |
| 035 | 0.11068 | 1.10362 | 9.971 | 1.035 |  |
| 036 | 0.08148 | 1.17652 | 14.439 | 1.031 |  |

## Answers

1. Camera-calibration geometry coverage: **multi-pose and usable, but not uniformly dense at all sensor edges** (u/v center spans 0.335/0.301 of the sensor).
2. K,D frame-selection stability: **LOO changes remain small**; bootstrap distributions are summarized above and keep k3 fixed at zero.
3. Intrinsics-induced truth change: median candidate P95 **0.3523 mm**; bootstrap 95%-candidate P95 median **1.0944 mm**; worst **1.1765 mm**.
4. Sensor-edge amplification: present only as a modest, non-universal effect (median 1.005×, max 1.038×).
5. Enough to explain current frame-dependent residual: **yes**; even the median candidate ratio is about **4.932**, while the median 95%-candidate tail ratio is **15.276**. Direction is not stable: same signed bias in only **15/30** frames, rho **0.253** (p=0.177).
6. Next step: continue with **corner extraction / camera-model bias audit** before changing the laser surface; do not alter formal K/D from this diagnostic alone.

## 027

- 027 Cone RMSE: **0.36880 mm**; median candidate P95 Δlambda **0.33029 mm**; bootstrap 95%-candidate P95 **1.04303 mm**; ratio **2.828**.
- 027 bootstrap propagation is larger than its Cone RMSE (ratio above 1), so under this frame-selection uncertainty model it is not a clean intrinsics-independent exception.

## Conclusion

`INTRINSICS_TRUTH_STABILITY = C. LOW`.
The descriptive gates are: HIGH when typical and worst propagated uncertainty remain below 0.25×/0.50× of Cone RMSE; MODERATE when below 0.75×/1.00×; LOW otherwise or when bootstrap coverage is insufficient.

Generated figures: `camera_calibration_corner_coverage.png`, `intrinsics_bootstrap_distribution.png`, `delta_lambda_vs_sensor_v.png`, `intrinsics_uncertainty_vs_cone_residual.png`, and `top_middle_bottom_truth_sensitivity.png`.
