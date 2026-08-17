# Task 6H-2 — Decoupled camera pose augmentation + M2 stability audit

`NEW_POSE_DECOUPLING = PASS`
`DECOUPLED_CAMERA_GAIN = D. NEGATIVE`
`FREEZE_M2_FOR_LASER_AB = NO`

0815 camera-candidate stage used only chess 041–048. 0815 laser/nolaser were not opened. Laser propagation used only old FIT 001–018 and 025–036; Validation was not opened. Formal K/D, distortion model, Cone, and Steger were not modified.

## Part A — measured 0815 geometry

| metric | 0815 result |
|---|---:|
| depth range (mm) | 649.971–681.969 |
| tilt range (deg) | 8.149–28.545 |
| high tilt >=18° | 5 |
| tilt directions | +pitch;+roll;-pitch;-roll |
| tilt-depth Spearman | -0.9048 |
| depth clusters | 2 (near=4, far=4) |
| cluster medians (mm) | 653.218, 680.826 |
| cross-depth matched-tilt pairs | 2 |
| same-depth direction pairs | 12 |

## M0 / M1-core / M2 calibration

| candidate | poses | global RMSE (px) | fx | fy | cx | cy |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 18 | 0.118739 | 7347.865114 | 7348.841147 | 2070.216530 | 1512.436454 |
| M1-core | 22 | 0.128318 | 7348.795134 | 7349.749125 | 2072.119771 | 1509.661416 |
| M2 | 30 | 0.135887 | 7350.235559 | 7351.251646 | 2073.935865 | 1511.007220 |

## LOO stability

| candidate | P50 | P90 | P95 | max (mm) |
|---|---:|---:|---:|---:|
| M0 | 0.0387093 | 0.263771 | 0.33741 | 0.399967 |
| M1-core | 0.117255 | 0.181689 | 0.246742 | 0.306194 |
| M2 | 0.237189 | 0.267913 | 0.307026 | 0.344105 |

M2 relative to M1-core: LOO P95 change = -24.43%, max change = -12.38%. New 0815 omission maximum = 0.274823 mm.

## Fixed-coverage corner-noise MC

| candidate | centered global P95 median | centered P95 tail | centered max (mm) |
|---|---:|---:|---:|
| M0 | 0.132794 | 0.173754 | 0.178306 |
| M1-core | 0.0626172 | 0.15609 | 0.166476 |
| M2 | 0.148626 | 0.168038 | 0.170195 |

## Coverage comparison

| candidate | depth range (mm) | tilt range (deg) | apparent-size range | tilt-depth Spearman |
|---|---:|---:|---:|---:|
| M0 | 63.601 | 22.2323 | 0.258762 | -0.576883 |
| M1-core | 63.601 | 24.1161 | 0.298219 | -0.616036 |
| M2 | 63.601 | 26.0115 | 0.298219 | -0.674305 |

## Decision

- 0815 provides two separated depth clusters, multiple high-tilt poses, four tilt-direction labels, and matched tilt across depth; Part A is therefore not a FAIL.
- M2 is recommended for laser old-vs-new A/B only when all four decision gates pass. The reported freeze flag is the gate result, not a modification of formal K/D.

## Outputs

- `0815_pose_characterization.csv`
- `new_pose_decoupling_report.csv`
- `m0_m1_m2_intrinsics.csv`
- `m0_m1_m2_loo_stability.csv`
- `m0_m1_m2_corner_mc.csv`
- `m0_m1_m2_coverage.csv`
- `new_frame_leverage.csv`
- `provenance.json
