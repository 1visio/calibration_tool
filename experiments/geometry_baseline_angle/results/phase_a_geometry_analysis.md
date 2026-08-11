# Phase-A baseline × laser-angle geometry analysis

## Interpretation boundary

This report uses the frozen extraction chain and the final `geometry_master_summary.csv`. No Steger, band, reference, ROI, trim, or metric definition is changed. `baseline_scale_reading` is the mechanical support scale, not the measured camera–laser optical baseline.

`sigma_z_pred_combined_mm` is predicted height repeatability derived from image-space temporal repeatability and geometric sensitivity. It is not final 3D measurement accuracy. Final accuracy must be verified after formal calibration using independent gauge blocks or traceable standards.

B00_A05 remains `invalid_fov`; every numeric matrix keeps that cell as NaN/gray and no interpolation is performed.

## 3 × 4 matrices

### sensitivity_combined_px_per_mm

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 1.25118 | 0.93679 |
| 10 | 1.56889 | 1.22807 | 0.89692 |
| 15 | 1.51667 | 1.19321 | 0.85414 |
| 20 | 1.46085 | 1.13824 | 0.81522 |

### sigma_z_pred_combined_mm

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 0.01420 | 0.02303 |
| 10 | 0.01880 | 0.02119 | 0.03205 |
| 15 | 0.01881 | 0.02479 | 0.03433 |
| 20 | 0.01519 | 0.02260 | 0.02746 |

### sensitivity_h10

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 1.24515 | 0.93406 |
| 10 | 1.54911 | 1.22124 | 0.88551 |
| 15 | 1.50398 | 1.18669 | 0.84709 |
| 20 | 1.44706 | 1.12495 | 0.81174 |

### sensitivity_h30

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 1.25720 | 0.93953 |
| 10 | 1.58867 | 1.23491 | 0.90832 |
| 15 | 1.52935 | 1.19972 | 0.86120 |
| 20 | 1.47464 | 1.15154 | 0.81870 |

### sigma_pixel_h10_p95_px

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 0.01592 | 0.01963 |
| 10 | 0.03276 | 0.02820 | 0.03398 |
| 15 | 0.02871 | 0.02435 | 0.02954 |
| 20 | 0.02437 | 0.02860 | 0.02551 |

### sigma_pixel_h30_p95_px

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 0.01965 | 0.02277 |
| 10 | 0.02620 | 0.02398 | 0.02311 |
| 15 | 0.02859 | 0.03456 | 0.02855 |
| 20 | 0.02001 | 0.02284 | 0.01906 |

### reference_cv_interior_rmse_px

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---:|---:|---:|
| 5 | invalid_fov | 0.12743 | 0.11960 |
| 10 | 0.19641 | 0.17441 | 0.14005 |
| 15 | 0.19925 | 0.19593 | 0.16767 |
| 20 | 0.25772 | 0.20770 | 0.19955 |

### H1 detectability/status (diagnostic only)

| laser_angle_deg \ baseline_scale_reading | 0 | 5 | 12.5 |
|---:|---|---|---|
| 5 | invalid_fov | available | available |
| 10 | available | available | warning/unavailable |
| 15 | available | available | warning/unavailable |
| 20 | available | available | available |

H1 never enters `sensitivity_combined_px_per_mm` or `sigma_z_pred_combined_mm`.

## Fixed baseline: angle 5 → 10 → 15 → 20

### baseline_scale_reading = 0

- sensitivity combined: invalid_fov → 1.56889 → 1.51667 → 1.46085
- sigma_Z predicted combined: invalid_fov → 0.01880 → 0.01881 → 0.01519
- sigma_pixel H10 P95: invalid_fov → 0.03276 → 0.02871 → 0.02437
- sigma_pixel H30 P95: invalid_fov → 0.02620 → 0.02859 → 0.02001
- reference CV RMSE: invalid_fov → 0.19641 → 0.19925 → 0.25772

Angle 5 is infeasible because the laser leaves the camera FOV. Across the captured 10–20° conditions, sensitivity decreases with angle, while predicted sigma_Z is nearly unchanged from 10° to 15° and improves at 20°. H10 sigma_pixel improves toward 20°; reference CV worsens at 20°. H1 is available for every captured condition.

### baseline_scale_reading = 5

- sensitivity combined: 1.25118 → 1.22807 → 1.19321 → 1.13824
- sigma_Z predicted combined: 0.01420 → 0.02119 → 0.02479 → 0.02260
- sigma_pixel H10 P95: 0.01592 → 0.02820 → 0.02435 → 0.02860
- sigma_pixel H30 P95: 0.01965 → 0.02398 → 0.03456 → 0.02284
- reference CV RMSE: 0.12743 → 0.17441 → 0.19593 → 0.20770

Sensitivity decreases monotonically with angle. Predicted sigma_Z is best at 5°, worsens through 15°, then partially recovers at 20°. H10/H30 sigma_pixel are non-monotonic. Reference CV worsens monotonically with angle. H1 remains available throughout.

### baseline_scale_reading = 12.5

- sensitivity combined: 0.93679 → 0.89692 → 0.85414 → 0.81522
- sigma_Z predicted combined: 0.02303 → 0.03205 → 0.03433 → 0.02746
- sigma_pixel H10 P95: 0.01963 → 0.03398 → 0.02954 → 0.02551
- sigma_pixel H30 P95: 0.02277 → 0.02311 → 0.02855 → 0.01906
- reference CV RMSE: 0.11960 → 0.14005 → 0.16767 → 0.19955

Sensitivity decreases monotonically with angle. Predicted sigma_Z is best at 5°, worst at 15°, and partially recovers at 20°. Pixel repeatability is non-monotonic; reference CV worsens monotonically with angle. H1 is available at 5° and 20°, warning at 10°, and unavailable for formal H1 statistics at 15° because of ROI-trim sensitivity.

## Fixed angle: baseline scale 0 → 5 → 12.5

### laser_angle_deg = 5

- sensitivity combined: invalid_fov → 1.25118 → 0.93679
- sigma_Z predicted combined: invalid_fov → 0.01420 → 0.02303
- sigma_pixel H10 P95: invalid_fov → 0.01592 → 0.01963
- sigma_pixel H30 P95: invalid_fov → 0.01965 → 0.02277
- reference CV RMSE: invalid_fov → 0.12743 → 0.11960

The scale-0 condition is invalid_fov. From scale 5 to 12.5, sensitivity decreases and predicted sigma_Z worsens, while reference CV improves slightly. Both feasible H1 observations are available.

### laser_angle_deg = 10

- sensitivity combined: 1.56889 → 1.22807 → 0.89692
- sigma_Z predicted combined: 0.01880 → 0.02119 → 0.03205
- sigma_pixel H10 P95: 0.03276 → 0.02820 → 0.03398
- sigma_pixel H30 P95: 0.02620 → 0.02398 → 0.02311
- reference CV RMSE: 0.19641 → 0.17441 → 0.14005

Sensitivity decreases strongly as the scale increases; predicted sigma_Z worsens. Reference CV improves with scale. H10 sigma_pixel is non-monotonic, H30 sigma_pixel improves, and H1 becomes warning at scale 12.5.

### laser_angle_deg = 15

- sensitivity combined: 1.51667 → 1.19321 → 0.85414
- sigma_Z predicted combined: 0.01881 → 0.02479 → 0.03433
- sigma_pixel H10 P95: 0.02871 → 0.02435 → 0.02954
- sigma_pixel H30 P95: 0.02859 → 0.03456 → 0.02855
- reference CV RMSE: 0.19925 → 0.19593 → 0.16767

Sensitivity decreases and predicted sigma_Z worsens with scale. Reference CV improves. H10/H30 sigma_pixel are non-monotonic; H1 is unavailable at scale 12.5 but remains diagnostic-only.

### laser_angle_deg = 20

- sensitivity combined: 1.46085 → 1.13824 → 0.81522
- sigma_Z predicted combined: 0.01519 → 0.02260 → 0.02746
- sigma_pixel H10 P95: 0.02437 → 0.02860 → 0.02551
- sigma_pixel H30 P95: 0.02001 → 0.02284 → 0.01906
- reference CV RMSE: 0.25772 → 0.20770 → 0.19955

Sensitivity decreases and predicted sigma_Z worsens with scale. Reference CV improves. Pixel repeatability is non-monotonic, with scale 12.5 giving the lowest H30 sigma_pixel P95. H1 remains available for all three baselines.

## Cross-factor conclusions

- At every fixed angle with valid data, increasing `baseline_scale_reading` lowers combined sensitivity and generally worsens predicted sigma_Z, while reference CV improves. This is a measured Phase-A trade-off, not a claim about actual optical baseline because the scale reading is not the measured baseline.
- At fixed baseline 5 and 12.5, sensitivity decreases monotonically as laser angle increases. The scale-0 captured subset shows the same decline from 10° to 20°.
- Pixel repeatability is not monotonic in either factor, so sigma_Z cannot be inferred from sensitivity alone.
- FOV feasibility is categorical in the available data: 11 conditions were captured in view; B00_A05 is invalid_fov. No quantitative FOV-margin measurement exists in the summary, so the report does not invent one.
- All 11 captured configurations have complete H10/H30 formal statistics and `needs_manual_review=false`. Existing primary warnings are diagnostic stable-plateau warnings and do not invalidate trim3-median formal statistics.
