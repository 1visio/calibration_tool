# Phase-A hierarchical candidate selection

No weighted composite score is used.

## Selection hierarchy

1. Exclude `invalid_fov` and `failed`. B00_A05 is excluded as invalid_fov; no captured configuration failed.
2. Check H10/H30 formal completeness and primary warnings. All 11 captured configurations are formally complete; diagnostic stable-plateau warnings are retained but are not hard failures.
3. Prefer lower `sigma_z_pred_combined_mm`.
4. When sigma_Z is close, prefer higher `sensitivity_combined_px_per_mm`.
5. Use reference CV, H1 detectability, warning cleanliness, and known FOV feasibility as supporting evidence. Quantitative FOV margin is not available and is not fabricated.

## Selected candidates

### recommended_primary: `B05_A05`

- sigma_Z=0.01420 mm, sensitivity=1.25118 px/mm, reference CV RMSE=0.12743 px, H1=available, quality=pass_with_diagnostic_warning.
- Lowest predicted sigma_Z among all valid configurations. It remains the primary even though its sensitivity is below the scale-0 candidates, because sigma_Z has priority in the declared hierarchy.

### recommended_backup: `B00_A20`

- sigma_Z=0.01519 mm, sensitivity=1.46085 px/mm, reference CV RMSE=0.25772 px, H1=available, quality=pass_with_diagnostic_warning.
- Second-lowest predicted sigma_Z and substantially higher sensitivity than the primary. It carries a diagnostic-only H30 stable-plateau warning and belongs to the scale-0 family where the 5° condition was invalid_fov, so next-stage FOV margin must be checked explicitly.

### recommended_validation_candidates

- `B00_A10` — sigma_Z=0.01880 mm, sensitivity=1.56889 px/mm, reference CV RMSE=0.19641 px, H1=available, quality=pass_with_diagnostic_warning. It provides the highest combined sensitivity and third-lowest predicted sigma_Z, with a diagnostic-only H10 plateau warning.
- `B00_A15` — sigma_Z=0.01881 mm, sensitivity=1.51667 px/mm, reference CV RMSE=0.19925 px, H1=available, quality=clean. Its sigma_Z is almost identical to B00_A10 while its summary status is clean; retaining both tests whether the small sensitivity advantage at 10° survives formal calibration and independent standards.

```yaml
recommended_primary: B05_A05
recommended_backup: B00_A20
recommended_validation_candidates:
  - B00_A10
  - B00_A15
```

## Why other configurations are not retained in the first validation set

- B05_A10/A15/A20 have higher predicted sigma_Z and lower sensitivity than B05_A05; they do not improve the primary two metrics within the same mechanical baseline scale.
- B12p5 conditions provide better reference CV in several comparisons, but their combined sensitivity is lower and predicted sigma_Z is higher. Reference CV is a fifth-layer supporting metric and cannot override both primary metrics without downstream calibration evidence.
- B00_A05 remains invalid_fov and is never interpolated or reconsidered as a candidate.

## Required next-stage interpretation

The retained four configurations are candidates, not a final structure decision. `sigma_z_pred_combined_mm` predicts height repeatability from image repeatability and geometric sensitivity; it is not final 3D measurement accuracy. Each retained structure must complete formal camera/laser calibration and then be tested with independent gauge blocks or standards before a final geometry is selected.
