# Phase-A analysis plots summary

## Figure 09 — Sensitivity vs laser angle

- Scale 5, 5°→20° sensitivity change: -9.03%.
- Scale 12.5, 5°→20° sensitivity change: -12.98%.
- Scale 0, 10°→20° sensitivity change: -6.89%.
- Within the captured range, combined sensitivity decreases as laser angle increases for all three mechanical scale settings.
- B00_A05 remains invalid_fov/NaN and is not interpolated.

## Figure 10 — Sensitivity–repeatability trade-off

- Lowest predicted sigma_Z: B05_A05 = 0.01420 mm.
- Highest combined sensitivity: B00_A10 = 1.56889 px/mm.
- The plot should be interpreted as a trade-off map, not as proof of final 3D accuracy.

## Figure 11 — H10 vs H30 consistency

- Mean relative difference: 1.566%.
- Median relative difference: 1.651%.
- Maximum relative difference: 2.544% at B12p5_A10.
- All valid configurations are below 3% relative difference.
- This supports local consistency of image-space height sensitivity over the tested 10–30 mm range; it is not a full linearity calibration.

## Interpretation limits

- baseline_scale_reading is a mechanical support scale, not measured optical baseline.
- sigma_z_pred_combined_mm is predicted height repeatability, not final 3D measurement accuracy.
- B00_A05 is a real invalid_fov outcome and remains missing without interpolation.
