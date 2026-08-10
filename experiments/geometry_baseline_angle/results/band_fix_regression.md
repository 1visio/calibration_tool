# Phase-A band fix regression

## Comparison scope

- Before: `results_v0/geometry_master_summary.csv`
- After: `results/geometry_master_summary.csv`
- H1 valid fractions: each current `data/<config_id>/analysis/multiheight_analysis.json`
- Band change: Phase-A multiheight uses the shared Steger extractor with `auto_band ∪ reference_band`.
- No ranking or heatmap was generated.
- `valid configs` below means captured configurations whose H10 and H30 formal statistics are both available after the fix.

## Summary

| Item | Count | Configs |
|---|---:|---|
| valid configs | 11 | all captured configs |
| H1 unavailable before | 7 | B00_A10, B00_A15, B00_A20, B05_A05, B05_A10, B05_A20, B12p5_A15 |
| H1 recovered configs | 6 | B00_A10, B00_A15, B00_A20, B05_A05, B05_A10, B05_A20 |
| H1 still unavailable configs | 1 | B12p5_A15 |
| warning configs after | 9 | B00_A10, B00_A20, B05_A05, B05_A10, B05_A15, B12p5_A05, B12p5_A10, B12p5_A15, B12p5_A20 |
| failed configs after | 0 | - |
| invalid_fov configs | 1 | B00_A05 |

The six band-truncated H1 cases recovered. `B12p5_A15` remains unavailable for formal H1 statistics because its ROI trim sensitivity is about 3.218%, above the 2% rule; its H1 valid-column fraction is 0.9655, so this is not a detection-band failure.

## H1 result after the fix

| Config | Available before | Status after | Valid-column fraction after | Recovered |
|---|---:|---|---:|---:|
| B00_A10 | no | ok | 1.0000 | yes |
| B00_A15 | no | ok | 1.0000 | yes |
| B00_A20 | no | ok | 1.0000 | yes |
| B05_A05 | no | ok | 1.0000 | yes |
| B05_A10 | no | ok | 0.9643 | yes |
| B05_A15 | yes | ok | 1.0000 | no |
| B05_A20 | no | ok | 1.0000 | yes |
| B12p5_A05 | yes | ok | 1.0000 | no |
| B12p5_A10 | yes | warning | 1.0000 | no |
| B12p5_A15 | no | unavailable | 0.9655 | no |
| B12p5_A20 | yes | ok | 1.0000 | no |

## Primary sensitivity regression

Relative change is `(after - before) / abs(before) * 100%`. A row is considered stable when both comparable H10 and H30 changes are below 1% in absolute value.

| Config | H10 change | H30 change | Combined change | Result |
|---|---:|---:|---:|---|
| B00_A10 | 0.0000% | 0.0000% | 0.0000% | pass |
| B00_A15 | newly available | 0.0000% | newly available | not comparable |
| B00_A20 | newly available | 0.0000% | newly available | not comparable |
| B05_A05 | -0.0114% | 0.0000% | -0.0057% | pass |
| B05_A10 | +0.3574% | 0.0000% | +0.1774% | pass |
| B05_A15 | 0.0000% | 0.0000% | 0.0000% | pass |
| B05_A20 | 0.0000% | 0.0000% | 0.0000% | pass |
| B12p5_A05 | 0.0000% | 0.0000% | 0.0000% | pass |
| B12p5_A10 | 0.0000% | 0.0000% | 0.0000% | pass |
| B12p5_A15 | 0.0000% | 0.0000% | 0.0000% | pass |
| B12p5_A20 | 0.0000% | 0.0000% | 0.0000% | pass |

Among configurations with both before and after primary results, the largest H10/H30 sensitivity change is +0.3574% (`B05_A10` H10), below the 1% criterion. This supports the conclusion that the fix primarily removes search-domain truncation without materially changing the formal center displacement estimate.

## Repeatability regression

| Config | H10 sigma_pixel P95 change | H30 sigma_pixel P95 change | Combined sigma_Z change |
|---|---:|---:|---:|
| B00_A10 | 0.0000% | 0.0000% | +0.0020% |
| B00_A15 | newly available | 0.0000% | newly available |
| B00_A20 | newly available | 0.0000% | newly available |
| B05_A05 | -0.5867% | 0.0000% | -0.2689% |
| B05_A10 | -21.0730% | 0.0000% | -14.5913% |
| B05_A15 | 0.0000% | 0.0000% | 0.0000% |
| B05_A20 | 0.0000% | 0.0000% | 0.0000% |
| B12p5_A05 | 0.0000% | 0.0000% | 0.0000% |
| B12p5_A10 | 0.0000% | 0.0000% | 0.0000% |
| B12p5_A15 | 0.0000% | 0.0000% | 0.0000% |
| B12p5_A20 | 0.0000% | 0.0000% | 0.0000% |

`B05_A10` is the only notable repeatability change: H10 sigma_pixel P95 decreases from 0.0357327 px to 0.0282027 px (-21.0730%), and combined sigma_Z decreases from 0.0248056 mm to 0.0211861 mm (-14.5913%). This is an improvement rather than degradation, but it means the expanded search band changed the per-frame H10 repeatability statistics even though the formal H10 sensitivity changed by only +0.3574%.

## Frozen B12p5 validation

For `B12p5_A05`, `B12p5_A10`, `B12p5_A15`, and `B12p5_A20`:

- H10 sensitivity relative change: 0% for all four.
- H30 sensitivity relative change: 0% for all four.
- H10/H30 sigma_pixel P95 relative change: 0% for all four.
- sensitivity_combined relative change: 0% for all four.
- sigma_z_pred_combined relative change: 0% for all four.

The band fix therefore does not alter the previously normal B12p5 formal results at the precision stored in the summary CSV.
