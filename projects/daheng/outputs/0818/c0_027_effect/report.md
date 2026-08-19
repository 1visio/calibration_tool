# C0 frame027 effect: Operational-35 vs Full-36 controlled FIT-only A/B

`C0_027_EFFECT = MIXED`

## Decision summary

- Same-35-pose grouped-CV pose improvement ratio (C0-35 RMSE better than C0-36): **0.600 (21/35)**.
- Global CV RMSE: C0-36 **0.086550 mm** → C0-35 **0.086170 mm**; improvement **0.438%**.
- Global CV P95: C0-36 **0.174755 mm** → C0-35 **0.172811 mm**; improvement **1.112%**.
- Global CV P99: C0-36 **0.233768 mm** → C0-35 **0.230296 mm**; improvement **1.485%**.
- Full-fit surface delta direction is `lambda(C0-35) - lambda(C0-36)` on the same 31,500 operational raw rays: RMSE **0.004575 mm**, P95 **0.006785 mm**, Max **0.025692 mm**.
- Operational-35 geometry-only audit: **PASS** against its own 35-pose reference domain; 027's excluded normal is not a required coverage target.

## Scope and hard constraints

- Current Frozen Full-36 Quadratic C0 remains byte-for-byte untouched and is loaded only as baseline.
- C0-35 candidate is the only newly fitted production-like surface; only frame027 is excluded from the raw Full-36 FIT table.
- CV is FIT-only on the same 35 operational poses with deterministic 6-fold pose grouping; the Full-36 arm keeps 027 in each fold's training set, while the Operational-35 arm excludes it.
- Validation is not read. C1 is not imported or trained. No production config is modified.
- This result is an A/B audit only; it does not replace Frozen C0-36.

## Artifact provenance / reuse audit

| artifact | action | status |
|---|---|---|
| Frozen Full-36 C0 YAML | LOADED_ONLY | SHA256 `113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27`; no frozen fit/write |
| Raw Full-36 FIT points | REUSED_EXISTING | `ddafce3b4482731d7c753f3c3d329386c08ca985c787c649a29d1613e53c8442`; 36 poses / 32,400 points; formal FIT-only table |
| Formal extraction config | REUSED_EXISTING | SHA256 `2241737d68276dbdfb226f5285b7ae77dad07be2eaef02ed66966d6b6206cebf`; `full_board_physical`, inset 0 |
| Existing Full-36 grouped-CV artifacts | PROTOCOL_REFERENCE_ONLY | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\report.md` and pointwise reference; not substituted for same-35 controlled CV because their held-out domain includes 027 |
| Existing PnP geometry artifact | REUSED_EXISTING | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\pose_geometry_audit\pose_geometry_metrics.csv`; geometry columns only |
| C0-35 quadratic candidate | NEW_FIT | raw FIT points excluding only 027; same QuadraticGraphModel.fit protocol |
| Operational-35 CV / surface delta / geometry re-range | NEW_CALCULATION | generated in this directory |
| Validation / C1 | NOT_READ / NOT_TRAINED | excluded by design |

## Fixed fitting protocol

- Model: `quadratic_graph`, dependent axis X, independent axes Y/Z.
- Frame-balanced weights, existing robust 10-iteration QuadraticGraphModel fit, ridge `1e-10`; no new tuning or weighting.
- Formal mask: `full_board_physical`, inset `0 mm`; v domain `[0,3000)` and 100 px reporting bins.
- The C0-35 full-fit YAML is written as a candidate artifact only; it is not copied into the Frozen C0 directory.

## Controlled grouped-CV comparison

| metric | C0-36 (027 kept in training) | C0-35 (027 excluded) | C0-35 minus C0-36 / improvement |
|---|---:|---:|---:|
| Bias / mm | -0.00188 | -0.00060 | Δ 0.00127 |
| MAE / mm | 0.06725 | 0.06741 | Δ 0.00016 |
| RMSE / mm | 0.08655 | 0.08617 | Δ -0.00038; 0.438% |
| P95 abs / mm | 0.17475 | 0.17281 | Δ -0.00194; 1.112% |
| P99 abs / mm | 0.23377 | 0.23030 | Δ -0.00347; 1.485% |
| worst-v RMSE / mm | 0.17892 | 0.17010 | Δ -0.00882; 4.928% |
| worst-v P95 abs / mm | 0.25128 | 0.24291 | Δ -0.00837; 3.332% |
| v-bias range / mm | 0.28599 | 0.27452 | Δ -0.01148 (-4.014%) |
| pose improvement ratio | — | — | **0.600 (21/35)** |

P95/P99/Max use absolute error; CV Bias/MAE/RMSE are pooled over the identical held-out operational point set. The point identity equality is checked in code and recorded by hash in `c0_35_vs_36_cv_folds.json`.

## Full-fit surface delta on unified Operational-35 ray grid

- Grid: raw `full_fit_points.csv` rays for the same 35 poses, 900 rows per pose, 31,500 rows total.
- Both full-fit surfaces use the same rays and the same plane root-selection hint; invalid pairs are not silently dropped (this run requires 100% valid pair rate).
- `c0_surface_delta.csv` reports global and each 100 px v-bin metrics.

| global delta metric | value |
|---|---:|
| Bias / mm | -0.00136 |
| MAE / mm | 0.00353 |
| RMSE / mm | 0.00458 |
| P95 abs / mm | 0.00679 |
| P99 abs / mm | 0.01594 |
| Max abs / mm | 0.02569 |

## Operational-35 geometry-only audit

- 35-pose reference normal diameter: **46.126°**; excluded 027 tilt: **26.648°**.
- v coverage: **300/300** 10 px cells and **30/30** 100 px bins; edge minimum **11** poses.
- depth/lambda span ratios: **1.0000 / 1.0000** relative to Operational-35 itself.
- LOO failures against the Operational-35 reference: **3/35**; this is a redundancy diagnostic, not a reason to delete a pose in this A/B.

## Classification rule and interpretation

- `NEGLIGIBLE`: surface delta is within RMSE/P95/Max ≤ 0.010/0.030/0.100 mm and no primary CV change reaches 2%.
- `MATERIAL`: surface delta reaches the declared material diagnostic margin or a primary global CV metric reaches 2%.
- `MIXED`: global CV and surface are negligible, but tail-v/pose evidence still changes at the diagnostic level (or dimensions disagree).
- This run: surface-negligible=True, surface-material=False, CV-change-material=True, mixed-signs=False.
- **Conclusion: `C0_027_EFFECT = MIXED`.** This is an audit classification only; the current Frozen C0-36 remains the production baseline and is not replaced by C0-35.

## Outputs

- `c0_35_vs_36_cv.csv`
- `c0_35_vs_36_pose_metrics.csv`
- `c0_35_vs_36_v_bins.csv`
- `c0_surface_delta.csv`
- `operational35_geometry_audit.md`
- `report.md`
