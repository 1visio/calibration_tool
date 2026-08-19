# FIT-only 1D C1 grouped-CV on Frozen Full-36 Quadratic C0

C1_STATUS = PARTIAL
FRAME027_C1_DECISION = QUARANTINE_PENDING_RECAPTURE

## Scope and boundary

- 本轮只做 FIT grouped-CV；没有冻结生产 C1、修改生产配置或重新拟合 Quadratic C0。
- Validation 未读取；027 保留在 Full-36 artifact 中，也没有永久删除或按 residual 删除其他 pose/point。
- 输入为现有 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\quadratic_residual_observability\quadratic_residual_points.csv`：Full-36、36 poses、32,400 points；C0 为 frozen `quadratic_graph`。

## Artifact provenance / reuse audit

| artifact | action | status | evidence |
|---|---|---|---|
| Full-36 residual/ray/PCA-s artifact | REUSED_EXISTING | CONFIRMED | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\quadratic_residual_observability\quadratic_residual_points.csv`; stored `residual_mm`, `residual_centered_mm`, `pca_s`, `v_px`; no re-extraction |
| Frozen Quadratic C0 | LOADED_ONLY | CONFIRMED | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c0_freeze\quadratic_graph.yaml`; sha256 `113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27`; no `fit()` |
| Residual audit | READ_ONLY_ASSERTED | CONFIRMED | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\quadratic_residual_observability\audit_summary.json` says `validation_read=false`, `c0_refit=false`, `c1_fit=false` |
| Existing 0817 C1 output | REFERENCE_ONLY | EXCLUDED | It used Frozen Cone and a different 30-frame/26,663-point artifact; not reused as a result |
| Validation | NOT_READ | EXCLUDED | No Validation path is an input to this run |

## Fixed method

- C1 target: stored frame-median-centered residual `r_centered = residual_mm - frame_residual_median_mm`; correction is `lambda_final = lambda_quadratic + F(s)`, so evaluated residual is `r - F(s)`.
- Candidate basis: cubic B-spline with interior knots 3/4/5 (`C1_3k`, `C1_4k`, `C1_5k`), common Full-36 PCA-s domain, 100 px v-bins.
- Fitting: frame-balanced weights (each training frame total weight=1), Huber IRLS (`k=1.345`), fixed second-order difference penalty `lambda=0.1`.
- CV: deterministic 6-fold pose-grouped round-robin. A is normal Full-36 CV with 027 retained; B is a separate 35-pose grouped-CV with 027 excluded from C1 training/evaluation. The no-027 full-fit model is then evaluated on 027 separately as the held-out stress test. No point-wise random split.
- Model assessment does not use Max for selection. Max is reported only for frame027 stress rows.

## Candidate comparison

| scenario | candidate | RMSE C0→C1 / % | P95 C0→C1 / % | P99 C0→C1 / % | worst-v RMSE / % | worst-v P95 / % | v-bias range C0→C1 / mm | pose ratio | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| full36_grouped_cv | C1_3k | 8.56 | 13.45 | -2.42 | 25.46 | 15.60 | 0.2848→0.1542 | 0.889 | PASS |
| full36_grouped_cv | C1_4k | 10.01 | 15.29 | -0.89 | 25.47 | 14.44 | 0.2848→0.1384 | 0.917 | PASS |
| full36_grouped_cv | C1_5k | 10.68 | 15.72 | -0.13 | 25.34 | 15.39 | 0.2848→0.1327 | 0.917 | PASS |
| full36_grouped_cv_non027 | C1_3k | 13.46 | 15.45 | 13.78 | 33.91 | 17.23 | 0.2897→0.1516 | 0.914 | PASS |
| full36_grouped_cv_non027 | C1_4k | 15.57 | 17.40 | 15.22 | 35.39 | 16.17 | 0.2897→0.1411 | 0.943 | PASS |
| full36_grouped_cv_non027 | C1_5k | 16.35 | 18.06 | 16.25 | 36.68 | 17.09 | 0.2897→0.1350 | 0.943 | PASS |
| exclude027_grouped_cv_non027 | C1_3k | 13.46 | 15.65 | 13.93 | 38.31 | 18.28 | 0.2897→0.1582 | 0.914 | PASS |
| exclude027_grouped_cv_non027 | C1_4k | 15.54 | 17.51 | 15.36 | 39.57 | 15.27 | 0.2897→0.1433 | 0.943 | PASS |
| exclude027_grouped_cv_non027 | C1_5k | 16.35 | 18.31 | 16.02 | 40.64 | 22.11 | 0.2897→0.1381 | 0.943 | PASS |

- Follow-up candidate for reporting: **C1_3k** (lowest knot count among Full-36 gate-passing candidates when available; otherwise best non-Max score). Passing Full-36 candidates: `C1_3k, C1_4k, C1_5k`.
- `full36_grouped_cv_non027` and `exclude027_grouped_cv_non027` use the same remaining 35 poses, so their difference isolates the effect of including 027 in C1 training.

## Frame027 stress test

| candidate | scenario | training | RMSE C0→C1 / % | P95 C0→C1 / % | P99 C0→C1 / % | Max C0→C1 / mm |
|---|---|---|---:|---:|---:|---:|
| C1_3k | full36_grouped_cv_027_heldout | 30 frames (027 excluded) | -1.54 | 2.52 | 2.48 | 0.7229→0.7059 |
| C1_3k | full36_fullfit_027_in_sample | 36 frames (027 included) | -0.05 | 3.44 | 3.25 | 0.7229→0.7014 |
| C1_3k | exclude027_fullfit_027_heldout | 35 frames (027 excluded) | -1.55 | 2.50 | 2.44 | 0.7229→0.7062 |
| C1_4k | full36_grouped_cv_027_heldout | 30 frames (027 excluded) | -1.35 | 2.00 | 1.98 | 0.7229→0.7090 |
| C1_4k | full36_fullfit_027_in_sample | 36 frames (027 included) | 0.12 | 2.62 | 2.46 | 0.7229→0.7064 |
| C1_4k | exclude027_fullfit_027_heldout | 35 frames (027 excluded) | -1.39 | 1.76 | 1.74 | 0.7229→0.7107 |
| C1_5k | full36_grouped_cv_027_heldout | 30 frames (027 excluded) | -0.86 | 2.42 | 2.06 | 0.7229→0.7110 |
| C1_5k | full36_fullfit_027_in_sample | 36 frames (027 included) | 0.66 | 2.98 | 2.45 | 0.7229→0.7103 |
| C1_5k | exclude027_fullfit_027_heldout | 35 frames (027 excluded) | -0.92 | 2.15 | 1.77 | 0.7229→0.7132 |

Max 在这里仅作为 027 的诊断输出，没有参与候选模型选择或状态门控。

## F(s) sensitivity to frame027

| candidate | curve RMS difference / mm | curve max difference / mm | curve P95 difference / mm |
|---|---:|---:|---:|
| C1_3k | 0.0071 | 0.0168 | 0.0164 |
| C1_4k | 0.0070 | 0.0157 | 0.0153 |
| C1_5k | 0.0069 | 0.0157 | 0.0151 |

`c1_curves_with_without_027.png` shows both fitted curves and their difference. Curve materiality flag=False using RMS ≥ 0.010 mm or max ≥ 0.025 mm; these are diagnostic thresholds, not Max-based model selection.

## Interpretation

- Remaining-35 training-effect summary for `C1_3k`: global RMSE change when excluding 027 = 0.00%; P95 change = 0.23%; pose-ratio change = 0.000.
- `027` C1 stress no-gain flag=True. This is based on held-out RMSE/P95, not Max.
- `FRAME027_C1_DECISION` is **QUARANTINE_PENDING_RECAPTURE**. The decision combines curve sensitivity, same-35-pose generalization, and whether a model trained without 027 generalizes back to 027; it is not an instruction to delete the frame.
- Operational recommendation: keep the original 027 artifact immutable; if the decision requests quarantine/recapture, quarantine is a future data-quality label pending same-pose recapture, not a deletion performed by this run.

## Outputs

- `c1_candidate_comparison.csv`: aggregate Full-36, remaining-35, and stress summaries.
- `c1_pose_cv_metrics.csv`: per-pose grouped-CV metrics for C0 and C0+C1.
- `c1_v_bin_metrics.csv`: 100 px v-bin metrics for C0 and C0+C1.
- `c1_curves_with_without_027.png`: fitted F(s) curves with and without 027 plus ΔF(s).
- `frame027_stress_test.csv`: 027 held-out and in-sample diagnostic metrics, including Max only here.

## Scope exclusions

- No Validation data was opened.
- No Quadratic C0 was refit; no 2D C1 was fit; no production configuration was changed.
- No pose or point was deleted or removed from the reused artifact.
