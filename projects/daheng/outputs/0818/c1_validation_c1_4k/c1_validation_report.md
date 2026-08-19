# Frozen Operational-35 C1_4k independent Validation

`C1_VALIDATION_STATUS = PASS`

## Scope and controls

本轮仅评估冻结模型，不进行 `fit()`、PCA、knots/coefficients/penalty 调整，也不读取 Validation 之外的训练数据来改变模型。frame027 不在本 Validation 集中，也没有被删除；其冻结状态仍为 `EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN`，理由为“超出实际工作姿态域”。本轮没有修改生产配置。

- Validation poses: `019, 020, 021, 022, 023, 024, 037, 038, 039, 040, 055, 056, 057, 058, 059, 060`；共 `14400` 点，16 pose，900 点/pose。
- 分组：019–024、037–040、055–060；pooled-16。
- C0：复用 `validation_points_used.csv` 中已由 Frozen Quadratic C0 产生的 `lambda_pred_quadratic_graph_mm`，没有在 Validation 上重新拟合或重新估计 C0。
- C1：直接加载冻结 JSON 的 Full-36 PCA center/axis/domain、cubic B-spline knots/coefficients；按冻结协议先将 raw `s` clamp 到 `[domain_min, domain_max]`，再精确求值，禁止 spline 外推。
- 评价权重：frame-balanced；每个 pose 总权重相等。P95/P99 是对应 frame-balanced 权重下的 absolute residual quantile。
- v-bin：沿用既有 QC 的固定 100 px bins `[0, 3000)`；out-of-range 点保留在 global/pose 汇总并单独列出。
- Top/Middle/Bottom：固定 reporting regions 分别为 `[0,300)`、`[300,2700)`、`[2700,3000)` px；不是基于 Validation 调整的选择。

## Artifact provenance / reuse audit

本轮复用现有点级 Validation artifact 和冻结模型；不复用旧的 C1 Validation 数值结果。Validation reuse audit 已确认旧的 0817 C1 artifacts excluded，当前点 artifact 为正式 16-pose Validation 输入。Frozen C1 的 `validation_read=false`、`c0_refit=false`、`production_config_modified=false`，且其 PCA 明确为 Full-36（包含027），没有重算 Operational-35 PCA。

| artifact | path | SHA-256 |
| --- | --- | --- |
| Frozen C1 JSON | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\frozen_c1_4k.json` | `4bf2bb08235ab76af118a85557445303a6ef635e657c55520d45fb23d476aee4` |
| C1 LUT | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\c1_4k_lut.csv` | `d09c4586848d2a1ccf20f2deabffc0e81df925128cdddd4ca5b878d146e680d2` |
| C1 freeze manifest | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\c1_freeze_manifest.json` | `88c7b580c5f0b1e2d57e312156c0b842a1ad487cc18c8cc92700172011037b28` |
| Frozen C0 YAML | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c0_freeze\quadratic_graph.yaml` | `113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27` |
| Validation points | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\final_validation_qc\validation_points_used.csv` | `463f2c76d8a5662f4f154d8020ca333b2e0130feba22651b5e0dabc6c2f255db` |
| Validation reuse audit | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\final_validation_qc\validation_artifact_reuse_audit.csv` | `bb833b2638465ce179215493c424c44a18a1b2867914c83e07c448e8db706311` |

Frozen C1 parameter SHA: `91ecf9de35e16b7e5c6f9c3264ca8222a765eb9419e6f16138189bd21aeb3815`. LUT 与精确 frozen spline 的 2049 个网格点最大绝对差：`5.55111512313e-16 mm`（阈值 0.001 mm）。

## Pooled and validation-group metrics

“improvement”定义为 `100 * (C0 - C0+C1) / C0`；Max 只作诊断，不参与 status 判断。

| scope | point_count | pose_count | c0_bias_mm | c1_bias_mm | c0_mae_mm | c1_mae_mm | c0_rmse_mm | c1_rmse_mm | rmse_improvement_pct | c0_p95_abs_mm | c1_p95_abs_mm | p95_abs_mm_improvement_pct | c0_p99_abs_mm | c1_p99_abs_mm | p99_abs_mm_improvement_pct | c0_worst_v_rmse_mm | c1_worst_v_rmse_mm | c0_worst_v_p95_abs_mm | c1_worst_v_p95_abs_mm | c0_v_bias_range_mm | c1_v_bias_range_mm | pose_improvement_ratio | clamp_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_019_024 | 5400 | 6 | -0.00376 | 0.00626 | 0.06480 | 0.06229 | 0.08051 | 0.07753 | 3.70919 | 0.15089 | 0.14666 | 2.80503 | 0.18619 | 0.17998 | 3.33926 | 0.14254 | 0.13191 | 0.21869 | 0.20689 | 0.18743 | 0.14465 | 0.66667 | 0.00000 |
| validation_037_040 | 3600 | 4 | -0.06174 | -0.05605 | 0.08043 | 0.06631 | 0.10002 | 0.08043 | 19.58741 | 0.19459 | 0.15099 | 22.40482 | 0.23950 | 0.19063 | 20.40464 | 0.19035 | 0.14378 | 0.26412 | 0.21486 | 0.29013 | 0.14186 | 1.00000 | 0.00000 |
| validation_055_060 | 5400 | 6 | -0.01569 | -0.01383 | 0.07636 | 0.05298 | 0.10062 | 0.06623 | 34.18269 | 0.21354 | 0.12481 | 41.54943 | 0.25460 | 0.16663 | 34.54981 | 0.23508 | 0.09986 | 0.28231 | 0.16012 | 0.41380 | 0.18120 | 1.00000 | 0.00019 |
| pooled_16 | 14400 | 16 | -0.02273 | -0.01685 | 0.07304 | 0.05980 | 0.09343 | 0.07427 | 20.50666 | 0.18992 | 0.14193 | 25.26813 | 0.24100 | 0.18124 | 24.79568 | 0.19803 | 0.10821 | 0.27168 | 0.20536 | 0.34781 | 0.13890 | 0.87500 | 0.00007 |

Pooled-16 关键结果：RMSE `0.093428 -> 0.074269 mm`（`20.507%`），P95 `0.189917 -> 0.141929 mm`（`25.268%`），P99 `0.240997 -> 0.181240 mm`（`24.796%`）。Pooled pose RMSE improvement ratio 为 `0.875`。

## Top / Middle / Bottom

三段是否同时受益：RMSE = `True`，P95 = `True`。

| scope | point_count | c0_rmse_mm | c1_rmse_mm | rmse_improvement_pct | c0_p95_abs_mm | c1_p95_abs_mm | p95_abs_mm_improvement_pct | pose_improvement_ratio | clamp_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top | 682 | 0.15934 | 0.07135 | 55.22352 | 0.25313 | 0.12733 | 49.69922 | 0.83333 | 0.00000 |
| middle | 13115 | 0.08308 | 0.07382 | 11.14151 | 0.16542 | 0.14279 | 13.68231 | 0.87500 | 0.00000 |
| bottom | 602 | 0.15079 | 0.07498 | 50.27656 | 0.22460 | 0.12162 | 45.84894 | 1.00000 | 0.00000 |

## Pose-level paired C0 -> C0+C1

逐 pose 的 RMSE/P95 paired 结果如下；`rmse_improved` 和 `p95_improved` 是逐 pose 的布尔判断，不以 Max 选模。

| pose_id | group | c0_rmse_mm | c1_rmse_mm | rmse_improvement_pct | c0_p95_abs_mm | c1_p95_abs_mm | p95_abs_mm_improvement_pct | rmse_improved | p95_improved | clamp_count | clamp_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 019 | validation_019_024 | 0.08894 | 0.06412 | 27.90483 | 0.16680 | 0.12384 | 25.75678 | True | True | 0 | 0.00000 |
| 020 | validation_019_024 | 0.07721 | 0.07806 | -1.09411 | 0.15951 | 0.15667 | 1.78368 | False | True | 0 | 0.00000 |
| 021 | validation_019_024 | 0.06195 | 0.05707 | 7.88915 | 0.13113 | 0.10697 | 18.42187 | True | True | 0 | 0.00000 |
| 022 | validation_019_024 | 0.11842 | 0.12685 | -7.11478 | 0.16362 | 0.17551 | -7.26570 | False | False | 0 | 0.00000 |
| 023 | validation_019_024 | 0.06495 | 0.06012 | 7.44218 | 0.12832 | 0.12272 | 4.36113 | True | True | 0 | 0.00000 |
| 024 | validation_019_024 | 0.05424 | 0.05383 | 0.76586 | 0.10104 | 0.10467 | -3.60183 | True | False | 0 | 0.00000 |
| 037 | validation_037_040 | 0.09033 | 0.07008 | 22.42569 | 0.17688 | 0.12838 | 27.42109 | True | True | 0 | 0.00000 |
| 038 | validation_037_040 | 0.09632 | 0.07057 | 26.73945 | 0.16381 | 0.12917 | 21.14532 | True | True | 0 | 0.00000 |
| 039 | validation_037_040 | 0.12062 | 0.10923 | 9.43595 | 0.22758 | 0.18123 | 20.36709 | True | True | 0 | 0.00000 |
| 040 | validation_037_040 | 0.08961 | 0.06366 | 28.95660 | 0.18345 | 0.13453 | 26.66552 | True | True | 0 | 0.00000 |
| 055 | validation_055_060 | 0.09101 | 0.05294 | 41.83576 | 0.20261 | 0.10828 | 46.55643 | True | True | 1 | 0.00111 |
| 056 | validation_055_060 | 0.12245 | 0.07594 | 37.98136 | 0.24981 | 0.14095 | 43.57973 | True | True | 0 | 0.00000 |
| 057 | validation_055_060 | 0.10798 | 0.06924 | 35.87487 | 0.22504 | 0.12821 | 43.02953 | True | True | 0 | 0.00000 |
| 058 | validation_055_060 | 0.09024 | 0.06672 | 26.06242 | 0.17446 | 0.12634 | 27.58552 | True | True | 0 | 0.00000 |
| 059 | validation_055_060 | 0.10901 | 0.07622 | 30.07861 | 0.21874 | 0.13131 | 39.96678 | True | True | 0 | 0.00000 |
| 060 | validation_055_060 | 0.07606 | 0.05187 | 31.80509 | 0.16760 | 0.10305 | 38.51423 | True | True | 0 | 0.00000 |

## s-domain clamp

全体 clamp `1/14400` = `0.006944%`。这些点使用 domain edge 的 frozen spline 值，不做外推；clamp 点与非 clamp 点单独统计如下。Global/pose 指标仍包含全部 Validation 点。

| scope | point_count | pose_count | c0_rmse_mm | c1_rmse_mm | rmse_improvement_pct | c0_p95_abs_mm | c1_p95_abs_mm | p95_abs_mm_improvement_pct | clamp_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clamped_only | 1 | 1 | 0.20719 | 0.02423 | 88.30374 | 0.20719 | 0.02423 | 88.30374 | 1.00000 |
| in_domain_only | 14399 | 16 | 0.09341 | 0.07427 | 20.49458 | 0.18990 | 0.14193 | 25.25940 | 0.00000 |

## Status rule and conclusion

本轮 status 规则固定写明如下：`FAIL` 若 pooled-16 RMSE 或 P95 没有改善；若 pooled-16 两者均改善但任一 validation group、Top/Middle/Bottom 稳定性条件或 pooled pose improvement ratio（至少 0.5）不满足，则为 `PARTIAL`；只有 pooled、三个 Validation group、三个 v regions 的 RMSE/P95 均改善且至少一半 pose 的 RMSE 改善，才为 `PASS`。Clamp 不被隐式当作失败，但按上节单独报告。

因此本次结论为 **`C1_VALIDATION_STATUS = PASS`**。这只是独立 Validation 结论，不会自动写入生产配置。

## Generated outputs

- `c1_validation_summary.csv`
- `c1_validation_pose_metrics.csv`
- `c1_validation_v_bins.csv`
- `c1_validation_residual_vs_v.png`
- `c1_validation_residual_vs_s.png`
- `c1_validation_report.md`
