# Full-36 Quadratic / Circular Cone 配对稳定性检验

`C0_PAIRED_STATUS = UNRESOLVED`

## 结论摘要

- 使用已有 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison` 的两份 pointwise held-out prediction；没有重新拟合模型。
- Q/C 均覆盖 36 个 held-out pose、32400 个相同点、6-fold pose-grouped CV；点级配对通过。
- 按 pose 的 RMSE win count：Quadratic 17，Cone 19，Tie 0；按 100 px v-bin：Quadratic 23，Cone 7，Tie 0。
- Global RMSE Δ(Q−C)=-0.00180 mm；95% pose-bootstrap CI 见下表。
- `C0_PAIRED_STATUS = UNRESOLVED`：判定要求 Global 与共同 worst-region 的 RMSE/P95 CI 同时不跨 0，并且 pose/bin 方向一致；当前结果未满足单一模型的全部条件。

## Artifact provenance / reuse audit

- Pointwise 来源：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\cv_pointwise_quadratic_graph.csv` 与 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\cv_pointwise_circular_cone.csv`。
- 交叉核对：`grouped_cv_model_comparison.csv` 的 Q/C pooled RMSE/P95 与 pointwise 重算一致；`per_v_bin_cv_metrics.csv` 用于 v-bin 结果复核。
- 两模型使用相同 FIT pose、相同 held-out fold、相同 frame_key/点坐标；mask、weighting、6-fold pose-grouped protocol 均继承 0817 artifact。
- 只读取 FIT grouped-CV artifacts；未读取 Validation，未训练 C1。

| check | result |
|---|---|
| Q/C rows | 32400 / 32400 |
| paired rows | 32400 |
| paired poses | 36 |
| folds | 6 |
| valid paired points | 32400 |
| finite paired points | 32400 |
| v domain observed | 0.0–2999.1 px |
| point identity | frame_key + deterministic point_index; coordinate/fold/frame checks passed |

## Pooled paired metrics

Negative Δ(Q−C) means Quadratic has lower error.

| metric | Quadratic | Cone | Δ(Q−C) |
|---|---:|---:|---:|
| Global MAE / mm | 0.07191 | 0.07237 | -0.00045 |
| Global RMSE / mm | 0.09909 | 0.10089 | -0.00180 |
| Global P95 / mm | 0.18438 | 0.19054 | -0.00616 |

## Pose-unit bootstrap 95% CI

Bootstrap replicates=10000, seed=20260818; each replicate resamples pose IDs with replacement, never individual points.共同 worst-region=`v_0000_0100`。

| scope | metric | observed Δ(Q−C) / mm | bootstrap mean | 95% CI low | 95% CI high | pose units |
|---|---|---:|---:|---:|---:|---:|
| Global | rmse | -0.00180 | -0.00174 | -0.00501 | 0.00163 | 36 |
| Global | mae | -0.00045 | -0.00046 | -0.00313 | 0.00221 | 36 |
| Global | p95 | -0.00616 | -0.00626 | -0.01872 | 0.00482 | 36 |
| WorstRegion | rmse | -0.02697 | -0.02527 | -0.05033 | 0.00426 | 11 |
| WorstRegion | mae | -0.02700 | -0.02558 | -0.04903 | 0.00167 | 11 |
| WorstRegion | p95 | -0.04204 | -0.04129 | -0.07758 | 0.01338 | 11 |

## Spatial consistency across v

- Q better in 23/30 populated v-bins by pooled RMSE; Cone better in 7/30. Bin direction is therefore not uniformly Quadratic-favored.
- Q-favored pose squared-error advantage top-5 pose fraction=65.2% of all Q-favored squared-error reduction; this is a concentration diagnostic, not a point bootstrap.
- Q-favored v-bin squared-error total=15.814 mm²; strongest Q-favored bins are: v_0000_0100, v_0600_0700, v_0500_0600, v_1700_1800, v_0700_0800。

| v-bin | pose count | Q win poses | Cone win poses | Δ RMSE / mm | Δ P95 / mm | winner |
|---|---:|---:|---:|---:|---:|---|
| v_0000_0100 | 11 | 7 | 4 | -0.02697 | -0.04204 | Quadratic |
| v_0100_0200 | 18 | 10 | 8 | -0.00056 | -0.02822 | Quadratic |
| v_0200_0300 | 22 | 10 | 12 | 0.00994 | 0.00943 | Cone |
| v_0300_0400 | 26 | 12 | 14 | 0.00502 | -0.00328 | Cone |
| v_0400_0500 | 30 | 15 | 15 | -0.00296 | -0.00208 | Quadratic |
| v_0500_0600 | 30 | 18 | 12 | -0.00732 | -0.01053 | Quadratic |
| v_0600_0700 | 31 | 17 | 14 | -0.00561 | -0.00918 | Quadratic |
| v_0700_0800 | 31 | 14 | 17 | -0.00369 | 0.00104 | Quadratic |
| v_0800_0900 | 33 | 17 | 16 | -0.00234 | -0.00100 | Quadratic |
| v_0900_1000 | 35 | 17 | 18 | -0.00053 | -0.00587 | Quadratic |
| v_1000_1100 | 36 | 12 | 24 | 0.00020 | -0.00393 | Cone |
| v_1100_1200 | 36 | 14 | 22 | 0.00002 | -0.00455 | Cone |
| v_1200_1300 | 36 | 23 | 13 | -0.00099 | 0.00079 | Quadratic |
| v_1300_1400 | 36 | 21 | 15 | -0.00232 | -0.00459 | Quadratic |
| v_1400_1500 | 36 | 19 | 17 | -0.00151 | -0.00018 | Quadratic |
| v_1500_1600 | 36 | 19 | 17 | -0.00183 | -0.00497 | Quadratic |
| v_1600_1700 | 36 | 12 | 24 | -0.00181 | 0.01391 | Quadratic |
| v_1700_1800 | 36 | 15 | 21 | -0.00374 | -0.01547 | Quadratic |
| v_1800_1900 | 36 | 16 | 20 | -0.00287 | -0.00233 | Quadratic |
| v_1900_2000 | 36 | 19 | 17 | -0.00091 | -0.00051 | Quadratic |
| v_2000_2100 | 30 | 17 | 13 | -0.00144 | -0.00470 | Quadratic |
| v_2100_2200 | 25 | 11 | 14 | -0.00048 | -0.00606 | Quadratic |
| v_2200_2300 | 23 | 13 | 10 | -0.00151 | -0.00814 | Quadratic |
| v_2300_2400 | 24 | 13 | 11 | 0.00115 | 0.02386 | Cone |
| v_2400_2500 | 24 | 14 | 10 | -0.00015 | 0.00868 | Quadratic |
| v_2500_2600 | 24 | 17 | 7 | -0.00154 | 0.00291 | Quadratic |
| v_2600_2700 | 20 | 9 | 11 | 0.00222 | 0.00831 | Cone |
| v_2700_2800 | 18 | 7 | 11 | -0.00432 | -0.01653 | Quadratic |
| v_2800_2900 | 15 | 6 | 9 | -0.00471 | -0.04482 | Quadratic |
| v_2900_3000 | 11 | 5 | 6 | 0.00463 | -0.01624 | Cone |

## Interpretation

Quadratic 的 pooled Full-36 指标略低，但配对检验要求同时具备 pose、v-bin 和 bootstrap CI 的方向一致性。若 CI 跨 0，或优势集中在少数 pose/bin，则只能视为轻微候选优势，不能将 C0 冻结为 Quadratic。
当前判定：`C0_PAIRED_STATUS = UNRESOLVED`。

## Scope exclusions

- 未重新拟合 Quadratic/Cone；未运行 Plane；未读取 Validation；未训练 C1。
- bootstrap 单位为 pose，禁止逐点 bootstrap。

## Outputs

- `paired_pose_model_comparison.csv`
- `paired_v_bin_comparison.csv`
- `paired_bootstrap_summary.csv`
- `paired_model_difference.png`
- `report.md`
