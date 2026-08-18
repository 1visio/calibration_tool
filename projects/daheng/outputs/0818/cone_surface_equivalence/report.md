# Cone operational surface equivalence

`CONE_SURFACE_STATUS = EQUIVALENT`

## 结论摘要

- 在相同 Full-36 held-out operational ray/v 域（32400 rays、36 poses、v=0.0..2999.1 px）上，复用四档 Cone fold 参数计算 lambda；没有重新拟合。
- 3000 参数重建与既有 pointwise lambda 的 RMSE/P95/Max 差异为 0.000000/0.000000/0.000000 mm，说明求交/root 选择路径复现一致。
- 参数漂移没有按同等幅度转化为 operational surface 漂移；需以以下 Δlambda 指标判断，而不是仅看 apex/angle 参数差异。
- 总体判定：`CONE_SURFACE_STATUS = EQUIVALENT`；各档：6000=EQUIVALENT, 12000=EQUIVALENT, all_feasible=EQUIVALENT。

## Provenance / reuse audit

- ray/v domain：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\full_fit_v_coverage_audit\full_fit_points.csv`；3000 baseline：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\cv_pointwise_circular_cone.csv`。
- 6000/12000/all 参数：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\cone_sampling_sensitivity\cone_sampling_fold_parameters.json`；每折 root hint：既有 `cv_fold_model_parameters.json`。
- 每个 sampling setting 的 fold model 只作用于对应 held-out pose rays；所有 setting 使用完全相同的 ray、v、fold、point identity。
- 未读取 Validation；未运行 Quadratic；未训练 C1；没有调用任何拟合/优化过程。

| check | result |
|---|---|
| operational rays | 32400 |
| poses / folds | 36 / 6 |
| v domain | 0.0–2999.1 px |
| 3000 baseline reconstruction max error | 0.000000 mm |
| parameter operation | loaded only; no refit |

## Global Δlambda vs 3000

负值仅表示新 sampling 的 lambda 小于 3000 baseline；指标是绝对 surface prediction difference，不是 board residual。

| setting | status | Δlambda bias / mm | Δlambda RMSE / mm | Δlambda P95 / mm | Δlambda Max / mm | valid pair rate |
|---|---|---:|---:|---:|---:|---:|
| 6000 | EQUIVALENT | -0.000489 | 0.002139 | 0.003720 | 0.014596 | 1.0000 |
| 12000 | EQUIVALENT | -0.000137 | 0.002430 | 0.004636 | 0.016530 | 1.0000 |
| all_feasible | EQUIVALENT | -0.000024 | 0.002774 | 0.005079 | 0.015765 | 1.0000 |

## Δlambda 随 v 分布

逐 100 px bin 的明细保存在 `cone_surface_equivalence.csv` 的 `pooled_v_bin` 行；图中同时给出 RMSE/P95/Max。重点观察是否在边缘或少数 bin 出现局部漂移。

| setting | largest-bin RMSE / mm | v-bin | largest-bin P95 / mm | v-bin | largest-bin Max / mm | v-bin |
|---|---:|---|---:|---|---:|---|
| 6000 | 0.006318 | v_2900_3000 | 0.014007 | v_2900_3000 | 0.014596 | v_2900_3000 |
| 12000 | 0.007090 | v_2900_3000 | 0.015803 | v_2900_3000 | 0.016530 | v_2900_3000 |
| all_feasible | 0.007130 | v_2900_3000 | 0.015042 | v_2900_3000 | 0.015765 | v_2900_3000 |

## Parameter drift vs surface drift

| setting | max axis drift / deg | max apex drift / mm | max half-angle drift / deg | Δlambda RMSE / mm | Δlambda P95 / mm | interpretation |
|---|---:|---:|---:|---:|---:|---|
| 6000 | 0.143 | 59.209 | 0.142 | 0.002139 | 0.003720 | parameter drift, surface nearly unchanged |
| 12000 | 0.244 | 100.901 | 0.244 | 0.002430 | 0.004636 | parameter drift, surface nearly unchanged |
| all_feasible | 0.253 | 104.358 | 0.252 | 0.002774 | 0.005079 | parameter drift, surface nearly unchanged |

## Operational equivalence gates

本次使用的诊断 gate（不是新的标定验收规范）：
- `EQUIVALENT`：所有新 sampling 的 Δlambda RMSE ≤ 0.01 mm、P95 ≤ 0.03 mm、Max ≤ 0.10 mm。
- `MARGINALLY_DIFFERENT`：未达到 equivalent，但所有新 sampling 的 RMSE ≤ 0.05 mm、P95 ≤ 0.10 mm、Max ≤ 0.50 mm。
- `MATERIAL_DIFFERENCE`：任一指标超过 marginal gate。

## Scope exclusions

- 不重新拟合；不读取 Validation；不运行 Quadratic；不训练 C1。
- 仅使用实际 Full-36 FIT ray/v operational domain，不引入 synthetic rays 或新的采样域。

## Outputs

- `cone_surface_equivalence.csv`
- `delta_lambda_vs_v.png`
- `report.md`
