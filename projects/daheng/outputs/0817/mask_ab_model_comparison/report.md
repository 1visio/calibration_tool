# 001–018 board mask A/B 三模型拟合比较

`MASK_FIX_EFFECT = MODERATE`

## Scope

- 仅读取 FIT 001–018 的 chess / nolaser / laser 三联图；未读取 Validation，也未读取 025–036、049–054。
- FIT root：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\fit`
- 内参：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`（来自用户指定的 intrinsics 路径）
- 配置基线：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`
- Old：inner-corner convex hull，`margin_px=-2`。
- New：PnP 投影完整棋盘物理边界，11×8 内角点、20 mm 格距，X=[-20,220] mm、Y=[-20,160] mm，inset=0 mm。
- 两个分支均使用同一 PnP、内参、Steger、vertical 每 row 单点、continuity、每帧最多 900 点和 frame-balanced weighting；仅切换 board mask。
- Old/New 各自从头拟合 global_plane、quadratic_graph、circular_cone；没有训练 C1，也没有覆盖 Frozen C0。
- residual 定义：模型从同一分支激光像素反算的点到对应 PnP 棋盘真实平面的有符号距离（mm）。

## Extracted FIT support

| mask | frames | points | min v / px | max v / px | min u / px | max u / px |
|---|---:|---:|---:|---:|---:|---:|
| Old | 18 | 16102 | 242.00 | 2731.98 | 1836.56 | 2131.67 |
| New | 18 | 16200 | 36.00 | 2961.04 | 1800.96 | 2135.46 |

`point_count` 是各 mask 分支自己的重新提取结果；由于 mask 改变了采样点集合，Old/New 的误差比较不是 point-wise 配对比较，而是同一 FIT 帧集合上的独立拟合/评价比较。

## Global comparison

| model | Old RMSE | New RMSE | ΔRMSE (Old-New) | Old P95 | New P95 | Old max | New max |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_plane | 0.22641 | 0.29824 | -0.07183 | 0.42863 | 0.54155 | 0.91171 | 1.41122 |
| quadratic_graph | 0.07447 | 0.08197 | -0.00750 | 0.14429 | 0.15854 | 0.56258 | 0.39337 |
| circular_cone | 0.07410 | 0.08147 | -0.00738 | 0.14035 | 0.15678 | 0.56688 | 0.43173 |

## Worst v-bin comparison

| model | Old worst bin | Old RMSE | New RMSE at Old bin | Δ at Old bin | New worst bin | New worst RMSE |
|---|---|---:|---:|---:|---|---:|
| global_plane | v_2700_2800 | 0.80463 | 0.96687 | -0.16224 | v_2900_3000 | 1.14375 |
| quadratic_graph | v_0200_0300 | 0.24835 | 0.11941 | 0.12894 | v_2900_3000 | 0.15183 |
| circular_cone | v_0200_0300 | 0.23882 | 0.11422 | 0.12461 | v_0000_0100 | 0.16213 |

## 100 px v-bin detail

完整 Global 与 100 px bin 指标见 `mask_ab_model_comparison.csv`。每个 bin 均记录 Bias、MAE、RMSE、P95、Max abs、有效交点率和点数。

## Effect rule

- SIGNIFICANT：至少 2/3 模型同时满足 Global RMSE 和 Old worst-v-bin RMSE 相对改善 ≥10%，且没有 ≥10% 的 Global 或 ≥20% 的 Old worst-bin 明显恶化。
- MODERATE：至少一个模型在 Global 或 Old worst-v-bin 达到 ≥10% 改善，且未出现所有模型 Global 同时明显恶化。
- WEAK：不满足上述条件。该标签描述本次 FIT-only mask A/B 影响，不代表 Validation 泛化结论。

- 统计得到的 Global RMSE 中位相对改善：-10.08%。
- 统计得到的 Old worst-v-bin RMSE 中位相对改善：51.92%。
- 同时达到阈值的模型数：0；至少一项达到阈值的模型数：2。

## Artifacts

- `mask_ab_model_comparison.csv`：Old/New × 三模型的 Global 与 v=0–3000、100 px bins 指标。
- `residual_vs_v_mask_ab.png`：Old/New 两个 mask 分支的 residual-v 散点与 100 px 中位趋势。
- `models/old/`、`models/new/`：各分支独立拟合的三模型参数，仅作本次 A/B 审计记录。

结论：`MASK_FIX_EFFECT = MODERATE`。本报告仅隔离 mask 对 001–018 FIT-only 拟合的影响；是否改善独立数据，仍需单独的冻结 Validation/标准件验证。
