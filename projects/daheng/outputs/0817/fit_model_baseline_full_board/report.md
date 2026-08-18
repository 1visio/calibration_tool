# FIT-only 三模型基线（完整棋盘物理 mask）

`FIT_MODEL_WINNER = QUADRATIC`

## Scope

- 仅读取 FIT：001–018、025–036、049–054，共 36 帧；没有读取任何 Validation 图像。
- 配置：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`；mask mode 固定为 `full_board_physical`，inset=0.0 mm。
- 棋盘：11×8 内角点、20 mm；物理边界为 X=[-20,220] mm、Y=[-20,160] mm。
- Steger、vertical 每 row 单点、continuity、900 点上限和 frame-balanced weighting 沿用正式流程。
- 有效 FIT 标定点：32400；有效 frame：36。
- residual 定义：模型射线重建点到该点对应 PnP 棋盘真平面的有符号距离（mm）。
- residual-v/s 中的 s 只使用冻结 PCA 定义：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`；不应用或训练 C1。
- 当前 Frozen C0 未覆盖、未修改。

## Region definition

- Top: `0 <= v < 300`；Middle: `300 <= v < 2700`；Bottom: `2700 <= v < 3000`。
- Global/区域指标均评价同一个“全 FIT 拟合”的模型，不对 Top/Middle/Bottom 单独重新拟合模型。

## Global / regional metrics

| model | region | n | frames | bias / mm | MAE / mm | RMSE / mm | P95 / mm | max abs / mm | valid rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CONE | Global | 32400 | 36 | 0.00205 | 0.06962 | 0.09673 | 0.17898 | 0.64381 | 1.0000 |
| CONE | Top | 1886 | 22 | 0.10582 | 0.11816 | 0.13939 | 0.23551 | 0.47589 | 1.0000 |
| CONE | Middle | 29048 | 36 | -0.00058 | 0.06565 | 0.09297 | 0.16454 | 0.64381 | 1.0000 |
| CONE | Bottom | 1466 | 18 | -0.07941 | 0.08576 | 0.10264 | 0.18851 | 0.43011 | 1.0000 |
| PLANE | Global | 32400 | 36 | -0.02390 | 0.26677 | 0.33732 | 0.68521 | 1.48530 | 1.0000 |
| PLANE | Top | 1886 | 22 | -0.45626 | 0.45644 | 0.47204 | 0.66583 | 0.89865 | 1.0000 |
| PLANE | Middle | 29048 | 36 | 0.04966 | 0.22123 | 0.25877 | 0.43864 | 0.90211 | 1.0000 |
| PLANE | Bottom | 1466 | 18 | -0.92525 | 0.92525 | 0.94932 | 1.28142 | 1.48530 | 1.0000 |
| QUADRATIC | Global | 32400 | 36 | 0.00324 | 0.06990 | 0.09647 | 0.17603 | 0.63139 | 1.0000 |
| QUADRATIC | Top | 1886 | 22 | 0.10820 | 0.12666 | 0.14615 | 0.23010 | 0.51374 | 1.0000 |
| QUADRATIC | Middle | 29048 | 36 | 0.00074 | 0.06528 | 0.09213 | 0.16363 | 0.63139 | 1.0000 |
| QUADRATIC | Bottom | 1466 | 18 | -0.08242 | 0.08858 | 0.10009 | 0.16194 | 0.41005 | 1.0000 |

## FIT-only residual trends

| model | variable | slope | rho | binned median range / mm | max adjacent jump / mm | within-bin RMSE / mm |
|---|---|---:|---:|---:|---:|---:|
| PLANE | v | -0.0000541 | -0.11725 | 1.35185 | 0.29519 | 0.10308 |
| PLANE | s | -0.3975666 | -0.11740 | 1.35253 | 0.29738 | 0.10328 |
| QUADRATIC | v | -0.0000016 | -0.01208 | 0.27473 | 0.09111 | 0.07984 |
| QUADRATIC | s | -0.0118810 | -0.01224 | 0.27391 | 0.09123 | 0.07982 |
| CONE | v | 0.0000011 | 0.00806 | 0.28229 | 0.09056 | 0.07951 |
| CONE | s | 0.0077073 | 0.00792 | 0.28159 | 0.09035 | 0.07952 |

## Circular Cone parameter diagnostic

| subset | status | frames | points | success | cost | half angle / deg | axis angle to global / deg | apex / mm |
|---|---|---:|---:|---|---:|---:|---:|---|
| global | OK | 36 | 32400 | True | 1.24412 | 88.95474 | 0.00000 | [-131.78611, 15.52716, 275.44477] |
| fit_001_018 | OK | 18 | 16200 | True | 0.90296 | 88.96615 | 0.03328 | [-134.32017, 2.95802, 266.82381] |
| fit_025_036 | OK | 12 | 10800 | True | 1.82635 | 86.78775 | 2.17286 | [-372.71907, 33.74853, -500.00000] |
| fit_049_054 | OK | 6 | 5400 | True | 0.74206 | 89.50774 | 0.57985 | [-62.31747, 39.36219, 500.00000] |

- Cone stability diagnostic: `FAIL`; subset half-angle range=2.71998 deg, maximum axis deviation=2.17286 deg.
- 该稳定性检查是 FIT 子集敏感性诊断，不是 Validation 泛化证明。

## Candidate selection

评分同时考虑 Global RMSE、Top/Bottom worst RMSE、edge P95、Top/Bottom bias range 和 v/s 分箱趋势幅度；Cone 若子集参数不稳定会增加惩罚。

| candidate | score | Global RMSE | edge worst RMSE | edge worst P95 | edge bias range | trend penalty |
|---|---:|---:|---:|---:|---:|---:|
| QUADRATIC | 0.59034 | 0.09647 | 0.14615 | 0.23010 | 0.19063 | 0.27473 |
| CONE | 1.58764 | 0.09673 | 0.13939 | 0.23551 | 0.18523 | 0.28229 |
| PLANE | 2.83810 | 0.33732 | 0.94932 | 1.28142 | 0.46900 | 1.35253 |

FIT-only 候选结论：`FIT_MODEL_WINNER = QUADRATIC`。
该结论只用于选择下一步 C0 候选，不能替代独立 Validation；进入实际使用前仍需在冻结 Validation 或标准件数据上复核。

## Artifacts

- `models/global_plane.yaml`, `models/quadratic_graph.yaml`, `models/circular_cone.yaml`：三模型参数。
- `model_parameters.json`：三模型参数及 Cone 子集诊断。
- `calibration_points_fit.csv`：新 mask 下的 FIT 标定点。
- `model_comparison_fit.csv`：Global/Top/Middle/Bottom 指标。
- `residual_vs_v.png`、`residual_vs_s.png`：FIT-only 残差趋势。
