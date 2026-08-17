# FIT ray residual spatial observability

## 结论

- `RAY_SUPPORT = WEAK_2D`
- `SPATIAL_RESIDUAL = MODERATE`
- 下一步建议：`C1 only`

本轮使用 30 个 FIT 帧（001–018、025–036），有效 ray 共 **26,663** 个；frame 027 保留在统计中。未打开、未读取、未评分 Validation 019–024、037–040。

## 数据与计算边界

- K/D：冻结 M0 runtime K/D；`(u,v) → (xn,yn)` 使用 `cv2.undistortPoints`。
- Truth：当前 FIT 棋盘 PnP plane，`lambda_truth = -d / (ray · normal)`，取 camera-Z/lambda。
- Cone：冻结 Circular Cone production path；`lambda_cone` 只做重放，不重新拟合。
- 残差定义：`delta_lambda = lambda_truth - lambda_cone`。
- 未训练 spline、未生成 `F(s)`/`F(s,t)`、未修改 K/D 或 Cone；约 300 mm 仅作为当前水平物理视场约束，不创建新的 `ROI_300`。

## Ray support / PCA

PCA center = (-0.0080211875, -0.0094126829); s-axis = (-0.010119852, 0.99994879); t-axis = (-0.99994879, -0.010119852).
主方向解释方差 **0.9833**，次方向 **0.0167**，sqrt eigenvalue ratio **7.67**。
全局 robust span（P05–P95）：s = **0.26537416**，t = **0.033145701**，t/s = **0.1249**。
逐帧 t/s robust-span ratio 中位数 **0.0107**；达到 0.05/0.10 的帧比例为 **0.367 / 0.233**。

判据：`CLEAR_2D` 需要次方向解释方差 ≥0.10、全局 t/s ≥0.15、逐帧 t/s 中位数 ≥0.10 且 ≥70% 帧达到 0.10；较弱但非退化的 2D 支持归为 `WEAK_2D`，其余为 `1D`。

## Residual low-frequency trends

低频趋势使用每个 predictor 的 12 个等样本 bin 的 bin-mean explained fraction 和 bin-mean peak-to-peak；它们是描述性统计，不是 correction fit。

| residual | predictor | global low-frequency amplitude (mm) | binned EV | linear slope over span (mm) | frame sign consistency | median frame EV |
|---|---:|---:|---:|---:|---:|---:|
| raw | s | 0.133384 | 0.1795 | -0.00331066 | 0.433 | 0.6661 |
| raw | t | 0.0609446 | 0.0342 | 0.005453 | 0.300 | 0.6095 |
| raw | u | 0.0733474 | 0.0381 | -0.00528284 | 0.400 | 0.6466 |
| raw | v | 0.133327 | 0.1798 | -0.00329983 | 0.433 | 0.6661 |
| frame median subtracted | s | 0.130249 | 0.2636 | 0.022201 | 0.567 | 0.6661 |
| frame median subtracted | t | 0.0754434 | 0.0720 | -0.0508145 | 0.700 | 0.6095 |
| frame median subtracted | u | 0.0719427 | 0.0719 | 0.0498323 | 0.600 | 0.6466 |
| frame median subtracted | v | 0.130414 | 0.2633 | 0.022277 | 0.567 | 0.6661 |

frame median 的跨帧范围为 **0.465353 mm**；全体 raw residual P05–P95 为 **-0.120103–0.135989 mm**，去 frame median 后为 **-0.113878–0.107338 mm**。
frame 027 的 residual median = **0.379911 mm**；保留它用于最终结论，同时给出去 027 的 centered s/t amplitude：s **0.12223 mm**，t **0.0435645 mm**。

## PnP truth uncertainty comparison

采用已有 PnP truth uncertainty 参考带 **0.025–0.033 mm**。去 frame median 后，候选空间结构的最大证据方向为 **s**，低频幅度 **0.169483 mm**，约为上限的 **5.14×**；逐帧同号比例 **0.567**。

这里的比较只用于量级判断：若去中位数后的重复空间变化不超过约 0.025–0.033 mm，不能把它稳健地解释为 Cone 的独立空间误差；超过该带且跨帧同向重复，才进入 STRONG/MODERATE。

## Decision gates

- `SPATIAL_RESIDUAL = STRONG`：低频幅度 ≥ 0.099 mm、逐帧同号 ≥ 0.60、逐帧 median binned EV ≥ 0.10。
- `SPATIAL_RESIDUAL = MODERATE`：低频幅度 ≥ 0.050 mm、逐帧同号 ≥ 0.40、逐帧 median binned EV ≥ 0.03。
- 否则为 `WEAK`，建议 `STOP`；即便 residual 有信号，只有 `CLEAR_2D` ray support 且 t 方向超过 uncertainty 才建议 `C1 + C2`，否则先做 `C1 only`。

## Reproducibility

- FIT data root: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane`
- measurement config: `D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\measure_tool_daheng_0811.yaml`
- frozen Circular provenance: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\circular_vs_elliptical_cone\provenance.json`
- formal Cone range file: `D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- 详细逐点数据见 `fit_ray_residual_points.csv`；全局、去 027、逐帧及 frame aggregate 趋势见 `spatial_residual_observability.csv`；PCA/support 与分类证据见 `ray_support_summary.json`。
