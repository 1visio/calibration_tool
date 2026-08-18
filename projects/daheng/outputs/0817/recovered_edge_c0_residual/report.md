# Frozen Circular Cone C0 residual audit for recovered edge FIT points

`C0_EDGE_STATUS = NEED_C0_REFIT`

## Scope and residual definition

- 只打开旧 FIT `001–018、025–036`（`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane`）和新 FIT `049–054`（`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\fit`）的 108 张三联图；没有枚举或读取 Validation。
- 两组数据均使用新完整棋盘物理 mask：PnP 投影 `X=[-20,220] mm`、`Y=[-20,160] mm`，0 mm inset、无腐蚀。
- 保持 PnP、Steger、`vertical` 每 row 单点、continuity 和 900 点上限不变；没有拟合或修改 C0/C1。
- Frozen C1 artifact：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`；SHA-256 = `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`。C1 仅用于固定 PCA `s` 坐标。
- `old_mask_existing`：new-mask effective point 落在旧 inner-corner hull + margin -2 mask 内；`new_mask_recovered`：该点落在旧 mask 外。
- 残差定义为相机 ray 深度误差：`residual_mm = lambda_truth - lambda_cone`；`lambda_truth` 来自当前 frame 的 PnP plane-ray truth，`lambda_cone` 来自 Frozen Circular Cone C0 production reconstruction。

## Frozen Top/Middle/Bottom regions

| region | sensor v interval |
|---|---:|
| Top | [0, 300) px |
| Middle | [300, 2700) px |
| Bottom | [2700, 3000) px |

## Bias / RMSE / P95 / max

| dataset | point class | region | n | frames | bias (mm) | RMSE (mm) | P95 abs (mm) | max abs (mm) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| old_fit | old_mask_existing | Top | 69 | 6 | 0.1996 | 0.2337 | 0.4757 | 0.5385 |
| old_fit | old_mask_existing | Middle | 21466 | 30 | 0.0083 | 0.0972 | 0.1586 | 0.6226 |
| old_fit | old_mask_existing | Bottom | 125 | 7 | -0.0707 | 0.0780 | 0.1127 | 0.1421 |
| old_fit | new_mask_recovered | Top | 959 | 15 | 0.2004 | 0.2141 | 0.3226 | 0.5826 |
| old_fit | new_mask_recovered | Middle | 2784 | 29 | -0.0019 | 0.1096 | 0.1715 | 0.7183 |
| old_fit | new_mask_recovered | Bottom | 882 | 14 | -0.1183 | 0.1345 | 0.2235 | 0.2848 |
| new_fit | old_mask_existing | Top | 75 | 3 | 0.2130 | 0.2340 | 0.3366 | 0.3689 |
| new_fit | old_mask_existing | Middle | 3699 | 6 | 0.0334 | 0.0752 | 0.1500 | 0.3701 |
| new_fit | old_mask_existing | Bottom | 21 | 1 | -0.0399 | 0.0521 | 0.0871 | 0.1199 |
| new_fit | new_mask_recovered | Top | 402 | 3 | 0.2499 | 0.2600 | 0.3547 | 0.3856 |
| new_fit | new_mask_recovered | Middle | 578 | 6 | -0.0196 | 0.0742 | 0.1175 | 0.2877 |
| new_fit | new_mask_recovered | Bottom | 295 | 2 | -0.0814 | 0.0914 | 0.1504 | 0.1824 |

## Recovered-edge cross-frame consistency and trends

状态判定只使用 `new_mask_recovered` 且 region 为 Top/Bottom 的点；Middle recovery 保留在前面的区域表中作为诊断，不混入 edge decision。

| dataset | recovered n | frames | frame bias range (mm) | frame RMSE range (mm) | s rho | s slope (mm/unit s) | s binned range | s within-bin RMSE | s max adjacent jump | v rho | v slope (mm/px) | v binned range | v within-bin RMSE | v max adjacent jump |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_fit | 1841 | 26 | 0.5328 | 0.2983 | -0.838 | -0.8927 | 0.3606 | 0.0677 | 0.2937 | -0.839 | -0.000122 | 0.3614 | 0.0677 | 0.2932 |
| new_fit | 697 | 5 | 0.4389 | 0.2622 | -0.855 | -0.8896 | 0.3721 | 0.0597 | 0.2952 | -0.855 | -0.000121 | 0.3721 | 0.0598 | 0.2952 |
| combined | 2538 | 31 | 0.5328 | 0.2983 | -0.843 | -0.8976 | 0.3844 | 0.0674 | 0.2878 | -0.843 | -0.000122 | 0.3848 | 0.0673 | 0.2878 |

### Region-specific cross-frame consistency

| dataset | region | recovered n | frames | frame bias range (mm) | frame RMSE range (mm) | s binned range | s within-bin RMSE | s max adjacent jump |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old_fit | Top | 959 | 15 | 0.2311 | 0.2459 | 0.1265 | 0.0705 | 0.0679 |
| old_fit | Bottom | 882 | 14 | 0.2525 | 0.1439 | 0.1093 | 0.0586 | 0.0452 |
| new_fit | Top | 402 | 3 | 0.1635 | 0.1629 | 0.1426 | 0.0707 | 0.1216 |
| new_fit | Bottom | 295 | 2 | 0.0649 | 0.0630 | 0.0772 | 0.0377 | 0.0459 |
| combined | Top | 1361 | 18 | 0.2311 | 0.2459 | 0.2095 | 0.0711 | 0.0659 |
| combined | Bottom | 1177 | 16 | 0.2525 | 0.1439 | 0.1036 | 0.0583 | 0.0452 |

### Recovered-edge aggregate (Top/Bottom only)

- n=2538，bias=0.0647 mm，RMSE=0.1878 mm，P95 abs=0.3164 mm，max abs=0.5826 mm。
- s trend: binned range=0.3844 mm，within-bin RMSE=0.0674 mm，最大相邻 bin 跳变=0.2878 mm（包含 Top/Bottom 区域间隔）。
- v trend: binned range=0.3848 mm，within-bin RMSE=0.0673 mm，最大相邻 bin 跳变=0.2878 mm（包含 Top/Bottom 区域间隔）。
- frame bias range=0.5328 mm，frame RMSE range=0.2983 mm；分 region 后 Top/Bottom 仍分别为 0.2311/0.2525 mm（old FIT），新 FIT Top/Bottom 为 0.1635/0.0649 mm。

## Decision rule

- PnP truth uncertainty reference: 0.025–0.033 mm；本审计使用上限 0.033 mm。
- STABLE：Top/Bottom recovered 点的 P95、bin 内残差和跨 frame bias range 均不超过 0.033 mm。
- CORRECTABLE_BY_C1：Top/Bottom recovered 残差超过 uncertainty，但 s/v 低频 trend 后的 bin 内 RMSE ≤0.050 mm、跨 frame bias range ≤0.080 mm、无明显大跳变（≤0.080 mm），说明主要是平滑空间项。
- NEED_C0_REFIT：跨 frame 不一致、bin 内噪声或跳变超过上述范围。以上仅为诊断判据，没有训练 C1。

- 结论：`C0_EDGE_STATUS = NEED_C0_REFIT`。
- 下一步建议：新增边缘点表现出非平滑或跨 frame 不一致，先重新评估/拟合 C0，再考虑 C1。

## Artifacts

- `recovered_edge_c0_residual.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\recovered_edge_c0_residual\recovered_edge_c0_residual.csv`
- `residual_vs_s.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\recovered_edge_c0_residual\residual_vs_s.png`
- `residual_vs_v.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\recovered_edge_c0_residual\residual_vs_v.png`
- `report.md`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\recovered_edge_c0_residual\report.md`
