# Full-board final centerline continuity audit

`OUTER_BOARD_LINE_CONTINUITY = PASS`

## Scope and method

- 只打开 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\fit` 下的 FIT `049–054` 三联图；没有打开任何 Validation 图像。
- 使用理论完整棋盘边界 `X=[-20,220] mm`、`Y=[-20,160] mm` 的投影 mask，重新执行现有 Steger、`vertical` 每 row 单点、continuity 和 900 点上限。
- 输出 overlay 只画最终中心线：蓝色 = 完整棋盘 mask 最终线，洋红色 = 当前正式 inner-hull mask 最终线；红色 = inner-corner hull，黄色 = 理论外边界；不画 raw candidates。
- 未拟合或修改 Cone/C1/K/D；正式 mask 未修改。

## Boundary continuity

| frame | full line v range | current line v range | boundary transitions | max u jump / px | max slope delta | jump outliers | boundary status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 049 | [953.0, 2977.0] | [1280.0, 2751.9] | 2 | 0.108 | 0.01670 | 0 | true |
| 050 | [1008.0, 2982.0] | [1227.0, 2760.0] | 2 | 0.088 | 0.03082 | 0 | true |
| 051 | [485.9, 2999.0] | [674.0, 2771.0] | 2 | 0.186 | 0.11976 | 0 | true |
| 052 | [0.0, 1975.0] | [181.0, 1749.0] | 2 | 0.145 | 0.01316 | 0 | true |
| 053 | [28.0, 2001.0] | [247.0, 1762.0] | 2 | 0.164 | 0.01485 | 0 | true |
| 054 | [14.0, 1947.0] | [242.0, 1742.1] | 2 | 0.123 | 0.01129 | 0 | true |

## 052–054 top dual-candidate branch

Branch A/B is defined geometrically per row: A = smaller u (left branch), B = larger u (right branch). The dual region is the first persistent raw-candidate dual band from the top, with a >100 px gap separating later isolated bands.

| frame | dual-region v | dual rows | branch A fit RMSE/P95 | branch B fit RMSE/P95 | selected A | selected B | ambiguous | switches | final choice |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 052 | [0.0, 179.0] | 33 | 0.124/0.244 | 0.161/0.312 | 0 (0.0%) | 101 (100.0%) | 0 (0.0%) | 0 | `B_right` |
| 053 | [33.0, 243.0] | 63 | 3.970/10.868 | 0.083/0.110 | 0 (0.0%) | 165 (100.0%) | 0 (0.0%) | 0 | `B_right` |
| 054 | [18.0, 237.0] | 121 | 2.796/1.348 | 0.139/0.260 | 0 (0.0%) | 123 (100.0%) | 0 (0.0%) | 0 | `B_right` |

## Conclusion

- Inner-hull boundary continuity overall: `PASS`。
- 052–054 top dual region final selection consistency: `STABLE_B_right`；报告中的 B_right 表示较大 u 的右侧分支。
- 如果 selected branch 在同一帧发生切换、或 boundary u jump 成为局部异常值，则不能把外圈点与正式中心线视为同一条稳定中心线。

## Artifacts

- `final_line_overlay_049_054.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\final_line_continuity_049_054\final_line_overlay_049_054.png`
- `boundary_continuity.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\final_line_continuity_049_054\boundary_continuity.csv`
