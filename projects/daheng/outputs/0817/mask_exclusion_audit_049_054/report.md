# Board-mask edge exclusion audit for FIT 049–054

`EDGE_POINTS_LOST_BY_BOARD_MASK = YES`

## Scope and method

- 只打开 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\fit` 下的 `chess/laser/nolaser 049–054.tif`；没有打开任何 Validation 图像。
- 棋盘 PnP：11×8 内角点，格距 20 mm；理论外边界使用物理坐标 `X=[-20,220] mm`、`Y=[-20,160] mm`，再用 PnP 投影回图像。
- 红色 = 检测内角点的原始 convex hull；黄色 = 理论完整棋盘外边界；绿色 = 全图 mask=True 时的 Steger 原始候选；蓝色 = 当前 `board_inner_mask(margin_px=-2)` 后沿用正式连续性筛选/900 点上限得到的最终点。
- 当前正式 mask 未修改；Cone/C1 未用于本审计计算、未拟合或修改；Steger 参数和 `vertical` 方向沿用现有流程。
- `raw_complete_outside_hull` 是“理论完整棋盘内、但当前 inner-corner hull 外”的原始 Steger 候选；同时报告 mask-free 连续性筛选后的同类点，避免把单纯噪声候选误解为稳定激光线。

## Per-frame result

| frame | PnP RMSE / px | raw candidates | masked candidates before selection | final points | complete∩outside hull raw | v range / px | complete∩outside hull mask-free selected | v range / px |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 049 | 0.1580 | 3028 | 1122 | 741 | 659 | [953.0, 2977.0] | 186 | [954.0, 2977.0] |
| 050 | 0.1334 | 2980 | 1409 | 900 | 376 | [1008.0, 2982.0] | 131 | [1010.0, 2981.0] |
| 051 | 0.1656 | 2843 | 1654 | 900 | 598 | [485.9, 2999.0] | 210 | [487.1, 2999.0] |
| 052 | 0.1481 | 3392 | 1543 | 900 | 599 | [-0.0, 1975.0] | 143 | [0.0, 1974.0] |
| 053 | 0.1322 | 2679 | 1061 | 864 | 422 | [28.0, 2001.0] | 133 | [29.0, 2000.0] |
| 054 | 0.0988 | 3309 | 1534 | 900 | 550 | [14.0, 1947.0] | 138 | [14.0, 1946.0] |

## Conclusion

- raw Steger 候选中落在完整棋盘内、但当前 inner-corner hull 外的总数：`3204`。
- 落在完整棋盘内、但当前正式 eroded mask 外的总数：`3297`；其中仅由 `margin_px=-2` 腐蚀额外排除、但仍在原始 hull 内的数量：`93`。
- 经过 mask-free 的同一连续性筛选后，仍落在完整棋盘内、但 hull 外的总数：`941`。
- `EDGE_POINTS_LOST_BY_BOARD_MASK = YES`。
- 这说明当前 mask 确实排除了至少一部分理论完整棋盘区域内的 Steger 候选；是否为真实可见激光点，应结合 overlay 中绿色点是否形成连续激光线判断。

## Artifacts

- `mask_exclusion_summary.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\mask_exclusion_audit_049_054\mask_exclusion_summary.csv`
- `mask_overlay_049.png` … `mask_overlay_054.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\mask_exclusion_audit_049_054`
