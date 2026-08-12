# 90° 旋转等价测试

- 图像：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\obs\test_350\fit\laser 003.tif`
- 尺寸：`4096×3000`，`uint8`
- 对照：原图 `scan_axis=row`（vertical pipeline）。
- 候选：旋转 90° 后 `scan_axis=column`（horizontal pipeline），再映射回原图坐标。
- 逐点阈值：`0.020000 px`；低频告警阈值：`0.050000 px`。
- 总结论：**PASS**。

| rotation | native | rotated | common | only native | only rotated | max abs Δp px | P95 abs Δp px | low-freq max abs bin mean px | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clockwise | 2242 | 2242 | 2242 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | PASS |
| counterclockwise | 2242 | 2242 | 2242 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | PASS |

低频量使用固定 v 宽度分箱后的 `mean(Δu)`；逐点明细见 `point_deltas.csv`，分箱明细见 `low_frequency_bins.csv`。
当前 `row` 路径会先转置原图，再复用 columnwise 核心；因此本测试主要验证 row adapter、旋转/坐标恢复以及正式入口的端到端方向等价性。
