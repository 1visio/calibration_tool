# Corrected Top/Bottom edge support audit for FIT 049–054

`EDGE_SUPPORT_CORRECTED = PARTIAL`

## Scope and mapping

- 只打开 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\fit` 下 FIT `049–054` 的 18 张图；没有打开 Validation。
- 使用上一轮相同的 after-mask：PnP 投影完整棋盘物理边界 `X=[-20,220] mm`、`Y=[-20,160] mm`，0 mm inset、无像素腐蚀。
- 保持 Steger、continuity、Frozen Circular Cone 有效性筛选和 Frozen C1 PCA `s` 不变；未重新拟合 Cone/C1。
- Frozen C1 artifact SHA-256 = `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`。

| corrected edge | frame IDs | actual assignment |
|---|---|---|
| Top | 052–054 | v=[30,520], s=[-0.191,-0.127] safety target |
| Bottom | 049–051 | v=[2760,2990], s=[0.181,0.209] safety target |

## Group-level corrected support

这里的 joint 统计要求同一个有效点同时落在目标 v 区间和目标 s 区间；conditional span 则要求用于覆盖 v 的点同时满足 s 目标，反之亦然。

| edge | effective | v range | s range | observed joint n / fraction / status | safety joint n / fraction / status |
|---|---:|---:|---:|---:|---:|
| Top | 2700 | [0.0, 2001.0] | [-0.196547, 0.075983] | 539 / 0.200 / FULL | 707 / 0.262 / FULL |
| Bottom | 2370 | [485.9, 2982.0] | [-0.130431, 0.209839] | 166 / 0.070 / FULL | 269 / 0.114 / PARTIAL |

### Joint conditional spans

| edge | domain | v span among s-in-domain | s span among v-in-domain | overlap fractions (v, s) |
|---|---|---:|---:|---:|
| Top | observed | [85.0, 467.0] | [-0.184897, -0.132818] | (1.000, 0.998) |
| Top | safe | [41.0, 509.0] | [-0.192411, -0.125863] | (0.955, 1.000) |
| Bottom | observed | [2810.0, 2937.0] | [0.186550, 0.203769] | (0.992, 0.989) |
| Bottom | safe | [2771.0, 2975.0] | [0.179582, 0.209839] | (0.887, 1.000) |

## Per-frame joint counts

| frame | corrected edge | effective | observed joint n/status | safety joint n/status |
|---:|---|---:|---:|---:|
| 049 | Bottom | 900 | 95 / FULL | 154 / PARTIAL |
| 050 | Bottom | 900 | 71 / FULL | 115 / PARTIAL |
| 051 | Bottom | 570 | 0 / NONE | 0 / NONE |
| 052 | Top | 900 | 166 / FULL | 216 / FULL |
| 053 | Top | 900 | 178 / FULL | 247 / FULL |
| 054 | Top | 900 | 195 / FULL | 244 / FULL |

## Decision

- `EDGE_SUPPORT_CORRECTED = PARTIAL`。判定使用 corrected group 的同点 joint support，不使用独立 v/s extrema 通过。
- FULL：同一条件集合下，v 与 s 的 conditional span overlap 均 ≥ 95%；PARTIAL：存在目标矩形内有效点但未达到该联合覆盖率；NONE：目标矩形内无有效点。
- 本轮没有重新采图、没有读取 Validation、没有拟合或修改 Cone/C1。

## Artifacts

- `corrected_edge_support.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\corrected_edge_support_049_054\corrected_edge_support.csv`
- `report.md`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\corrected_edge_support_049_054\report.md`
