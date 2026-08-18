# Old FIT mask support comparison

`OLD_FIT_EDGE_SUPPORT_AFTER_MASK_FIX = PARTIAL`

## Scope and frozen processing

- 只打开 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane` 下 FIT `001–018` 和 `fit_edge_extension/fit` 下 FIT `025–036` 的 90 张图；没有枚举或打开 Validation。
- Old mask：inner-corner convex hull + `margin_px=-2`。
- New mask：PnP 投影完整棋盘物理边界 `X=[-20,220] mm`、`Y=[-20,160] mm`，0 mm inset、无额外腐蚀。
- 两种 mask 均使用相同 Steger、`vertical` 每 row 单点、continuity 和 900 点上限；有效点使用相同 Frozen Circular Cone validity filter。
- Frozen C1 artifact：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`；SHA-256 = `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`。C1 仅用于固定 PCA `s`，没有重新拟合。
- 旧 FIT 没有可靠的 frame-to-Top/Bottom 映射；Top/Bottom 作为冻结 sensor-domain target，在全部 FIT union 和每帧上按同点 `v/s` 联合条件统计。

## Frozen target domains

| edge | observed v | observed s | safety v | safety s |
|---|---:|---:|---:|---:|
| Top | [86, 467] | [-0.185001, -0.132862] | [30, 520] | [-0.191, -0.127] |
| Bottom | [2810, 2938] | [0.186364, 0.203712] | [2760, 2990] | [0.181, 0.209] |

`joint` 表示同一个有效点同时落入目标 v 和目标 s 区间；FULL 使用同条件集合的 v/s conditional span overlap ≥ 95%，避免用互不对应的独立 min/max 误判。

## All FIT union

| target edge | old effective | new effective | old v range | new v range | old s range | new s range | old observed joint/status | new observed joint/status | old safety joint/status | new safety joint/status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top | 26663 | 26285 | [239.0, 2874.0] | [27.0, 2999.1] | [-0.164123, 0.194958] | [-0.193065, 0.212035] | 912 / PARTIAL | 2179 / FULL | 1273 / PARTIAL | 2670 / FULL |
| Bottom | 26663 | 26285 | [239.0, 2874.0] | [27.0, 2999.1] | [-0.164123, 0.194958] | [-0.193065, 0.212035] | 37 / PARTIAL | 444 / FULL | 61 / PARTIAL | 698 / PARTIAL |

## Points recoverable by replacing the old mask

恢复点定义为：new-mask effective point 的像素坐标落在 old mask 外；这比单纯比较 old/new 总点数更直接地隔离 mask 排除效应。

| target edge | observed recovered points | safety recovered points |
|---|---:|---:|
| Top | 1438 | 1639 |
| Bottom | 415 | 649 |

## Interpretation

- Old mask union decision = `PARTIAL`；new full-board mask union decision = `PARTIAL`。
- `OLD_FIT_EDGE_SUPPORT_AFTER_MASK_FIX = PARTIAL`。
- 结论：旧 mask 确实排除了部分边缘点（见 recovered counts），新 mask 将 Top 提升到 FULL、Bottom 提升到 observed FULL，但 Bottom safety 仍为 PARTIAL；因此旧 mask 是部分原因，不是唯一原因。

## Artifacts

- `old_fit_mask_support_comparison.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\old_fit_mask_support_comparison\old_fit_mask_support_comparison.csv`
- `old_fit_support_coverage.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\old_fit_mask_support_comparison\old_fit_support_coverage.png`
- `report.md`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\old_fit_mask_support_comparison\report.md`
