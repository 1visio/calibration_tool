# Laser-plane mask fix coverage audit for FIT 049–054

`EDGE_SUPPORT_AFTER_MASK_FIX = INSUFFICIENT`

## Scope and mask definitions

- 只打开 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\fit` 下 FIT `049–054` 的 18 张图；没有打开 `validation/055–060` 或旧 Validation。
- Before：当前 `board_inner_mask(inner-corner hull, margin_px=-2)`。
- After：PnP 投影的完整 12×9 方格物理边界 `X=[-20,220] mm`、`Y=[-20,160] mm`，不做任何像素腐蚀。该 polygon 不扩展到棋盘印刷区之外的白边或铝框。
- 两种 mask 均使用相同 Steger、`vertical` 每 row 单点、continuity、900 点上限，并使用相同 Frozen Circular Cone 有效性筛选；`s` 使用 Frozen C1 PCA 定义。
- 未重新拟合或修改 K/D、Cone、C1；C1 仅用于固定 PCA `s` 坐标和 frozen domain 对照。
- Frozen C1 artifact：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`；SHA-256 = `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`。

安全目标域（沿用上一轮 coverage plan，仅用于覆盖判定）：

| edge | safe v range (px) | safe s range |
|---|---:|---:|
| Top | [30, 520] | [-0.191, -0.127] |
| Bottom | [2760, 2990] | [0.181, 0.209] |

## Top/Bottom effective support

| edge | before effective / Steger | after effective / Steger | before v range | after v range | before s range | after s range | after safe v/s |
|---|---:|---:|---:|---:|---:|---:|---|
| Top | 2273 / 2541 | 2370 / 2700 | [674.0, 2760.0] | [485.9, 2982.0] | [-0.104750, 0.179494] | [-0.130431, 0.209839] | false |
| Bottom | 2664 / 2664 | 2700 / 2700 | [181.0, 1762.0] | [0.0, 2001.0] | [-0.171835, 0.043441] | [-0.196547, 0.075983] | false |

## Per-pose effective counts

| frame | edge | before effective | after effective | delta | before safe v/s | after safe v/s |
|---:|---|---:|---:|---:|---|---|
| 049 | Top | 741 | 900 | 159 | false | false |
| 050 | Top | 900 | 900 | 0 | false | false |
| 051 | Top | 632 | 570 | -62 | false | false |
| 052 | Bottom | 900 | 900 | 0 | false | false |
| 053 | Bottom | 864 | 900 | 36 | false | false |
| 054 | Bottom | 900 | 900 | 0 | false | false |

## Decision

- `EDGE_SUPPORT_AFTER_MASK_FIX = INSUFFICIENT`。判定以 Top/Bottom edge union 的 effective points 同时覆盖 v 和 s safe target 为准。
- After Top safe target covered = `false`；After Bottom safe target covered = `false`。
- 若 after 仍为 INSUFFICIENT，说明仅扩大棋盘有效 mask 不能补足真实 operational Top/Bottom 位置；问题还包括采集 pose/domain 覆盖不足。

## Artifacts

- `mask_before_after.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\mask_fix_coverage_049_054\mask_before_after.csv`
- `new_fit_support_coverage.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\mask_fix_coverage_049_054\new_fit_support_coverage.png`
- `report.md`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\mask_fix_coverage_049_054\report.md`
