# New FIT 049–054 coverage audit

`NEW_FIT_SUPPORT_COVERAGE = INSUFFICIENT`
`QUALITY_GATE = WARNING`

## Scope

- 只打开：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\fit` 下的 FIT `049–054`，共 6 个 pose、18 张图；Top/Bottom 分组来自用户指令：049–051 = Top，052–054 = Bottom。
- 本轮没有打开 `validation/055–060`，也没有读取旧 Validation 019–024、037–040；`validation_read = false`。
- 使用 Frozen M0 K/D、Frozen Circular Cone 和既有 Frozen C1_4k PCA `s`；没有重新拟合 K/D、Cone、PCA 或 C1。
- Frozen C1：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`；artifact SHA-256 = `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`；parameter SHA-256 = `be7c316c91b54ac9b13a1ff2485a9ca2ceebf6644a57a91ffca98f858f726037`。
- 真实 operational domain 取上一轮 20 mm / 50 mm height-position union；本轮只复用其数值，不重新打开标准件图像。

## Frozen and target domains

- 旧 FIT support：v = `[238.984, 2874.006] px`；s = `[-0.164123, 0.194958]`。
- Frozen C1 s domain：`[-0.164122564, 0.194958485]`。

| edge | observed v | observed s | safe v (+margin) | safe s (+margin) | new FIT v | new FIT s | safe v/s covered |
|---|---:|---:|---:|---:|---:|---:|---|
| Top | [86.0, 467.0] | [-0.185001, -0.132862] | [30.0, 520.0] | [-0.191, -0.127] | [674.0, 2760.0] | [-0.104750, 0.179494] | false |
| Bottom | [2810.0, 2938.0] | [0.186364, 0.203711] | [2760.0, 2990.0] | [0.181, 0.209] | [181.0, 1762.0] | [-0.171835, 0.043441] | false |

## Per-pose support

| frame | edge | points | v range / px | s range | old support v extrapolation | old support s extrapolation |
|---:|---|---:|---:|---:|---:|---:|
| 049 | Top | 741 | [1280.0, 2751.9] | [-0.022081, 0.178475] | 0 (0.0%) | 0 (0.0%) |
| 050 | Top | 900 | [1227.0, 2760.0] | [-0.029390, 0.179494] | 0 (0.0%) | 0 (0.0%) |
| 051 | Top | 632 | [674.0, 2114.9] | [-0.104750, 0.091701] | 0 (0.0%) | 0 (0.0%) |
| 052 | Bottom | 900 | [181.0, 1749.0] | [-0.171835, 0.041742] | 31 (3.4%) | 31 (3.4%) |
| 053 | Bottom | 864 | [247.0, 1762.0] | [-0.162912, 0.043441] | 0 (0.0%) | 0 (0.0%) |
| 054 | Bottom | 900 | [242.0, 1742.1] | [-0.163543, 0.040615] | 0 (0.0%) | 0 (0.0%) |

## Acquisition quality

- 仅读取 `frames.csv` 的前 18 个 FIT 数据行（049–054）；读取行数 = 18，Validation 行读取 = `false`。
- 质量警告行数：12 / 18；因此 `QUALITY_GATE = WARNING`。
- frame 049: `laser:dynamic_range_low; nolaser:image_too_dark;dynamic_range_low`
- frame 050: `laser:dynamic_range_low; nolaser:image_too_dark;dynamic_range_low`
- frame 051: `laser:dynamic_range_low; nolaser:image_too_dark;dynamic_range_low`
- frame 052: `laser:dynamic_range_low; nolaser:image_too_dark;dynamic_range_low`
- frame 053: `laser:dynamic_range_low; nolaser:image_too_dark;dynamic_range_low`
- frame 054: `laser:dynamic_range_low; nolaser:image_too_dark;dynamic_range_low`

## Processing sanity

| frame | valid points | PnP RMSE / px | laser intensity mean / DN | stripe contrast mean / DN |
|---:|---:|---:|---:|---:|
| 049 | 741 | 0.1580 | 132.2821 | 138.4453 |
| 050 | 900 | 0.1334 | 65.7000 | 67.2222 |
| 051 | 632 | 0.1656 | 111.8149 | 116.0269 |
| 052 | 900 | 0.1481 | 68.1711 | 71.7767 |
| 053 | 864 | 0.1322 | 112.9271 | 115.6725 |
| 054 | 900 | 0.0988 | 95.0422 | 97.0400 |

## Decision

- `NEW_FIT_SUPPORT_COVERAGE = INSUFFICIENT`：至少一个 edge 未覆盖 observed operational domain，或新 FIT 没有足够有效点。
- Top safe target covered = `false`；Bottom safe target covered = `false`。
- 几何 support 若为 SUFFICIENT，表示新 FIT 的 v/s 点集范围已覆盖 operational domain 及安全余量；这不等于新图像已经通过激光信号质量验收。
- `QUALITY_GATE = WARNING`：新 FIT metadata 存在 dynamic_range_low / image_too_dark 警告；几何覆盖结果可用作 support audit，但不应视为采集质量通过。
- 下一步：只有在采集质量问题得到确认/修正后，才建议将 049–054 作为新的 FIT 输入重新冻结 C1；本轮没有执行该拟合。

## Artifacts

- `new_fit_support_coverage.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\new_fit_support_coverage_049_054\new_fit_support_coverage.csv`
- `new_fit_support_points.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\new_fit_support_coverage_049_054\new_fit_support_points.csv`
- `new_fit_support_summary.json`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\new_fit_support_coverage_049_054\new_fit_support_summary.json`
- `new_fit_support_coverage.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\new_fit_support_coverage_049_054\new_fit_support_coverage.png`
