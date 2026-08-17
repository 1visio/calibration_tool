# Frozen C0 / C1_4k standard-object full-FOV acceptance

`FULL_FOV_ACCURACY = PASS`
`C1_PRODUCTION = YES`

## Scope and frozen boundary

- 仅读取四个标准件目录：`D:\Docs\linelaserscan\0704line-laser-3d-scanner\laser_measurement_tool\output_daheng_0811` 下的 `frame_061303_measure`、`frame_065292_measure`、`frame_063995_measure`、`frame_062878_measure`。
- 未读取 laser-plane Validation（019–024、037–040），未重新拟合 K/D 或 Cone，未训练新 correction。
- C0 = Frozen Circular Cone；C1 = Frozen `C1_4k`，即 `lambda_cone + F(s)`；PCA s、knots、penalty 和 region definition 均未修改。
- C0/C1 对每个位置使用完全相同的既有 `laser_center.csv` 及其中已对应的 baseline/height `u,v` 子集；本轮没有重新提取 laser center。
- 工程真值按该组约 50 mm 量块的固定标称值 **50.000 mm** 计算。四个目录的 `result.json` 未含 nominal 字段，因此此值是本报告必须显式记录的外部标准件规格假设。

## Frozen provenance

- C1 model: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`
- C1 model SHA-256: `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`
- C1 parameter SHA-256: `be7c316c91b54ac9b13a1ff2485a9ca2ceebf6644a57a91ffca98f858f726037`
- Frozen C0 provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`
- Frozen formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`

## Position-level accuracy

误差定义：`(height point Zg - fitted local baseline Zg) - nominal height`；Bias 为带符号均值，P95/Max 为绝对误差。

| position | v median | model | n | Bias (mm) | MAE (mm) | RMSE (mm) | P95 (mm) | Max abs (mm) | within 0.2 mm |
|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| Top | 429.5 | C0 | 64 | 0.010972 | 0.012416 | 0.016030 | 0.029202 | 0.034315 | True |
| Top | 429.5 | C1_4k | 64 | -0.041098 | 0.041098 | 0.042846 | 0.061554 | 0.065448 | True |
| Middle_Upper | 1272.0 | C0 | 77 | -0.125476 | 0.125476 | 0.127847 | 0.158910 | 0.164759 | True |
| Middle_Upper | 1272.0 | C1_4k | 77 | -0.119740 | 0.119740 | 0.122271 | 0.153525 | 0.160075 | True |
| Middle_Lower | 2084.0 | C0 | 62 | -0.131964 | 0.131964 | 0.132393 | 0.149160 | 0.157695 | True |
| Middle_Lower | 2084.0 | C1_4k | 62 | -0.131674 | 0.131674 | 0.132096 | 0.148616 | 0.157135 | True |
| Bottom | 2846.5 | C0 | 74 | -0.196788 | 0.196788 | 0.197683 | 0.222736 | 0.228256 | False |
| Bottom | 2846.5 | C1_4k | 74 | -0.164161 | 0.164161 | 0.165281 | 0.193179 | 0.194456 | True |

## Full-field consistency comparison

| model | global RMSE | global MAE | worst-position RMSE | worst-position P95 | position bias range | position RMSE range | all positions <= 0.2 mm |
|---|---:|---:|---:|---:|---:|---:|:---:|
| C0 | 0.137716 | 0.119857 | 0.197683 | 0.222736 | 0.207760 | 0.181653 | False |
| C1_4k | 0.125632 | 0.116108 | 0.165281 | 0.193179 | 0.123063 | 0.122435 | True |

- C1 global RMSE change vs C0: **-8.774%**; global MAE change: **-3.128%**。负值表示改善。
- C1 worst-position RMSE change: **-16.391%**；position bias range change: **-40.767%**；position RMSE range change: **-32.600%**。

## Decision notes

- C1 全部四个位置 Max abs error 均 满足 0.2 mm 目标；C0 的结果为 至少一个位置超标。
- C1 最坏位置 RMSE 改善，位置 Bias range 收窄，位置 RMSE range 收窄。
- Middle_Upper/Middle_Lower 相对 C0 的 RMSE 未出现明显退化（阈值为 2% 或 0.005 mm 的较宽者）。
- Top 位置 C0→C1 RMSE：0.016030 → 0.042846 mm；这是 退化但仍在 0.2 mm 目标内，不能描述为每个位置都改善。
- Bottom 位置 C0→C1 RMSE：0.197683 → 0.165281 mm；其 Max abs error 为 0.194456 mm。

C1 的生产建议以全位置 Max abs error <= 0.2 mm、最坏位置 RMSE、位置间 Bias/RMSE range 和 Middle 区域不发生明显退化共同判断，而不是只看 pooled global RMSE。

## Artifacts

- `standard_object_accuracy.csv`、`regional_consistency.csv`：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\standard_object_accuracy`
