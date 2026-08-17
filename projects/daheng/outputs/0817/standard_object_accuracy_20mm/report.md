# Frozen C0 / C1_4k 20 mm standard-object full-FOV acceptance

`FULL_FOV_ACCURACY = FAIL`
`C1_PRODUCTION = NO`

## Scope and frozen boundary

- 仅读取 `D:\Docs\linelaserscan\0704line-laser-3d-scanner\laser_measurement_tool\output_daheng_0811` 下明确指定的 8 个标准件目录；保留 8 个独立位置，按 height_points 的 v 中位数排序。
- 未读取 laser-plane Validation（019–024、037–040），未重新拟合 K/D 或 Cone，未训练新 correction。
- C0 = Frozen Circular Cone；C1 = Frozen `C1_4k`，即 `lambda_cone + F(s)`；PCA s、knots、penalty 和 region definition 均未修改。
- C0/C1 使用完全相同的既有 `laser_center.csv` 及对应 baseline/height `u,v` 子集；本轮没有重新提取 laser center。
- 工程真值固定为 **20.000 mm**，由用户提供；四个目录的 `result.json` 均未包含 nominal 字段。

## Frozen provenance

- C1 model: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`
- C1 model SHA-256: `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`
- C1 parameter SHA-256: `be7c316c91b54ac9b13a1ff2485a9ca2ceebf6644a57a91ffca98f858f726037`
- Frozen C0 provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`
- Frozen formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`

## Eight position results

误差定义：`(height point Zg - fitted local baseline Zg) - 20.000 mm`；Bias 为带符号均值，P95/Max 为绝对误差。

| position | directory | region | v median | model | n | Bias | MAE | RMSE | P95 | Max abs | <=0.2 mm |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| P01_v125.0 | frame_000974_measure | top | 125.0 | C0 | 78 | -0.115495 | 0.115495 | 0.117782 | 0.149983 | 0.156626 | True |
| P01_v125.0 | frame_000974_measure | top | 125.0 | C1_4k | 78 | -0.245415 | 0.245415 | 0.248485 | 0.300824 | 0.321391 | False |
| P02_v346.0 | frame_004021_measure | middle | 346.0 | C0 | 71 | 0.007791 | 0.012938 | 0.015604 | 0.027624 | 0.039342 | True |
| P02_v346.0 | frame_004021_measure | middle | 346.0 | C1_4k | 71 | -0.041888 | 0.041946 | 0.045788 | 0.073058 | 0.080130 | True |
| P03_v849.0 | frame_005772_measure | middle | 849.0 | C0 | 69 | -0.050435 | 0.050435 | 0.053000 | 0.077296 | 0.084991 | True |
| P03_v849.0 | frame_005772_measure | middle | 849.0 | C1_4k | 69 | -0.050746 | 0.050746 | 0.053262 | 0.076317 | 0.084120 | True |
| P04_v1467.5 | frame_007020_measure | middle | 1467.5 | C0 | 74 | -0.062587 | 0.062587 | 0.067953 | 0.116644 | 0.120739 | True |
| P04_v1467.5 | frame_007020_measure | middle | 1467.5 | C1_4k | 74 | -0.060500 | 0.060500 | 0.065783 | 0.113186 | 0.117122 | True |
| P05_v2006.5 | frame_008310_measure | middle | 2006.5 | C0 | 76 | -0.036839 | 0.036839 | 0.039871 | 0.061648 | 0.067217 | True |
| P05_v2006.5 | frame_008310_measure | middle | 2006.5 | C1_4k | 76 | -0.037849 | 0.037849 | 0.040781 | 0.062399 | 0.067892 | True |
| P06_v2516.5 | frame_009614_measure | middle | 2516.5 | C0 | 68 | -0.078656 | 0.078656 | 0.079590 | 0.095089 | 0.105500 | True |
| P06_v2516.5 | frame_009614_measure | middle | 2516.5 | C1_4k | 68 | -0.071961 | 0.071961 | 0.072979 | 0.089315 | 0.098054 | True |
| P07_v2593.0 | frame_011317_measure | middle | 2593.0 | C0 | 68 | -0.091933 | 0.091933 | 0.094912 | 0.124238 | 0.131315 | True |
| P07_v2593.0 | frame_011317_measure | middle | 2593.0 | C1_4k | 68 | -0.082098 | 0.082098 | 0.085034 | 0.111899 | 0.119764 | True |
| P08_v2905.0 | frame_012686_measure | bottom | 2905.0 | C0 | 61 | -0.070933 | 0.070933 | 0.075384 | 0.112949 | 0.127979 | True |
| P08_v2905.0 | frame_012686_measure | bottom | 2905.0 | C1_4k | 61 | -0.038524 | 0.040794 | 0.048167 | 0.085026 | 0.101317 | True |

## Region and full-field consistency

| model | global RMSE | global MAE | worst-position RMSE | worst-position P95 | position Bias range | position RMSE range | all positions <=0.2 mm |
|---|---:|---:|---:|---:|---:|---:|:---:|
| C0 | 0.074616 | 0.065071 | 0.117782 | 0.149983 | 0.123286 | 0.102178 | True |
| C1_4k | 0.108103 | 0.081310 | 0.248485 | 0.300824 | 0.207567 | 0.207704 | False |

| region | C0 RMSE | C1 RMSE | C0 P95 | C1 P95 | C1 RMSE change |
|---|---:|---:|---:|---:|---:|
| Top | 0.117782 | 0.248485 | 0.149983 | 0.300824 | 110.970% |
| Middle | 0.063485 | 0.062142 | 0.114258 | 0.103703 | -2.115% |
| Bottom | 0.075384 | 0.048167 | 0.112949 | 0.085026 | -36.104% |

- C1 global RMSE change vs C0: **44.879%**；global MAE change: **24.954%**。负值表示改善。
- C1 worst-position RMSE change: **110.970%**；position Bias range change: **68.362%**；position RMSE range change: **103.276%**。

## Decision notes

- C1 的 8 个位置中 7/8 个满足 0.2 mm Max abs error 目标；C0 为 8/8。
- C1 最坏位置为 P01_v125.0（frame_000974_measure），RMSE 0.248485 mm；C0 最坏位置为 P01_v125.0，RMSE 0.117782 mm。
- 最坏位置 RMSE、位置 Bias range 和位置 RMSE range 均未同时改善；Middle 区域 未出现明显退化。
- 8 个位置中 C1 的 frozen s domain 外推点数合计为 144；外推由冻结模型定义允许，未在本轮调整。

C1 的生产建议同时考虑所有位置的 Max abs error、最坏位置 RMSE、位置间 Bias/RMSE range 和 Middle 区域是否明显退化，而不是只看 pooled global RMSE。

## Artifacts

- `standard_object_accuracy.csv`、`regional_consistency.csv`：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\standard_object_accuracy_20mm`
