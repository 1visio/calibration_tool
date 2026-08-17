# Frozen C1_4k independent Validation

`C1_INDEPENDENT_VALIDATION = PARTIAL`

是否值得进入实际标准件/高度恢复全视场验收：**CONDITIONAL**。该结论只针对本次 frozen Validation，不代表跳过后续验收。

## Frozen boundary

- Frozen model：`C1_4k`，4 interior knots、8 basis、cubic、penalty = **0.1**。
- frozen_c1_model.json SHA-256：`fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`；parameter SHA-256：`be7c316c91b54ac9b13a1ff2485a9ca2ceebf6644a57a91ffca98f858f726037`。
- FIT：30 frames（001–018、025–036），26,663 rays；frame-balanced total weight = 1 per frame。
- Frozen PCA s domain：[-0.16412256362, 0.194958484593]；s/t、knots、penalty、region definition 均未调整。
- C0 = Frozen Circular Cone；C1 = `lambda_cone + F(s)`；truth = checkerboard PnP plane ray intersection。
- 未重新拟合 K/D 或 Cone；不做 C2/C3；最终成功的完整 Validation 评价在 frozen JSON 写入后执行一次，之后不再重评。
- 执行记录：此前两次代码级路径/字段错误在写出 metrics/verdict 前中止，未改变 frozen 参数；本报告只采用随后成功的完整评价结果。

## Validation scope

- Validation frames：019, 020, 021, 022, 023, 024, 037, 038, 039, 040；成功评价 pass 处理 triplet image：**30**（每帧 chess/nolaser/laser 一次）。
- `validation_per_frame.csv` 共 **10** 行；raw image manifest/hash 均通过。
- Top/Middle/Bottom 固定为 `v∈[0,300) / [300,2700) / [2700,3000)`；没有按 Validation 结果改变分区。
- Region support：Top 有效 frame/point = **3/25**；Middle = **10/8960**；Bottom = **2/15**。Top/Bottom edge evidence 偏 sparse。

## Aggregate comparison (frame-equal)

| region | C0 RMSE | C1 RMSE | RMSE improvement | C0 P95 | C1 P95 | P95 improvement | C0 bias range | C1 bias range | frame RMSE improve fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 0.0873945 | 0.0850092 | 2.729% | 0.162514 | 0.157259 | 3.234% | 0.21519 | 0.225888 | 0.900 |
| top | 0.0994503 | 0.0805619 | 18.993% | 0.144581 | 0.165157 | -14.232% | 0.0537738 | 0.0552257 | 0.667 |
| middle | 0.0872687 | 0.0849808 | 2.622% | 0.162347 | 0.157174 | 3.186% | 0.218784 | 0.226525 | 0.900 |
| bottom | 0.126006 | 0.110881 | 12.003% | 0.205319 | 0.186545 | 9.144% | 0.0457747 | 0.0435383 | 1.000 |
| worst_region | 0.0931921 | 0.0896009 | 3.854% | 0.156157 | 0.143185 | 8.307% | 0.269938 | 0.250198 | 0.800 |

- Global edge/middle ratio median：C0 **1.40683** → C1 **1.23555**；改善 **12.175%**。
- Global frame-equal RMSE improvement median：**4.765%**；P95 improvement median：**10.988%**。

## Top / Bottom / Middle decision

- Top：RMSE improvement **18.993%**，P95 improvement **-14.232%**，逐 frame RMSE 正改善比例 **0.667**。
- Bottom：RMSE improvement **12.003%**，P95 improvement **9.144%**，逐 frame RMSE 正改善比例 **1.000**。
- Top/Bottom 同时稳健改善：**NO**；Top 的 RMSE/P95 = **18.993%/-14.232%**，Bottom = **12.003%/9.144%**。
- Middle：RMSE improvement **2.622%**，P95 improvement **3.186%**；判定阈值为不恶化超过 **2.0%**。

## Fixed verdict gates

- PASS requires global RMSE improvement ≥ 5.0% and non-worse P95; both Top and Bottom RMSE/P95 positive with ≥0.50 frame positive fraction; Middle degradation ≤2.0%; worst-region degradation ≤2.0%; edge/middle ratio degradation ≤2.0%; bias-range degradation ≤5.0%。
- Gate details：global_good=False; top_good=False; bottom_good=True; middle_ok=True; worst_good=True; edge_middle_ok=True; bias_range_ok=True。
- `C1_INDEPENDENT_VALIDATION = PARTIAL`：这是一次 frozen Validation 评价结果，不能据此重新调整模型。

## Provenance

- points：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\spatial_residual_observability\fit_ray_residual_points.csv`，SHA-256 `ea68251e05e1d472db7e25bb2090b2094f2e813f04dd5475fea1c06e4af01f8f`。
- PCA summary：`a01dca9079cf8985c4a7d3e97235a6e4d6249751ef633d3d2928ec3ab6a51c83`。
- Frozen provenance：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\circular_vs_elliptical_cone\provenance.json`，SHA-256 `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`。
- Formal Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`，SHA-256 `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。
