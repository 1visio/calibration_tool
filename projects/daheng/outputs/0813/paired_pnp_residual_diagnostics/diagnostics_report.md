# Paired PnP laser residual diagnostics

## 结论

**ONE_DIMENSIONAL_BV = PARTIAL**

本结论只评价一维 `v` 相关残差是否具有重复性；没有构建正式 LUT，也没有应用 compensation。
每帧 reference plane 唯一来自同编号 chess 的独立 PnP，禁止且未执行激光点自拟合平面。

## 关键统计

- fit frame pair correlation median（overlap ≥ 100）：0.515258 （45 对）。
- fit median `b(v)` explained residual energy：0.564570。
- validation 对冻结 fit median profile 的 observed explained energy：0.364294；validation 未参与 profile 计算。
- fit residual std(v) median（sample count ≥ 5）：0.074664 mm。
- fit sign consistency median（sample count ≥ 5）：0.857143。
- residual bin sample count：fit=17564，validation=6049。

## 判定规则

`SUPPORTED` 要求 fit median pair correlation ≥0.7、fit explained energy ≥0.7、validation observed explained energy ≥0.5、fit 至少半数帧覆盖 ≥50% 图像行、且对应行的 median sign consistency ≥0.8。若未全部满足，但 fit correlation 与 explained energy 均 ≥0.3、validation explained energy 为正且覆盖 ≥25%，判为 `PARTIAL`；否则 `NOT_SUPPORTED`。
- fit median pair correlation >= 0.7: FAIL
- fit explained energy >= 0.7: FAIL
- validation explained energy using fit profile >= 0.5: FAIL
- fit >=half-frame coverage >= 50% of image rows: PASS
- fit median sign consistency >= 0.8: PASS

当前数据存在明确的一维共同成分，但强支持门槛未通过：fit profile 只解释约56% energy，validation 只保留约36%；且 fit top/bottom 的 median std(v) 明显高于 middle，帧间整体偏置也在变化。因此 `PARTIAL` 表示可继续研究的一维成分，不表示足以发布 LUT。

## 分 split 统计

| split | frames | residual samples | mean (mm) | median (mm) | std (mm) | RMSE (mm) | P95 abs (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit | 10 | 17564 | -0.068448 | -0.089026 | 0.174267 | 0.187223 | 0.373741 |
| validation | 3 | 6049 | -0.114072 | -0.122788 | 0.147431 | 0.186399 | 0.347162 |

## Top / middle / bottom

以下按 3000 px 图像高度固定划分 top 10%、middle 80%、bottom 10%。`std(v)`、sign consistency 与 diagnostic bias 只汇总至少有2帧样本的行。

| split | region | residual bins | row count ≥2 | median samples/row | median std(v) mm | median sign consistency | median abs bias mm |
|---|---|---:|---:|---:|---:|---:|---:|
| fit | top_0_299 | 1605 | 300 | 5.00 | 0.253295 | 1.000000 | 0.182442 |
| fit | middle_300_2699 | 13979 | 2397 | 6.00 | 0.068126 | 1.000000 | 0.093650 |
| fit | bottom_2700_2999 | 1980 | 300 | 6.00 | 0.192468 | 0.833333 | 0.306180 |
| validation | top_0_299 | 684 | 300 | 2.00 | 0.116568 | 0.666667 | 0.043271 |
| validation | middle_300_2699 | 4734 | 1950 | 2.00 | 0.026830 | 1.000000 | 0.124411 |
| validation | bottom_2700_2999 | 631 | 195 | 3.00 | 0.038125 | 1.000000 | 0.252614 |

## 逐帧 residual

| frame | split | extracted | reconstructed | mean mm | std mm | RMSE mm | P95 abs mm |
|---:|---|---:|---:|---:|---:|---:|---:|
| 001 | fit | 1705 | 1702 | 0.024202 | 0.211553 | 0.212871 | 0.639746 |
| 002 | fit | 1799 | 1790 | 0.013135 | 0.172454 | 0.172906 | 0.280124 |
| 003 | fit | 1609 | 1530 | 0.014112 | 0.105485 | 0.106391 | 0.280456 |
| 004 | fit | 1728 | 1579 | -0.068669 | 0.203349 | 0.214570 | 0.529271 |
| 005 | fit | 2872 | 2854 | -0.066743 | 0.093526 | 0.114886 | 0.198669 |
| 006 | fit | 1893 | 1581 | -0.061853 | 0.155924 | 0.167698 | 0.344175 |
| 007 | fit | 1968 | 1374 | -0.136189 | 0.131228 | 0.189092 | 0.345729 |
| 008 | fit | 1871 | 1847 | -0.127370 | 0.177740 | 0.218627 | 0.432751 |
| 009 | fit | 1868 | 1815 | -0.165078 | 0.173383 | 0.239366 | 0.410311 |
| 010 | fit | 1632 | 1492 | -0.113817 | 0.187010 | 0.218869 | 0.452862 |
| 011 | validation | 1694 | 1673 | -0.108232 | 0.116719 | 0.159152 | 0.268078 |
| 012 | validation | 2626 | 2592 | -0.104434 | 0.181411 | 0.209293 | 0.448491 |
| 013 | validation | 1799 | 1784 | -0.133551 | 0.112174 | 0.174390 | 0.293187 |

## 冻结链路与 residual 定义

- Measurement config：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\measure_tool_daheng_0811.yaml`
- PnP audit：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\pnp_reference_audit\paired_pnp_reference_audit.csv`
- Laser model：`circular_cone`。
- Steger：`{"sigma": 1.5, "threshold": 30.0, "deriv_thresh": 0.5, "roi_margin": 48, "roi_max_height": 512, "scan_axis": "row"}`。
- Reconstruction：`{"parallel_epsilon": 1e-09, "quadratic_epsilon": 1e-12, "min_camera_depth_mm": 630.0, "max_camera_depth_mm": 715.0, "model_range_margin_mm": 2.0}`。
- Ground compensation 配置为 `null`，本轮没有 compensation。
- 对 ground-frame PnP 平面 `nx Xg + ny Yg + nz Zg + d = 0`，逐点计算 `Z_plane=-(nx Xg+ny Yg+d)/nz`，再计算 `residual_z=Zg-Z_plane`。这是 ground Z 方向的有符号误差，不是正交点面距离。
- `residual_v_statistics.csv` 中的 fit median 是未平滑、未插值的诊断统计，不得直接当作正式 LUT。
