# Task 6G — Camera calibration pose observability / coverage audit

`CAMERA_COVERAGE_WEAKNESS = B. TILT`

本审计只读取正式 camera FIT chess 001–018、激光 FIT 001–018/025–036，以及 6E/6F 派生 CSV。Validation 未打开；正式 K/D、畸变模型、PnP flags、Cone 和 Steger 均未修改。

## 关键结论

Full-18 fixed-coverage reference candidate-global P95 = 0.122387 mm。6E LOO 的 truth influence 按 propagated P95 排序，最高影响 frame 为：002, 001, 010, 003, 017。

受控 pair ablation 显示，删去两帧后的敏感性主要由 depth/tilt/apparent-size 的耦合覆盖决定；sensor-position 不是唯一主导项。

## LOO 高影响帧

| rank | omitted frame | truth P95 median (mm) | truth P95 max (mm) | depth (mm) | tilt (deg) | apparent area |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 002 | 0.399968 | 0.426583 | 664.715 | 21.0096 | 0.29786 |
| 2 | 001 | 0.32637 | 0.34845 | 664.769 | 24.7658 | 0.270498 |
| 3 | 010 | 0.236942 | 0.247793 | 680.471 | 7.3762 | 0.272619 |
| 4 | 003 | 0.190167 | 0.201621 | 664.022 | 18.9486 | 0.297852 |
| 5 | 017 | 0.0894335 | 0.0992639 | 671.057 | 17.0879 | 0.30573 |

## Coverage observability

| dimension | range | occupied bins | LOO rho | pair P95 median (mm) |
|---|---:|---:|---:|---:|
| depth | 63.601 | 4 | -0.469556 | 0.0569996 |
| tilt | 22.2323 | 5 | 0.72549 | 0.144254 |
| apparent_size | 0.258762 | 3 | 0.240454 | 0.0242644 |
| sensor_position | 0.197864 | 5 | -0.0856553 | 0.0251494 |

## Minimal stable pose set

以 full-18 的 1.25×（0.152984 mm）作为‘接近’阈值；候选来自 6F 的 coverage-matched subsets，不重新选择 Cone 参数。
- 12/18: 0/8 个候选通过；最佳 P95=0.172292 mm。
- 14/18: 0/8 个候选通过；最佳 P95=0.283161 mm。
- 16/18: 5/8 个候选通过；最佳 P95=0.0294344 mm。
- 18/18: 0/0 个候选通过；最佳 P95=nan mm。

## 建议

当前 18 帧不需要立即全部重采；若需要降低 coverage-loss tail，最值得补充的是：

1. 近/远 depth 两端且不伴随同方向 tilt 的姿态；
2. 高 tilt（最好正交方向各一组）姿态；
3. 大/小 apparent board size 与 sensor edge 的组合姿态；
4. 将 depth、tilt、size 解耦的 matched control 姿态。

## 输出

- `camera_pose_geometry.csv`
- `frame_influence_ranking.csv`
- `targeted_ablation.csv`
- `coverage_observability.csv`
- `minimal_stable_pose_sets.csv`
