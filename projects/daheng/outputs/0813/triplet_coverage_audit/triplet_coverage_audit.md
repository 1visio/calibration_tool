# Daheng 0811 Circular Cone 原始三联图几何覆盖审计

**NO_LASER_MODEL_FIT = TRUE**

本报告只读取已有三联图、manifest、0811 输出和正式内参；重新检测棋盘并独立计算 PnP ray–plane truth。没有调用任何 Cone 拟合/优化，不改写正式参数。

## 最终判定

**RAY_DEPTH_COVERAGE = PARTIAL**

## Provenance chain

- triplet frames：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\frames.csv`（实际路径由 frames.csv 提供，不按文件名猜测）。
- dataset manifest：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\dataset_manifest.yaml`；fit/validation 角色和 task_id 从 manifest/frames.csv 交叉核验。
- 0811 laser-model config：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`。
- 0811 stage run：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\laser_model\stage_run.yaml`；model 参数为 stage arguments 中的 `circular_cone`。
- 0811 model output：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\laser_model\laser_model.yaml`；本审计只读取 metrics，不读取或使用 Cone 参数进行计算。
- 0811 正式 runtime Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`；SHA-256（运行前后相同）=`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。
- laser centre UV：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\laser_model\calibration_points.csv` 的 `calibration_points.csv`；这是 0811 三联图 Steger 输出的实际中心记录。
- PnP intrinsics/distortion：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`；角点检测/PnP 复用正式 board-only 实现。

实际进入 0811 Cone 拟合的 frame：**001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018**（18 帧，train）。
未进入 0811 Cone 拟合的 frame：**019, 020, 021, 022, 023, 024**（6 帧，validation；只用于已有 0811 验证，不进入 Cone 参数估计）。

## 独立 PnP 与 ray-plane 方法

- 每个 frame 使用对应 chess image 的 11×8 内角点、20 mm 方格、正式 camera matrix/distortion；检测策略为 SB，失败时 classic + cornerSubPix；`SOLVEPNP_ITERATIVE` 后使用 `solvePnPRefineLM`（可用时）。
- camera-frame 棋盘平面写为 `n_c · X_c + d_c = 0`，其中 `n_c` 单位化并指向相机、`d_c > 0`。board center 是棋盘内角点网格几何中心的 PnP 位置；`board_tilt_deg = acos(-n_c,z)`，0° 表示正对相机。
- 对每个 recorded laser centre `(u,v)`，使用 `cv2.undistortPoints` 得到 `r=[x_n,y_n,1]`，再计算 `lambda_truth = -d_c/(n_c·r)`，`[Xc,Yc,Zc] = lambda_truth*r`。这里 lambda 是 z=1 非归一化 camera ray 的尺度，因此 `Zc=lambda`。
- 该 truth 不使用 Circular Cone 重建结果；仅依赖棋盘 PnP、相机内参/畸变和记录的 laser centre UV。

## Coverage summary

- 有效 PnP：24/24；PnP RMSE median/P95/max = 0.10194 / 0.15421 / 0.15902 px。
- 全部独立 ray-plane truth 点：21502；u=[1836.559, 2131.671] px，v=[241.998, 2731.978] px。
- 全部 lambda_truth：[631.549, 713.678] mm，span=82.129 mm；全部 Zc_truth 同范围（ray z=1）。
- laser UV 训练支持模型范围：v=[241.998, 2731.978] px。
- 全部 board-center Zc：[648.685, 712.278] mm，span=63.593 mm。
- board tilt：[2.522, 24.738]°，span=22.216°。
- 每帧 lambda span：[1.023, 65.532] mm。

| v region | points | unique frames | lambda span | Zc span | multi-frame 30px sub-bin fraction |
|---|---:|---:|---:|---:|---:|
| v=0–299 | 60 | 5 | 61.608 | 61.608 | 0.500 |
| v=2700–2999 | 9 | 1 | 0.083 | 0.083 | 0.000 |

`unique_frame_count` 和 30 px sub-bin 的多帧比例用于判断同一/相邻 v 是否有多姿态深度支持；没有用 Cone residual 或最终量块结果反推阈值。

## Per-frame geometry

| frame | split | PnP RMSE px | board center Zc mm | tilt deg | laser points | lambda span mm | u span px | v span px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 001 | fit | 0.15677 | 664.670 | 24.738 | 822 | 58.895 | 224.581 | 1420.901 |
| 002 | fit | 0.13619 | 664.636 | 21.017 | 890 | 1.695 | 8.273 | 1505.981 |
| 003 | fit | 0.13971 | 664.023 | 18.949 | 900 | 65.532 | 215.759 | 2106.011 |
| 004 | fit | 0.09289 | 648.685 | 3.113 | 900 | 3.184 | 30.450 | 1576.514 |
| 005 | fit | 0.09421 | 652.886 | 2.884 | 900 | 6.083 | 44.696 | 2255.978 |
| 006 | fit | 0.08897 | 702.960 | 2.522 | 900 | 1.203 | 18.696 | 1465.012 |
| 007 | fit | 0.10861 | 706.978 | 2.588 | 900 | 1.123 | 20.710 | 1453.024 |
| 008 | fit | 0.09277 | 705.255 | 4.585 | 900 | 6.708 | 39.624 | 1735.469 |
| 009 | fit | 0.07455 | 703.948 | 3.819 | 900 | 6.370 | 3.680 | 1454.021 |
| 010 | fit | 0.10450 | 680.471 | 7.374 | 900 | 1.560 | 22.002 | 1493.001 |
| 011 | fit | 0.10506 | 678.956 | 12.553 | 900 | 30.098 | 87.005 | 1484.894 |
| 012 | fit | 0.08731 | 705.164 | 4.507 | 900 | 9.298 | 42.060 | 1456.548 |
| 013 | fit | 0.08902 | 712.278 | 2.649 | 900 | 1.907 | 25.863 | 2042.020 |
| 014 | fit | 0.09280 | 709.875 | 2.652 | 900 | 1.246 | 22.299 | 1440.605 |
| 015 | fit | 0.10636 | 672.639 | 14.246 | 900 | 22.546 | 95.541 | 1489.563 |
| 016 | fit | 0.11331 | 668.362 | 17.942 | 900 | 62.115 | 242.474 | 2095.529 |
| 017 | fit | 0.11369 | 671.026 | 17.083 | 890 | 41.863 | 165.861 | 1461.015 |
| 018 | fit | 0.08857 | 709.153 | 3.034 | 900 | 5.913 | 5.147 | 2073.912 |
| 019 | validation | 0.11960 | 704.711 | 4.374 | 900 | 8.522 | 10.526 | 1457.967 |
| 020 | validation | 0.09938 | 709.417 | 2.637 | 900 | 1.470 | 19.509 | 1661.008 |
| 021 | validation | 0.12898 | 675.376 | 12.449 | 900 | 1.023 | 18.918 | 1513.968 |
| 022 | validation | 0.15902 | 659.358 | 24.222 | 900 | 5.385 | 41.128 | 2137.999 |
| 023 | validation | 0.09526 | 710.496 | 2.646 | 900 | 1.254 | 20.868 | 1437.999 |
| 024 | validation | 0.09931 | 671.016 | 2.594 | 900 | 1.734 | 25.141 | 1736.008 |

## Interpretation

A. v<300 与 v>2700 均有真值，但覆盖不对称：top 0–299 有 60 点、5 个 frame；bottom 2700–2999 只有 9 点、1 个 frame。
B. 同一/相邻 v 是否有多个 lambda：中心 300–2699 的 300px bins 基本由 10–24 个 frame 支持；bottom 的 30px sub-bin 多帧比例为 0.000，因此底部没有可用于区分 ray-depth gain 的多姿态深度交叉。
其余 region 的 `unique_frame_count`、30 px sub-bin 多帧比例和 lambda span 见 `triplet_ray_depth_support.csv`；这些数值直接来自独立 ray-plane truth。
C. 是否只是单一水平/单深度流形：看 per-frame board center Zc、tilt、每帧 lambda span 以及 v–lambda 图；多姿态 depth/tilt/position 共同改变 ray-plane 交点。
D. 是否有距离、倾角、位置变化：逐帧 geometry CSV 和 pose distribution 图给出原始范围；本审计不把 Cone 参数拟合成 coverage 证据。
E. 能否约束 ray-depth gain：若同一/相邻 v 由多 frame 支持且 lambda_truth 跨 frame 有明显 span，则数据在几何上不仅是零平面约束；这仍不等价于保证 6 参数 Cone 数值可辨识。

因此本轮对‘原始数据是否足以用于 multi-pose ray-depth refined Circular Cone’的结论为 **PARTIAL**：覆盖存在，但至少一个关键维度不足，需补采数据后再做 ray-depth refined Cone。

## Not performed

- 未重新提取激光中心、未重新拟合任何 laser model、未执行 Cone 优化、未使用 Circular Cone residual 选择 frame 或调整覆盖阈值。
- 未将任何结果写回 `calibration_daheng_0811` 或其它正式标定配置。
