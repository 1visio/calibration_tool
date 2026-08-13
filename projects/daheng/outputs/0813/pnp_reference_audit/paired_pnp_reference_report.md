# Paired PnP reference audit

## 结论

- 严格配对核验：**通过**。
- PnP：**13/13 成功 (100.0%)**。
- 本审计只读取棋盘图建立 PnP 真值；laser 图只用于配对、清单与 SHA-256 核验。
- 未读取激光像素建立参考平面，未生成 `b(v)`，未执行 compensation。

## 关键统计

| split | PnP success | reprojection RMSE median / P95 / max (px) | tilt median / P95 / max (deg) | board center height median / min / max (mm) | height range / std / P95 abs dev (mm) |
|---|---:|---:|---:|---:|---:|
| fit | 10/10 | 0.084708 / 0.091591 / 0.092166 | 0.029609 / 0.045372 / 0.045389 | -0.013019 / -0.046993 / 0.014842 | 0.061835 / 0.020025 / 0.031921 |
| validation | 3/3 | 0.085094 / 0.087212 / 0.087447 | 0.044290 / 0.054939 / 0.056122 | -0.023123 / -0.047210 / -0.017493 | 0.029717 / 0.012888 / 0.022242 |
| overall | 13/13 | 0.085094 / 0.091400 / 0.092166 | 0.034781 / 0.049682 / 0.056122 | -0.017493 / -0.047210 / 0.014842 | 0.062052 / 0.019709 / 0.030764 |

`height variation` 在这里同时报告 range、population std 和相对中位数的 P95 绝对偏差；`board_height_at_center_mm` 是棋盘内角点网格几何中心经 PnP 后转换到 ground frame 的 `Zg`。

## 逐帧结果

| frame | split | pair | corners | PnP | RMSE px | tilt deg | center Zg mm | method |
|---:|---|---|---:|---|---:|---:|---:|---|
| 001 | fit | OK | 88 | OK | 0.080726 | 0.023814 | -0.026826 | SB |
| 002 | fit | OK | 88 | OK | 0.087381 | 0.031636 | -0.042432 | SB |
| 003 | fit | OK | 88 | OK | 0.075027 | 0.021591 | -0.046993 | SB |
| 004 | fit | OK | 88 | OK | 0.092166 | 0.027581 | -0.022071 | classic+cornerSubPix |
| 005 | fit | OK | 88 | OK | 0.081623 | 0.023155 | 0.007093 | SB |
| 006 | fit | OK | 88 | OK | 0.082762 | 0.038543 | -0.016856 | classic+cornerSubPix |
| 007 | fit | OK | 88 | OK | 0.090630 | 0.022413 | 0.014842 | classic+cornerSubPix |
| 008 | fit | OK | 88 | OK | 0.077315 | 0.045351 | -0.005596 | classic+cornerSubPix |
| 009 | fit | OK | 88 | OK | 0.086654 | 0.034781 | -0.009183 | classic+cornerSubPix |
| 010 | fit | OK | 88 | OK | 0.090889 | 0.045389 | 0.008465 | SB |
| 011 | validation | OK | 88 | OK | 0.087447 | 0.042090 | -0.047210 | SB |
| 012 | validation | OK | 88 | OK | 0.072619 | 0.056122 | -0.017493 | SB |
| 013 | validation | OK | 88 | OK | 0.085094 | 0.044290 | -0.023123 | SB |

## 失败与异常

无。13 组严格配对，13 帧 PnP 全部成功。

## 配对证据

每个编号要求 chess/laser 文件各一份，并要求 `frames.csv` 中相同 `pose_id` 下 `role=chess` 与 `role=laser` 各恰好一行、记录的相对文件名一致、两份文件 SHA-256 与采集记录一致，且 laser 的采集时间不早于 chess。详细布尔值与 capture gap 见 CSV。

## 方法与坐标定义

- 数据：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\extrinsics0813`
- 冻结内参：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`
- 冻结 ground 变换：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\ground_extrinsics\camera_ground_extrinsics.yaml`
- 棋盘：11 × 8 内角点，20.0 mm 方格。
- 复用实现：`calibration/src/calibrate_ground_extrinsics_board_only.py` 的 `load_intrinsics`、`chessboard_object_points`、`detect_chessboard`；检测策略为 SB，失败时 classic + cornerSubPix；PnP 为 `SOLVEPNP_ITERATIVE`，可用时再 `solvePnPRefineLM`。
- camera-frame 棋盘平面为 `n_c · X_c + d_c = 0`；通过冻结的 `T_ground_from_camera` 作平面协向量变换 `pi_g = T^{-T} pi_c`，归一化并令 `n_g · +Zg >= 0`。CSV 的 `plane_nx..plane_d` 均为 ground-frame 系数，单位法向，`plane_d` 单位 mm。
- `tilt_deg = acos(clamp(n_board_ground · [0,0,1], -1, 1))`。

## 输出

- `paired_pnp_reference_audit.csv`
- `paired_pnp_reference_report.md`
- `pnp_pose_summary.png`
