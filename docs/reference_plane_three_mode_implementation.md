# Ground-bias residual 三种 reference plane 模式实现

## 范围

本次修改只扩展 31 帧 residual diagnostics：

- 未生成或修改 ground-bias LUT；
- 未修改 reconstruction、相机内参、circular-cone laser model 或 Steger；
- 原有 `residual_frame_v_heatmap.png`、逐 `v` 统计和相关性输出仍以
  `self_fitted` 为 baseline，避免改变既有诊断口径；
- 新增三模式逐帧对照表和汇总表。

诊断实现位于：
`projects/daheng/outputs/ground_bias_v_diagnostics/generate_diagnostics.py`。

## `reference_plane_mode`

诊断实现支持以下三个枚举值，并在默认的 31 帧诊断入口中依次运行三者，
形成严格对照：

### `self_fitted`

每帧独立复用现有 ground-bias 稳健平面 residual 实现
`generate_ground_bias_compensation.frame_plane_residual`：

```text
Z_ref,i(Xg,Yg) = a_i Xg + b_i Yg + c_i
r_i = Zg - Z_ref,i(Xg,Yg)
```

平面参数采用 iterative MAD rejection 拟合；当前阈值为 3.5，最多 8 次迭代。
输出 residual 是沿 ground `Zg` 方向的 signed vertical residual，不是正交点面距离。

### `fixed_normal_per_frame_offset`

固定法向为 ground `+Zg`，禁止拟合 `a`、`b`，仅由该帧点估计高度：

```text
Z_ref,i = median(Zg_i)
r_i = Zg_i - median(Zg_i)
```

因此每帧的整体高度 offset 被移除，但该帧数据中的 tilt 不会被平面拟合吸收。

### `fixed_ground_plane`

所有帧共同使用冻结 ground 坐标系的零平面：

```text
Z_ref = Z0
r_i = Zg_i - Z0
```

`Z0` 不是经验设为 0。实现读取
`projects/daheng/outputs/0811/ground_extrinsics/camera_ground_extrinsics.yaml`，并同时要求：

1. 单位明确为 `mm`；
2. `coordinate_convention.zero_surface` 明确为 `checkerboard pattern surface`；
3. ground 原点明确位于 `checkerboard pattern plane`；
4. 存在有效的 `T_ground_from_camera` 和 camera-frame 参考平面系数；
5. 将 camera-frame 平面按
   `plane_ground = inv(T_ground_from_camera).T @ plane_camera` 变换后，数值上必须平行
   ground XY 且得到 `Zg=0`。

上述检查全部通过后才采用 `Z0 = 0.0 mm`；任一语义、单位、矩阵、平面或数值一致性
检查失败都会停止诊断，不会猜测 Z0。

该 `Z0` 表示 0811 外参定义的棋盘图案基准面，不应泛化解释为任意物理地面高度。
外参文件自身记录的独立 validation `Zg` RMSE 为约 0.0536 mm、最大绝对误差约
0.1573 mm，应作为 fixed-ground 结果的参考不确定度。

## 新增输出

`reference_plane_mode_comparison_per_frame.csv` 每帧包含：

- `frame_id`
- `self_fit_a`、`self_fit_b`、`self_fit_c`
- `self_fit_condition_number`
- `apparent_tilt_deg`
- `fixed_normal_offset_mm`
- `fixed_ground_offset_error_mm`
- `point_count`

其中：

```text
apparent_tilt_deg = degrees(atan(sqrt(a_i^2 + b_i^2)))
fixed_ground_offset_error_mm = median(Zg_i) - Z0
```

`apparent_tilt_deg` 只是激光重建点窄带自拟合得到的表观倾角，不能当作棋盘真实机械倾角。
本批数据的 self-fit design condition number 中位数约 3024、最大约 9875，也说明窄带
几何对二维平面参数的约束较弱。

`reference_plane_mode_comparison_summary.csv` 给出三种模式的 residual count、mean、
median、MAE、RMS、P95 absolute residual、逐帧 RMS 中位数、pair correlation 中位数和
共同 median profile 的 explained-energy fraction。结构化结果也写入
`diagnostics_summary.json`。

31 帧本次对照的主要数值为：

| reference mode | MAE / mm | RMS / mm | P95 abs / mm | median pair correlation | median profile explained energy |
|---|---:|---:|---:|---:|---:|
| `self_fitted` | 0.04700 | 0.06248 | 0.13033 | 0.4705 | 0.4839 |
| `fixed_normal_per_frame_offset` | 0.10064 | 0.15676 | 0.38610 | 0.2978 | 0.2896 |
| `fixed_ground_plane` | 0.12301 | 0.16417 | 0.33617 | 0.2978 | 0.3271 |

`fixed_normal_per_frame_offset` 与 `fixed_ground_plane` 的 pair correlation 相同，是因为
相关系数对每帧加减常数不敏感；二者的绝对 residual 指标不同。

## Synthetic tests

新增 `calibration/tests/test_reference_plane_diagnostics_modes.py`，验证：

- `self_fitted` 能恢复已知 `a,b,c` 并去除该平面；
- `fixed_normal_per_frame_offset` 精确执行 `Z-median(Z)`，且不会去除 tilt；
- `fixed_ground_plane` 精确执行 `Z-Z0`；
- `apparent_tilt_deg` 公式正确；
- 只有显式、坐标一致的棋盘零平面定义才能提供 Z0，含糊来源会报错停止。

与现有 compensation-axis 测试一并执行：

```text
python -m pytest calibration/tests/test_reference_plane_diagnostics_modes.py \
  calibration/tests/test_ground_bias_compensation_axis.py -q
```

结果：`13 passed`。

