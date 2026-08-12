# 地面点云逐点轴向补偿结果报告

## 1. 运行摘要

- 输入目录：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\ground_bias_v_experiment_0812\input`
- 建表帧数：25
- 评估帧数：6
- 评估方式：independent holdout
- 有效补偿列数：2967
- 补偿轴：`v`
- 图像行 v 范围：18.000–2999.000 px
- 趋势保留方式：`linear`
- 跨帧聚合方式：`median`
- MAD 离群剔除阈值：3.5
- 平滑窗口：1

## 2. 关键结果

| 指标 | 补偿前 | 补偿后 |
|---|---:|---:|
| 平均剖面去趋势峰谷值 P–V | 0.768653 mm | 0.925362 mm |
| 平均剖面去趋势 RMS | 0.114827 mm | 0.112966 mm |
| 单帧 P–V 中位数 | 0.732964 mm | 0.742262 mm |
| 单帧 RMS 中位数 | 0.145868 mm | 0.129240 mm |

平均剖面 P–V 降低比例：-20.39%

补偿表各列跨帧重复性 σ 中位数：0.052048 mm。

## 3. 输出文件

- `ground_bias_table.npy`：Python 使用的完整补偿表与元数据。
- `ground_bias_table.csv`：可用 Excel 查看和绘图的逐轴补偿表。
- `z_profile_before_after.png`：补偿前后平均 Z 剖面及去趋势残差。
- `pointcloud_before_after.png`：补偿前后点云三维对比图。
- `ground_cloud_before.npz`、`ground_cloud_after.npz`：补偿前后合并点云。
- `ground_cloud_before.ply`、`ground_cloud_after.ply`：可在 CloudCompare 中打开的 ASCII PLY（若未使用 `--no-ply`）。
- `compensation_metrics.json`：结构化运行参数与评估指标。

## 4. 补偿定义

默认 `--trend linear` 时，先对 31 帧平均地面剖面拟合线性趋势，再将残差作为逐列系统偏差：

```text
bias(v) = mean_Z(v) - linear_trend(v)
Z_corrected = Z_raw - bias(v)
```

这样会保留地面的整体高度和倾斜，只消除随图像列稳定重复的弯曲误差。

## 5. 结果解释注意事项

1. 当 `--validation-count 0` 时，补偿表与评估使用同一批帧，补偿后的平均剖面会非常平，这是建表数据上的自评估结果，不代表独立测量精度。
2. 建议额外运行一次 `--validation-count 6`，以前 25 帧建表、后 6 帧独立验证，观察补偿对未参与建表帧的效果。
3. 该补偿表绑定当前机械和光学状态。相机、激光器、基线、工作距离、焦距或光圈发生变化后必须重新采集标准平面并重建 LUT。
4. 补偿消除的是固定系统偏差的综合结果，不能单独证明误差来自激光 smile、承载板不平或外参残差。
5. 如果标准平面本身存在明显不平，补偿表会同时吸收这部分形貌。最终精度实验应使用平面度已知的基准板。
