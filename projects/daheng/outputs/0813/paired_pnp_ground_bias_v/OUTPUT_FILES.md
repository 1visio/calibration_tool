# OUTPUT_FILES

- `ground_bias_table.csv`：3000行实验表，含bias、BUILD样本数、工作区和support标志；unsupported bias留空。
- `ground_bias_table.npy`：同一表的结构化数组；unsupported bias为NaN且`support=false`。
- `compensation_metrics.json`：完整配置、隔离声明、逐帧/聚合指标、局部block和判定。
- `holdout_residual_before_after.png`：三个独立holdout逐帧before/after曲线。
- `holdout_residual_frame_v_heatmap_before_after.png`：独立holdout before/after热图。
- `holdout_metrics_per_frame.csv`：三个holdout逐帧Bias/MAE/RMSE/P95/P-V/std/sign mixing。
- `compensation_support.png`：BUILD支持数、support mask与未平滑experimental b(v)。
- `compensation_report.md`：工程结论和边界。
- `OUTPUT_FILES.md`：本文件。

本目录不包含正式runtime LUT。工作区外、上下边缘和内部support缺口均不补偿；无插值、外推或平滑。
