# OUTPUT_FILES

本目录只包含 paired-PnP residual 诊断，不包含正式 LUT 或任何 compensation 产物。

- `residual_frame_v_heatmap.png`：13帧 residual_z 的 frame×v 热图；fit/validation 由横线分隔。
- `residual_v_median_sigma.png`：fit-only 诊断 median b(v)、fit std(v)、validation 独立观察和样本数。
- `residual_v_statistics.csv`：每个 v 的 fit/validation 分离统计；fit median 不是正式 LUT。
- `frame_residual_correlation.csv`：split 内无重复帧对相关性及共同样本数。
- `per_frame_residual_metrics.csv`：逐帧提取、重建、PnP plane 与 residual 指标。
- `diagnostics_summary.json`：机器可读的冻结参数、provenance、支持度、相关性、能量和区域统计。
- `diagnostics_report.md`：结论、判定规则和关键表格。
- `OUTPUT_FILES.md`：本文件。

明确未执行：激光点 reference-plane 自拟合、median profile 平滑/插值、正式 LUT 构建、compensation。
