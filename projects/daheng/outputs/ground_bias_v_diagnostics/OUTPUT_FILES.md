# Ground-bias reference-mode diagnostics 输出文件说明

本目录所有结果均基于现有 laser 001–031，未执行补偿、未调整 smooth window。

| 文件 | 文件体现什么 | 应看哪些指标 | 不能得出什么结论 | 推荐放组会报告 |
|---|---|---|---|---|
| `reference_modes/self_fitted/residual_frame_v_heatmap.png` | self_fitted 下逐帧 residual(v) 热图 | 重复波形、符号翻转、覆盖空白 | 只能描述该 reference 定义下的 residual；不能给出真实棋盘姿态 | 是 |
| `reference_modes/self_fitted/residual_v_median_sigma.png` | self_fitted 下跨帧 median、±1 sigma、sample count | median 波形、离散度、支持数 | 不能证明波形来源或补偿有效性 | 是 |
| `reference_modes/self_fitted/residual_v_statistics.csv` | self_fitted 下逐 v 完整数值统计 | sample count、median、std、MAD、P95 | 不能把 sparse v 行当稳定系统误差 | 否，适合作为备查数据 |
| `reference_modes/self_fitted/frame_residual_correlation.csv` | self_fitted 下所有 frame pair 共同支持与相关系数 | common_sample_count 和 correlation | 低共同支持相关性不可靠；相关性不反映常数 offset | 否，适合作为备查数据 |
| `reference_modes/self_fitted/diagnostics_summary.json` | self_fitted 的结构化口径和指标；fixed_ground 另含 Z0 provenance | comparison_metrics、metric_definitions | 不是人工结论，也不是 compensation 配置 | 否，机器复核用 |
| `reference_modes/fixed_normal_per_frame_offset/residual_frame_v_heatmap.png` | fixed_normal_per_frame_offset 下逐帧 residual(v) 热图 | 重复波形、符号翻转、覆盖空白 | 只能描述该 reference 定义下的 residual；不能给出真实棋盘姿态 | 是 |
| `reference_modes/fixed_normal_per_frame_offset/residual_v_median_sigma.png` | fixed_normal_per_frame_offset 下跨帧 median、±1 sigma、sample count | median 波形、离散度、支持数 | 不能证明波形来源或补偿有效性 | 是 |
| `reference_modes/fixed_normal_per_frame_offset/residual_v_statistics.csv` | fixed_normal_per_frame_offset 下逐 v 完整数值统计 | sample count、median、std、MAD、P95 | 不能把 sparse v 行当稳定系统误差 | 否，适合作为备查数据 |
| `reference_modes/fixed_normal_per_frame_offset/frame_residual_correlation.csv` | fixed_normal_per_frame_offset 下所有 frame pair 共同支持与相关系数 | common_sample_count 和 correlation | 低共同支持相关性不可靠；相关性不反映常数 offset | 否，适合作为备查数据 |
| `reference_modes/fixed_normal_per_frame_offset/diagnostics_summary.json` | fixed_normal_per_frame_offset 的结构化口径和指标；fixed_ground 另含 Z0 provenance | comparison_metrics、metric_definitions | 不是人工结论，也不是 compensation 配置 | 否，机器复核用 |
| `reference_modes/fixed_ground_plane/residual_frame_v_heatmap.png` | fixed_ground_plane 下逐帧 residual(v) 热图 | 重复波形、符号翻转、覆盖空白 | 只能描述该 reference 定义下的 residual；不能给出真实棋盘姿态 | 是 |
| `reference_modes/fixed_ground_plane/residual_v_median_sigma.png` | fixed_ground_plane 下跨帧 median、±1 sigma、sample count | median 波形、离散度、支持数 | 不能证明波形来源或补偿有效性 | 是 |
| `reference_modes/fixed_ground_plane/residual_v_statistics.csv` | fixed_ground_plane 下逐 v 完整数值统计 | sample count、median、std、MAD、P95 | 不能把 sparse v 行当稳定系统误差 | 否，适合作为备查数据 |
| `reference_modes/fixed_ground_plane/frame_residual_correlation.csv` | fixed_ground_plane 下所有 frame pair 共同支持与相关系数 | common_sample_count 和 correlation | 低共同支持相关性不可靠；相关性不反映常数 offset | 否，适合作为备查数据 |
| `reference_modes/fixed_ground_plane/diagnostics_summary.json` | fixed_ground_plane 的结构化口径和指标；fixed_ground 另含 Z0 provenance | comparison_metrics、metric_definitions | 不是人工结论，也不是 compensation 配置 | 否，机器复核用 |
| `reference_mode_comparison.csv` | 三种 reference mode 的统一数值口径 | pair correlation、explained energy、std、sign mixing、三区域 std | 不能把任一模式自动认定为真实棋盘平面 | 是 |
| `reference_mode_comparison.png` | 四个关键一致性指标的并排柱图 | A/B/C 相对变化 | 不能代替逐 v 曲线和覆盖检查 | 是 |
| `repositioning_effects.csv` | 31 帧 offset、表观倾角和 self-fit condition number | offset 范围；tilt 与 condition 的异常帧 | apparent tilt 不是机械倾角 | 可选；报告正文优先放统计摘要 |
| `reference_mode_comparison_report.md` | A–D 问题的定量回答、Z0 来源和限制 | 四个结论段、offset/tilt 统计 | 不是正式补偿验收报告 | 是 |
| `OUTPUT_FILES.md` | 本文件；输出导航和解释边界 | 推荐列和文件用途 | 不包含新的数据分析 | 否 |
| `diagnostics_report.md` | 早先 self_fitted baseline 的详细 A–E 报告，现保留兼容 | self_fitted 的区域、符号翻转、边缘结论 | 不能代表 B/C；统一比较应看新 report | 不推荐单独使用 |
| `diagnostics_summary.json` | 早先 baseline 加初版三模式摘要，现保留兼容 | provenance 和旧 baseline 指标 | 不应替代各模式独立 summary | 否 |
| `residual_frame_v_heatmap.png` | 根目录 legacy self_fitted 热图 | 同 self_fitted 子目录热图 | 不能代表 B/C | 不推荐；使用子目录版本 |
| `residual_v_median_sigma.png` | 根目录 legacy self_fitted 曲线 | 同 self_fitted 子目录曲线 | 不能代表 B/C | 不推荐；使用子目录版本 |
| `residual_v_statistics.csv` | 根目录 legacy self_fitted 统计 | 同 self_fitted 子目录 CSV | 不能代表 B/C | 否 |
| `frame_residual_correlation.csv` | 根目录 legacy self_fitted pair correlation | 同 self_fitted 子目录 CSV | 不能代表 B/C | 否 |
| `per_frame_plane_fit_diagnostics.csv` | self-fit 平面参数、条件数和 inlier 信息 | condition number、inlier count | `a,b` 不能解释成真实机械姿态 | 否 |
| `reference_plane_mode_comparison_per_frame.csv` | 前一轮三模式逐帧初版对照，保留兼容 | self-fit 参数、offset | 新分析应优先使用 `repositioning_effects.csv` | 否 |
| `reference_plane_mode_comparison_summary.csv` | 前一轮三模式初版总体对照，保留兼容 | MAE、RMS、P95 | 指标口径少于新统一比较 | 否 |

组会建议的最小文件组合：

1. `reference_mode_comparison.png`
2. `reference_mode_comparison_report.md`
3. 三张 `reference_modes/*/residual_v_median_sigma.png`
4. 如需展示帧间结构，再选三张 heatmap；不要只展示 self_fitted。
