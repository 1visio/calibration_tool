# Circular Cone FIT stability outputs

本目录只包含 FIT-only frame jackknife 与 weak-direction profile；没有 validation 结果，也没有候选 YAML。

| 文件 | 作用 | 边界 |
|---|---|---|
| cone_frame_jackknife.csv | 10 折训练/留一帧 exact 指标与参数移动 | 不是 validation |
| cone_frame_jackknife_regions.csv | 留一帧 top/middle/bottom 与每300px区域 | 仅 FIT |
| cone_frame_jackknife.png | 留一帧 RMSE 与 explained 曲线 | 组会图，不含 validation |
| cone_weak_direction_profile.csv | 最弱 SVD 方向的 exact profile | 不是新优化结果 |
| cone_weak_direction_profile.png | weak-direction loss/profile 图 | bounds 限制的一维扫描 |
| cone_fit_stability_report.md | 本阶段主报告 | 不授权写回参数 |
| OUTPUT_FILES.md | 输出索引 | 不增加科学证据 |
