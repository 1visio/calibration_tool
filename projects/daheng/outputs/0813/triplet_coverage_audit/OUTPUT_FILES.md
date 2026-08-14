# Triplet coverage audit outputs

| 文件 | 内容 | 主要边界 |
|---|---|---|
| triplet_provenance.csv | frames.csv/manifest/config/stage 与实际 fit/validation 追踪 | 不表示重新拟合 |
| triplet_frame_geometry.csv | 每帧独立 PnP、棋盘平面、board center、tilt、ray-depth 汇总 | 不是 Cone residual |
| triplet_ray_depth_support.csv | 300px v-bin、外推区、多帧/多 lambda 支持统计 | descriptive coverage only |
| triplet_uv_support.png | 原始 laser centre `(u,v)` 覆盖图 | 不是模型预测图 |
| triplet_v_lambda_support.png | 独立 PnP ray-plane truth 的 `(v,lambda)` 覆盖 | 不证明参数可辨识 |
| triplet_pose_depth_distribution.png | 每帧 board Zc、tilt、lambda span | 不用于筛帧 |
| triplet_coverage_audit.md | provenance、方法、覆盖结论与限制 | 未做 laser model fit |
| OUTPUT_FILES.md | 输出索引 | 不增加证据 |
