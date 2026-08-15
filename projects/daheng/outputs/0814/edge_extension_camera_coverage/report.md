# Task 6H-0 — Edge-extension chess coverage audit

`EDGE_EXTENSION_REUSABLE_FOR_CAMERA_CALIBRATION = PARTIAL`

本审计只打开 baseline chess 001–018 与 extension chess 025–036；使用正式 K/D 做 PnP pose characterization。未读取 laser/Validation，未重新估计 K/D，未拟合 Cone。

## 分类

HIGH_VALUE=4，USEFUL=4，REDUNDANT=4，REJECT_QUALITY=0。

建议加入 M1 的 extension frame：026, 027, 028, 035, 031, 033, 032, 030。

## 覆盖变化

| dimension | baseline range | extension range | new low | new high | range increase |
|---|---:|---:|---:|---:|---:|
| depth | 648.677–712.278 | 655.813–710.498 | 0 | 0 | -0.1402 |
| tilt | 2.53347–24.7658 | 2.62391–26.6496 | 0 | 1 | 0.08066 |
| apparent_size | 0.246413–0.505175 | 0.267707–0.544631 | 0 | 1 | 0.07019 |
| sensor_u | 0.342337–0.677024 | 0.312969–0.487462 | 2 | 0 | -0.4786 |
| sensor_v | 0.338838–0.640289 | 0.372223–0.578157 | 0 | 0 | -0.3169 |
| normalized_radius | 0.185217–0.288314 | 0.193048–0.267289 | 0 | 0 | -0.2799 |

Tilt/depth and tilt/size decoupling checks:
- tilt_depth: baseline Spearman=-0.5769, extension Spearman=-0.6014; near-zero would indicate tilt/depth decoupling.
- tilt_apparent_size: baseline Spearman=0.2487, extension Spearman=0.1049; near-zero would indicate tilt/size decoupling.

## Candidate ranking

| rank | frame | category | novelty | nearest baseline | new dimensions | tilt-direction distance | PnP RMSE (px) |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 026 | HIGH_VALUE | 3.115 | 008 | apparent_size | 3.147 | 0.1652 |
| 2 | 027 | HIGH_VALUE | 1.877 | 015 | tilt | 7.953 | 0.1749 |
| 3 | 028 | HIGH_VALUE | 1.524 | 016 | sensor_u | 13.82 | 0.1878 |
| 4 | 035 | HIGH_VALUE | 1.521 | 002 | sensor_u | 0.6104 | 0.1352 |
| 5 | 031 | USEFUL | 1.626 | 016 |  | 6.179 | 0.1146 |
| 6 | 033 | USEFUL | 1.424 | 003 |  | 5.19 | 0.1217 |
| 7 | 032 | USEFUL | 1.138 | 016 |  | 4.542 | 0.13 |
| 8 | 030 | USEFUL | 1.068 | 011 |  | 5.907 | 0.1304 |
| 9 | 034 | REDUNDANT | 1.849 | 016 |  | 1.417 | 0.1 |
| 10 | 029 | REDUNDANT | 1.747 | 011 |  | 0.3323 | 0.1065 |
| 11 | 036 | REDUNDANT | 1.223 | 006 |  | 0.03016 | 0.0918 |
| 12 | 025 | REDUNDANT | 0.7963 | 003 |  | 2.826 | 0.1573 |

## 判断

结论为 PARTIAL：025–036 能补强 tilt、apparent-size 和一部分 sensor-u edge coverage，但没有新增 depth 范围，且 extension 的 tilt–depth Spearman 仍约 -0.60，不能视为独立 depth/tilt 约束。025–036 是否可用于 M1 的判断仅基于棋盘几何覆盖和 PnP 质量，不代表已经验证重新标定后的激光 truth 或 laser surface。HIGH_VALUE/USEFUL candidate 应与原 001–018 一起作为 coverage augmentation；REDUNDANT candidate 不增加明显约束，但不等于图像质量差。

## 输出

- `camera_candidate_pose_coverage.csv`
- `baseline_vs_extension_coverage.csv`
- `candidate_ranking.csv`
- `provenance.json`
