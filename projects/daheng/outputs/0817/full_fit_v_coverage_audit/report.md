# 全 FIT v 覆盖与 pose 冗余审核

`FULL_V_COVERAGE = SUFFICIENT`
`FIT_REDUNDANCY = HIGH`

## Scope

- 仅读取 FIT 001–018、025–036、049–054 三联图，共 36 pose；未读取 Validation。
- 使用统一 new `full_board_physical` mask：11×8 内角点、20 mm 格距、X=[-20,220] mm、Y=[-20,160] mm、inset=0 mm。
- 配置：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`；内参：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`。
- 仅做 PnP + Steger + per-row single point + continuity + ray/棋盘平面求交；不拟合 Plane/Quadratic/Cone/C1。
- `lambda_truth_mm = Zc_mm / ray_z`，即激光像素射线与对应 PnP 棋盘平面的真实交点参数；`Zc_mm` 为相机坐标深度。

## Extraction summary

- 有效点：32400；v∈[0,3000) 内点：32400；域外点：0。
- 实际 v 范围：0.03–2999.06 px；域内 v 范围：0.03–2999.06 px。
- lambda_truth 范围：599.916–713.864 mm，span=113.948 mm。
- Z 深度范围：599.916–713.864 mm，span=113.948 mm。
- populated bin 的中位点数：1107.5；sparse 阈值：276.9 点/bin。

## v-bin support

| status | bin count | interpretation |
|---|---:|---|
| unsupported | 0 | no FIT point in the 100 px bin |
| single-frame | 0 | only one pose contributes |
| sparse | 0 | low point count or ≤2 contributing poses |
| well-supported | 3 | multi-pose, non-sparse support |
| highly-redundant | 27 | ≥18 contributing poses |

- unsupported bins: none
- single-frame bins: none
- sparse bins: none
- highly-redundant bins: v_0100_0200, v_0200_0300, v_0300_0400, v_0400_0500, v_0500_0600, v_0600_0700, v_0700_0800, v_0800_0900, v_0900_1000, v_1000_1100, v_1100_1200, v_1200_1300, v_1300_1400, v_1400_1500, v_1500_1600, v_1600_1700, v_1700_1800, v_1800_1900, v_1900_2000, v_2000_2100, v_2100_2200, v_2200_2300, v_2300_2400, v_2400_2500, v_2500_2600, v_2600_2700, v_2700_2800

Coverage decision rules：
- `SUFFICIENT`：30 个 bin 全部有点，无 single-frame/sparse bin，且 ≥80% bin 为 well-supported 或 highly-redundant。
- `PARTIAL`：最多 2 个、总跨度不超过 200 px 的边缘 unsupported bin，且至少 80% bin 有点；或存在 single/sparse 但未达到 insufficient 条件。
- `INSUFFICIENT`：unsupported 超过上述边界，或有效 v 覆盖比例低于 80%。

## Pose redundancy

- pose overlap rate 中位数：1.000；均值：1.000。
- HIGH pose：36 / 36；没有 exclusive v-bin 的 pose：36。
- pose 的 `overlap_rate` 定义为：该 pose 所占 v-bin 中，有其他 pose 同时贡献的 bin 比例；exclusive v-bin 是该 pose 在该 bin 中唯一贡献者的覆盖。
- overall redundancy 规则：median overlap≥0.85 且至少一半 pose 为 HIGH → HIGH；median≤0.55 且 HIGH 少于 25% → LOW；其余 → MODERATE。

- HIGH redundancy poses：001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 025, 026, 027, 028, 029, 030, 031, 032, 033, 034, 035, 036, 049, 050, 051, 052, 053, 054
- zero-new-coverage poses：001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 025, 026, 027, 028, 029, 030, 031, 032, 033, 034, 035, 036, 049, 050, 051, 052, 053, 054

Pose 级别的完整 v 范围、exclusive coverage、overlap rate、pairwise overlap 和 redundancy class 见 `pose_redundancy.csv`。

## Interpretation

- 本审核只描述当前 FIT 采样支持和姿态冗余，不自动删除任何 pose。
- highly-redundant 表示统计上存在大量重叠，不等同于数据无效；是否减少 pose 仍需结合 PnP 姿态差异、残差和独立 Validation 决定。
- 每个 v-bin 的点数、frame IDs、frame contribution ratio、lambda_truth/Z 深度跨度见 `full_fit_v_coverage.csv`。

结论：`FULL_V_COVERAGE = SUFFICIENT`；`FIT_REDUNDANCY = HIGH`。
