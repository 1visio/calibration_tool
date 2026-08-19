# Operational 35-pose C1 FIT-only model selection

C1_OPERATIONAL_MODEL = C1_4k
C1_FIT_STATUS = READY_FOR_VALIDATION

## Scope decision

- frame027 状态：`EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN`。排除理由：**超出实际工作姿态域**；不是 residual-based deletion。
- 027 原始 Full-36 residual artifact 保留不动；本轮只在 C1 development/evaluation domain 中使用剩余 35 pose。
- 本轮没有读取 Validation、没有重拟合 Quadratic C0、没有重跑同协议 grouped-CV、没有做 2D C1、没有修改生产配置。

## Artifact provenance / reuse audit

| artifact | action | status | evidence |
|---|---|---|---|
| Frozen Quadratic C0 | LOADED_ONLY / reused by source artifact | CONFIRMED | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c0_freeze\quadratic_graph.yaml`; sha256 `113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27` |
| Full-36 residual artifact | REUSED_EXISTING | CONFIRMED | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\quadratic_residual_observability\quadratic_residual_points.csv`; hash matched source manifest; 027 retained |
| Existing 35-pose C1 grouped-CV | REUSED_EXISTING | CONFIRMED | `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_frozen_quadratic_grouped_cv`; scenario `exclude027_grouped_cv_non027`; 6 folds; no refit in this run |
| frame027 | EXCLUDED_FROM_OPERATIONAL_DOMAIN | CONFIRMED | reason is actual working-pose domain, not residual deletion |
| Validation | NOT_READ | EXCLUDED | no Validation path opened |

## Reused 35-pose grouped-CV comparison

| candidate | RMSE C0→C1 / % | P95 C0→C1 / % | P99 C0→C1 / % | worst-v RMSE / % | worst-v P95 / % | v-bias range C0→C1 / mm | pose RMSE improvement ratio | operational gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C1_3k | 13.46 | 15.65 | 13.93 | 38.31 | 18.28 | 0.290→0.158 | 0.914 | PASS |
| C1_4k | 15.54 | 17.51 | 15.36 | 39.57 | 15.27 | 0.290→0.143 | 0.943 | PASS |
| C1_5k | 16.35 | 18.31 | 16.02 | 40.64 | 22.11 | 0.290→0.138 | 0.943 | PASS |

## Pose-level paired comparison

Positive values mean the higher-knot candidate is better than the baseline candidate.

| comparison | RMSE better poses | RMSE median / P05 / P95 gain % | P95 better poses | P95 median / P05 / P95 gain % |
|---|---:|---:|---:|---:|
| C1_4k vs C1_3k | 32/35 (0.914) | 2.93 / -0.34 / 5.33 | 32/35 (0.914) | 3.20 / -0.80 / 5.13 |
| C1_5k vs C1_3k | 32/35 (0.914) | 4.31 / -0.57 / 7.43 | 32/35 (0.914) | 4.44 / -1.31 / 8.47 |
| C1_5k vs C1_4k | 28/35 (0.800) | 1.30 / -0.96 / 2.98 | 28/35 (0.800) | 1.53 / -1.64 / 4.42 |

### Selection rule

- 4k stable-gain gate: paired RMSE/P95 better-pose fraction ≥ 0.80, and both median gains ≥ 0.50%. Result: `True`.
- 5k saturation gate: paired 5k-vs-4k median RMSE/P95 gain < 2.00% and global RMSE increment < 1.50%. Result: `True`.
- **4k has stable paired pose gains over 3k and 5k has saturated incremental gains over 4k.**

当前数值支持 C1_4k：相对 3k，RMSE/P95 均在 32/35 pose 改善；5k 相对 4k 的 RMSE/P95 median 增量为 1.30%/1.53%，继续改善 pose 为 28/35、28/35，收益已明显递减。

## Scope and handoff

- 本报告输出的是最终 FIT-only 候选：`C1_4k`；`C1_FIT_STATUS = READY_FOR_VALIDATION` 仅表示可以进入下一步 Validation，不表示生产冻结或已通过 Validation。
- 027 仍保留在 Full-36 residual artifact；正式 operational domain label 只作用于 C1 development/evaluation，不改变原始数据文件。
- 下一步应使用独立 Validation 验证该候选，届时另建版本化产物；本轮不写入生产配置。

## Outputs

- `c1_operational35_model_comparison.csv`：35-pose aggregate 与选择字段。
- `c1_operational35_pose_paired.csv`：C0→候选及候选间逐 pose paired RMSE/P95。
- `c1_operational35_v_bins.csv`：复用的 100 px v-bin metrics。
- `c1_model_selection_report.md`：本报告。
