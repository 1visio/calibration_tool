# Full-v FIT pose-grouped CV 三模型比较

`C0_MODEL_CANDIDATE = UNRESOLVED`

## Scope

- 仅读取 FIT 001–018、025–036、049–054，共 36 pose；未读取任何 Validation 图像。
- FIT 点：32400；内参：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`；配置：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`。
- mask：`full_board_physical`，11×8 内角点、20 mm 格距，X=[-20,220] mm、Y=[-20,160] mm、inset=0 mm。
- Steger、vertical per-row single point、continuity、每帧点数限制和 frame-balanced weighting 与正式流程保持一致。
- 未增加 v-density weighting；未训练 C1；未覆盖历史 Frozen C0。

## Grouped CV design

- 采用 6-fold pose-grouped CV；每个 fold 完整留出 pose，训练与评价之间没有同一 pose 的点交叉。
- fold assignment 为按 frame ID 排序后的确定性 round-robin；每 fold 6 个 held-out pose。

| fold | held-out frames | train frames |
|---:|---|---:|
| 0 | 001, 007, 013, 025, 031, 049 | 30 |
| 1 | 002, 008, 014, 026, 032, 050 | 30 |
| 2 | 003, 009, 015, 027, 033, 051 | 30 |
| 3 | 004, 010, 016, 028, 034, 052 | 30 |
| 4 | 005, 011, 017, 029, 035, 053 | 30 |
| 5 | 006, 012, 018, 030, 036, 054 | 30 |

## Pooled grouped-CV comparison

以下指标来自所有 held-out pose 的 pooled predictions；模型选择同时使用 Global、worst-v-bin、v-bias range 和 fold/frame 稳定性，不使用训练集 global RMSE 单独选择。

| model | Global bias | Global RMSE | Global P95 | worst v-bin | worst RMSE | worst P95 | v-bias range | fold RMSE std | score |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| quadratic_graph | 0.00271 | 0.09909 | 0.18438 | v_0000_0100 | 0.17423 | 0.24810 | 0.28363 | 0.02463 | 0.01271 |
| circular_cone | 0.00009 | 0.10089 | 0.19054 | v_0000_0100 | 0.20120 | 0.29013 | 0.31599 | 0.02411 | 0.02359 |
| global_plane | -0.02794 | 0.34129 | 0.69232 | v_2900_3000 | 1.09887 | 1.38509 | 1.41209 | 0.02820 | 1.00000 |

## Worst v-bin detail

### global_plane

| v_bin       |   point_count |   unique_frame_count |   bias_mm |   rmse_mm |   p95_mm |   max_abs_mm |
|:------------|--------------:|---------------------:|----------:|----------:|---------:|-------------:|
| v_2900_3000 |           345 |                   11 |  -1.07842 |   1.09887 |  1.38509 |      1.54230 |
| v_2800_2900 |           580 |                   15 |  -0.96588 |   0.98863 |  1.32408 |      1.49181 |
| v_2700_2800 |           541 |                   18 |  -0.81670 |   0.83583 |  1.13640 |      1.23928 |

### quadratic_graph

| v_bin       |   point_count |   unique_frame_count |   bias_mm |   rmse_mm |   p95_mm |   max_abs_mm |
|:------------|--------------:|---------------------:|----------:|----------:|---------:|-------------:|
| v_0000_0100 |           323 |                   11 |   0.15805 |   0.17423 |  0.24810 |      0.34047 |
| v_0100_0200 |           643 |                   18 |   0.12240 |   0.16008 |  0.24415 |      0.42235 |
| v_0700_0800 |          1003 |                   31 |  -0.12558 |   0.14204 |  0.24507 |      0.37501 |

### circular_cone

| v_bin       |   point_count |   unique_frame_count |   bias_mm |   rmse_mm |   p95_mm |   max_abs_mm |
|:------------|--------------:|---------------------:|----------:|----------:|---------:|-------------:|
| v_0000_0100 |           323 |                   11 |   0.18482 |   0.20120 |  0.29013 |      0.37881 |
| v_0100_0200 |           643 |                   18 |   0.11961 |   0.16064 |  0.27237 |      0.39333 |
| v_0700_0800 |          1003 |                   31 |  -0.13117 |   0.14574 |  0.24403 |      0.38908 |

## Full 36-pose reference

三模型全量拟合参数保存在 `candidate_models/full_fit/`；该 fit 仅作为最大数据量 reference，不用于 grouped-CV 选择。
- `global_plane.yaml`
- `quadratic_graph.yaml`
- `circular_cone.yaml`
- `model_parameters.json`

## Selection rule

score = 0.25×Global RMSE + 0.30×worst-v-bin RMSE + 0.15×worst-v-bin P95 + 0.15×v-bias range + 0.10×fold RMSE std + 0.05×frame RMSE std；各项先在三模型间 min-max 归一化。最高分与第二名差距小于 0.05 时判为 `UNRESOLVED`。
本次选择判据：top score gap=0.0109 < 0.05。

## Artifacts

- `grouped_cv_model_comparison.csv`：fold 级与 pooled Global CV 指标及 worst-v-bin/trend 汇总。
- `per_v_bin_cv_metrics.csv`：三模型每个 100 px v-bin 的 Bias、RMSE、P95、Max 和 pose/fold 覆盖。
- `residual_vs_v_cv.png`：pooled grouped-CV 的 residual-v Bias/RMSE 趋势。

结论：`C0_MODEL_CANDIDATE = UNRESOLVED`。该结论只基于 FIT pose-grouped CV；进入独立 Validation 或标准件验收前，不冻结生产模型。
