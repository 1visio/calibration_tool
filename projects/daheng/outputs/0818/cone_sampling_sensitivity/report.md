# Full-36 Circular Cone fit_max_points 敏感性

`CONE_SAMPLING_STATUS = UNSTABLE`

## 结论摘要

- 3000 点基线直接复用 0817 artifact，未重跑；新增计算仅为 6000、12000、all feasible。
- all feasible 使用每个训练 fold 的全部 27000 个训练点；除 `fit_max_points` 外，Cone 配置、fold、frame-balanced weighting、初值策略保持不变。
- 3000→all feasible：Global RMSE 0.10089→0.10081 mm，Global P95 0.19054→0.19058 mm，worst-v RMSE 0.20120→0.20064 mm。
- 判定：`UNSTABLE`。all-feasible 参数/性能路径不稳定：monotonic_rmse=False, monotonic_worst=False, axis_change=0.253 deg, apex_change=104.358 mm。

## Artifact reuse audit

- 配置：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`，只读；正式配置中的 `fit_max_points: 3000` 未修改。
- 3000 点结果：复用 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison` 下的 `grouped_cv_model_comparison.csv`、`per_v_bin_cv_metrics.csv`、`cv_pointwise_circular_cone.csv` 和 `cv_fold_model_parameters.json`。
- 新档只调用 Circular Cone fit/evaluate；没有运行 Quadratic、Plane candidate 或 Validation。既有每折 plane 参数仅作为 Cone 所需的固定 orientation/root hint，未调用 PlaneModel.fit。
- 训练数据为 Full-36 FIT、full_board_physical、inset=0 mm、frame-balanced weighting、6-fold pose-grouped CV；没有 v-density weighting。

## Performance

| sampling | fit_max_points | selected points/fold | Global RMSE | Global P95 | worst-v-bin | worst RMSE | worst P95 | fold RMSE std | v-bias range | status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 3000 | 3000 | 3000 | 0.10089 | 0.19054 | v_0000_0100 | 0.20120 | 0.29013 | 0.02411 | 0.31599 | REUSED_EXISTING |
| 6000 | 6000 | 6000 | 0.10079 | 0.19059 | v_0000_0100 | 0.20107 | 0.28709 | 0.02435 | 0.31521 | COMPUTED_MISSING |
| 12000 | 12000 | 12000 | 0.10085 | 0.19056 | v_0000_0100 | 0.20119 | 0.28661 | 0.02435 | 0.31544 | COMPUTED_MISSING |
| all_feasible | 27000 | 27000 | 0.10081 | 0.19058 | v_0000_0100 | 0.20064 | 0.28398 | 0.02435 | 0.31507 | COMPUTED_MISSING |

## Parameter changes and stability

`axis_angle_vs_3000` uses the absolute axis dot product; apex change is Euclidean camera-coordinate distance.

| sampling | axis range / deg | apex spread / mm | half-angle range / deg | max axis change vs 3000 / deg | max apex change / mm | max half-angle change / deg | cost CV / % | success rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.982 | 408.281 | 0.968 | 0.000 | 0.000 | 0.000 | 10.98 | 1.00 |
| 6000 | 1.075 | 446.951 | 1.063 | 0.143 | 59.209 | 0.142 | 10.87 | 1.00 |
| 12000 | 1.183 | 491.209 | 1.170 | 0.244 | 100.901 | 0.244 | 11.02 | 1.00 |
| all_feasible | 1.178 | 489.118 | 1.166 | 0.253 | 104.358 | 0.252 | 10.99 | 1.00 |

## Decision rule

- `SATURATED_AT_3000`：3000→all feasible 的 Global RMSE、Global P95、worst-v RMSE、worst-v P95 改变量分别小于 0.001、0.003、0.005、0.01 mm，且参数路径稳定。
- `BENEFITS_FROM_MORE_POINTS`：上述前三项主要误差指标达到预设实质改善，且参数/性能路径稳定。
- `UNSTABLE`：拟合失败、性能随采样数明显非单调，或出现大幅 axis/apex/half-angle 漂移。

## Scope exclusions

- 未重跑 3000 点；未运行 Quadratic/Plane；未读取 Validation；未训练 C1。
- 未修改正式 YAML 配置；所有新采样档均为内存配置副本。

## Outputs

- `cone_sampling_sensitivity.csv`
- `cone_sampling_performance.png`
- `parameter_stability.csv`
- `report.md`
