# Curated-14 / Robust-18 / Full-36 Quadratic-Cone 泛化比较

`RECOMMENDED_FIT_SET = FULL_36`

`C0_MODEL_STATUS = UNRESOLVED`

## 结论摘要

- 推荐 FIT 集：`FULL_36`。判据：Robust-18 相对 Full-36 的剩余差距仍为 0.01164 mm，未达到预设饱和阈值。
- C0：`UNRESOLVED`。Full-36 既有同协议结果的 Quadratic/Cone top-score gap 为 0.0109，小于既定 0.05 决策阈值；因此不能仅凭子集结果把生产 C0 冻结为某一模型。
- Robust-18 的四个新增 pose 是既有几何审核固定的 `005, 026, 028, 050`；本次没有根据 residual 删除或替换任何 pose。

## Provenance / reuse audit

- Full-36 直接复用 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\grouped_cv_model_comparison.csv`，未重跑。
- 复用的协议：`full_board_physical`、inset=0 mm、Steger、vertical per-row single point、continuity、每帧点数限制、frame-balanced weighting、无 v-density weighting、6-fold pose-grouped CV、sorted frame-id round-robin。
- Curated-14/Robust-18 仅读取已生成的 `full_fit_points.csv`，按固定 pose 集做缺失的 Quadratic/Cone CV；没有读取图像，也没有读取 Validation。
- 为满足“不运行 Plane”，子集 CV 不调用 `PlaneModel.fit`。模型实现所需的 root-selection / orientation hint 复用 Full-36 的既有 fold-plane 参数；这些参数不作为候选模型、指标或选择依据。
- Historical-18 没有发现同协议 Q/C CV 结果，仅作已有几何/历史参考，没有补跑。

## Pooled grouped-CV metrics

`global_p95_mm` 是所有 held-out pose pooled prediction 的绝对误差 P95；worst-v-bin 在 100 px bin 中按 RMSE 最大者确定。

| FIT set | model | status | Global RMSE | Global P95 | worst-v-bin | worst RMSE | worst P95 | v-bias range | fold RMSE std | frame RMSE std |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| CURATED_14 | circular_cone | COMPUTED_MISSING | 0.12219 | 0.21992 | v_1700_1800 | 0.19893 | 0.53707 | 0.29018 | 0.04262 | 0.06217 |
| CURATED_14 | quadratic_graph | COMPUTED_MISSING | 0.12292 | 0.21506 | v_1700_1800 | 0.19486 | 0.52381 | 0.25216 | 0.04260 | 0.06081 |
| ROBUST_18 | circular_cone | COMPUTED_MISSING | 0.11118 | 0.19804 | v_0000_0100 | 0.18303 | 0.30119 | 0.30081 | 0.04101 | 0.05416 |
| ROBUST_18 | quadratic_graph | COMPUTED_MISSING | 0.11073 | 0.19518 | v_1700_1800 | 0.17355 | 0.50277 | 0.27482 | 0.03907 | 0.05129 |
| FULL_36 | circular_cone | REUSED_EXISTING | 0.10089 | 0.19054 | v_0000_0100 | 0.20120 | 0.29013 | 0.31599 | 0.02411 | 0.04314 |
| FULL_36 | quadratic_graph | REUSED_EXISTING | 0.09909 | 0.18438 | v_0000_0100 | 0.17423 | 0.24810 | 0.28363 | 0.02463 | 0.04036 |

## Model parameter stability

稳定性只描述每个 FIT 集内部六个训练折得到的参数散布，不使用 residual 删除 pose。

| FIT set | model | axis consistency | beta rel std / % | center std / mm | cone axis range / deg | cone apex spread / mm | cone half-angle range / deg | success rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CURATED_14 | circular_cone | NA | NA | NA | 1.274 | 501.442 | 1.259 | 1.00 |
| CURATED_14 | quadratic_graph | 1.00 | 7.65 | 2.836 | NA | NA | NA | NA |
| ROBUST_18 | circular_cone | NA | NA | NA | 0.550 | 225.305 | 0.536 | 1.00 |
| ROBUST_18 | quadratic_graph | 1.00 | 6.80 | 2.903 | NA | NA | NA | NA |
| FULL_36 | circular_cone | NA | NA | NA | 0.982 | 408.281 | 0.968 | 1.00 |
| FULL_36 | quadratic_graph | 1.00 | 9.70 | 2.761 | NA | NA | NA | NA |

## 14 → 18 → 36 性能收益

- **quadratic_graph**：Global RMSE 0.12292 → 0.11073 → 0.09909 mm；14→18 改善 0.01219 mm，14→36 总改善 0.02383 mm，Robust-18 回收 51.2%。
  worst-v-bin RMSE 0.19486 → 0.17355 → 0.17423 mm；worst-v P95 0.52381 → 0.50277 → 0.24810 mm。
- **circular_cone**：Global RMSE 0.12219 → 0.11118 → 0.10089 mm；14→18 改善 0.01101 mm，14→36 总改善 0.02130 mm，Robust-18 回收 51.7%。
  worst-v-bin RMSE 0.19893 → 0.18303 → 0.20120 mm；worst-v P95 0.53707 → 0.30119 → 0.29013 mm。

## Fold detail / reproducibility

Curated-14 与 Robust-18 的 fold 级结果保存在脚本运行目录的内存计算中，并将 pooled comparison 写入 `fit_set_model_comparison.csv`；Full-36 的 fold 明细继续由 0817 原始 artifact 提供。

| FIT set | model | fold RMSE mean / mm | fold RMSE std / mm |
|---|---|---:|---:|
| CURATED_14 | circular_cone | 0.10723 | 0.04262 |
| CURATED_14 | quadratic_graph | 0.10891 | 0.04260 |
| ROBUST_18 | circular_cone | 0.10334 | 0.04101 |
| ROBUST_18 | quadratic_graph | 0.10361 | 0.03907 |

## Scope exclusions

- 未运行 Plane candidate；未训练 C1；未读取 Validation；未使用 Plane/Quadratic/Cone residual 构造或删除 pose。
- 结果是 FIT pose-grouped CV 的泛化 proxy，不替代独立 Validation 或标准件验收。

## Output

- `artifact_reuse_audit.csv`
- `fit_set_model_comparison.csv`
- `fit_size_performance.png`
- `report.md`
