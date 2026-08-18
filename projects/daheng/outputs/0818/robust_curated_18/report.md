# Robust-Curated-18 几何互补审计

POSE_DIVERSITY = SUFFICIENT
RECOMMENDED_ROBUST_CURATED_FIT_SIZE = 18

## 结论

固定 Curated-14，不删除已有 pose；新增 4 个 pose：005, 026, 028, 050。
Robust-Curated-18：001, 005, 006, 010, 013, 015, 017, 025, 026, 027, 028, 031, 049, 050, 051, 052, 053, 054。

选择只使用 pose_geometry_metrics.csv、pair_pose_similarity.csv、当前 full-board-physical FIT 点和既有 geometry gates。
没有拟合 Plane/Quadratic/Cone，没有读取 Validation，也没有读取或使用任何模型 residual。

## 选择方法

- 候选池：其余 22 个 pose；穷举组合数：7315（C(22,4)）。
- 硬排除任何新增 pose 参与的 strict near-duplicate pair；剩余安全组合：4692。
- 优先级：四个边缘 band 的最弱独立 frame 数与低/高边缘平衡度 → normal cover 最大角度 → depth/lambda 极值重复支持 → excitation span → 与 Curated-14 的低 geometric similarity。
- edge 前沿 key（min-band, low/high balance, total, low total, high total）：(6, 13, 28, 15, 13)。
- edge 前沿中的 normal 最佳最大角度：3.805°。
- Robust-18 内部 strict near-duplicate pair：1（其中 Curated-14 原有 1，新增 pose 涉及 0）。

## 新增 pose 的几何作用

| pose | 作用 |
|---|---|
| 005 | edge low/high bins 0/2; depth bin 0; lambda bins 3; max similarity 0.251 |
| 026 | edge low/high bins 1/2; depth bin 1; lambda bins 0;1;2;3;4;5;6; max similarity 0.461 |
| 028 | edge low/high bins 2/0; depth bin 2; lambda bins 2;3; max similarity 0.435 |
| 050 | edge low/high bins 0/2; depth bin 3; lambda bins 4;5; max similarity 0.298 |

## 几何覆盖比较

| 指标 | Curated-14 | Robust-18 | 变化 |
|---|---:|---:|---:|
| v 10 px occupied | 300 | 300 | +0 |
| v 100 px occupied | 30 | 30 | +0 |
| edge 0–100 / 100–200 | 5 / 7 | 6 / 9 | +1 / +2 |
| edge 2800–2900 / 2900–3000 | 4 / 3 | 7 / 6 | +3 / +3 |
| edge minimum | 3 | 6 | +3 |
| low/high edge total | 12 / 7 | 15 / 13 | +3 / +6 |
| normal cover max / ° | 4.720 | 3.805 | 0.914 |
| normal diameter / ° | 51.014 | 51.014 | 0.000 |
| depth span / mm | 66.232 | 66.232 | 0.000 |
| lambda span / mm | 113.699 | 113.699 | 0.000 |
| depth extreme low/high support | 2 / 1 | 3 / 1 | +1 / +0 |
| lambda extreme low/high support | 2 / 5 | 3 / 5 | +1 / +0 |
| max similarity(new, Curated-14) | — | 0.461 | lower is more complementary |

Robust-18 仍覆盖全部 300 个 10 px cells、全部 30 个 100 px bins，并通过既有 geometry gates。
normal cover 从 4.720° 降至 3.805°；depth 极值支持 low/high = 2/1 → 3/1；lambda 极值支持 low/high = 2/5 → 3/5。

## 近重复复核

Robust-Curated-18 没有新增 pose 参与既有 strict near-duplicate pair。高相似候选 036 虽能补高 depth 极值，但与 Curated-14 的 006、013 构成 strict near-duplicate，因此按约束排除。

| 新增 pose | Curated pose | similarity | normal Δ / ° | translation Δ / mm | v Jaccard |
|---|---|---:|---:|---:|---:|
| 026 | 025 | 0.461 | 4.327 | 27.743 | 0.833 |
| 028 | 052 | 0.435 | 4.720 | 67.819 | 0.714 |
| 026 | 027 | 0.350 | 8.875 | 57.604 | 0.655 |
| 050 | 010 | 0.298 | 3.364 | 106.641 | 0.414 |
| 005 | 010 | 0.251 | 9.709 | 47.305 | 0.690 |
| 050 | 052 | 0.245 | 6.426 | 95.813 | 0.333 |
| 005 | 053 | 0.205 | 14.209 | 79.916 | 0.633 |
| 050 | 006 | 0.201 | 12.975 | 52.208 | 0.667 |

## 输入与输出

- 输入 metrics：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\pose_geometry_audit\pose_geometry_metrics.csv。
- 输入 pair similarity：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\pose_geometry_audit\pair_pose_similarity.csv。
- 当前点表：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\full_fit_v_coverage_audit\full_fit_points.csv；mask provenance = full_board_physical, inset=0 mm。
- 当前 100 px reference：30/30 bins populated，minimum frame multiplicity = 11。
- 输出：robust_curated_18_ids.json、robust18_geometry_comparison.csv、robust18_v_coverage.png、report.md。

POSE_DIVERSITY = SUFFICIENT
RECOMMENDED_ROBUST_CURATED_FIT_SIZE = 18
