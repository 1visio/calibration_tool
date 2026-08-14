# Triplet local metric-scale / height-gain observability audit

**NO_CONE_FIT = TRUE**  
**FIT_ONLY_FOR_DECISION = TRUE**

**METRIC_GAIN_COVERAGE = PARTIAL**

本审计使用 Task 2 的 frame geometry、300 px support summary，以及 `calibration_points.csv` 中记录的逐点 laser centre UV；逐点 `lambda_truth` 重新按同一 PnP plane 与正式内参计算。FIT 001–018 才能影响 observability 分类；VALIDATION 019–024 只单独显示。

## Provenance and isolation

- Task 2 coverage directory：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\triplet_coverage_audit`。
- recorded UV source：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\laser_model\calibration_points.csv`；没有重新提取 Steger。
- formal measurement config / M0 path：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\measure_tool_daheng_0811.yaml`。
- formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。
- M0 comparison calls production `reconstruct_uv_to_ground()` only; no parameter, residual, weight, threshold, or candidate is optimized.
- bootstrap resampling unit is frame, not point; fixed seed and fixed replicate count are recorded in `triplet_local_gain_bootstrap.csv`.

## Fixed classification rules

分类优先级：
1. no points → `UNSUPPORTED`；
2. one frame → `SINGLE_FRAME_ONLY`；
3. frame-median u span < 10 px → `U_EXCITATION_WEAK`；
4. frame-median lambda span < 2 mm → `DEPTH_EXCITATION_WEAK`；
5. 至少 5 frames、u span ≥ 50 px、lambda span ≥ 5 mm、condition ≤ 1000、bootstrap relative interval ≤ 1 → `WELL_CONSTRAINED`；
6. 其余有至少 3 frames 的 bin → `SPARSE_BUT_INFORMATIVE`。
这些阈值是预先声明的几何/统计可观测性规则，不由量块、Cone residual 或 validation 结果反推。

## FIT classification summary

| classification | FIT bin count (all scales) | interpretation |
|---|---:|---|
| WELL_CONSTRAINED | 139 | 可稳定估计局部 gain |
| SPARSE_BUT_INFORMATIVE | 7 | 有跨帧激励，但斜率不够稳定/密集 |
| DEPTH_EXCITATION_WEAK | 0 | 跨帧 depth/lambda 变化不足 |
| U_EXCITATION_WEAK | 0 | 跨帧 u 基线不足 |
| SINGLE_FRAME_ONLY | 6 | 无法用跨帧关系约束 gain |
| UNSUPPORTED | 28 | 没有 FIT 点 |

## Top / middle / bottom

| region | scales present | strongest FIT evidence | gap interpretation |
|---|---|---|---|
| top_0_299 | 30px, 60px, 100px | best 30px: SPARSE_BUT_INFORMATIVE, frames=4, frame-u-span=212.1445, frame-lambda-span=61.5824, slope=0.296982 | 核心已有证据；需补采 0–240 UNSUPPORTED, 240–270 SINGLE_FRAME_ONLY |
| middle_300_2699 | 30px, 60px, 100px | best 30px: WELL_CONSTRAINED, frames=18, frame-u-span=215.4649, frame-lambda-span=62.6084, slope=0.298436 | 核心已有证据；需补采 2670–2700 SINGLE_FRAME_ONLY |
| bottom_2700_2999 | 30px, 60px, 100px | best 30px: SINGLE_FRAME_ONLY, frames=1, frame-u-span=0.0000, frame-lambda-span=0.0000, slope=n/a | 需补采 2700–2760 SINGLE_FRAME_ONLY, 2760–3000 UNSUPPORTED |

## Gain comparison with frozen M0

`slope_dlambda_du` 是按 frame median `(u,lambda_truth)` 的 Theil–Sen slope；`m0_point_derivative_median`（同 CSV 的 `m0_local_gain_dlambda_du`）是正式 M0 在同一 UV 处以 ±0.5 px 数值差分得到的 pointwise `d(lambda_M0)/du` 中位数；M0 没有被优化。

| scale | v region | truth slope median | truth bootstrap P05–P95 | M0 derivative median | truth–M0 slope | classification |
|---:|---|---:|---:|---:|---:|---|
| 30 | top (270–300) | 0.296982 | 0.281460–0.317004 | 0.306617 | -0.009635 | SPARSE_BUT_INFORMATIVE |
| 30 | middle (1650–1680) | 0.294901 | 0.288422–0.302938 | 0.295028 | -0.000127 | WELL_CONSTRAINED |
| 30 | bottom (2700–2730) | n/a | n/a–n/a | 0.263330 | n/a | SINGLE_FRAME_ONLY |
| 60 | top (240–300) | 0.297910 | 0.285857–0.317004 | 0.302022 | -0.004112 | SPARSE_BUT_INFORMATIVE |
| 60 | middle (1620–1680) | 0.294955 | 0.288135–0.302848 | 0.295061 | -0.000106 | WELL_CONSTRAINED |
| 60 | bottom (2700–2760) | n/a | n/a–n/a | 0.263333 | n/a | SINGLE_FRAME_ONLY |
| 100 | top (200–300) | 0.297910 | 0.281159–0.310837 | 0.302022 | -0.004112 | SPARSE_BUT_INFORMATIVE |
| 100 | middle (1600–1700) | 0.294974 | 0.288150–0.302455 | 0.295059 | -0.000085 | WELL_CONSTRAINED |
| 100 | bottom (2700–2800) | n/a | n/a–n/a | 0.263333 | n/a | SINGLE_FRAME_ONLY |

## Acquisition gaps

- **无需补采（当前证据已足够）：** 30 px 的 `WELL_CONSTRAINED` 核心为 `v=300–2610`，top `v=270–300` 已有跨帧基本激励；60/100 px 结果见下方精确区间表。
- **需要补采：** `SINGLE_FRAME_ONLY` 或 `UNSUPPORTED` 区间；首要缺口是覆盖这些 v 的新 frame，而不是在同一 frame 增加密集点。
- `DEPTH_EXCITATION_WEAK`：缺跨帧 board depth / lambda span；应增加不同棋盘距离或倾角的 frame。
- `U_EXCITATION_WEAK`：缺跨帧 median-u baseline；应让同一 v 区域在不同 pose 下横向落点分离。
- `SPARSE_BUT_INFORMATIVE`：已有基本 gain 方向，但 slope CI 或设计条件数不足；优先增加相邻 pose，而不是重复同一 pose 的密集点。
- `UNSUPPORTED`：该 v 区域 FIT 没有点；必须直接采集覆盖该 v 的棋盘+激光三联图。

## Exact FIT intervals and missing excitation

下表按每个 bin scale 合并连续区间；`SPARSE_BUT_INFORMATIVE` 不是硬缺口，但代表应优先补相邻pose以收窄 slope CI。

| scale | v interval | classification | 缺什么 |
|---:|---|---|---|
| 30px | 0–240 | `UNSUPPORTED` | 缺 frame；该区同时没有 u span 和 lambda span |
| 30px | 240–270 | `SINGLE_FRAME_ONLY` | 缺 frame；单帧无法形成跨frame u/lambda span |
| 30px | 270–300 | `SPARSE_BUT_INFORMATIVE` | 已有基本激励；增加相邻pose可降低斜率不确定度 |
| 30px | 2610–2670 | `SPARSE_BUT_INFORMATIVE` | 已有基本激励；增加相邻pose可降低斜率不确定度 |
| 30px | 2670–2760 | `SINGLE_FRAME_ONLY` | 缺 frame；单帧无法形成跨frame u/lambda span |
| 30px | 2760–3000 | `UNSUPPORTED` | 缺 frame；该区同时没有 u span 和 lambda span |
| 60px | 0–240 | `UNSUPPORTED` | 缺 frame；该区同时没有 u span 和 lambda span |
| 60px | 240–300 | `SPARSE_BUT_INFORMATIVE` | 已有基本激励；增加相邻pose可降低斜率不确定度 |
| 60px | 2640–2700 | `SPARSE_BUT_INFORMATIVE` | 已有基本激励；增加相邻pose可降低斜率不确定度 |
| 60px | 2700–2760 | `SINGLE_FRAME_ONLY` | 缺 frame；单帧无法形成跨frame u/lambda span |
| 60px | 2760–3000 | `UNSUPPORTED` | 缺 frame；该区同时没有 u span 和 lambda span |
| 100px | 0–200 | `UNSUPPORTED` | 缺 frame；该区同时没有 u span 和 lambda span |
| 100px | 200–300 | `SPARSE_BUT_INFORMATIVE` | 已有基本激励；增加相邻pose可降低斜率不确定度 |
| 100px | 2600–2700 | `SPARSE_BUT_INFORMATIVE` | 已有基本激励；增加相邻pose可降低斜率不确定度 |
| 100px | 2700–2800 | `SINGLE_FRAME_ONLY` | 缺 frame；单帧无法形成跨frame u/lambda span |
| 100px | 2800–3000 | `UNSUPPORTED` | 缺 frame；该区同时没有 u span 和 lambda span |

## Limits

- 局部线性/二次关系只是 observability diagnostic，不是补偿模型。
- 该审计证明的是局部 metric-scale 激励是否存在，不证明 Circular Cone 六参数良好可辨识，也不证明 M0 正确。
- VALIDATION 表格和图只用于独立显示，不能用于选择缺口、阈值或候选。
