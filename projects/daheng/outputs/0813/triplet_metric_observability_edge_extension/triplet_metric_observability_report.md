# Combined edge-extension metric-scale / height-gain observability audit

**EDGE_COVERAGE_WITHIN_FORMAL_DOMAIN = CLOSED**  
**CAN_ENTER_NEXT_STEP = YES**  
**NO_CONE_FIT = TRUE**

## Scope and provenance

- 原始 Task 2 coverage：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\triplet_coverage_audit`；原始 FIT 001–018 = 16102 points，原始 VALIDATION 019–024 = 5400 points。
- 新 FIT extension：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\fit_edge_extension`，frame 025–036 = 10800 points。
- 新 validation holdout：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\validation_edge_holdout`，frame 037–040 = 3600 points；只作最终独立评价。
- 合并后 FIT frames：001,002,003,004,005,006,007,008,009,010,011,012,013,014,015,016,017,018,025,026,027,028,029,030,031,032,033,034,035,036；合并后 validation frames：019,020,021,022,023,024,037,038,039,040。
- 新图像使用 recorded `frames.csv` 与 manifest SHA；validation manifest 的 split tag 误写为 `fit`，本审计按用户指定目录角色将 037–040 固定为 validation，并在 provenance CSV 保留该不一致。
- Formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。
- UV 使用正式 Steger 配置重新提取；PnP + ray-plane truth 独立计算；M0 只调用 production `reconstruct_uv_to_ground()`，没有优化或写回。

## Combined coverage summary

| split | frames | points | v bin envelope (30px) | interpretation |
|---|---:|---:|---|---|
| FIT | 30 | 26902 | 180.000–2900.000 | used for observability decision |
| VALIDATION | 10 | 9000 | 200.000–2800.000 | frozen holdout only |

新 FIT extension 的实际 laser UV v 范围为 [238.984, 2874.006] px；新 validation holdout 为 [267.007, 2744.985] px。
正式 0811 FIT 的原始 v 工作域为 [241.998, 2731.978] px；本报告把 FIT 的‘edge closed’定义为该正式工作域内 top `[v_min,300)` 与 bottom `(2700,v_max]` 的 30px bins 均有跨帧可用激励。
validation status 只作冻结 holdout 诊断，不参与 edge-closure 判定；该定义也不宣称整个相机 v=0–2999 都有数据，工作域之外仍单独列为 full-sensor gap。

## Edge closure

| edge | FIT status | FIT frames | validation status | validation frames |
|---|---|---|---|---|
| top | CLOSED | 004,014,015,018,025,028,029,030 | OPEN | 019,038,039 |
| bottom | CLOSED | 005,026,031,032,033,035,036 | OPEN | 037,040 |

FIT 30px edge classifications outside/around the formal domain:

| scale | v interval | classification |
|---:|---|---|
| 30px | 0–210 | `UNSUPPORTED` |
| 30px | 2880–3000 | `UNSUPPORTED` |
| 30px | 210–240 | `SINGLE_FRAME_ONLY` |
| 30px | 2790–2880 | `SINGLE_FRAME_ONLY` |
| 30px | 240–270 | `SPARSE_BUT_INFORMATIVE` |
| 30px | 2760–2790 | `SPARSE_BUT_INFORMATIVE` |
| 60px | 0–180 | `UNSUPPORTED` |
| 60px | 2880–3000 | `UNSUPPORTED` |
| 60px | 180–240 | `SINGLE_FRAME_ONLY` |
| 60px | 2820–2880 | `SINGLE_FRAME_ONLY` |
| 60px | 2760–2820 | `SPARSE_BUT_INFORMATIVE` |
| 100px | 0–200 | `UNSUPPORTED` |
| 100px | 2900–3000 | `UNSUPPORTED` |
| 100px | 2800–2900 | `SINGLE_FRAME_ONLY` |

## Previous result versus combined result

- 原始结果为 `PARTIAL`：中部可观测，但 top 仅稀疏、bottom 单帧/无数据。
- 合并后新增 025–036 为 FIT，显著增加了边缘的跨帧 `u–lambda` 激励；037–040 在两个正式边缘工作域提供独立 holdout。
- 在正式 0811 工作域内，FIT top 与 bottom 均达到 CLOSED；validation holdout 作为冻结评价集保留独立覆盖状态，不参与该 FIT 决策。其 top 的 240–270 bin 与 bottom 的 2730–2760 bin 仍是单帧，不能宣称 holdout 自身全边缘闭合。

## Local gain and M0

中部与边缘的 `slope_dlambda_du`、frame bootstrap P05/P50/P95、design condition、M0 local gain 均在 `triplet_metric_observability.csv`；validation 行不参与任何 FIT 决策。
M0 ±0.5px 差分在合并点中有 243 个 derivative 无效；无效点未被静默删除，truth observability 仍保留，M0 仅在有效点上报告。

## Acquisition conclusion

- 正式工作域内的 FIT 已不需要继续为‘是否有边缘多帧深度激励’补采；可以进入冻结数据上的 sensitivity / local reoptimization 设计。validation 037–040 仍只用于最终冻结预测评价。
- 若目标扩大为整个 0–2999 sensor height，则仍需补采 top 0–239 与 bottom 2874–2999；这属于扩展工作域，不是当前 0811 formal-domain closure 的阻塞项。
- 进入下一步时仍不得把 validation 037–040 用于求 DeltaTheta、选权重或调阈值；它们只用于最终冻结预测评价。

## Limits

- 这是局部几何可观测性审计，不是 Cone 非线性拟合，也不证明新增数据一定降低实际 residual。
- 新采集 laser 图像 manifest 标记有 dynamic_range_low 等质量 warning；Steger/PnP 均成功，但应在后续 sensitivity 后继续保留该质量 provenance。
