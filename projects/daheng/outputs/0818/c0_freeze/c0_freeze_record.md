# C0 Freeze Record

`C0_FREEZE_STATUS = FROZEN`

## 1. 冻结结论

当前阶段性生产候选冻结为：

`FINAL_C0_STATUS = QUADRATIC`

冻结对象是 Full-36 FIT 上已经生成的 `quadratic_graph` 参数，不是重新拟合的副本。冻结目录中的 `quadratic_graph.yaml` 与 0817 正式 Full-36 Quadratic source YAML 的 SHA-256 相同：

`113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27`

本次只建立可追溯记录和同内容的 YAML copy；没有重新拟合 Quadratic/Cone，没有训练 C1，没有读取或修改原始 FIT/Validation，也没有修改正式生产测量配置。

## 2. 冻结输入与协议

### Full-36 FIT

FIT pose IDs 为：

`001–018, 025–036, 049–054`

共 36 poses、32400 FIT points。正式几何域使用：

- mask：`full_board_physical`
- mask inset：`0 mm`
- laser orientation：`vertical`
- weighting：`frame-balanced`
- operational v domain：`0–3000 px`
- v-bin：100 px

### Intrinsics 与 extraction protocol

- Intrinsics：`projects/daheng/outputs/0811/intrinsics/calibration_result.yaml`
- Intrinsics SHA-256：`d162d581ffd12df510b15e4edd42536a97abb4dc7d883352b0be76cb8c65f9b0`
- Formal extraction config：`configs/laser_model_fit_config.daheng.yaml`
- Extraction config SHA-256：`2241737d68276dbdfb226f5285b7ae77dad07be2eaef02ed66966d6b6206cebf`
- Steger：`sigma=1.5`、`min_intensity=8.0`、`min_response=0.8`、`max_subpixel_offset=0.60`
- Continuity：二次多项式、`threshold=2.0 px`
- 每图点数限制：`min=80`、`max=900`

上述 protocol 字段与哈希已写入 `c0_freeze_manifest.json`；完整输入审计见 `artifact_provenance.csv`。

## 3. 决策链

### 3.1 mask 修正与几何域

前序几何审核将正式板域固定为 `full_board_physical`、`inset=0 mm`，以保持完整物理板面和完整 v 工作域的可观测性。Curated-14 与 Robust-18 只用于几何代表性和安全余量检查；没有使用 Plane/Quadratic/Cone residual 删除 pose。

### 3.2 Full-36 与 FIT 集决策

几何审核首先固定 Curated-14，再加入 4 个几何互补 pose 形成 Robust-18；随后对 Curated-14、Robust-18、Full-36 做同协议 pose-grouped CV。现有比较记录推荐 `FULL_36`：

- Quadratic Global RMSE：Curated-14 `0.12292` → Robust-18 `0.11073` → Full-36 `0.09909` mm；
- Cone Global RMSE：Curated-14 `0.12219` → Robust-18 `0.11118` → Full-36 `0.10089` mm；
- Full-36 提供最低 pooled global error，并保留完整 FIT 几何多样性。

因此冻结的 C0 来源明确为 Full-36，而不是 Curated-14 或 Robust-18。

### 3.3 三模型与 Full-36 grouped CV

Full-36 既有 6-fold pose-grouped CV 直接复用。Plane 不进入当前生产候选；其结果不用于本冻结决定。

Full-36 的 pooled CV 为：

| model | Global RMSE / mm | Global P95 / mm | worst-v RMSE / mm | v-bias range / mm | fold RMSE std / mm |
|---|---:|---:|---:|---:|---:|
| Quadratic | 0.09909 | 0.18438 | 0.17423 | 0.28363 | 0.02463 |
| Cone | 0.10089 | 0.19054 | 0.20120 | 0.31599 | 0.02411 |

Grouped CV 只说明 Quadratic 有轻微 pooled 优势；由于原有预设差异阈值和跨 pose 稳定性条件未完全满足，单靠 grouped CV 时状态仍记录为 `C0_MODEL_STATUS = UNRESOLVED`。

### 3.4 配对 grouped/paired CV

Q/C 使用相同 held-out pose、fold、frame/point identity 的 pointwise predictions 做配对检查，没有重新拟合：

- pose RMSE wins：Quadratic `17`，Cone `19`；
- 100 px v-bin wins：Quadratic `23`，Cone `7`；
- Global RMSE Δ(Q−C)：`-0.00180 mm`；
- pose bootstrap 95% CI 跨 0；
- common worst-region 的 RMSE/P95 bootstrap CI 也跨 0。

所以 paired CV 记录为 `C0_PAIRED_STATUS = UNRESOLVED`。这表示 Quadratic 是轻微候选优势，不把 CV 结果误写成跨 pose 的确定性证明。

### 3.5 Cone sampling 与 operational surface

Cone 的 `fit_max_points=3000` 结果直接复用；6000、12000、all-feasible 的敏感性审计保持其余配置、fold、frame-balanced weighting 和初值策略不变。

- 3000 → all-feasible Global RMSE：`0.10089 → 0.10081 mm`；
- Global P95：`0.19054 → 0.19058 mm`；
- 参数路径出现 axis/apex/half-angle 漂移，sampling status 为 `CONE_SAMPLING_STATUS = UNSTABLE`；
- 在 Full-36 operational ray/v domain 上，各档相对 3000 的 Δlambda 均满足既有诊断 equivalent gates，记录为 `CONE_SURFACE_STATUS = EQUIVALENT`。

结论是：增加 Cone sampling 没有带来足以改变模型选择的 operational surface 改善；Cone 仍保留为未选候选，不回写为 C0。

### 3.6 独立 Validation：Primary 优先

Q/C 参数均冻结自 Full-36 YAML。Validation 只做 frozen prediction/evaluation，不重新拟合、不调参。Primary 055–060 是最终模型选择依据：

| model | Bias / mm | RMSE / mm | P95 / mm | Max / mm | worst-v RMSE / mm | worst-v P95 / mm | pose RMSE wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quadratic | -0.014469 | 0.094036 | 0.196720 | 0.461213 | 0.219383 | 0.262694 | 3/6 |
| Cone | -0.028302 | 0.098009 | 0.197712 | 0.552490 | 0.180442 | 0.275984 | 3/6 |

Primary 同点配对结果：

- ΔGlobal RMSE (Q−C)：`-0.003973 mm`
- ΔGlobal P95 (Q−C)：`-0.000992 mm`
- Quadratic absolute-error better fraction：`0.591667`
- 汇总 6 项指标投票：Quadratic `5`，Cone `1`

Quadratic 在 Primary 的 global RMSE/P95/Max、absolute bias 和 worst-v P95 更低；Cone 仅在 worst-v RMSE 更低，pose RMSE 为 3:3。按预先固定的 Primary-only 规则，最终选择 Quadratic。该 Primary 结论不由 Historical 或 pooled 结果事后改变。

Historical regression 与 all-validation 仅作一致性检查：

| scope | model | Bias / mm | RMSE / mm | P95 / mm | Max / mm |
|---|---|---:|---:|---:|---:|
| 019–024 | Quadratic | -0.005182 | 0.077971 | 0.145219 | 0.259251 |
| 019–024 | Cone | -0.006625 | 0.081179 | 0.148741 | 0.262116 |
| 037–040 | Quadratic | -0.060924 | 0.099032 | 0.190544 | 0.385406 |
| 037–040 | Cone | -0.069277 | 0.103475 | 0.206741 | 0.425742 |
| All 16 poses | Quadratic | -0.022600 | 0.089709 | 0.181801 | 0.461213 |
| All 16 poses | Cone | -0.030417 | 0.093543 | 0.186812 | 0.552490 |

这些历史/pooled 数值不作为覆盖 Primary 决策的第二投票。

## 4. Cone 与旧 Cone-C1 的最终状态

`circular_cone` 是已评估但未选中的 C0 candidate：

`CONE_FINAL_STATUS = UNSELECTED_CANDIDATE`

它的 Full-36 grouped CV、paired stability、sampling sensitivity 和 operational surface 结果继续作为可追溯参考；不覆盖冻结的 Quadratic。

旧 Cone-C1 artifact 的状态为：

`OLD_CONE_C1_STATUS = HISTORICAL_REFERENCE_ONLY`

旧 Cone-C1 不作为当前 C0 来源，不作为当前补偿基准，也不重新训练。

## 5. 后续补偿接口

后续补偿统一定义为：

`lambda_final = lambda_quadratic + delta_lambda`

其中 `lambda_quadratic` 必须来自本 freeze record 绑定的 Frozen Full-36 Quadratic，`delta_lambda` 是独立、版本化、另有 provenance 的补偿项。旧 Cone-C1 仅作历史参考，不能替代 `lambda_quadratic` 或隐式改变本冻结记录。

## 6. 可追溯性与变更规则

- Frozen YAML：`c0_freeze/quadratic_graph.yaml`
- Freeze manifest：`c0_freeze/c0_freeze_manifest.json`
- Provenance audit：`c0_freeze/artifact_provenance.csv`
- Source Full-36 Quadratic：`outputs/0817/grouped_cv_model_comparison/candidate_models/full_fit/quadratic_graph.yaml`
- Source Quadratic SHA-256 与 freeze copy SHA-256 相同，均为 `113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27`

任何模型参数、FIT 集、mask、intrinsics、extraction protocol 或补偿定义的变化，都必须新建版本化 freeze record；不得静默覆盖本目录中的冻结文件。

`C0_FREEZE_STATUS = FROZEN`
