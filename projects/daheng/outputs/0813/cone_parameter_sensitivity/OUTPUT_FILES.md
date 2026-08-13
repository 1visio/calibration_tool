# Circular Cone sensitivity outputs

**PARAMETER_ERROR_CAN_EXPLAIN = PARTIAL**

本目录只有线性化 sensitivity / identifiability 产物；没有候选 Cone YAML，也没有非线性重优化结果。

| 文件 | 文件体现什么 | 主要看什么 | 能得出什么 | 不能得出什么 | 是否适合组会 |
|---|---|---|---|---|---|
| cone_parameter_sensitivity.csv | Theta0、三组 DeltaTheta/scale、Jacobian 列 norm、selected step 与精确 invalid indices | 增量是否跨越多个解释尺度；三 weighting 是否给出一致方向 | 参数变化量与局部一阶敏感度 | 非线性新参数是否有效或可发布 | 是，建议放参数增量摘要 |
| cone_jacobian_singular_values.csv | 三种加权 FIT Jacobian 的 raw/scaled singular values、rank、condition | 最小奇异值比例、effective rank、condition number | 局部可辨识维数和近退化强度 | 具体哪个物理参数单独可信 | 是，适合一页谱图/表 |
| cone_parameter_coupling.csv | column cosine、covariance correlation、最小两个右奇异向量 | apex-alpha 与 axis-apex 的高相关和最小向量组成 | 局部耦合方向 | 因果归因或全局参数唯一性 | 附录；组会被追问时使用 |
| cone_residual_explainability.csv | FIT/VALIDATION global 的 before/after 指标与 explained energy | validation 是否保持正 explained fraction | 冻结线性预测的全局解释力 | 真实非线性重优化效果 | 是，核心结果表 |
| cone_region_explainability.csv | top/middle/bottom、每300px bin、0811外推区的三 weighting 指标 | 边缘 after RMSE、explained fraction、invalid count | 边缘结构是否落在参数切空间 | Cone 模型形式一定充分/不足 | 是，建议筛选关键区域 |
| cone_linearized_prediction.png | FIT/VALIDATION 的 baseline 与三组冻结线性预测 residual-v 曲线 | validation 边缘是否同步靠近0 | 空间结构改善是否跨 split | 点级误差分布和非线性有效性 | 是，主图 |
| cone_singular_values.png | 三种 weighting 的相对 scaled singular spectrum | 谱尾是否塌陷、weighting 是否改变可辨识性 | 近退化是否稳健存在 | 参数误差能解释多少 residual | 是，可辨识性主图 |
| cone_sensitivity_report.md | 方法、隔离、step 稳定性、SVD、耦合、区域解释力与最终判定 | 首行判定及其 validation/edge 证据 | 本轮完整线性诊断结论 | 正式参数变更建议的最终批准 | 是，组会讲稿底稿 |
| OUTPUT_FILES.md | 九个产物的阅读索引和边界 | 按问题快速定位主文件 | 每个产物适用范围 | 任何新增科学证据 | 是，作为入口页 |

建议组会顺序：`cone_linearized_prediction.png` → `cone_residual_explainability.csv` → `cone_singular_values.png` → 报告中的 coupling/edge 表。
