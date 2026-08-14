# Circular Cone FIT nonlinear outputs

**FIT_SURFACE_RESULT = SUCCESS**  
**PARAMETER_CONVERGENCE = UNRESOLVED_BOUNDARY**

| 文件 | 文件体现什么 | 主要看什么 | 不能得出什么 |
|---|---|---|---|
| cone_nonlinear_fit_trace.csv | 每个 exact trial 的半径、阻尼、预测/实际下降、接受状态 | 检查 optimizer 是否真实收敛 | 不含 validation |
| cone_nonlinear_fit_candidates.csv | Theta0 与三组实验 candidate/delta/scale | 查看参数移动和 weighting 差异 | 不是可部署 YAML |
| cone_nonlinear_fit_metrics.csv | 三 candidate × 三 metric weighting 的 global 指标 | 查看全局改善及交叉稳健性 | 不能证明 validation |
| cone_nonlinear_fit_regions.csv | top/middle/bottom、300px bins、外推区指标 | 查看边缘是否改善 | 仅 FIT |
| cone_nonlinear_fit_singular_values.csv | 最终 scaled Jacobian singular spectrum | 查看最终条件数和有效秩 | 不能单独解释物理参数 |
| cone_nonlinear_fit_coupling.csv | 最终 column cosine、covariance correlation、弱向量 | 查看 apex-alpha/axis-apex 耦合 | 不是参数置信区间 |
| cone_nonlinear_fit_jacobian_stability.csv | 最终点三尺度 finite-difference 复核 | 确认最终 Jacobian 数值稳定 | 不是全局模型验证 |
| cone_candidate_surface_consistency.csv | 不同 weighting candidates 的 FIT surface residual 差异 | 区分表面稳定与参数漂移 | 不含新姿态 |
| cone_nonlinear_fit_prediction.png | 残差-v、收敛轨迹、区域 RMSE | 组会主图 | 不含 validation |
| cone_nonlinear_fit_report.md | 方法、结果、辨识性与 FIT 决策门 | 本阶段主报告 | 不授权写回正式参数 |
| OUTPUT_FILES.md | 输出索引 | 快速导航 | 不增加证据 |
