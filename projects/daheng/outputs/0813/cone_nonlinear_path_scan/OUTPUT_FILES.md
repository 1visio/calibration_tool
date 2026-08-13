# Circular Cone nonlinear path-scan outputs

**PATH_SCAN_DECISION = DAMPED_RELINEARIZATION_REQUIRED**

| 文件 | 内容 | 主要用途 | 边界 |
|---|---|---|---|
| cone_nonlinear_path_scan.csv | 三方向各 t 的 exact/linear global 指标、invalid、bounds 和参数值 | 检查真实 loss 路径及线性化误差 | 不是 optimizer trace |
| cone_nonlinear_path_regions.csv | top/middle/bottom、每300px bin、外推区的 exact path 指标 | 检查改善是否只来自中部 | 仅 FIT |
| cone_nonlinear_path_scan.png | exact 与 linear RMSE 路径及 nonlinearity gap | 组会主图 | 不含 validation |
| cone_nonlinear_path_scan_report.md | 隔离、关键数值、区域表现与决策门结论 | 本步骤主报告 | 不证明新参数可发布 |
| OUTPUT_FILES.md | 文件索引 | 快速导航 | 不增加科学证据 |
