# Task 3B-2 — Local parameterization Full-FIT 稳定性验证

**FIT_ONLY = TRUE**
**VALIDATION_OPENED = FALSE**
**PRODUCTION_CONE_MODIFIED = FALSE**

## 数据与流程

- FIT: 001–018 + 025–036，共 30 frame；Validation 019–024 + 037–040 未读取。
- 使用与 Task 3A 相同的 FIT sampling、frame-equal weighting、formal Cone residual、soft_l1、evaluation grid 和 v 工作域。
- M0 先通过 Task 3B-1 的 `legacy_to_local()` 转换作为局部优化初值；local residual 每次转换回 legacy 后直接调用正式 CircularConeModel residual。
- Formal Cone SHA-256 before/after: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac` / `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。

## 1. Full-FIT 结果

- `M_local_fullfit` status=`2`, success=`True`, objective cost=`1.38594559`。

| region | M0 RMSE / mm | M_local_fullfit RMSE / mm | M0 P95 / mm | M_local P95 / mm |
|---|---:|---:|---:|---:|
| global | 0.03040761 | 0.03024313 | 0.05039031 | 0.0502611 |
| top_formal_edge | 0.06753284 | 0.0734948 | 0.1428585 | 0.1505639 |
| middle_formal | 0.03016235 | 0.03001722 | 0.05009512 | 0.04976173 |
| bottom_formal_edge | 0.03566015 | 0.02317064 | 0.04974798 | 0.03364559 |

## 2. Jacobian / SVD 对照

- Legacy condition number: `133578.427`；local condition number: `13774.6636`；改善倍数（legacy/local）：`9.6974`。
- Legacy effective rank: `6/6`；local effective rank: `6/6`。
- Local weakest normalized loading: `theta_axis=-0.0177, phi_axis=-0.0010, c1=-0.6978, c2=-0.0369, rho_ref=+0.6932, q=+0.1759`。
- Local weakest direction mapped back to legacy has apex/alpha normalized norm `0.720924`；映射结果见 `local_fullfit_result.json`。
- Local SVD uses physical interpretation scales `[1°, 1°, 10 mm, 10 mm, 10 mm, dq(0.1°)]`, where `dq = |d cot(alpha)/d alpha| * 0.1°` at the local solution; this is column scaling only, not regularization.

## 3. Frame jackknife

- Local leave-one-FIT-frame-out count: `30`。
- Legacy max grid P95: `0.01735861` mm；local max grid P95: `0.01735644` mm。
- Legacy median grid P95: `0.006580907` mm；local median grid P95: `0.006578855` mm。
- 沿 v 的 local/legacy prediction drift 见 `local_jackknife_prediction_vs_v.csv` 及对应图；这只是 FIT stability，不是 validation accuracy。

## 4. 对问题的回答

1. condition number 改善：从 `133578` 到 `13774.7`，改善倍数 `9.6974`。
2. apex–alpha 弱方向：local 坐标中不再显式出现 apex/alpha，但映射回 legacy 后的 apex/alpha loading norm 为 `0.720924`；因此应以 local SVD 与 mapped physical direction 一起判断，而不是宣称几何弱方向已经消失。
3. local 参数稳定性：local normalized jackknife delta L2 median/max=`2.15242`/`12.8725`，legacy 为 `1.55577`/`9.28764`；因此 local 的条件数改善没有转化为所有参数坐标上的 jackknife 缩小。完整参数表见 `local_frame_jackknife.csv`。
4. surface prediction：local 与 legacy 均保持同一量级，局部坐标转换没有改变曲面几何；jackknife 差异由优化参数稳定性产生。
5. top-edge residual：仍存在；本任务没有添加任何 correction，也没有解决它。

## 5. Gate

可以停止在本任务；不自动进入 residual compensation。下一步只能在人工确认后进行局部参数化的进一步对照研究。

Outputs: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\cone_local_fullfit`
