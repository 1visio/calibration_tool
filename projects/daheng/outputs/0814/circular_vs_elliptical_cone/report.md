# Task 5A — Circular Cone vs Elliptical Cone 模型形式对照

**FIT_ONLY = TRUE**  
**VALIDATION_OPENED = FALSE**  
**PRODUCTION_M0_MODIFIED = FALSE**  
**EMPIRICAL_CORRECTION_ADDED = FALSE**

`MODEL_FORM_RESULT = C. Elliptical improves but仍不足`

## 模型与公平性约束

- 主拟合：001–018 + 025–036，临时排除027，共29帧；027仅在两个冻结候选上单独测试。
- 两模型统一最小化 `r_lambda=lambda_model-lambda_truth`，使用相同frame-balanced sampling、frame-equal weighting、soft_l1、f_scale、least_squares和max_nfev。
- 两模型共用正nappe、固定相机深度[100,1500] mm和固定truth-domain z hint选根；拟合时不使用候选模型各自的z_valid_range过滤，避免产生不同objective样本集。
- Circular参数化：axis line + apex axial location + one circular slope，共6 DOF。
- Elliptical使用严格二次锥 `x_perp^T H x_perp-axial^2=0`，H为2×2 SPD；共8 DOF。Circular是其嵌套子模型 `H=q^2 I`，只放宽圆对称的两个DOF。
- 没有b(v)、spline、polynomial、LUT或任何u/v位置项。
- Circular诊断求交与正式reconstruct公共有效点最大lambda差=`4.092726e-12` mm，1e-6 gate=`True`。
- 正式M0 SHA-256 before/after：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac` / `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。

## Full-FIT optimizer

| model | DOF | success | robust cost | raw MSE | selected | nfev |
|---|---:|---|---:|---:|---:|---:|
| Circular | 6 | True | 6.55977 | 0.005923659 | 2987 | 223 |
| Elliptical | 8 | True | 6.054675 | 0.005342886 | 2987 | 543 |

- Circular half angle=`88.87412` deg。
- Elliptical principal half angles=`89.95` / `89.75777` deg，transverse orientation=`-177.7682` deg。
- Elliptical是否命中89.95°主半角硬边界：`True`。命中边界意味着该方向接近退化，不能仅凭Full-FIT训练改善认定模型形式已充分。

## 主29帧 frame-equal 指标

| region | model | Bias mm | RMSE mm | P95 mm | RMSE change vs Circular |
|---|---|---:|---:|---:|---:|
| global | Circular | 0.0006757738 | 0.07452205 | 0.1432403 | +0.000% |
| global | Elliptical | 0.001380244 | 0.07058411 | 0.1370377 | -5.284% |
| top_formal_edge | Circular | -0.183165 | 0.2283767 | 0.4835225 | +0.000% |
| top_formal_edge | Elliptical | -0.09492984 | 0.1645084 | 0.3965589 | -27.966% |
| middle_formal | Circular | 0.001076434 | 0.07323242 | 0.1419128 | +0.000% |
| middle_formal | Elliptical | 0.001020735 | 0.0691387 | 0.134981 | -5.590% |
| bottom_formal_edge | Circular | 0.06741053 | 0.07702411 | 0.12869 | +0.000% |
| bottom_formal_edge | Elliptical | 0.1300844 | 0.1381151 | 0.2035654 | +79.314% |

## 027 sensitivity（不参与模型选择）

| model | Bias mm | RMSE mm | P95 mm |
|---|---:|---:|---:|
| Circular | -0.2949384 | 0.3667674 | 0.5762695 |
| Elliptical | -0.2894576 | 0.3605891 | 0.5640244 |

## Jackknife稳定性与最终判断

- Elliptical 29折 prediction-grid P95 median/max=`0.004807896` / `0.009161749` mm；失败数=`0`。
- Circular normalized parameter-delta L2 median/max=`6.474023` / `22.42215`；Elliptical=`0.002387186` / `0.2799526`。参数值及逐参数delta见 `frame_cv_comparison.csv`。
- Elliptical参数delta较小需要结合主半角命中硬边界理解；边界会限制该方向漂移，因此模型稳定性主要以prediction-grid drift和held-out RMSE判断。
- 配对heldout RMSE改善中位数=`5.088%`；改善frame数=`21/29`。
- Top RMSE改善=`27.966%`，Bottom=`-79.314%`，Middle=`5.590%`，top/bottom bias asymmetry改善=`10.201%`。

1. Top是否明显下降：`True`。
2. Bottom是否同步改善：`False`。
3. Middle是否基本不恶化：`True`。
4. top+/bottom− asymmetry是否明显减弱：`False`。
5. Elliptical改善在frame jackknife中是否稳定：`True`。

**最终：`MODEL_FORM_RESULT = C. Elliptical improves but仍不足`。**

该结果只授权下一步研究判断，不授权部署Elliptical Cone，也不是Validation结论。

### 固定判据

- B：top RMSE reduction>=20%, bottom>=10%, middle degradation<=5%, asymmetry reduction>=25%, jackknife median heldout reduction>=10%。
- C：at least one global/top/bottom RMSE reduction>=10% and jackknife median heldout reduction>0, but B is not met。
- A：stable comparison with no material improvement gate met。
- D：fit/jackknife failure, incomplete pairing, or elliptical max jackknife grid P95>0.10mm。

Outputs: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\circular_vs_elliptical_cone`
