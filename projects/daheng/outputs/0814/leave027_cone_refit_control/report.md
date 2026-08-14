# Task 4C — Leave-027-out Circular Cone refit control

**FIT_ONLY = TRUE**  
**VALIDATION_OPENED = FALSE**  
**PRODUCTION_M0_MODIFIED = FALSE**  
**CORRECTION_OR_PRIOR_ADDED = FALSE**

## 控制变量

- Baseline FIT：001–018 + 025–036（30帧）；diagnostic FIT 只临时排除027（29帧）。
- 完全复用 Task 3B-2 的 local parameterization、正式 Cone residual、固定采样、frame-equal weighting、soft_l1、bounds、x_scale、optimizer 和 evaluation grid。
- M0 作为初值且全程冻结；M_local_fullfit 直接读取 Task 3B-2 产物，没有重算或覆盖。
- 正式 Cone SHA-256 before/after：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac` / `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。
- 独立29帧 refit 与 Task 3B-2 既有 jackknife(027) 参数最大绝对差：`0`；cost差：`0`。

## 29帧优化结果

- status=`2`, success=`True`, message=``ftol` termination condition is satisfied.`。
- selected points=`2987`, objective cost=`0.8231963`, objective MSE=`0.000275593`。
- M_leave027 相对 M_local_fullfit 的 local normalized delta L2=`11.1763`。
- 冻结 evaluation grid 上 `lambda_leave-lambda_full`：P95=`0.01610403` mm，max=`0.0187389` mm。

### Cone参数变化（M_leave027 − M_local_fullfit）

| parameterization | parameter | delta | unit |
|---|---|---:|---|
| local | theta_axis | -0.003104679 | rad |
| local | phi_axis | -0.0002734854 | rad |
| local | c1 | -78.09978 | mm |
| local | c2 | -6.41444 | mm |
| local | rho_ref | 77.71512 | mm |
| local | q | 0.003060496 | cot(alpha) |
| legacy | A_x | -23.21067 | mm |
| legacy | A_y | 6.237596 | mm |
| legacy | A_z | -74.63733 | mm |
| legacy | alpha | -0.003059147 | rad |

这些参数变化大部分沿 Task 3B-2 已知的几何弱方向；物理影响应以 evaluation-grid 的 delta_lambda 为准，不能把远端 apex 数十毫米漂移直接解释成测量面移动数十毫米。

## 全部30帧 truth 上的 e_lambda 对照（point-equal）

`e_lambda = lambda_truth - lambda_model`。把027仍保留在评估集中，确保这里只改变拟合模型，不改变评估数据。

| region | model | bias / mm | RMSE / mm | P95 / mm |
|---|---|---:|---:|---:|
| global | M0 | -0.002292116 | 0.09917692 | 0.1637558 |
| global | M_local_fullfit | 0.002784108 | 0.09851705 | 0.1624624 |
| global | M_leave027 | 0.009148406 | 0.09923494 | 0.160284 |
| top_formal_edge | M0 | 0.1728811 | 0.2128032 | 0.4489313 |
| top_formal_edge | M_local_fullfit | 0.1950642 | 0.2315664 | 0.4731607 |
| top_formal_edge | M_leave027 | 0.1930043 | 0.2298081 | 0.4708694 |
| middle_formal | M0 | -0.002190166 | 0.09849961 | 0.1628701 |
| middle_formal | M_local_fullfit | 0.002619712 | 0.09794542 | 0.1612372 |
| middle_formal | M_leave027 | 0.009082202 | 0.09866299 | 0.1591785 |
| bottom_formal_edge | M0 | -0.1105182 | 0.1187904 | 0.1698429 |
| bottom_formal_edge | M_local_fullfit | -0.06920532 | 0.07698235 | 0.1147966 |
| bottom_formal_edge | M_leave027 | -0.07510036 | 0.08193253 | 0.1175739 |

## Edge判断

- Top 相对30帧 M_local_fullfit：RMSE `0.2315664` → `0.2298081` mm（`-0.759%`），P95 `0.4731607` → `0.4708694` mm；**仅有轻微数值改善，不能视为top residual被解决**。
- Bottom 相对30帧 M_local_fullfit：RMSE `0.07698235` → `0.08193253` mm（`+6.430%`），P95 `0.1147966` → `0.1175739` mm；**相对30帧模型变差**。
- Bottom 相对 M0 的 RMSE 为 `0.1187904` → `0.08193253` mm；Task 3B-2 的 bottom 改善**保留**。
- M_leave027 的 edge bias 仍为 top `0.1930043` mm、bottom `-0.07510036` mm；top+/bottom− 结构**仍存在**。

## 最终回答

1. **027 是否显著拉偏 Full-FIT Cone：是，有可测的参数与曲面影响。** 参数坐标变化需要结合 grid surface drift 解读，不能仅凭弱方向中的大参数漂移下结论。
2. **排除027后：top 只有不到1%的轻微数值改善，系统正偏仍在；bottom相对30帧模型变差，但相对M0的改善仍保留。**
3. **选择 B：027 有影响，但 top/bottom 系统残差仍存在。**

这是一项 FIT-only diagnostic control，不是删除027的授权，也不是 validation accuracy 结论。

Outputs: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\leave027_cone_refit_control`
