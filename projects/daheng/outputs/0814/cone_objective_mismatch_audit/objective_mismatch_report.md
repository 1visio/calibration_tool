# Task 4D — Circular Cone objective mismatch audit

**FIT_ONLY = TRUE**  
**VALIDATION_OPENED = FALSE**  
**MODEL_REFIT = FALSE**  
**PRODUCTION_M0_MODIFIED = FALSE**  
**CORRECTION_ADDED = FALSE**

`OBJECTIVE_MISMATCH = WEAK`

## 定义与边界

- 主分析：FIT 001–018、025–036，临时排除027，共29帧；027只作为 sensitivity case 单列。
- 模型：正式 M0 与 Task 4C 的 M_leave027，二者均冻结；没有重新优化。
- `e_surface = radial/tan(alpha) - axial`，与原 Circular Cone fit objective 的标量surface residual一致。
- `e_lambda = lambda_cone - lambda_truth`，lambda为正式 reconstruction 返回的相机Z。
- 相机射线使用 `k=[x_n,y_n,1]`，其中 `(x_n,y_n)=cv2.undistortPoints(u,v)`。
- 解析局部预测：`e_lambda_linear = -e_surface/(grad(F)·k)`；几何放大使用 `|dZ/d(surface distance)|=|grad(F)|/|grad(F)·k|`。
- 固定异常诊断阈值：`|e_surface|≤0.02` mm 且 `|e_lambda|≥0.1` mm；阈值未按结果调整。
- truth点与正式ray的最大一致性误差：`0` mm。
- 正式 M0 SHA-256 before/after：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac` / `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。

## 主分析：29帧 frame-equal 区域统计

| model | region | surface RMSE mm | depth RMSE mm | depth/surface | corr(surface,depth) | geom amp median/P95 | linear prediction explained | small-surface large-depth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | global | 0.0229541 | 0.07522149 | 3.277039 | -0.9993636 | 3.297084/3.437 | 1 | 0 (0.000%) |
| M0 | top_formal_edge | 0.06343223 | 0.2024929 | 3.192272 | -0.9994193 | 3.346599/3.444528 | 1 | 0 (0.000%) |
| M0 | middle_formal | 0.02257287 | 0.0739765 | 3.27723 | -0.9993558 | 3.294469/3.436996 | 1 | 0 (0.000%) |
| M0 | bottom_formal_edge | 0.03459166 | 0.1158873 | 3.350151 | -0.9989496 | 3.329626/3.434638 | 1 | 0 (0.000%) |
| M_leave027 | global | 0.02250621 | 0.07372178 | 3.275619 | -0.9993866 | 3.297632/3.437781 | 1 | 0 (0.000%) |
| M_leave027 | top_formal_edge | 0.0683482 | 0.2185625 | 3.197779 | -0.9993052 | 3.346932/3.444998 | 1 | 0 (0.000%) |
| M_leave027 | middle_formal | 0.02215955 | 0.07260908 | 3.276649 | -0.9993846 | 3.294963/3.437777 | 1 | 0 (0.000%) |
| M_leave027 | bottom_formal_edge | 0.0242415 | 0.08062354 | 3.325848 | -0.9985214 | 3.326005/3.431335 | 1 | 0 (0.000%) |

- 负相关符号来自本任务定义 `e_lambda=cone-truth`；surface residual 的正方向与求交深度误差相反。相关系数绝对值才表示一致程度。
- 解析一阶预测的global误差RMSE：M0=`9.162659e-09` mm，M_leave027=`4.646311e-09` mm，说明局部几何公式与正式求交数值一致。
- M0 top/middle surface RMSE比=`2.81011`，对应depth RMSE比=`2.73726`；M_leave027分别为 `3.08437` / `3.01013`。边缘增大在surface residual中已经存在。
- 正式求交无效点：M0=`238`，M_leave027=`239`；逐点CSV保留这些行，没有静默删除。

## 027 sensitivity case（不进入主结论）

| model | surface RMSE mm | depth RMSE mm | corr | geom amp median/P95 | linear prediction explained |
|---|---:|---:|---:|---:|---:|
| M0 | 0.1105837 | 0.3613049 | -0.9996166 | 3.205379/3.327125 | 1 |
| M_leave027 | 0.113502 | 0.3707193 | -0.9995956 | 3.206551/3.327226 | 1 |

## 判断

- 两模型的最大 edge/middle 中位几何放大比为 `1.01582`；固定small-surface/large-depth条件的最大frame-weighted比例为 `0.000%`。
- M0与M_leave027是否给出相同相关性等级：`True`；两模型分类细节写入报告末尾的判据说明。
- M0 global surface/depth Pearson=`-0.9993636`，M_leave027=`-0.9993866`。

## 最终回答

1. **原 surface objective 与最终 ray-depth 精度：在当前FIT工作域内基本一致。**
2. **边缘大误差是否属于几何误差放大：不是主要解释。**
3. **下一步选择 C：objective mismatch 不足以解释残差，需要继续研究模型形式。**

027只用于独立sensitivity展示，没有混入上述判定。该结论是FIT-only objective诊断，不是Validation或部署结论。

### 固定分类规则

- STRONG：edge/middle median amplification >=1.5 and small-surface/large-depth weighted fraction >=1%。
- PARTIAL：edge/middle median amplification >=1.2 or small-surface/large-depth weighted fraction >=0.1%。
- WEAK：neither gate is met。

M0 alpha=`89.07255` deg；M_leave027 alpha=`88.71046` deg。

Outputs: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\cone_objective_mismatch_audit`
