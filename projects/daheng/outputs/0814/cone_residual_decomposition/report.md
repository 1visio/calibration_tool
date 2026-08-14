# Task 4A — FIT-only Circular Cone 位置相关残差分解

**FIT_ONLY = TRUE**
**VALIDATION_OPENED = FALSE**
**PRODUCTION_CONE_MODIFIED = FALSE**

## 数据与定义

- FIT: 001–018 + 025–036，共 30 frame；Validation 019–024 + 037–040 未读取。
- 主分析模型：Task 3B-2 的 `M_local_fullfit`；M0 仅作参考。
- `e_lambda = lambda_truth - lambda_cone`；lambda_truth 为独立 PnP ray-plane truth 的相机 Z。
- 固定使用 30/60/100 px 三种 v-bin；bin 内按 frame 平衡拟合 `e = b + delta_g*(u-u_ref)`，bootstrap 以 frame 为重采样单位（1000 次，seed=4101）。
- Formal Cone SHA-256 before/after: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac` / `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。

## 全局与区域 residual

| model | region | bias / mm | RMSE / mm | P95 / mm | frame mean std / mm |
|---|---|---:|---:|---:|---:|
| M0 | top_formal_edge | 0.1728811 | 0.2128032 | 0.4489313 | 0.1383313 |
| M0 | middle_formal | -0.002190166 | 0.09849961 | 0.1628507 | 0.06548886 |
| M0 | bottom_formal_edge | -0.1105182 | 0.1187904 | 0.1662461 | 0.0480466 |
| M_local_fullfit | top_formal_edge | 0.1950642 | 0.2315664 | 0.4731607 | 0.1394403 |
| M_local_fullfit | middle_formal | 0.002619712 | 0.09794542 | 0.1612345 | 0.06529509 |
| M_local_fullfit | bottom_formal_edge | -0.06920532 | 0.07698235 | 0.1127965 | 0.02976051 |

## 诊断回答

1. **Top residual：** `M_local_fullfit` top bias=`0.195064` mm、RMSE=`0.231566` mm；60 px top bin 的 b=`0.149286` mm，|delta_g|=`0.000658431` mm/px，在该 bin 的 u span 上对应约 `0.165467` mm 的 gain 变化。按 60 px bin 的能量分解，offset-only 解释 `0.6717`，加入 gain 后为 `0.7235`；因此 top 以 offset 为主，但 gain 不是可忽略项。
2. **Bottom：** bias=`-0.0692053` mm、RMSE=`0.0769823` mm；60 px bottom bin 的 gain span 约 `0.0727076` mm，offset-only / offset+gain energy explained 为 `0.7714` / `0.8208`；结构方向与 top 不同且幅度不对称。
3. **Top/bottom asymmetry：** 明显；top frame-mean std=`0.13944` mm，bottom=`0.0297605` mm。
4. **一维 b(v)：** local offset-only energy explained fractions（30/60/100 px）为 `0.2226, 0.2184, 0.2090`；60 px 分区为 top/middle/bottom=`0.6717`/`0.2083`/`0.7714`。b(v) 对边缘共同偏置有效，但中部只解释约五分之一，不能解释主要全局 residual。加入局部 gain 后全局为 `0.2376, 0.2319, 0.2177`。
5. **u-dependent residual：** 全局 frame-balanced u slope=`-2.31376e-06` mm/px，weighted RMSE=`0.0983961` mm；全局斜率接近零不代表局部没有 u 结构，边缘 bin 的 `delta_g(v)` 与 bootstrap 区间见 CSV/图，且边缘加入 gain 后仍只有限改善。

## v / frame 依赖

- local 全局 v slope=`1.56168e-06` mm/px，weighted RMSE=`0.0983919` mm。
- local frame mean residual 范围=`-0.101041` 到 `0.287692` mm，frame std=`0.0652326` mm。
- 固定分箱的 b(v) 曲线跨 v 多次变号且峰谷约达 0.1 mm 量级；|e_lambda|≥0.3 mm 的 `519` 个点主要集中在 frame `027`（`475` 个），v 范围约 `273.987`–`2221.97` px，说明存在 frame/位置耦合而非单一全局线性趋势。
- Reconstruction invalid counts: M0=`238`, M_local_fullfit=`239`；invalid rows保留在 `residual_points.csv`，未静默删除。

## 下一步选择

- **推荐 D：残差结构更复杂，暂不建立 correction。** 当前证据支持 top 的 b(v) 共同偏置和一定 u/gain 结构，但固定 bin 的诊断尚不足以授权建立 correction；本轮没有拟合或部署任何 b(v)/gain correction。
- 若后续继续，应先由人工确认 residual decomposition，再单独定义 correction 模型与独立 validation 方案。

Outputs: `projects\daheng\outputs\0814\cone_residual_decomposition`
