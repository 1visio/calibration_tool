# Task 4B — Frame 027 异常影响诊断

**FIT_ONLY = TRUE**  
**VALIDATION_OPENED = FALSE**  
**CONE_REFIT = FALSE**  
**PRODUCTION_CONE_MODIFIED = FALSE**

## 最终结论

`FRAME027_TRUTH_CONSISTENCY = PASS`

`FRAME027_INFLUENCE = STRONG`

`NEXT_STEP = B`

027 的原始三联图 provenance、SHA、PnP、Steger UV、ray 定义和 ray-plane 交点均自洽；没有发现足以判为 truth 错误的证据。027 对全局 FIT residual 的影响很强，但它不覆盖 top<300 或 bottom>2700，因此排除后 top+/bottom− 结构原样保留。下一步应继续 objective mismatch / residual 研究，而不是先删除 027。

## 1. 原始三联图与 truth 审计

- 三张图必须同属 FIT manifest 的同一 pose ID，实际文件 SHA-256 必须匹配 frames.csv。
- PnP 使用正式 intrinsics/distortion、11×8 内角点、20 mm 方格；laser center 使用 Task 3/4 沿用的 Steger 配置。
- ray 定义为 `r=[x_n,y_n,1]`，其中 `(x_n,y_n)=cv2.undistortPoints(u,v)`；`lambda_truth=-d/(n·r)`，因此 `Zc=lambda_truth`。
- 027 chess→laser 间隔 `14.5256` s；邻近帧统计见下表。这个间隔不能数学证明棋盘绝无物理移动，但并不比邻近帧更长。
- 027 复算 truth 与 Task 4A 的最大 UV/lambda 差为 `0` px / `0` mm；平面方程最大残差 `1.13687e-13` mm。

| frame | SHA all pass | chess→laser s | PnP RMSE px | points | u span px | v span px | lambda span mm | laser coverage | FWHM p50/p95 px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 025 | True | 16.8591 | 0.156195 | 900 | 250.667 | 2095 | 74.1093 | 0.619 | 3/6 |
| 026 | True | 41.2007 | 0.165339 | 900 | 241.104 | 2446.95 | 71.1716 | 0.618 | 3/7 |
| 027 | True | 14.5256 | 0.174137 | 900 | 211.217 | 1421.96 | 63.0356 | 0.732333 | 4/6 |
| 028 | True | 26.9674 | 0.18364 | 900 | 16.7852 | 2284 | 2.02026 | 0.667 | 3/4 |
| 029 | True | 40.6414 | 0.10517 | 900 | 42.1737 | 2187.02 | 5.41518 | 0.679 | 2/3 |
| 030 | True | 42.8994 | 0.130974 | 900 | 43.9579 | 2126.01 | 6.27074 | 0.589 | 2/4 |

027 的 `dynamic_range_low` 与邻近激光帧一致，是暗背景窄激光线采集的共同 warning；027 无丢图、错 ID、hash mismatch、PnP 超阈值、无效 ray-plane 点或 Task 4A truth 不一致。

## 2. Frame 027 residual

| model | valid | bias mm | RMSE mm | P95 mm | corr(u,e) | corr(v,e) | corr(u,v) | v slope mm/px | v-line R² | offset energy explained | offset+v-tilt explained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 891 | 0.281217 | 0.361305 | 0.570011 | 0.958031 | 0.959378 | 0.999969 | 0.000517874 | 0.920406 | 0.605807 | 0.968625 |
| M_local_fullfit | 890 | 0.287692 | 0.362026 | 0.570321 | 0.956191 | 0.9575 | 0.999969 | 0.000501359 | 0.916806 | 0.631505 | 0.969344 |

027 不是单纯常数 offset：M_local 的平均偏置为 `0.287692` mm，但沿 v 的线性项还能解释大量变化，单变量 v-line R²=`0.916806`。由于该帧激光轨迹上 corr(u,v)=`0.999969`，无法从单帧稳定地区分 u-tilt 与 v-tilt；只能判定为明显的整帧 offset + 沿条纹方向的 tilt。

## 3. Leave-027-out（模型完全冻结）

- M_local 全局 point residual RMSE：`0.098517` → `0.0742621` mm。
- 027 占 M_local 原始总 residual energy 的 `45.08%`，因此影响判为 **STRONG**。固定阈值：energy share>25% 或 RMSE 降幅>20% 为 STRONG；>10% 为 MODERATE；否则 WEAK。

### Top / middle / bottom

| region | original bias/RMSE mm | leave-027 bias/RMSE mm | conclusion |
|---|---:|---:|---|
| top_formal_edge | 0.195064/0.231566 | 0.195064/0.231566 | exactly unchanged |
| middle_formal | 0.00261971/0.0979454 | -0.00731586/0.0732103 | changed |
| bottom_formal_edge | -0.0692053/0.0769823 | -0.0692053/0.0769823 | exactly unchanged |

### 30/60/100 px residual explainability（M_local_fullfit）

| bin | original offset | original offset+gain | leave offset | leave offset+gain |
|---:|---:|---:|---:|---:|
| 30 | 0.222647 | 0.237619 | 0.408692 | 0.435352 |
| 60 | 0.218395 | 0.231888 | 0.401169 | 0.424818 |
| 100 | 0.208992 | 0.217659 | 0.384012 | 0.404454 |

排除 027 会显著降低中部的 frame-specific 大残差，也会改变其覆盖 v 范围内的 b(v)/delta_g(v)；但 top 和 bottom 没有任何 027 点，所以 top 正偏、bottom 负偏不是 027 制造的。

## 边界与下一步

- `PASS` 仅表示已记录数据与计算链路一致；三联图之间棋盘是否发生肉眼不可见的微小移动，现有暗场 nolaser/laser 图无法独立证明。
- 027 不应被永久删除；如果要进一步区分姿态时序移动与 Cone/objective mismatch，应补做同一姿态的短时重复三联图或在低曝光 laser 图中加入可追踪的板面标记。
- 本轮未重拟合 Cone、未建立 correction、未读取 validation。
- Formal Cone SHA-256 before/after: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。

Outputs: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\frame027_audit`
