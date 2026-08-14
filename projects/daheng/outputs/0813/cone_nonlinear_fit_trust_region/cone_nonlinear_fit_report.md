# Circular Cone FIT-only damped trust-region reoptimization

**FIT_SURFACE_RESULT = SUCCESS**  
**PARAMETER_CONVERGENCE = UNRESOLVED_BOUNDARY**

本步骤只优化 FIT 001–010。VALIDATION 011–013 未打开、未重建、未评分。所有 candidate 仅在内存和本实验 CSV 中存在，没有写出或覆盖任何正式 Cone YAML。

## 方法与隔离

- objective：`mean_w(r_z^2)`，其中每次 candidate 都通过正式 `reconstruct_uv_to_ground()` 重建，并在新的 `(Xg,Yg)` 上重新评价 paired PnP plane。
- 三种 weighting 独立从同一个正式 `Theta0` 启动；`v_region_equal` 是预先指定的主分支。
- scaled trust radius：初值 0.1，范围 [1e-07,1]；LM damping 只用于 trust-region step，不进入最终 objective。
- 接受阈值 actual/predicted >= 0.1；invalid 或越界 trial 直接拒绝；每次接受后重算 Jacobian。
- 主结果使用 L2；没有 robust loss、参数先验或正则化。
- FIT frozen-pixel SHA-256：`33ac7b3ca72a1a7e83c86fec1d49bc9bd54f773e79d7005812e6d09dd7a24660`。

## Optimizer outcome

| weighting | classification | status | accepted/trials | final radius | final gradient inf |
|---|---|---|---:|---:|---:|
| point_equal | SUCCESS | max_accepted_steps | 80/96 | 3.815e-06 | 2.425e-04 |
| frame_equal | SUCCESS | max_accepted_steps | 80/97 | 9.537e-07 | 6.016e-05 |
| v_region_equal | SUCCESS | max_accepted_steps | 80/97 | 9.537e-07 | 6.088e-05 |

## Matching-weight global exact residual

| weighting | before RMSE | after RMSE | before P95 | after P95 | energy explained |
|---|---:|---:|---:|---:|---:|
| point_equal | 0.187223 | 0.034159 | 0.373761 | 0.070773 | 0.966712 |
| frame_equal | 0.190432 | 0.033671 | 0.379989 | 0.072462 | 0.968737 |
| v_region_equal | 0.192985 | 0.034900 | 0.408968 | 0.075788 | 0.967296 |

## Matching-weight top / middle / bottom

| weighting | region | before RMSE | after RMSE | energy explained |
|---|---|---:|---:|---:|
| point_equal | top_0_299 | 0.349411 | 0.060187 | 0.970329 |
| point_equal | middle_300_2699 | 0.130371 | 0.028136 | 0.953426 |
| point_equal | bottom_2700_2999 | 0.303276 | 0.042725 | 0.980153 |
| frame_equal | top_0_299 | 0.365154 | 0.058931 | 0.973954 |
| frame_equal | middle_300_2699 | 0.131247 | 0.027763 | 0.955255 |
| frame_equal | bottom_2700_2999 | 0.308161 | 0.043466 | 0.980105 |
| v_region_equal | top_0_299 | 0.379494 | 0.058480 | 0.976253 |
| v_region_equal | middle_300_2699 | 0.130962 | 0.027272 | 0.956635 |
| v_region_equal | bottom_2700_2999 | 0.302006 | 0.053011 | 0.969189 |

## Final parameter displacement

表内为 `(Theta_final-Theta0)/interpretation_scale`；不同分支的弱参数不能直接平均。

| parameter | point_equal | frame_equal | v_region_equal |
|---|---:|---:|---:|
| theta_axis | +41.4469 | +41.4523 | +41.992 |
| phi_axis | -0.861625 | -0.865771 | -0.84233 |
| A_x | -22.3454 | -22.3528 | -23.0611 |
| A_y | +7.48308 | +7.58918 | +5.93764 |
| A_z | +17.2967 | +17.2967 | +17.2967 |
| alpha | +4.47345 | +4.46921 | +4.70524 |

## Final local identifiability

| weighting | effective rank | condition number | smallest/large singular ratio | strongest apex-alpha | strongest axis-apex |
|---|---:|---:|---:|---|---|
| point_equal | 6/6 | 614149 | 1.628e-06 | A_z/alpha=+0.998551 | phi_axis/A_y=-0.995257 |
| frame_equal | 6/6 | 615271 | 1.625e-06 | A_z/alpha=+0.998516 | phi_axis/A_y=-0.995399 |
| v_region_equal | 6/6 | 663449 | 1.507e-06 | A_z/alpha=+0.999038 | phi_axis/A_y=-0.993742 |

## Candidate surface consistency

| candidate pair | residual-surface RMS difference | P95 | scaled parameter distance L2 |
|---|---:|---:|---:|
| point_equal / frame_equal | 0.002474 | 0.004320 | 0.106666 |
| point_equal / v_region_equal | 0.005453 | 0.009260 | 1.80327 |
| frame_equal / v_region_equal | 0.003647 | 0.007386 | 1.8912 |

## FIT decision gate

**FIT_SURFACE_RESULT = SUCCESS**  
**PARAMETER_CONVERGENCE = UNRESOLVED_BOUNDARY**

- FIT_SURFACE SUCCESS：matching global explained >=0.80 且 top/middle/bottom 各 >=0.50。
- PARTIAL：matching global explained >=0.30 且三个大区均为正改善。
- 其他情况为 FAIL；`v_region_equal` 是总体判定的预注册主分支。
- FIT 表面目标上，Circular Cone 的真实非线性调整已经同时解释全局和上下边缘的主要 residual。
- 但本轮三个分支均为 max_accepted_steps，参数沿 apex/alpha 近退化谷漂移并触及 A_z 上界；因此参数并未收敛，不能把 candidate 当作已收敛参数。
- 这使“0811 参数/objective 不匹配”成为更强假设，但还不能查看 validation 后直接发布；下一步应先做 FIT frame jackknife 与弱方向 profile。
- 按计划在 FIT 收敛后、读取 VALIDATION 前停止。

## Provenance / 不变项

- Formal Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- Formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`
- Frozen baseline 最大复核误差：4.979e-10
- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 与正式 Cone 均未修改。
