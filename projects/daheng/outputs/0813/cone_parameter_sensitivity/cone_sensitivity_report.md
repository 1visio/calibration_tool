# Circular Cone paired-PnP sensitivity and local identifiability

**PARAMETER_ERROR_CAN_EXPLAIN = PARTIAL**

本轮只有当前 `Theta0` 附近的数值 Jacobian、加权线性最小二乘和 SVD；没有执行非线性重优化，没有把任何 `DeltaTheta` 写回 0811 或部署 Cone。

## 严格隔离与 residual

- FIT 001–010：baseline 17564 点；全部 step 比较、step 选择、Jacobian、weight 和 `DeltaTheta` 只来自 FIT。
- VALIDATION 011–013：baseline 6049 点；只在 FIT step 与三组 FIT `DeltaTheta` 冻结后计算最终线性预测。
- 冻结 Steger 像素 hash：`0a356781cc4c2d6e46da0033af6aa8349c7a3a1722eacd9554c0c2d4f1997b3a`。
- 每次 candidate 都只在内存替换 `calibration['laser_model']`，然后直接调用正式 `reconstruct_uv_to_ground()`。
- 主 residual：`r_z = Zg - (-(nx*Xg+ny*Yg+d)/nz)`；候选的 `Xg,Yg,Zg` 均重新重建，PnP plane Z 不是静态标签。
- Jacobian：`J=dr_z/dTheta`。求解符号为 `J DeltaTheta ≈ -r0`，线性预测为 `r_pred=r0+J DeltaTheta`。

## 参数化与解释尺度

`Theta=[theta_axis, phi_axis, A_x, A_y, A_z, alpha]`；angle 用 rad，apex 用 mm。axis 由球坐标生成并再次单位化。解释尺度只用于 SVD/变量数值缩放与 `DeltaTheta/scale` 展示，没有加入任何正则项。

| parameter | unit | Theta0 | interpretation scale | selected FIT step |
|---|---|---:|---:|---:|
| theta_axis | rad | 1.855709955 | 0.01745329252 | 1e-05 |
| phi_axis | rad | 0.01003015064 | 0.01745329252 | 1e-05 |
| A_x | mm | -115.5997074 | 10 | 0.01 |
| A_y | mm | 1.741889416 | 10 | 0.01 |
| A_z | mm | 327.0330738 | 10 | 0.01 |
| alpha | rad | 1.554609275 | 0.001745329252 | 1e-05 |

解释尺度：axis angles=1 degree，apex=10 mm，alpha=0.1 degree。

## FIT 三尺度 Jacobian 稳定性

每个参数固定测试 `0.3× / 1× / 3× base_step`。selected step 只按 FIT 导数一致性和 FIT invalid 数选定；下表不含任何 validation step 比较。

| parameter | step | selected | derivative RMS | rel. to selected | invalid +/- |
|---|---:|---|---:|---:|---:|
| theta_axis | 3e-06 | false | 1386.5 | 1.043e-09 | 0/0 |
| theta_axis | 1e-05 | true | 1386.5 | 0.000e+00 | 0/0 |
| theta_axis | 3e-05 | false | 1386.5 | 8.899e-09 | 0/0 |
| phi_axis | 3e-06 | false | 279.124 | 9.738e-10 | 0/0 |
| phi_axis | 1e-05 | true | 279.124 | 0.000e+00 | 0/0 |
| phi_axis | 3e-05 | false | 279.124 | 3.941e-10 | 0/0 |
| A_x | 0.003 | false | 3.2793 | 9.599e-11 | 0/0 |
| A_x | 0.01 | true | 3.2793 | 0.000e+00 | 0/0 |
| A_x | 0.03 | false | 3.2793 | 4.101e-11 | 0/0 |
| A_y | 0.003 | false | 0.0351038 | 8.760e-09 | 0/0 |
| A_y | 0.01 | true | 0.0351038 | 0.000e+00 | 0/0 |
| A_y | 0.03 | false | 0.0351038 | 2.758e-09 | 0/0 |
| A_z | 0.003 | false | 1.01721 | 2.964e-10 | 0/0 |
| A_z | 0.01 | true | 1.01721 | 0.000e+00 | 0/0 |
| A_z | 0.03 | false | 1.01721 | 1.150e-10 | 0/0 |
| alpha | 3e-06 | false | 1416.67 | 1.092e-09 | 0/0 |
| alpha | 1e-05 | true | 1416.67 | 0.000e+00 | 0/0 |
| alpha | 3e-05 | false | 1416.67 | 9.007e-09 | 0/0 |

FIT common invalid=0；VALIDATION fixed-step common invalid=0。每个参数 selected-step 的精确 split-local invalid indices 已记录在 `cone_parameter_sensitivity.csv`；没有因不同参数静默改变 residual 长度。

## 三种 weighting 的线性增量

- `point_equal`：所有 FIT 点等权。
- `frame_equal`：每个 FIT frame 总权重相同。
- `v_region_equal`：0–2999 按 300 px 分为 10 区；每个有数据区总权重相同，区内每个有数据 frame 总权重相同。

| parameter | point_equal delta/scale | frame_equal delta/scale | v_region_equal delta/scale |
|---|---:|---:|---:|
| theta_axis | +15.6344 | +15.672 | +15.6032 |
| phi_axis | -0.162926 | -0.164932 | -0.162887 |
| A_x | -11.1755 | -11.4754 | -13.1672 |
| A_y | +0.118038 | +0.168369 | +0.0402765 |
| A_z | +1.05564 | +0.226331 | -5.09557 |
| alpha | +0.428556 | +0.239448 | -0.994996 |

原生单位的 `DeltaTheta`、每列加权 Jacobian RMS norm 与 selected step 见 `cone_parameter_sensitivity.csv`。

## 局部可辨识性

SVD 对 `W^(1/2) J diag(scale) / sqrt(sum(W))` 计算；effective rank 阈值为 `S/S1 >= 1e-06`。

| weighting | effective rank | condition number | scaled singular values | smallest vector main components |
|---|---:|---:|---|---|
| point_equal | 6/6 | 293647 | 42.08, 4.87, 0.04628, 0.007999, 0.001488, 0.0001433 | A_z=+0.9322, A_x=+0.2921, alpha=+0.2129 |
| frame_equal | 6/6 | 293959 | 42.08, 4.859, 0.04619, 0.008125, 0.001473, 0.0001431 | A_z=+0.9323, A_x=+0.2919, alpha=+0.2129 |
| v_region_equal | 6/6 | 295001 | 42.08, 4.823, 0.0454, 0.00853, 0.00152, 0.0001426 | A_z=+0.9324, A_x=+0.2913, alpha=+0.2131 |

耦合重点（covariance correlation，scaled FIT Jacobian pseudoinverse）：

- `point_equal`：apex–alpha 最强 `A_z / alpha`=+0.999886；axis-angle–apex 最强 `phi_axis / A_y`=-0.996582。
- `frame_equal`：apex–alpha 最强 `A_z / alpha`=+0.999887；axis-angle–apex 最强 `phi_axis / A_y`=-0.996811。
- `v_region_equal`：apex–alpha 最强 `A_z / alpha`=+0.999880；axis-angle–apex 最强 `phi_axis / A_y`=-0.996948。

三种 weighting 在阈值 1e-06 下虽均为数值满秩 6/6，但 condition number=2.936e+05–2.950e+05，最小/最大 singular value=3.390e-06–3.405e-06；因此只能称为数值满秩，不能称为各物理参数良好可辨识。
`A_z–alpha` covariance correlation=+0.999880–+0.999887；`phi_axis–A_y`=-0.996948–-0.996582。前者接近完全耦合，后者表明 axis angle 与 apex 也强耦合。
当前 `alpha=89.072550°`，Cone 已接近平面极限；最弱方向（以 point_equal 为例）为 `A_z=+0.9322, A_x=+0.2921, alpha=+0.2129`。alpha 与 apex 同时进入最弱方向，支持存在近退化的判断；这不是仅由 `alpha≈89°` 单独推断。

完整 pairwise column cosine、covariance correlation 以及最小两个右奇异向量见 `cone_parameter_coupling.csv`。`alpha≈89°` 是否形成近退化方向，应结合最小奇异值比例与最小向量中的 alpha/apex 分量判断，不能只看单个相关系数。

## Global residual explainability

所有 VALIDATION 行都使用对应 weighting 在 FIT 得到并冻结的同一个 `DeltaTheta`。

| split | weighting | before RMSE | after RMSE | before bias | after bias | energy explained |
|---|---|---:|---:|---:|---:|---:|
| fit | point_equal | 0.187223 | 0.022907 | -0.068448 | +0.000000 | 0.985030 |
| fit | frame_equal | 0.190432 | 0.021037 | -0.068827 | +0.000000 | 0.987796 |
| fit | v_region_equal | 0.192985 | 0.021185 | -0.049843 | +0.000000 | 0.987949 |
| validation | point_equal | 0.186399 | 0.036908 | -0.114072 | +0.023007 | 0.960793 |
| validation | frame_equal | 0.182158 | 0.035205 | -0.115406 | +0.024114 | 0.962648 |
| validation | v_region_equal | 0.184439 | 0.035969 | -0.116435 | +0.025226 | 0.961968 |

## Top / middle / bottom（v_region_equal）

| split | region | samples | before RMSE | after RMSE | explained fraction |
|---|---|---:|---:|---:|---:|
| fit | top_0_299 | 1605 | 0.379494 | 0.025266 | 0.995567 |
| fit | middle_300_2699 | 13979 | 0.130962 | 0.019968 | 0.976752 |
| fit | bottom_2700_2999 | 1980 | 0.302006 | 0.025688 | 0.992765 |
| validation | top_0_299 | 684 | 0.245157 | 0.047407 | 0.962606 |
| validation | middle_300_2699 | 4734 | 0.156627 | 0.033534 | 0.954160 |
| validation | bottom_2700_2999 | 631 | 0.289515 | 0.041156 | 0.979792 |

## 0811 拟合支持之外的 paired points

0811 原标定点支持约为 `v=[241.998,2731.978]`。以下 paired 点全部保留；这里只单独汇总其线性 explainability，没有把它们从 FIT 解中排除。

| split | region | samples | before RMSE | after RMSE (v_region_equal) | explained fraction |
|---|---|---:|---:|---:|---:|
| fit | extrap_top_v_lt_242 | 1141 | 0.396770 | 0.028416 | 0.994871 |
| fit | extrap_bottom_v_gt_2732 | 1784 | 0.307259 | 0.026034 | 0.992821 |
| validation | extrap_top_v_lt_242 | 567 | 0.264764 | 0.047245 | 0.968159 |
| validation | extrap_bottom_v_gt_2732 | 535 | 0.292681 | 0.041618 | 0.979780 |

每 300 px bin、Bias/MAE/RMSE/P95/energy 的三 weighting 全量结果见 `cone_region_explainability.csv`；global 全量结果见 `cone_residual_explainability.csv`。

## 判定

**PARAMETER_ERROR_CAN_EXPLAIN = PARTIAL**

- FIT global explained fraction: point_equal=0.985030, frame_equal=0.987796, v_region_equal=0.987949。
- VALIDATION global explained fraction: point_equal=0.960793, frame_equal=0.962648, v_region_equal=0.961968。
- VALIDATION v_region_equal: top_0_299=0.962606, middle_300_2699=0.954160, bottom_2700_2999=0.979792。
- max |DeltaTheta/scale|=15.672。
- FIT common invalid rows=0; max step disagreement=9.007e-09。
- 关键区分：residual tangent space 的跨 split 解释力本身很强，且 validation 的 top/middle/bottom 均明显改善；但最优增量达到 `15.672×` 解释尺度，已经超出可信的局部线性邻域。
- 因而本轮能确认“存在共同的参数方向可解释 residual”，却不能仅凭该大步长线性外推确认真实 `Theta0+DeltaTheta` 仍能实现同样改善；结合约 3e5 的条件数与近完全参数耦合，参数误差结论保守判为 PARTIAL，而不是把 tangent-space 的高 explained fraction 直接判为 STRONG。
- 预先固定判据：STRONG 要求三 weighting 的 validation global explained fraction 均 >=0.50，且 validation 的 top/middle/bottom 在 v_region_equal 下均 >=0.30 并降低 RMSE，同时 `max|DeltaTheta/scale|<=5` 且 FIT 无 invalid；PARTIAL 要求至少一种 FIT global >=0.30、至少一种 validation global >=0.10，且 validation 三大区至少两区为正；否则 WEAK。大于尺度 20 倍的局部增量不判 PARTIAL。

该结论只评价当前参数切空间能否解释 residual；它不证明 `Theta0+DeltaTheta` 的非线性有效性，也不授权发布或写回新参数。

## Provenance / 不变项

- Measurement config：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\measure_tool_daheng_0811.yaml`
- PnP audit：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\pnp_reference_audit\paired_pnp_reference_audit.csv`
- Formal Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- Formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`
- Steger：`{'sigma': 1.5, 'threshold': 30.0, 'deriv_thresh': 0.5, 'roi_margin': 48, 'roi_max_height': 512, 'scan_axis': 'row'}`
- Reconstruction：`ReconstructionParams(parallel_epsilon=1e-09, quadratic_epsilon=1e-12, min_camera_depth_mm=630.0, max_camera_depth_mm=715.0, model_range_margin_mm=2.0)`
- 既有 frozen baseline 指标最大复核误差：4.979e-10
- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 和正式 Cone 均未修改。
