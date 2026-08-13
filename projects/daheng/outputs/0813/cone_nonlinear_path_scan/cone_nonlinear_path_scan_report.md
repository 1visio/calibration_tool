# Circular Cone exact nonlinear path scan

**FULL_LINEAR_STEP = REJECTED**  
**LOCAL_DESCENT_DIRECTION = CONFIRMED**  
**PATH_SCAN_DECISION = DAMPED_RELINEARIZATION_REQUIRED**

本步骤是 FIT-only 决策门，不是非线性优化。仅扫描 `Theta(t)=Theta0+t*DeltaTheta`；所有 candidate 只存在内存，并通过正式 `reconstruct_uv_to_ground()` 计算。

## 隔离与复现

- 只重建 FIT 001–010，共 17564 个冻结 laser points。
- VALIDATION 011–013 图像未打开、未重建、未评分，也没有参与路径、阈值或判定。
- FIT frozen-pixel SHA-256：`33ac7b3ca72a1a7e83c86fec1d49bc9bd54f773e79d7005812e6d09dd7a24660`。
- 三组 `DeltaTheta` 在本次 FIT-only 重算后与 sensitivity CSV 数值一致。
- 扫描点：`t=0…1/8` 以 `1/128` 细扫，另含 `3/16, 1/4, 1/2, 3/4, 1`。
- 保持正式 runtime depth/range/root-selection 配置；没有调用拟合脚本私有求交实现。

## Global exact result

| weighting | best t | baseline RMSE | best exact RMSE | exact RMSE at t=1 | exact explained at best | exact explained at t=1 | gap/baseline at best | invalid over path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| point_equal | 0.0625 | 0.187223 | 0.165144 | 16.483738 | 0.221946 | -7750.639195 | 0.435739 | 0 |
| frame_equal | 0.046875 | 0.190432 | 0.169747 | 20.958843 | 0.205445 | -12112.071328 | 0.334860 | 0 |
| v_region_equal | 0.023438 | 0.192985 | 0.182270 | 49.054601 | 0.107957 | -64610.986522 | 0.230565 | 0 |

## Best scanned point: top / middle / bottom

| weighting | region | exact RMSE | exact explained | invalid |
|---|---|---:|---:|---:|
| point_equal | top_0_299 | 0.391798 | -0.257335 | 0 |
| point_equal | middle_300_2699 | 0.094732 | 0.472004 | 0 |
| point_equal | bottom_2700_2999 | 0.232673 | 0.411405 | 0 |
| frame_equal | top_0_299 | 0.398086 | -0.188508 | 0 |
| frame_equal | middle_300_2699 | 0.099272 | 0.427892 | 0 |
| frame_equal | bottom_2700_2999 | 0.250420 | 0.339637 | 0 |
| v_region_equal | top_0_299 | 0.405585 | -0.142231 | 0 |
| v_region_equal | middle_300_2699 | 0.107603 | 0.324915 | 0 |
| v_region_equal | bottom_2700_2999 | 0.274041 | 0.176622 | 0 |

## Decision gate

**FULL_LINEAR_STEP = REJECTED**  
**LOCAL_DESCENT_DIRECTION = CONFIRMED**  
**PATH_SCAN_DECISION = DAMPED_RELINEARIZATION_REQUIRED**

- all scanned candidates retained every FIT intersection。
- best exact explained fraction: point_equal=0.221946, frame_equal=0.205445, v_region_equal=0.107957。
- t=1 exact explained fraction: point_equal=-7750.639195, frame_equal=-12112.071328, v_region_equal=-64610.986522。
- 判据：三条方向均存在 exact energy 下降且全路径无 invalid，才确认 LOCAL_DESCENT；三条方向在 t=1 仍保持正 exact explained fraction，才确认 FULL_LINEAR_STEP。
- 关键结果：三条方向在足够小的步长下都是下降方向，但完整线性步全部灾难性失效。这拒绝的是 one-shot `Theta0+DeltaTheta`，不是 Circular Cone，也不是小步非线性优化。
- 下一阶段若继续，必须采用 scaled/damped trust region，每次接受小步后重新计算 Jacobian；不得把本次 best path point 或 t=1 参数当作正式候选写回。
- 按研究顺序在此停下。

## 参数与不变项

以下增量只定义扫描方向，没有写出候选 YAML：

| parameter | point_equal delta | frame_equal delta | v_region_equal delta |
|---|---:|---:|---:|
| theta_axis (rad) | +0.272871283 | +0.273528622 | +0.272326781 |
| phi_axis (rad) | -0.00284359986 | -0.0028786114 | -0.00284291957 |
| A_x (mm) | -111.755478 | -114.75404 | -131.671627 |
| A_y (mm) | +1.18037531 | +1.6836853 | +0.402764515 |
| A_z (mm) | +10.5564018 | +2.26331238 | -50.9557274 |
| alpha (rad) | +0.000747970966 | +0.000417916024 | -0.00173659553 |

- Formal Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- Formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`
- Frozen baseline 最大复核误差：4.979e-10
- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 与正式 Cone 均未修改。
