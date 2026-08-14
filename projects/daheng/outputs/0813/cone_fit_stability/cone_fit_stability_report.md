# Circular Cone FIT stability: frame jackknife and weak-direction profile

**FIT_ONLY = TRUE**

本轮只使用 FIT 001–010；VALIDATION 011–013 未打开、未参与任何参数、步长、阈值或判定。正式 Cone 参数没有写回。

## Frame jackknife

- 10 折按帧留一；每折用 9 帧重新运行 scaled/damped trust-region，最多 20 个 accepted steps。
- weighting：point_equal, frame_equal, v_region_equal。
- held-out frame 只用于评估该折 candidate，没有参与该折优化。

| weighting | held-out RMSE median | held-out RMSE max | explained median | explained min | positive folds |
|---|---:|---:|---:|---:|---:|
| point_equal | 0.084881 | 0.095982 | 0.821274 | 0.306031 | 10/10 |
| frame_equal | 0.084686 | 0.094567 | 0.823913 | 0.293817 | 10/10 |
| v_region_equal | 0.083754 | 0.100973 | 0.815997 | 0.371712 | 10/10 |

| heldout frame | weighting | train RMSE | heldout RMSE | heldout explained | status | scaled delta L2 |
|---|---|---:|---:|---:|---|---:|
| 001 | point_equal | 0.076548 | 0.095982 | 0.796696 | max_accepted_steps | 17.441 |
| 001 | frame_equal | 0.077991 | 0.094567 | 0.802647 | max_accepted_steps | 17.445 |
| 001 | v_region_equal | 0.084581 | 0.082246 | 0.808132 | max_accepted_steps | 17.344 |
| 002 | point_equal | 0.077329 | 0.089299 | 0.733270 | max_accepted_steps | 17.459 |
| 002 | frame_equal | 0.078789 | 0.088060 | 0.740617 | max_accepted_steps | 17.46 |
| 002 | v_region_equal | 0.085682 | 0.072971 | 0.777452 | max_accepted_steps | 17.372 |
| 003 | point_equal | 0.077534 | 0.088629 | 0.306031 | max_accepted_steps | 17.431 |
| 003 | frame_equal | 0.078671 | 0.089405 | 0.293817 | max_accepted_steps | 17.433 |
| 003 | v_region_equal | 0.083080 | 0.100973 | 0.371712 | max_accepted_steps | 17.375 |
| 004 | point_equal | 0.077767 | 0.083831 | 0.847359 | max_accepted_steps | 17.425 |
| 004 | frame_equal | 0.079077 | 0.083110 | 0.849973 | max_accepted_steps | 17.429 |
| 004 | v_region_equal | 0.081779 | 0.098186 | 0.823862 | max_accepted_steps | 17.374 |
| 005 | point_equal | 0.081832 | 0.058053 | 0.744659 | max_accepted_steps | 17.431 |
| 005 | frame_equal | 0.081509 | 0.057931 | 0.745733 | max_accepted_steps | 17.434 |
| 005 | v_region_equal | 0.086331 | 0.064308 | 0.698607 | max_accepted_steps | 17.361 |
| 006 | point_equal | 0.077835 | 0.082605 | 0.757365 | max_accepted_steps | 17.438 |
| 006 | frame_equal | 0.079011 | 0.083253 | 0.753543 | max_accepted_steps | 17.441 |
| 006 | v_region_equal | 0.082805 | 0.085262 | 0.712308 | max_accepted_steps | 17.396 |
| 007 | point_equal | 0.079464 | 0.062900 | 0.889349 | max_accepted_steps | 17.42 |
| 007 | frame_equal | 0.081064 | 0.063261 | 0.888076 | max_accepted_steps | 17.42 |
| 007 | v_region_equal | 0.084777 | 0.060902 | 0.866242 | max_accepted_steps | 17.357 |
| 008 | point_equal | 0.078104 | 0.079985 | 0.866153 | max_accepted_steps | 17.425 |
| 008 | frame_equal | 0.079374 | 0.080079 | 0.865838 | max_accepted_steps | 17.428 |
| 008 | v_region_equal | 0.084738 | 0.078417 | 0.844981 | max_accepted_steps | 17.342 |
| 009 | point_equal | 0.077295 | 0.087073 | 0.867676 | max_accepted_steps | 17.411 |
| 009 | frame_equal | 0.078523 | 0.087881 | 0.865207 | max_accepted_steps | 17.416 |
| 009 | v_region_equal | 0.083169 | 0.091473 | 0.846060 | max_accepted_steps | 17.348 |
| 010 | point_equal | 0.077566 | 0.085932 | 0.845852 | max_accepted_steps | 17.429 |
| 010 | frame_equal | 0.078714 | 0.086119 | 0.845179 | max_accepted_steps | 17.432 |
| 010 | v_region_equal | 0.081888 | 0.100552 | 0.827510 | max_accepted_steps | 17.359 |

## Held-out top / middle / bottom

| weighting | region | held-out RMSE median | held-out explained median | positive folds |
|---|---|---:|---:|---:|
| point_equal | top_0_299 | 0.145898 | 0.802769 | 8/8 |
| point_equal | middle_300_2699 | 0.058132 | 0.801259 | 10/10 |
| point_equal | bottom_2700_2999 | 0.104879 | 0.885566 | 10/10 |
| frame_equal | top_0_299 | 0.142872 | 0.810674 | 8/8 |
| frame_equal | middle_300_2699 | 0.058425 | 0.797652 | 10/10 |
| frame_equal | bottom_2700_2999 | 0.103732 | 0.888716 | 10/10 |
| v_region_equal | top_0_299 | 0.134401 | 0.831105 | 8/8 |
| v_region_equal | middle_300_2699 | 0.059311 | 0.786129 | 10/10 |
| v_region_equal | bottom_2700_2999 | 0.104727 | 0.842006 | 10/10 |

## Weak-direction profile

对 full-FIT nonlinear candidate 的最终 Jacobian 做 scaled SVD；沿最小右奇异向量进行 exact production reconstruction 扫描。由于 candidate 已在 `A_z=500 mm` 上界，profile 是受 bounds 限制的一侧/非对称 profile。它不是新的优化结果。

| weighting | weak/strong singular ratio | profile t range | best RMSE | RMSE at t=0 | best explained vs Theta0 |
|---|---:|---|---:|---:|---:|
| point_equal | 1.628e-06 | [-8,0] | 0.034159 | 0.034159 | 0.966712 |
| frame_equal | 1.625e-06 | [-8,0] | 0.033671 | 0.033671 | 0.968737 |
| v_region_equal | 1.507e-06 | [-8,0] | 0.034900 | 0.034900 | 0.967296 |

## Interpretation

- 如果 jackknife 的 held-out residual 仍稳定改善，说明 FIT 上得到的是跨姿态共同表面，而不是单帧过拟合。
- 如果不同 weighting 的参数变化很大但 profile/重建表面变化很小，说明物理参数不可单独辨识，应把结论写成‘表面可拟合、参数有耦合’，不能发布某个 apex/alpha 数值。
- 本报告不评价 VALIDATION；只有在 FIT 稳定性核验完成后，才允许做一次冻结的 011–013 最终评价。

## Provenance / 不变项

- Formal Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- Formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`
- FIT frozen-pixel SHA-256：`33ac7b3ca72a1a7e83c86fec1d49bc9bd54f773e79d7005812e6d09dd7a24660`
- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 与正式 Cone 均未修改。
