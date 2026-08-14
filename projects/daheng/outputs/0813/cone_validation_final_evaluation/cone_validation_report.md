# Circular Cone frozen validation evaluation

**VALIDATION_ONLY = TRUE**

本报告只评价 FIT-only 产生的三个冻结 candidate；VALIDATION 没有参与 candidate、weight、step、阈值或任何选择。
没有执行 validation reoptimization，也没有写回正式 Cone。

## Candidate and global result

| candidate | metric weighting | before RMSE | after RMSE | before P95 | after P95 | explained fraction | invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| point_equal | point_equal | 0.186399 | 0.036572 | 0.347593 | 0.088672 | 0.961505 | 0 |
| point_equal | frame_equal | 0.182158 | 0.035532 | 0.315495 | 0.087916 | 0.961951 | 0 |
| point_equal | v_region_equal | 0.184439 | 0.035862 | 0.320946 | 0.087916 | 0.962194 | 0 |
| frame_equal | point_equal | 0.186399 | 0.035793 | 0.347593 | 0.088336 | 0.963126 | 0 |
| frame_equal | frame_equal | 0.182158 | 0.034809 | 0.315495 | 0.087689 | 0.963483 | 0 |
| frame_equal | v_region_equal | 0.184439 | 0.035065 | 0.320946 | 0.087684 | 0.963855 | 0 |
| v_region_equal | point_equal | 0.186399 | 0.035025 | 0.347593 | 0.090989 | 0.964693 | 0 |
| v_region_equal | frame_equal | 0.182158 | 0.033743 | 0.315495 | 0.087697 | 0.965686 | 0 |
| v_region_equal | v_region_equal | 0.184439 | 0.034091 | 0.320946 | 0.088705 | 0.965835 | 0 |

## Top / middle / bottom

| candidate | metric weighting | region | before RMSE | after RMSE | explained fraction |
|---|---|---|---:|---:|---:|
| point_equal | point_equal | top_0_299 | 0.247129 | 0.062225 | 0.936601 |
| point_equal | point_equal | middle_300_2699 | 0.154150 | 0.021306 | 0.980896 |
| point_equal | point_equal | bottom_2700_2999 | 0.297654 | 0.072240 | 0.941097 |
| point_equal | frame_equal | top_0_299 | 0.216512 | 0.054541 | 0.936542 |
| point_equal | frame_equal | middle_300_2699 | 0.156621 | 0.021937 | 0.980382 |
| point_equal | frame_equal | bottom_2700_2999 | 0.286226 | 0.071701 | 0.937247 |
| point_equal | v_region_equal | top_0_299 | 0.245157 | 0.062047 | 0.935945 |
| point_equal | v_region_equal | middle_300_2699 | 0.156627 | 0.021924 | 0.980408 |
| point_equal | v_region_equal | bottom_2700_2999 | 0.289515 | 0.071873 | 0.938370 |
| frame_equal | point_equal | top_0_299 | 0.247129 | 0.059711 | 0.941621 |
| frame_equal | point_equal | middle_300_2699 | 0.154150 | 0.020659 | 0.982040 |
| frame_equal | point_equal | bottom_2700_2999 | 0.297654 | 0.072215 | 0.941139 |
| frame_equal | frame_equal | top_0_299 | 0.216512 | 0.052248 | 0.941766 |
| frame_equal | frame_equal | middle_300_2699 | 0.156621 | 0.021255 | 0.981582 |
| frame_equal | frame_equal | bottom_2700_2999 | 0.286226 | 0.071736 | 0.937187 |
| frame_equal | v_region_equal | top_0_299 | 0.245157 | 0.059410 | 0.941274 |
| frame_equal | v_region_equal | middle_300_2699 | 0.156627 | 0.021207 | 0.981667 |
| frame_equal | v_region_equal | bottom_2700_2999 | 0.289515 | 0.071890 | 0.938340 |
| v_region_equal | point_equal | top_0_299 | 0.247129 | 0.056746 | 0.947275 |
| v_region_equal | point_equal | middle_300_2699 | 0.154150 | 0.020379 | 0.982522 |
| v_region_equal | point_equal | bottom_2700_2999 | 0.297654 | 0.071788 | 0.941832 |
| v_region_equal | frame_equal | top_0_299 | 0.216512 | 0.049643 | 0.947428 |
| v_region_equal | frame_equal | middle_300_2699 | 0.156621 | 0.020742 | 0.982461 |
| v_region_equal | frame_equal | bottom_2700_2999 | 0.286226 | 0.070031 | 0.940136 |
| v_region_equal | v_region_equal | top_0_299 | 0.245157 | 0.056334 | 0.947197 |
| v_region_equal | v_region_equal | middle_300_2699 | 0.156627 | 0.020830 | 0.982313 |
| v_region_equal | v_region_equal | bottom_2700_2999 | 0.289515 | 0.070551 | 0.940617 |

## Interpretation

- 三个 candidate 都来自 FIT；本表只回答它们在冻结 VALIDATION 上是否保持改善。
- 不根据 validation 表现挑选 candidate；point/frame/v_region 三者分别报告。
- 即使 validation 仍改善，也不能消除 FIT 中已经发现的 apex/alpha 参数耦合和 A_z 边界问题。
- stability analysis：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\cone_fit_stability`。

## Provenance / 不变项

- VALIDATION frozen-pixel SHA-256：`7f1790ccf1f1792ea413088a3cb4115d72cb1db97376d941e13dc35d28c2fa00`。
- Formal Cone：`D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- Formal Cone SHA-256（运行前后相同）：`478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`
- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 与正式 Cone 均未修改。
