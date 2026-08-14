# Circular Cone production-path sanity audit

- CANDIDATE_ROUNDTRIP = **PASS**
- ROOT_BRANCH_BEHAVIOR = **STABLE**
- HEIGHT_GAIN_COLLAPSE = **CONFIRMED**

## Scope and gate

本轮没有重新采集数据；paired UV 只按原始 provenance 调用了正式 Steger 入口重新生成并做 hash 校验。没有优化参数、没有修改 PnP、ground extrinsics 或 ROI；历史量块仍只读取保存的原始 `u,v`。
paired FIT/VALIDATION 原始图像仍在；按上一轮相同正式入口重新生成 UV，FIT/VALIDATION split pixel hash 均精确匹配，已固化为 frozen UV。

## Candidate round-trip

优化进程已结束；本轮 A（in-memory）使用冻结 `cone_nonlinear_fit_candidates.csv` 中的 final 参数直接构造 production model，B/C 则写出并通过正式 loader reload 实际 YAML。
历史量块上的内存模型→实际 YAML→loader reload→production reconstruction：PASS。每个字段均记录在 `candidate_roundtrip_audit.csv`；阈值为 max absolute difference ≤ 1e-9、valid mask 完全一致。
paired FIT/VALIDATION round-trip：**PASS**；实际 YAML reload 的逐点结果与 in-memory candidate 一致，并完成 paired RMSE replay。

实际候选 YAML：
- `M1_point_equal`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\cone_nonlinear_fit_trust_region\candidate_roundtrip_yaml\circular_cone_point_equal.yaml`
- `M2_frame_equal`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\cone_nonlinear_fit_trust_region\candidate_roundtrip_yaml\circular_cone_frame_equal.yaml`
- `M3_v_region_equal`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0813\cone_nonlinear_fit_trust_region\candidate_roundtrip_yaml\circular_cone_v_region_equal.yaml`

上一轮 paired 报告给出的匹配 RMSE（本轮已用 exact regenerated UV 和实际 YAML reload 重算）：
- FIT: point_equal 0.034159 mm, frame_equal 0.033671 mm, v_region_equal 0.034900 mm
- VALIDATION: point_equal 0.036572 mm, frame_equal 0.034809 mm, v_region_equal 0.034091 mm

## Historical root branch audit

`gauge_root_branch_audit.csv` 逐点保存两根、selected root、lambda、Zc、Zg 和 validity reason，并附 M0 对照字段。

| model | invalid fraction | root-switch fraction vs M0 | |Δlambda| P95 (mm) | |Δlambda| max (mm) |
|---|---:|---:|---:|---:|
| M0 | 0 | 0 | 0 | 0 |
| M1_point_equal | 0 | 0 | 40.5909 | 40.6595 |
| M2_frame_equal | 0 | 0 | 40.5968 | 40.6655 |
| M3_v_region_equal | 0 | 0 | 40.8059 | 40.8751 |

判定：`ROOT_BRANCH_BEHAVIOR = STABLE`。这里的 root-switch 是 production root array 的 selected index 改变；另有 lambda 分布用于区分连续参数变化与物理分支切换。

## Height gain audit

`height_gain_audit.csv` 的 median rows 使用每个历史 frame 的 baseline/top 原始 UV；local rows 在合并历史 UV 的三个 v 区域上，用正式重建计算 ±0.5 px 和 ±1.0 px 的中心差分 `dZg/du`。

| model | median replay height mean (mm) | mean |G/G_M0| | max |G/G_M0| |
|---|---:|---:|---:|
| M0 | 49.8998 | 1 | 1 |
| M1_point_equal | 9.4237 | 0.188821 | 0.188995 |
| M2_frame_equal | 9.42148 | 0.188775 | 0.188958 |
| M3_v_region_equal | 9.21872 | 0.1847 | 0.184884 |

代表性 `v1000_1999` 区域的局部中心差分（h=0.5 px；完整的三个 v 区域、两个步长见 CSV）：

| model | median dZg/du (mm/px) | ratio vs M0 |
|---|---:|---:|
| M0 | -0.31345 | 1 |
| M1_point_equal | -0.0558267 | 0.178104 |
| M2_frame_equal | -0.0558135 | 0.178062 |
| M3_v_region_equal | -0.0545963 | 0.174178 |

判定：`HEIGHT_GAIN_COLLAPSE = CONFIRMED`。候选与 M0 的中位数高度增益及局部 `dZg/du` 比率均被写入 CSV；这说明约 50 mm→约 9 mm 的塌缩是模型对 u→Zg 映射增益的变化，而不是依赖旧 XYZ 的伪造结果。

## Provenance

- base Cone: `D:\Docs\linelaserscan\0704line-laser-3d-scanner\laser_measurement_tool\configs\calibration_daheng_0811\circular_cone.yaml`
- reconstruction params: `ReconstructionParams(parallel_epsilon=1e-09, quadratic_epsilon=1e-12, min_camera_depth_mm=630.0, max_camera_depth_mm=715.0, model_range_margin_mm=2.0)`
- gauge source: `D:\Docs\linelaserscan\0704line-laser-3d-scanner\laser_measurement_tool\output_daheng_0811`
- frozen paired UV: regenerated from original images, split hashes exact, permanently saved
- no optimization or recapture was performed; image extraction was rerun only from the original files for exact hash regeneration

## Paired UV regeneration and exact replay

- `PAIRED_UV_REGENERATION = EXACT_MATCH`
- FIT 001–010: 10 frames, 17564 frozen points, hash `33ac7b3ca72a1a7e83c86fec1d49bc9bd54f773e79d7005812e6d09dd7a24660`。
- VALIDATION 011–013: 3 frames, 6049 frozen points, hash `7f1790ccf1f1792ea413088a3cb4115d72cb1db97376d941e13dc35d28c2fa00`。
- Previous artifact did not retain separate per-frame pixel hashes; this run stores per-frame hashes and proves the complete ordered split aggregates exactly. No parameter was adjusted to obtain the match.
- Steger provenance: original `extrinsics0813` laser images; `measure_tool_daheng_0811.yaml`; formal `create_extraction_params` / `extract_laser_center` entry; sigma=1.5, threshold=30, deriv_thresh=0.5, roi_margin=48, roi_max_height=512, scan_axis=row.
- PnP truth was reused from `pnp_reference_audit/paired_pnp_reference_audit.csv`; chess/laser pairing, manifest and image hashes remained valid.

- PnP consistency: **PASS**, max difference against prior per-frame metrics=0。

### Candidate replay

| model | split | weighting | samples | invalid | unweighted RMSE (mm) | matched RMSE (mm) | prior RMSE (mm) | difference (mm) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M1_point_equal | fit | point_equal | 17564 | 0 | 0.0341589895888 | 0.0341589895888 | 0.0341589895845 | 4.34753e-12 |
| M1_point_equal | validation | point_equal | 6049 | 0 | 0.0365716191628 | 0.0365716191628 | 0.0365716191628 | 2.76029e-14 |
| M2_frame_equal | fit | frame_equal | 17564 | 0 | 0.0342331014481 | 0.0336710173156 | 0.0336710173154 | 2.14946e-13 |
| M2_frame_equal | validation | frame_equal | 6049 | 0 | 0.0357932157281 | 0.0348094799115 | 0.0348094799115 | 4.71359e-14 |
| M3_v_region_equal | fit | v_region_equal | 17564 | 0 | 0.0342865798222 | 0.0349000651046 | 0.0349000651043 | 2.685e-13 |
| M3_v_region_equal | validation | v_region_equal | 6049 | 0 | 0.0350248338072 | 0.0340913114336 | 0.0340913114336 | 4.05717e-14 |

- `PAIRED_CANDIDATE_REPLAY = PASS`
- max RMSE difference = `4.34752928e-12 mm`。

永久保存：`paired_frozen_uv.csv`、`paired_frozen_uv_regenerated.csv`、`paired_pnp_reference.csv`、`paired_source_manifest.json`。后续实验应直接复用这些文件，不再从图像重新提取。
