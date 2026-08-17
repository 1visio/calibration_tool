# Frozen C1_4k boundary-policy comparison

`C1_BOUNDARY_POLICY = SAFE_GUARD_AVAILABLE`
`RECOMMENDED_POLICY = boundary_clamp`

## Scope and frozen boundary

- 仅读取 20 mm 与 50 mm 两组标准件目录，输入根目录：`D:\Docs\linelaserscan\0704line-laser-3d-scanner\laser_measurement_tool\output_daheng_0811`。
- C0/C1 使用完全相同的既有 `laser_center.csv` 及 baseline/height `u,v` 子集。
- 未读取 Validation（019–024、037–040），未重新拟合 C1、K/D 或 Cone，未修改 knots/penalty，未做 C2。
- Frozen s-domain：`[-0.16412256362, 0.194958484593]`。
- C1 三种策略：`raw_extrapolation`、`boundary_clamp`、`fallback_c0`；另列 `C0_baseline` 作为参考。

## Frozen provenance

- C1 model: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`
- C1 model SHA-256: `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`
- C1 parameter SHA-256: `be7c316c91b54ac9b13a1ff2485a9ca2ceebf6644a57a91ffca98f858f726037`
- Frozen C0 provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`
- Frozen formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`

## Dataset-level policy summary

| dataset | policy | global RMSE | global P95 | global Max | worst-position RMSE | worst-position P95 | position Bias range | position RMSE range | all positions <=0.2 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20mm | C0_baseline | 0.074616 | 0.125508 | 0.156626 | 0.117782 | 0.149983 | 0.123286 | 0.102178 | True |
| 20mm | raw_extrapolation | 0.108103 | 0.265138 | 0.321391 | 0.248485 | 0.300824 | 0.207567 | 0.207704 | False |
| 20mm | boundary_clamp | 0.063915 | 0.103734 | 0.119764 | 0.085034 | 0.113186 | 0.044249 | 0.044253 | True |
| 20mm | fallback_c0 | 0.081734 | 0.150537 | 0.173457 | 0.142456 | 0.168917 | 0.232577 | 0.101675 | True |
| 50mm | C0_baseline | 0.137716 | 0.215605 | 0.228256 | 0.197683 | 0.222736 | 0.207760 | 0.181653 | False |
| 50mm | raw_extrapolation | 0.125632 | 0.183025 | 0.194456 | 0.165281 | 0.193179 | 0.123063 | 0.122435 | True |
| 50mm | boundary_clamp | 0.125704 | 0.183024 | 0.194456 | 0.165485 | 0.193179 | 0.123259 | 0.122639 | True |
| 50mm | fallback_c0 | 0.128404 | 0.192168 | 0.230718 | 0.173071 | 0.222729 | 0.129989 | 0.130224 | False |

## Position-level policy results

详细 CSV 对每个位置给出 Bias、MAE、RMSE、P95、Max abs、超域点数、实际修正幅度和 0.2 mm 判定。下表列出 RMSE/P95/Max abs 及超域点数。

| dataset | position | directory | v median | policy | RMSE | P95 | Max abs | s outside count | correction mean abs | correction max abs | <=0.2 mm |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 20mm | P01_v125.0 | frame_000974_measure | 125.0 | C0_baseline | 0.117782 | 0.149983 | 0.156626 | 79 | 0.000000 | 0.000000 | True |
| 20mm | P01_v125.0 | frame_000974_measure | 125.0 | raw_extrapolation | 0.248485 | 0.300824 | 0.321391 | 79 | 0.387149 | 0.456333 | False |
| 20mm | P01_v125.0 | frame_000974_measure | 125.0 | boundary_clamp | 0.079490 | 0.105788 | 0.119639 | 79 | 0.218769 | 0.218769 | True |
| 20mm | P01_v125.0 | frame_000974_measure | 125.0 | fallback_c0 | 0.142456 | 0.168917 | 0.173457 | 79 | 0.000000 | 0.000000 | True |
| 20mm | P02_v346.0 | frame_004021_measure | 346.0 | C0_baseline | 0.015604 | 0.027624 | 0.039342 | 0 | 0.000000 | 0.000000 | True |
| 20mm | P02_v346.0 | frame_004021_measure | 346.0 | raw_extrapolation | 0.045788 | 0.073058 | 0.080130 | 0 | 0.108568 | 0.139650 | True |
| 20mm | P02_v346.0 | frame_004021_measure | 346.0 | boundary_clamp | 0.045788 | 0.073058 | 0.080130 | 0 | 0.108568 | 0.139650 | True |
| 20mm | P02_v346.0 | frame_004021_measure | 346.0 | fallback_c0 | 0.045788 | 0.073058 | 0.080130 | 0 | 0.108568 | 0.139650 | True |
| 20mm | P03_v849.0 | frame_005772_measure | 849.0 | C0_baseline | 0.053000 | 0.077296 | 0.084991 | 0 | 0.000000 | 0.000000 | True |
| 20mm | P03_v849.0 | frame_005772_measure | 849.0 | raw_extrapolation | 0.053262 | 0.076317 | 0.084120 | 0 | 0.024815 | 0.027795 | True |
| 20mm | P03_v849.0 | frame_005772_measure | 849.0 | boundary_clamp | 0.053262 | 0.076317 | 0.084120 | 0 | 0.024815 | 0.027795 | True |
| 20mm | P03_v849.0 | frame_005772_measure | 849.0 | fallback_c0 | 0.053262 | 0.076317 | 0.084120 | 0 | 0.024815 | 0.027795 | True |
| 20mm | P04_v1467.5 | frame_007020_measure | 1467.5 | C0_baseline | 0.067953 | 0.116644 | 0.120739 | 0 | 0.000000 | 0.000000 | True |
| 20mm | P04_v1467.5 | frame_007020_measure | 1467.5 | raw_extrapolation | 0.065783 | 0.113186 | 0.117122 | 0 | 0.033552 | 0.034631 | True |
| 20mm | P04_v1467.5 | frame_007020_measure | 1467.5 | boundary_clamp | 0.065783 | 0.113186 | 0.117122 | 0 | 0.033552 | 0.034631 | True |
| 20mm | P04_v1467.5 | frame_007020_measure | 1467.5 | fallback_c0 | 0.065783 | 0.113186 | 0.117122 | 0 | 0.033552 | 0.034631 | True |
| 20mm | P05_v2006.5 | frame_008310_measure | 2006.5 | C0_baseline | 0.039871 | 0.061648 | 0.067217 | 0 | 0.000000 | 0.000000 | True |
| 20mm | P05_v2006.5 | frame_008310_measure | 2006.5 | raw_extrapolation | 0.040781 | 0.062399 | 0.067892 | 0 | 0.001138 | 0.002206 | True |
| 20mm | P05_v2006.5 | frame_008310_measure | 2006.5 | boundary_clamp | 0.040781 | 0.062399 | 0.067892 | 0 | 0.001138 | 0.002206 | True |
| 20mm | P05_v2006.5 | frame_008310_measure | 2006.5 | fallback_c0 | 0.040781 | 0.062399 | 0.067892 | 0 | 0.001138 | 0.002206 | True |
| 20mm | P06_v2516.5 | frame_009614_measure | 2516.5 | C0_baseline | 0.079590 | 0.095089 | 0.105500 | 0 | 0.000000 | 0.000000 | True |
| 20mm | P06_v2516.5 | frame_009614_measure | 2516.5 | raw_extrapolation | 0.072979 | 0.089315 | 0.098054 | 0 | 0.002670 | 0.003684 | True |
| 20mm | P06_v2516.5 | frame_009614_measure | 2516.5 | boundary_clamp | 0.072979 | 0.089315 | 0.098054 | 0 | 0.002670 | 0.003684 | True |
| 20mm | P06_v2516.5 | frame_009614_measure | 2516.5 | fallback_c0 | 0.072979 | 0.089315 | 0.098054 | 0 | 0.002670 | 0.003684 | True |
| 20mm | P07_v2593.0 | frame_011317_measure | 2593.0 | C0_baseline | 0.094912 | 0.124238 | 0.131315 | 0 | 0.000000 | 0.000000 | True |
| 20mm | P07_v2593.0 | frame_011317_measure | 2593.0 | raw_extrapolation | 0.085034 | 0.111899 | 0.119764 | 0 | 0.005559 | 0.007672 | True |
| 20mm | P07_v2593.0 | frame_011317_measure | 2593.0 | boundary_clamp | 0.085034 | 0.111899 | 0.119764 | 0 | 0.005559 | 0.007672 | True |
| 20mm | P07_v2593.0 | frame_011317_measure | 2593.0 | fallback_c0 | 0.085034 | 0.111899 | 0.119764 | 0 | 0.005559 | 0.007672 | True |
| 20mm | P08_v2905.0 | frame_012686_measure | 2905.0 | C0_baseline | 0.075384 | 0.112949 | 0.127979 | 65 | 0.000000 | 0.000000 | True |
| 20mm | P08_v2905.0 | frame_012686_measure | 2905.0 | raw_extrapolation | 0.048167 | 0.085026 | 0.101317 | 65 | 0.053818 | 0.063654 | True |
| 20mm | P08_v2905.0 | frame_012686_measure | 2905.0 | boundary_clamp | 0.052821 | 0.087638 | 0.102357 | 65 | 0.045214 | 0.045225 | True |
| 20mm | P08_v2905.0 | frame_012686_measure | 2905.0 | fallback_c0 | 0.094645 | 0.132883 | 0.147602 | 65 | 0.001339 | 0.044980 | True |
| 50mm | P01_v429.5 | frame_065292_measure | 429.5 | C0_baseline | 0.016030 | 0.029202 | 0.034315 | 0 | 0.000000 | 0.000000 | True |
| 50mm | P01_v429.5 | frame_065292_measure | 429.5 | raw_extrapolation | 0.042846 | 0.061554 | 0.065448 | 0 | 0.047893 | 0.071441 | True |
| 50mm | P01_v429.5 | frame_065292_measure | 429.5 | boundary_clamp | 0.042846 | 0.061554 | 0.065448 | 0 | 0.047893 | 0.071441 | True |
| 50mm | P01_v429.5 | frame_065292_measure | 429.5 | fallback_c0 | 0.042846 | 0.061554 | 0.065448 | 0 | 0.047893 | 0.071441 | True |
| 50mm | P02_v1272.0 | frame_063995_measure | 1272.0 | C0_baseline | 0.127847 | 0.158910 | 0.164759 | 0 | 0.000000 | 0.000000 | True |
| 50mm | P02_v1272.0 | frame_063995_measure | 1272.0 | raw_extrapolation | 0.122271 | 0.153525 | 0.160075 | 0 | 0.030199 | 0.032875 | True |
| 50mm | P02_v1272.0 | frame_063995_measure | 1272.0 | boundary_clamp | 0.122271 | 0.153525 | 0.160075 | 0 | 0.030199 | 0.032875 | True |
| 50mm | P02_v1272.0 | frame_063995_measure | 1272.0 | fallback_c0 | 0.122271 | 0.153525 | 0.160075 | 0 | 0.030199 | 0.032875 | True |
| 50mm | P03_v2084.0 | frame_062878_measure | 2084.0 | C0_baseline | 0.132393 | 0.149160 | 0.157695 | 0 | 0.000000 | 0.000000 | True |
| 50mm | P03_v2084.0 | frame_062878_measure | 2084.0 | raw_extrapolation | 0.132096 | 0.148616 | 0.157135 | 0 | 0.000504 | 0.001020 | True |
| 50mm | P03_v2084.0 | frame_062878_measure | 2084.0 | boundary_clamp | 0.132096 | 0.148616 | 0.157135 | 0 | 0.000504 | 0.001020 | True |
| 50mm | P03_v2084.0 | frame_062878_measure | 2084.0 | fallback_c0 | 0.132096 | 0.148616 | 0.157135 | 0 | 0.000504 | 0.001020 | True |
| 50mm | P04_v2846.5 | frame_061303_measure | 2846.5 | C0_baseline | 0.197683 | 0.222736 | 0.228256 | 11 | 0.000000 | 0.000000 | False |
| 50mm | P04_v2846.5 | frame_061303_measure | 2846.5 | raw_extrapolation | 0.165281 | 0.193179 | 0.194456 | 11 | 0.039046 | 0.047852 | True |
| 50mm | P04_v2846.5 | frame_061303_measure | 2846.5 | boundary_clamp | 0.165485 | 0.193179 | 0.194456 | 11 | 0.038849 | 0.045225 | True |
| 50mm | P04_v2846.5 | frame_061303_measure | 2846.5 | fallback_c0 | 0.173071 | 0.222729 | 0.230718 | 11 | 0.032127 | 0.044998 | False |

## Decision

- `C1_BOUNDARY_POLICY = SAFE_GUARD_AVAILABLE`。
- 推荐策略：`boundary_clamp`。
- 同时满足两组全部位置 0.2 mm、安全性门槛的策略：boundary_clamp。
- 原始 extrapolation 的超域点总数：20 mm = 144, 50 mm = 11；它在 20 mm Top 的 worst Max abs = 0.321391 mm。
- boundary clamp 的两组 worst Max abs：20 mm = 0.119764 mm，50 mm = 0.194456 mm。
- fallback C0 的两组 all-position gate：20 mm = True，50 mm = False。

该结论只评价边界策略，不代表重新校准或放宽 C1 的 frozen 参数；若没有策略同时通过两组门槛，则说明当前 calibration domain 不足。

## Artifacts

- `boundary_policy_comparison.csv`：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_boundary_policy`
