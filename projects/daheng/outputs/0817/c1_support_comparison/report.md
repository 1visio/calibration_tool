# Frozen C1_4k v/s support comparison

`C1_SUPPORT_COVERAGE = INSUFFICIENT`
`NEED_TOP_BOTTOM_FIT = YES`

## Scope and frozen boundary

- FIT 输入：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\spatial_residual_observability\fit_ray_residual_points.csv`；只保留 `001–018、025–036`，共 26663 个实际 C1 拟合点、30 帧；frame 027 retained = `true`。
- C1 模型：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation\frozen_c1_model.json`；model SHA-256 = `fb702821a2156e7ec409b0a1c733fcc16a89eddc417f3fcd8a4ffbaaa7dbd5e4`；parameter SHA-256 = `be7c316c91b54ac9b13a1ff2485a9ca2ceebf6644a57a91ffca98f858f726037`。
- FIT CSV SHA-256 = `ea68251e05e1d472db7e25bb2090b2094f2e813f04dd5475fea1c06e4af01f8f`。
- 标准件只读取既有 20 mm、50 mm 测试目录的 `laser_center.csv`、`baseline_points.csv`、`height_points.csv`；未读取 Validation 019–024、037–040。
- 未重新拟合 C1、K/D 或 Cone，未修改 knots/penalty；s 使用 Frozen PCA 定义。
- `v` 支持域定义为 FIT 实际拟合点的全局 min/max；`s` 支持域定义为 Frozen C1 的 `domain_min/domain_max`，并与 FIT 实际 `pca_s` min/max 做一致性核对。
- 覆盖判定按轴独立：点在闭区间内为 interpolation，区间外为 extrapolation；超出距离是到最近域边界的距离。
- 标准件位置的主要判断使用 height subset；CSV 同时保留 laser-center、baseline、height 三类点的逐点状态。

## FIT training support

- FIT v support：`[238.984, 2874.006] px`。
- FIT s support：`[-0.164122564, 0.194958485]`。
- Frozen C1 s domain 与 FIT s min/max 一致：`true`。

| coordinate | p01 | p05 | p25 | median | p75 | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v / px | 354.994 | 517.975 | 966.990 | 1424.036 | 1876.985 | 2465.968 | 2660.012 | 238.984 | 2874.006 |
| s | -0.148335 | -0.126035 | -0.064891 | -0.002587 | 0.058998 | 0.139339 | 0.165739 | -0.164122564 | 0.194958485 |

## Standard-object height-point coverage

下表的 outside count/fraction 同时报告 v-domain 和 s-domain；`coverage` 是两者联合状态。

| dataset | position | v range / px | v outside | max v distance / px | s range | s outside | max s distance | coverage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 20mm | P01_v125.0 | [86.0, 164.0] | 79 (100.0%) | 152.984 | [-0.185001, -0.174351] | 79 (100.0%) | 0.020878 | extrapolation |
| 20mm | P02_v346.0 | [311.0, 381.0] | 0 (0.0%) | 0.000 | [-0.154288, -0.144738] | 0 (0.0%) | 0.000000 | interpolation |
| 20mm | P03_v849.0 | [812.0, 886.0] | 0 (0.0%) | 0.000 | [-0.085986, -0.075907] | 0 (0.0%) | 0.000000 | interpolation |
| 20mm | P04_v1467.5 | [1431.0, 1504.0] | 0 (0.0%) | 0.000 | [-0.001720, 0.008215] | 0 (0.0%) | 0.000000 | interpolation |
| 20mm | P05_v2006.5 | [1969.0, 2044.0] | 0 (0.0%) | 0.000 | [0.071511, 0.081726] | 0 (0.0%) | 0.000000 | interpolation |
| 20mm | P06_v2516.5 | [2480.0, 2553.0] | 0 (0.0%) | 0.000 | [0.141162, 0.151124] | 0 (0.0%) | 0.000000 | interpolation |
| 20mm | P07_v2593.0 | [2557.0, 2629.0] | 0 (0.0%) | 0.000 | [0.151670, 0.161498] | 0 (0.0%) | 0.000000 | interpolation |
| 20mm | P08_v2905.0 | [2872.0, 2938.0] | 64 (95.5%) | 63.994 | [0.194690, 0.203711] | 65 (97.0%) | 0.008753 | mixed |
| 50mm | P01_v429.5 | [392.0, 467.0] | 0 (0.0%) | 0.000 | [-0.143091, -0.132862] | 0 (0.0%) | 0.000000 | interpolation |
| 50mm | P02_v1272.0 | [1234.0, 1310.0] | 0 (0.0%) | 0.000 | [-0.028384, -0.018040] | 0 (0.0%) | 0.000000 | interpolation |
| 50mm | P03_v2084.0 | [2044.0, 2124.0] | 0 (0.0%) | 0.000 | [0.081872, 0.092771] | 0 (0.0%) | 0.000000 | interpolation |
| 50mm | P04_v2846.5 | [2810.0, 2883.0] | 9 (12.2%) | 8.994 | [0.186364, 0.196340] | 11 (14.9%) | 0.001381 | mixed |

## Standard-object laser-center coverage

laser-center 覆盖通常比 height subset 更宽；它反映原始测量行中所有可见点的域外比例，不替代上表的目标位置判断。

| dataset | position | laser-center v range | v outside | laser-center s range | s outside | coverage |
|---|---|---:|---:|---:|---:|---|
| 20mm | P01_v125.0 | [0.0, 2997.0] | 230 (10.6%) | [-0.196838, 0.211647] | 230 (10.6%) | mixed |
| 20mm | P02_v346.0 | [17.0, 2997.0] | 268 (12.1%) | [-0.194516, 0.211647] | 268 (12.1%) | mixed |
| 20mm | P03_v849.0 | [0.0, 2997.0] | 296 (13.8%) | [-0.196838, 0.211647] | 297 (13.9%) | mixed |
| 20mm | P04_v1467.5 | [0.0, 2997.0] | 267 (12.4%) | [-0.196883, 0.211647] | 268 (12.5%) | mixed |
| 20mm | P05_v2006.5 | [0.0, 2999.0] | 329 (15.4%) | [-0.196883, 0.211962] | 330 (15.5%) | mixed |
| 20mm | P06_v2516.5 | [0.0, 2999.0] | 258 (11.5%) | [-0.196882, 0.211962] | 259 (11.6%) | mixed |
| 20mm | P07_v2593.0 | [0.0, 2818.0] | 210 (9.3%) | [-0.196882, 0.187222] | 211 (9.4%) | mixed |
| 20mm | P08_v2905.0 | [0.0, 2999.0] | 329 (14.0%) | [-0.196883, 0.211962] | 331 (14.1%) | mixed |
| 50mm | P01_v429.5 | [125.0, 2997.0] | 169 (8.0%) | [-0.179768, 0.211647] | 170 (8.0%) | mixed |
| 50mm | P02_v1272.0 | [0.0, 2997.0] | 272 (12.8%) | [-0.196882, 0.211647] | 273 (12.9%) | mixed |
| 50mm | P03_v2084.0 | [0.0, 2999.0] | 306 (14.5%) | [-0.196882, 0.211962] | 307 (14.6%) | mixed |
| 50mm | P04_v2846.5 | [0.0, 2995.0] | 321 (13.9%) | [-0.196882, 0.211416] | 324 (14.0%) | mixed |

## Edge findings

- v≈125 对应 20mm P01_v125.0：FIT v 下界为 238.984 px，height v range 为 [86.0, 164.0] px，79/79 点 v 外推；s 也有 79/79 点外推。
- v≈2905 对应 20mm P08_v2905.0：FIT v 上界为 2874.006 px，height v range 为 [2872.0, 2938.0] px，64/67 点 v 外推；s 有 65/67 点外推。
- Top/Bottom height positions with s extrapolation: 20mm P01_v125.0, 20mm P08_v2905.0, 50mm P04_v2846.5。
- Top/Bottom height positions with v extrapolation: 20mm P01_v125.0, 20mm P08_v2905.0, 50mm P04_v2846.5。
- C1 实际自变量是 s，因此补采优先级以 s-domain gap 为准；v-domain gap 是对应的传感器位置证据。

## Decision

- `C1_SUPPORT_COVERAGE = INSUFFICIENT`。
- `NEED_TOP_BOTTOM_FIT = YES`。
- 判定依据：Top/Bottom height points 存在明显 s-domain 外推（至少一个边缘位置的 s 外推比例达到 25% 以上）。
- 建议补采/扩展 FIT：优先覆盖 20 mm Top（v≈125，s 约 -0.18）和 20 mm Bottom（v≈2905，s 约 +0.20）；50 mm Bottom 仍有小段 s 超域，也应作为边界余量一并覆盖。
- 当前 C1 参数本身没有被修改；在补采前，s-domain 外只能视为外推，不能当作已验证的 interpolation。

## Artifacts

- `c1_support_comparison.csv`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_support_comparison\c1_support_comparison.csv`
- `c1_support_comparison.png`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_support_comparison\c1_support_comparison.png`