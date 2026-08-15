# Task 6H-1 — Augmented camera calibration stability A/B

`EDGE_EXTENSION_CAMERA_GAIN = B. MODERATE`
推荐冻结 candidate：`M1-core`

本审计只读取 M0 chess 001–018、M1 extension chess 026/027/028/035/031/033/032/030，以及激光 FIT 001–018/025–036。Validation 未打开；正式 K/D 文件未修改，未更换 distortion model，未拟合 Cone。

## Fixed-coverage corner-noise MC

Raw values are relative to formal M0 truth; centered values are relative to each dataset's own full-data candidate and are the stability comparison.

| dataset | MC success | raw global P95 median (mm) | centered global P95 median (mm) | centered P95 tail (mm) | centered max (mm) | fx std (px) | k2 std |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3 | 0.0389737 | 0.0389737 | 0.0546164 | 0.0563545 | 0.42273 | 0.00135486 |
| M1-core | 3 | 0.108169 | 0.126712 | 0.17666 | 0.18221 | 1.56826 | 0.00507884 |
| M1-full | 3 | 0.176365 | 0.0528386 | 0.0892755 | 0.0933241 | 0.666953 | 0.00774008 |

## Camera-side LOO stability

| dataset | LOO frame-P95 median across omissions (mm) | P90 | P95 | max |
|---|---:|---:|---:|---:|
| M0 | 0.0387069 | 0.26377 | 0.33741 | 0.399968 |
| M1-core | 0.117254 | 0.181688 | 0.24674 | 0.306194 |
| M1-full | 0.154313 | 0.204741 | 0.265622 | 0.326454 |

The LOO tail and max quantify frame dependence; the raw M1-vs-M0 shift is not treated as MC uncertainty. M0 high-leverage omissions 002/001/010/003/017 and extension omissions are in `m0_m1_loo_stability.csv` and `extension_frame_leverage.csv`.

## Coverage and coupling

| dataset | dimension | range | min–max / Spearman |
|---|---|---:|---:|
| M0 | depth | 63.601 | 648.677–712.278 |
| M0 | tilt | 22.2323 | 2.53347–24.7658 |
| M0 | apparent_size | 0.258762 | 0.246413–0.505175 |
| M0 | sensor_u | 0.334687 | 0.342337–0.677024 |
| M0 | sensor_v | 0.301452 | 0.338838–0.640289 |
| M0 | tilt_depth | — | Spearman=-0.576883 |
| M0 | tilt_apparent_size | — | Spearman=0.24871 |
| M1-core | depth | 63.601 | 648.677–712.278 |
| M1-core | tilt | 24.1161 | 2.53347–26.6496 |
| M1-core | apparent_size | 0.298219 | 0.246413–0.544631 |
| M1-core | sensor_u | 0.364055 | 0.312969–0.677024 |
| M1-core | sensor_v | 0.301452 | 0.338838–0.640289 |
| M1-core | tilt_depth | — | Spearman=-0.616036 |
| M1-core | tilt_apparent_size | — | Spearman=0.274986 |
| M1-full | depth | 63.601 | 648.677–712.278 |
| M1-full | tilt | 24.1161 | 2.53347–26.6496 |
| M1-full | apparent_size | 0.298219 | 0.246413–0.544631 |
| M1-full | sensor_u | 0.364055 | 0.312969–0.677024 |
| M1-full | sensor_v | 0.301452 | 0.338838–0.640289 |
| M1-full | tilt_depth | — | Spearman=-0.631453 |
| M1-full | tilt_apparent_size | — | Spearman=0.316239 |

## 结论

M1 的判断同时考虑 camera-side LOO、fixed-coverage corner-noise MC 和覆盖耦合；training reprojection RMSE 不是选择依据。若 M1 降低了单帧 leverage 但 tilt-depth 相关仍高，则 observability 仅部分改善，不能宣称 depth/tilt 已解耦。

## 输出

- `m0_m1_intrinsics_comparison.csv`
- `m0_m1_loo_stability.csv`
- `m0_m1_corner_mc.csv`
- `extension_frame_leverage.csv`
- `m0_m1_coverage_comparison.csv`
- `provenance.json
