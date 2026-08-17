# C1_4k Top/Bottom edge-extension coverage plan

`EDGE_EXTENSION_PLAN = READY`

## Scope

- This is a planning-only result; C1 is not refit and no model parameter is changed.
- The 20 mm / 50 mm standard-object points are used only to define the real operating support domain, never as C1 training points.
- Source support CSV: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_support_comparison\c1_support_comparison.csv`; SHA-256 = `f804bcf6c2c35bde244e151237e5b921bd370a7ac32ed4cef72544d7aed65e88`.
- Current FIT: `001–018`, `025–036`; current Validation: `019–024`, `037–040`; frame 027 retained = `true`.
- Existing Validation raw data was not read in this planning run.

## Real standard-object operating range

The primary range is the height subset that defines each accepted standard-object position. The full laser-center cloud is reported in JSON for traceability but is not used as the target-position domain because it spans the complete visible scan line.

| edge | observed v range / px | observed s range | positions represented |
|---|---:|---:|---|
| Top | [86.0, 467.0] | [-0.185001, -0.132862] | 20mm P01_v125.0, 50mm P01_v429.5 |
| Bottom | [2810.0, 2938.0] | [0.186364, 0.203711] | 20mm P08_v2905.0, 50mm P04_v2846.5 |

- Current FIT support is v=[238.984, 2874.006] px and s=[-0.164122564, 0.194958485].
- Observed full laser-center union, for reference only: v=[0.0, 2999.0] px and s=[-0.196883, 0.211962].

## Safety margin

- Margin rule: max(10% of each edge span, 50 px v guard, 0.005 s guard). Final targets are rounded outward to 10 px / 0.001 s.

| edge | exact domain with margin (v px) | exact domain with margin (s) | recommended rounded target (v px) | recommended rounded target (s) |
|---|---:|---:|---:|---:|
| Top | [36.0, 517.0] | [-0.190215, -0.127648] | [30, 520] | [-0.191, -0.127] |
| Bottom | [2760.0, 2988.0] | [0.181364, 0.208711] | [2760, 2990] | [0.181, 0.209] |

## New FIT extension

- Dataset: `fit_edge_extension_v2`; split=`fit`; pose IDs `041–044`.
- Minimum: **4 poses total**, Top=2, Bottom=2.
- Top: 041 outer guard near 20 mm Top (v≈125, s≈−0.180), 042 inner bridge near 50 mm Top (v≈430, s≈−0.138).
- Bottom: 043 inner bridge near 50 mm Bottom (v≈2847, s≈+0.191), 044 outer guard near 20 mm Bottom (v≈2905, s≈+0.199).

| pose | edge | role | reference center v / px | reference center s |
|---|---|---|---:|---:|
| 041 | Top | outer_guard | 125.0 | -0.179676 |
| 042 | Top | inner_bridge | 429.5 | -0.137976 |
| 043 | Bottom | inner_bridge | 2846.5 | 0.191352 |
| 044 | Bottom | outer_guard | 2905.0 | 0.199200 |

## New extreme-edge Validation

- Dataset: `validation_edge_holdout_v2`; split=`validation`; pose IDs `045–048`.
- Bare minimum: 2 poses (one outer Top + one outer Bottom). Recommended: **4 poses**, Top=2, Bottom=2.
- 045/046 independently repeat the outer/inner Top target centers; 047/048 independently repeat the inner/outer Bottom target centers.

| pose | edge | role | reference center v / px | reference center s |
|---|---|---|---:|---:|
| 045 | Top | outer_edge_holdout | 125.0 | -0.179676 |
| 046 | Top | inner_edge_holdout | 429.5 | -0.137976 |
| 047 | Bottom | inner_edge_holdout | 2846.5 | 0.191352 |
| 048 | Bottom | outer_edge_holdout | 2905.0 | 0.199200 |

## Strict FIT / Validation separation

- FIT IDs 041–044 and Validation IDs 045–048 are disjoint and outside the current 001–040 registry.
- Do not reuse a raw image, camera frame, board pose, acquisition session, or file hash between the two sets.
- New Validation manifest must explicitly use `dataset_id: validation_edge_holdout_v2` and `split: validation`; do not copy the old `validation_edge_holdout` metadata inconsistency where the split remained `fit`.
- FIT and Validation intentionally target the same physical edge domain, but use independent poses/sessions. This is the correct separation for in-domain independent generalization; forcing disjoint v/s ranges would make Validation an extrapolation test instead.
- 045–048 must never enter the C1 fit CSV or a future C1 refit. Freeze the future model before reading them once.

## Directory structure

```text
D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\fit_edge_extension_v2/
  dataset_manifest.yaml          # dataset_id=fit_edge_extension_v2, split=fit
  fit/
    chess 041.tif ... chess 044.tif
    nolaser 041.tif ... nolaser 044.tif
    laser 041.tif ... laser 044.tif
D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\validation_edge_holdout_v2/
  dataset_manifest.yaml          # dataset_id=validation_edge_holdout_v2, split=validation
  validation/
    chess 045.tif ... chess 048.tif
    nolaser 045.tif ... nolaser 048.tif
    laser 045.tif ... laser 048.tif
```

## Post-acquisition acceptance gates

1. FIT 041–044 的实际 union 必须覆盖 Top/Bottom 推荐 rounded v/s domain，且每个 edge 至少由两个不同 pose 提供支持。
2. Validation 045–048 必须独立采集、质量通过、与 FIT 文件/hash 完全不交集。
3. 先冻结未来 C1，再一次性读取新 Validation；不得用 Validation 调 knots、penalty、PCA 或边界策略。

## Artifacts

- `edge_extension_plan.json`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\edge_extension_plan`
- `report.md`: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\edge_extension_plan`