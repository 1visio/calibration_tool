# Frozen Operational-35 C1_4k

C1_FREEZE_STATUS = FROZEN_FOR_VALIDATION
C1_REPRODUCTION = PASS

## Scope

- 仅对既定 Operational-35 执行一次 C1_4k full-fit；未运行 C1_3k/C1_5k、grouped-CV 或模型选择。
- Frozen Full-36 Quadratic C0 未重新拟合。
- frame027 未进入 C1 FIT，只用于 `exclude027_fullfit_027_heldout` reproduction stress check。
- Validation 未读取，生产配置未修改。

## Provenance / protocol

- Operational pose count: `35`；point count: `31500`。
- Operational IDs: `001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 025, 026, 028, 029, 030, 031, 032, 033, 034, 035, 036, 049, 050, 051, 052, 053, 054`。
- frame027: `EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN`；理由：**超出实际工作姿态域**。
- C0 SHA256: `113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27`。
- residual artifact SHA256: `6cff09a4036c815ef3931dd2e818650af0b6c3439b0bf8fb333827d9bd6d995e`。
- original fit script SHA256: `747ca1f81dbed283e4de04977048a41c0c0102e8170c854d081e3176829c633c`。
- git commit: `9ed98352bda139c5bd1bb26cc2185aa11a06d6d1`；worktree dirty: `True`。
- Protocol: cubic B-spline, degree 3, 4 interior knots, frame-balanced weighting, Huber IRLS, second-difference penalty 0.1.
- Target: `residual_centered_mm = residual_mm - frame_residual_median_mm`; correction sign: `residual - F(s)`; application: `lambda_final = lambda_quadratic + F(s)`.

## PCA-s reproduction

- PCA 使用 Full-36 `xn/yn`，包括 027；没有重新拟合 Operational-35 PCA。
- center: `(-0.0099487925627070629, -0.013651736422099612)`。
- axis_s: `(-0.0065829846382417512, 0.99997833192187335)`。
- s domain: `[-0.1923654071984672, 0.2166209347652564]`。
- recomputed `pca_s` vs stored `pca_s` max abs diff: `9.9746599868666408e-17`; tolerance `1e-12`。

## Frozen spline

- robust scale: `0.049474420454752595 mm`；robust iterations: `15`。
- training frames/points: `35` / `31500`。
- parameter SHA256: `91ecf9de35e16b7e5c6f9c3264ca8222a765eb9419e6f16138189bd21aeb3815`。
- 完整 knot vector、coefficients、PCA 参数和 protocol 已写入 `frozen_c1_4k.json`。

## frame027 reproduction

| metric | historical | reproduced | delta mm | tolerance mm | pass |
|---|---:|---:|---:|---:|---|
| rmse_mm | 0.34277693532781339 | 0.34277693532781339 | 0 | 1e-09 | PASS |
| p95_abs_mm | 0.58176842996733491 | 0.58176842996733491 | 0 | 1e-09 | PASS |
| p99_abs_mm | 0.63760172533971604 | 0.63760172533971604 | 0 | 1e-09 | PASS |
| max_abs_mm | 0.71068735887644252 | 0.71068735887644252 | 0 | 1e-09 | PASS |

`C1_REPRODUCTION = PASS`。

## LUT validation

- points: `2049`。
- grid round-trip max error: `1.0061396160665481e-16 mm`。
- dense linear-interpolation max error: `4.6109642465319567e-07 mm`。
- tolerance: `0.001 mm`；result: `PASS`。

## Outputs

- `frozen_c1_4k.json`：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\frozen_c1_4k.json
- `c1_4k_lut.csv`：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\c1_4k_lut.csv
- `c1_reproduction_check.csv`：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\c1_reproduction_check.csv
- `c1_freeze_manifest.json`：D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c1_4k_freeze\c1_freeze_manifest.json
- `c1_freeze_report.md`：本报告。
