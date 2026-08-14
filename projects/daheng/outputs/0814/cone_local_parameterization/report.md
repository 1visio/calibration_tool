# Task 3B-1 — Circular Cone 等价局部参数化与验证

**VALIDATION_OPENED = FALSE**
**PRODUCTION_CONE_MODIFIED = FALSE**
**LOCAL_PARAMETERIZATION_EQUIVALENCE = PASS**

## 数据与冻结项

- FIT-only: 001–018 + 025–036，共 30 frame；没有读取 019–024、037–040 的图像、点或 residual。
- Formal working domain: v=[241.998, 2731.978] px；evaluation grid 与 Task 3A 相同。
- Formal Cone SHA-256 before: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`；after: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`。
- P_ref 由 30 个 FIT frame centroid 等权得到：`[-5.400870389961511, -7.002832713450796, 681.3566154039942]` mm。

## 1. 数学定义

详见 `local_parameterization_definition.md`。新参数为 `[theta_axis, phi_axis, c1, c2, rho_ref, q]`，其中 `q=cot(alpha)`，并通过固定 P_ref 的 axis-normal 截面严格恢复 legacy apex 与 alpha。

## 2. 等价性结果

| model | max theta roundtrip error | axis dot | nappe preserved | max objective-vector diff | max lambda diff / mm |
|---|---:|---:|---:|---:|---:|
| M0 | 1.137e-13 | 1.000000000000000 | True | 3.553e-14 | 2.956e-12 |
| M_diag_fullfit | 1.776e-15 | 1.000000000000000 | True | 2.842e-14 | 0.000e+00 |

- Across M0 and M_diag_fullfit, maximum legacy→local→legacy parameter error: `1.137e-13` (native units).
- Across both models, maximum evaluation-grid `|lambda_legacy-lambda_roundtrip|`: `2.956e-12` mm; required threshold is `1.0e-6` mm.
- Objective cost and residual vector equivalence: `PASS`; ray intersection equivalence: `PASS`; axis/nappe equivalence: `PASS`.

## 3. Interpretation

- The local coordinates are a coordinate change only. They do not add a prior, regularization, v-dependent term, polynomial correction or residual compensation.
- The local coordinates are closer to the finite observed patch because the apex is represented through a nearby axis-line cross-section and local slope. This is a parameterization hypothesis; improved conditioning must be tested separately.
- The top-edge residual is deliberately not analyzed or corrected in this task.

## 4. Next step

- Task 3B-2 local-parameter Full-FIT + SVD + jackknife may proceed: `YES`.
- No diagnostic local parameter vector is written to the production Cone file.

Outputs are under `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0814\cone_local_parameterization`.
