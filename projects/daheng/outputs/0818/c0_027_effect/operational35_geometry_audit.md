# Operational-35 geometry-only coverage audit

`AUDIT_SCOPE = OPERATIONAL_35`

## Boundary

- FIT-only geometry audit；只使用现有 Full-36 PnP geometry artifact 与正式 `full_board_physical` mask 的 raw FIT point support。
- `frame027` 状态：`EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN`；原因是超出实际工作姿态域，不是 residual-based deletion。
- Operational reference domain = Full-36 minus 027，共 35 poses；027 保留在原始 Full-36 artifacts 中，本 audit 不修改它们。
- Normal coverage、depth/lambda span、translation bins 的 reference 全部重新由这 35 个 operational poses 建立；不再要求覆盖 027 的超大倾角法向。
- 不读取 Validation；不读取模型 residual；不拟合 Plane/Quadratic/Cone。

## Operational-35 reference range

- Pose count / points: **35 / 31500**
- Board tilt: **2.534–24.766°**; excluded 027: **26.648°**
- Operational normal pairwise diameter: **46.126°**
- Board-center Z: **646.045–712.276 mm**, span **66.232 mm**
- Lambda truth: **599.916–713.864 mm**, span **113.948 mm**

## Coverage gates evaluated against the 35-pose domain

| gate | Operational-35 result | pass |
|---|---:|:---:|
| v 10 px occupied cells | 300/300 | True |
| v 100 px occupied bins | 30/30 | True |
| edge frame count 0–100 / 100–200 / 2800–2900 / 2900–3000 | 11 / 17 / 15 / 11 | True |
| depth span ratio to operational reference | 1.0000 | True |
| lambda span ratio to operational reference | 1.0000 | True |
| depth/lambda bins | 7/8; 8/8 | True |
| normal cover max angle to operational reference | 0.0000° | True |
| normal diameter | 46.1262° / reference 46.1262° | True |
| translation X/Y bins | 3/3; 3/3 | True |
| overall geometry | — | **True** |

## Leave-one-operational-pose diagnostic

- Removing one of the 35 poses was checked against the same Operational-35 reference gates. Failed cases: **3/35**.

| removed pose | lost gate(s) |
|---|---|
| 015 | normal |
| 031 | normal |
| 051 | depth_lambda |

## Interpretation

本 audit 的 `normal_angle_diversity_ok` 只回答 Operational-35 是否覆盖自己的实际法向域；它不回答 027 所代表的超大倾角域是否仍被覆盖。Full-36 的原始 normal diameter 与原 audit 结论保持不变，未被本文件改写。

`GEOMETRY_ONLY_OPERATIONAL35 = PASS`

## Reuse/new calculation boundary

- Reused: existing Full-36 PnP geometry rows and raw Full-36 `full_board_physical` FIT point support.
- Newly calculated: 027-filtered support unions, operational re-ranged bins, normal matrix/reference, and leave-one-operational-pose checks.
