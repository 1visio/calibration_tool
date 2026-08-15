# Task 5D-2 — Matched-plane sensor-position audit

`SENSOR_POSITION_EFFECT = C. WEAK`

## Scope and boundary

- Data: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0814_v2`; 15 FIT triplets only. The manifest has no explicit Top/Middle/Bottom label, so regions are assigned by measured laser-line median sensor v (lower v = Top): Top=006,007,008,009,010, Middle=001,002,003,004,005, Bottom=011,012,013,014,015.
- Validation was not opened or used. No Circular Cone was fitted/refit, no production file was modified, and no compensation was created.
- Each triplet independently uses PnP plane, Steger laser center, ray-plane `lambda_truth`, and the frozen Circular Cone `e_lambda = lambda_truth - lambda_model`.
- Frozen Circular provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`.

## PnP plane consistency

- Reference normal: `[0.045503017, -0.008778728, -0.998925628]`; reference plane d=711.669288 mm.
- Maximum normal deviation: **0.012500 deg**; maximum |d deviation|: **0.007563 mm**; maximum PnP RMSE: **0.1180 px**.
- Plane consistency gate: **PASS** (normal ≤0.163659 deg, |d|≤0.08724 mm, PnP RMSE≤0.4 px).

## Top / Middle / Bottom residuals

| region | groups | v-center median (px) | bias (mm) | RMSE (mm) | P95 abs (mm) | a_frame mean (mm) | k_frame mean | same-v repeat P95 (mm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Top | 006,007,008,009,010 | 1282.4963 | 0.1189 | 0.1399 | 0.2532 | 0.1189 | -0.0114 | 0.0360 |
| Middle | 001,002,003,004,005 | 1507.9729 | 0.1029 | 0.1192 | 0.1951 | 0.1029 | 0.0326 | 0.0282 |
| Bottom | 011,012,013,014,015 | 1660.5009 | 0.1251 | 0.1393 | 0.2260 | 0.1251 | -0.0022 | 0.0348 |

## Same-v repeatability and matched-v comparison

- Top: 10 within-region pairs; same-v residual delta RMSE median/P95 = **0.0154 / 0.0182 mm**; delta P95 median/P95 = **0.0313 / 0.0360 mm**.
- Middle: 10 within-region pairs; same-v residual delta RMSE median/P95 = **0.0123 / 0.0147 mm**; delta P95 median/P95 = **0.0243 / 0.0282 mm**.
- Bottom: 10 within-region pairs; same-v residual delta RMSE median/P95 = **0.0141 / 0.0199 mm**; delta P95 median/P95 = **0.0290 / 0.0348 mm**.

| matched-v pair | delta bias (mm) | delta RMSE (mm) | delta P95 abs (mm) |
|---|---:|---:|---:|
| Top vs Middle | 0.0082 | 0.0470 | 0.0898 |
| Top vs Bottom | -0.0131 | 0.0522 | 0.1046 |
| Middle vs Bottom | -0.0213 | 0.0487 | 0.0944 |

## Interpretation

- Expected diagnostic pattern Top positive / Middle near zero / Bottom negative: **NOT OBSERVED**.
- Region bias range: **0.0221 mm**; maximum matched-v region-pair residual delta P95: **0.1046 mm**.
- Same-v repeatability maximum P95: **0.0360 mm**.
- Matched-v cross-region differences exceed repeat noise in places, but they do not form a consistent Top→Middle→Bottom offset; this is a reason to investigate a pose/translation-dependent term, not evidence for a fixed sensor-v residual.
- `residual_vs_v.png` shows all point residuals and the per-region median curves on their common sensor-v support.

`SENSOR_POSITION_EFFECT = C. WEAK`.
A STRONG result requires the expected sign reversal after the PnP planes pass consistency. A MODERATE result requires a substantial region bias separation with repeatable same-v residuals. Otherwise the data do not support a fixed sensor-position residual conclusion.

The laser frames carry the acquisition `dynamic_range_low` warning; this quality limitation is retained in the CSV provenance fields and does not authorize changing the extraction or model.
