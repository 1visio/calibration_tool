# FIT pose geometric observability audit

`POSE_DIVERSITY = SUFFICIENT`
`RECOMMENDED_CURATED_FIT_SIZE = 14`

## Scope and guardrails

- FIT only: 001–018, 025–036, 049–054 (36 poses).
- PnP R/t uses geometry columns only: existing FIT 001–018/025–036 pose records plus direct `SOLVEPNP_ITERATIVE + solvePnPRefineLM` for the six 049–054 chess images absent from that table.
- Current point input: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\full_fit_v_coverage_audit\full_fit_points.csv`; current 100 px aggregate: `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\full_fit_v_coverage_audit\full_fit_v_coverage.csv`.
- The point table is the current `full_board_physical` mask result (inset=0 mm); v support is recomputed from its retained points.
- No Validation image/file was opened. No Plane/Quadratic/Cone was fitted. No model residual was read or used for ranking, deletion, or selection.

## Geometry-only selection gates

- v continuity: every 10 px cell in [0, 3000) occupied, with all 100 px bins populated.
- Edge support: each of 0–100, 100–200, 2800–2900, 2900–3000 px has at least 2 selected poses.
- Depth/lambda excitation: all occupied full-range depth bins (7) and lambda bins (8) represented; both spans retain ≥95% of full FIT.
- Normal diversity: every full-pose normal is within 5° of a selected normal and selected normal diameter is within 1° of full.
- Translation diversity: all 3 equal-width board-center X and Y bins represented.

## Full FIT geometry range

- Board-center Z: 646.045–712.276 mm; span 66.232 mm.
- Camera-board distance: 648.178–714.391 mm.
- Translation norm: 619.386–734.092 mm.
- Board tilt: 2.534–26.648°.
- Lambda truth: 599.916–713.864 mm; span 113.948 mm.
- Current 0817 100 px reference: 30/30 populated bins; minimum frame multiplicity 11.

## Geometric near duplicates

Pairwise distance uses normal angle, board-center translation difference, board-center Z difference, 100 px v-support Jaccard, and lambda-span difference. The strict flag is: angle≤3°, translation≤45 mm, depth≤10 mm, v Jaccard≥0.75, lambda-span difference≤6 mm.

| pair | normal Δ / ° | translation Δ / mm | depth Δ / mm | v Jaccard | lambda-span Δ / mm | similarity |
|---|---:|---:|---:|---:|---:|---:|
| 011–054 | 1.212 | 15.894 | 2.691 | 0.905 | 2.958 | 0.712 |
| 013–036 | 0.029 | 40.008 | 1.777 | 0.926 | 0.095 | 0.692 |
| 006–013 | 0.112 | 16.275 | 9.296 | 0.769 | 0.724 | 0.683 |
| 006–036 | 0.090 | 29.577 | 7.519 | 0.769 | 0.629 | 0.665 |
| 029–034 | 2.157 | 22.604 | 0.231 | 0.833 | 5.479 | 0.558 |

Strict near-duplicate pairs: **5**; broader geometry candidates: **24**. Similar geometry does not imply identical v support: the pair CSV retains both 10 px and 100 px support overlap.

## Leave-one-pose check

All 36 leave-one-out cases were evaluated against the gates above. Cases with at least one geometric loss: **5**.

| deleted pose | lost gate(s) |
|---|---|
| 001 | normal |
| 015 | normal |
| 027 | normal |
| 031 | normal |
| 051 | depth/lambda |

The per-pose CSV contains the complete LOO metrics, including missing v cells, edge multiplicity, span ratios, normal nearest-cover angle, and translation-bin retention.

## Curated recommendation

- Solver: `greedy_local_prune+exact_pair_search_proven`; minimum-size proof for the declared linear geometry gates: **True**.
- Recommended IDs (14): **001, 006, 010, 013, 015, 017, 025, 027, 031, 049, 051, 052, 053, 054**.

| criterion | full 36 | curated | pass |
|---|---:|---:|:---:|
| v 10 px occupied cells | 300 | 300 | True |
| v 100 px occupied bins | 30 | 30 | True |
| minimum edge frame count | 11 | 3 | True |
| board-center Z span / mm | 66.232 | 66.232 | True |
| lambda truth span / mm | 113.948 | 113.699 | True |
| depth span ratio | 1.000 | 1.000 | True |
| lambda span ratio | 1.000 | 0.998 | True |
| normal cover max angle / ° | 0.000 | 4.720 | True |
| normal diameter / ° | 51.014 | 51.014 | True |
| translation X/Y bins | 3/3 | 3/3 | True |

## Historical 001–018 comparison

Historical is treated as the original 001–018 FIT pose set. The comparison is geometric only and uses the same current point table and gates.

| metric | Historical 001–018 | Full 36 | Curated |
|---|---:|---:|---:|
| pose count | 18 | 36 | 14 |
| v 10 px occupied cells | 294 | 300 | 300 |
| edge minimum frame count | 1 | 11 | 3 |
| board-center Z span / mm | 63.594 | 66.232 | 66.232 |
| lambda truth span / mm | 88.907 | 113.948 | 113.699 |
| normal diameter / ° | 43.546 | 51.014 | 51.014 |
| normal cover max to full / ° | 13.815 | 0.000 | 4.720 |

Historical 001–018 already contains the principal low/high tilt and near/far depth families, but the 025–036 and 049–054 extension poses provide the explicit v-edge and lambda/depth extremes needed by the complete-workdomain gates. The curated set therefore keeps only those extension poses that add a declared geometric bin or normal cover while removing geometry-near repeats.

## Files

- `pose_geometry_metrics.csv`: per-pose PnP geometry, v/lambda support and leave-one-out results.
- `pair_pose_similarity.csv`: all 630 pairwise geometry comparisons.
- `pose_similarity_matrix.png`: geometric similarity heatmap.
- `curated_fit_ids.json`: machine-readable selection and gate summary.
- `curated_v_coverage.png`: full-vs-curated support and excitation plot.

`POSE_DIVERSITY = SUFFICIENT`
`RECOMMENDED_CURATED_FIT_SIZE = 14`
