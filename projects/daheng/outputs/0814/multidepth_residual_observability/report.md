# Task 5B-1 — Multi-depth residual observability audit

`MULTIDEPTH_CORRECTION_FEASIBILITY = C. WEAK`

## Scope and safeguards

- Main FIT only: `001–018 + 025–036`, with `027` temporarily excluded (29 frames).
- Validation `019–024, 037–040` was not loaded.
- Models are read-only: formal `M0` and Task 5A's 29-frame Circular diagnostic model.
- Residual convention: `e_lambda = lambda_truth - lambda_model`.
- Analysed 26002 independent ray-plane truth points; invalid intersections: {'M0': 229, 'M_current29_circular': 229}.
- No correction, production parameter, spline, polynomial, or LUT was written.

## Fixed method

For each fixed 30/60/100 px v-bin, every frame has equal total weight. `lambda_ref` is the median of per-frame median truth depth in that bin. The depth slope is enabled only when at least 3 frames and at least 2 mm cross-frame median-depth span are present; otherwise the depth prediction falls back to offset-only. Uncertainty uses 500 frame bootstrap draws plus leave-one-frame-out.

The verdict uses the predeclared 60 px gates: STRONG requires global incremental explained fraction ≥20%, both edge RMSE reductions ≥15%, and ≥70% stable informative slopes. PARTIAL requires ≥5%, at least one edge ≥10%, and ≥40% stable slopes. Otherwise WEAK.

## Primary 60 px result (current 29-frame Circular)

| region | offset explained | offset+depth explained | incremental | RMSE reduction vs offset | frame-mean std after depth (mm) |
|---|---:|---:|---:|---:|---:|
| global | 0.393 | 0.417 | 0.024 | 0.020 | 0.0366 |
| top_formal_edge | 0.636 | 0.714 | 0.078 | 0.114 | 0.1236 |
| middle_formal | 0.379 | 0.403 | 0.024 | 0.020 | 0.0366 |
| bottom_formal_edge | 0.786 | 0.835 | 0.048 | 0.121 | 0.0171 |

## Observability and stability

- Informative 60 px bins: 42; stable b1 bins: 8 (19.0%).
- Informative-bin b1 range: -0.00251478 to 0.000934764 mm/mm.
- Across adjacent informative 60 px bins, b1 changes sign 6 times (16 positive, 26 negative); this is not a smooth, repeatable v-trend.
- Top has 1 depth-informative 60 px bin and 0 stable bin; bottom has 1 informative bin and 1 stable bin.
- Global frame-mean residual std after offset-only / after depth: 0.0367 / 0.0366 mm.
- A bin is called stable only if its bootstrap 90% interval excludes zero and both bootstrap and LOFO sign consistency are at least 80%.

### Bin-scale consistency

| bin width | informative bins | stable bins | stable fraction | global incremental | top RMSE reduction | bottom RMSE reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 30 px | 83 | 11 | 13.3% | 2.7% | 16.4% | 13.6% |
| 60 px | 42 | 8 | 19.0% | 2.4% | 11.4% | 12.1% |
| 100 px | 26 | 4 | 15.4% | 2.0% | 11.8% | 14.0% |

All three scales agree that the depth term adds only about 2–3% global explained energy, while stable-slope coverage remains only 13–19%. Edge in-sample improvement therefore does not amount to a globally observable correction.

### Read-only M0 cross-check

M0 reaches 2.6% global incremental explanation, with 15/42 stable 60 px slopes. Its top/bottom RMSE reductions are 9.3%/35.8%. The larger M0 bottom-only response is not mirrored at top and does not change the current-model verdict.

## Answers

1. Adding depth changes global explained energy by 2.4% beyond offset-only.
2. Top / bottom RMSE reductions beyond offset-only are 11.4% / 12.1%.
3. b1 is stable in only 19.0% of depth-informative 60 px bins and changes sign repeatedly. It is not yet a stable, smooth, cross-frame-repeatable function of v.
4. Frame dependence remains substantial after the in-sample depth decomposition (frame-mean std ratio 0.998).

## What this can and cannot establish

This FIT-only audit can show whether a low-order depth term is locally observable and repeatable across frames. It cannot establish generalization, choose a deployable v-parameterization, or justify production correction. Those require a separately frozen candidate followed by untouched validation.

`MULTIDEPTH_CORRECTION_FEASIBILITY = C. WEAK`
