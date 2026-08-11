# B05_A10 H10 band candidate-selection audit

## Scope and provenance

- Dataset: `B05_A10`, task: `multiheight`, frames: 50.
- H10 selected ROI: `[1857, 1899]`; formal trim3 ROI: `[1860, 1896]` (37 columns).
- Before auto band: `[793, 895)`.
- Reference band: `[886, 927)`.
- After final band: `[793, 927)`.
- Formal Steger config SHA-256: `4ffe544c002bfcdcfee494dbe2fa9c8dbfb75bef292ccc5f0156e6e0e2acbf96`.
- No Steger, ROI, band, reference, geometry summary, or analysis parameter was modified.

## Per-frame audit

- Before median-center range: 891.445643–891.536525 px.
- After median-center range: 891.394312–891.480529 px.
- Before valid-column count range: 37–37.
- After valid-column count range: 37–37.
- Before selected-response median range: 3.594369–3.936987.
- After selected-response median range: 4.020063–4.243510.
- Full per-frame median/P95 response statistics are stored as `frame_summary` rows in the CSV.

## Paired frame-column differences

- Paired accepted opportunities: 1850/1850 (100.0000%).
- `abs(center difference) < 0.05 px`: 739 (39.9459%).
- `0.05 <= abs(center difference) <= 0.2 px`: 1111 (60.0541%).
- `abs(center difference) > 0.2 px`: 0 (0.0000%).
- Maximum absolute center difference: 0.193985 px.
- Median signed center difference: -0.061191 px; P95 absolute difference: 0.144133 px.
- Same integer candidate row (floor): 95.6757%; same rounded row: 91.1892%.
- After selected response is stronger in 100.0000% of paired opportunities; response-ratio P50 is 1.064597.
- All paired shifts have one sign: `true`.
- Median H10 center is only 3.502 px above the old band bottom, versus 35.558 px above the final band bottom.

## Largest changes

| Frame | u | Before y | After y | Difference | Before response | After response | Raw peak y |
|---|---|---|---|---|---|---|---|
| 45 | 1860 | 892.070922 | 891.876936 | -0.193985 | 2.511163 | 3.314138 | 892.0 |
| 45 | 1861 | 892.013363 | 891.827112 | -0.186251 | 2.375922 | 3.148387 | 893.0 |
| 1 | 1860 | 892.057767 | 891.878310 | -0.179457 | 2.632547 | 3.430886 | 891.0 |
| 49 | 1860 | 892.040351 | 891.861018 | -0.179333 | 2.671230 | 3.484705 | 892.0 |
| 4 | 1860 | 892.033485 | 891.854506 | -0.178979 | 2.727612 | 3.562606 | 892.0 |
| 48 | 1860 | 892.042374 | 891.864519 | -0.177854 | 2.630029 | 3.427209 | 892.0 |
| 4 | 1861 | 891.996889 | 891.819509 | -0.177380 | 2.522832 | 3.323324 | 893.0 |
| 48 | 1861 | 891.999199 | 891.823126 | -0.176073 | 2.433339 | 3.202643 | 893.0 |

The raw-image audit figure overlays the old band, final band, both selected centers, and the raw intensity peak for these cases.

## Verdict

**A. legitimate_recovery_of_true_laser**

All paired selections remain within 0.2 px, the candidate row is preserved for more than 95% of pairs, every shift has the same sign, and the selected response becomes stronger for every pair. The raw overlays show both centers on the same physical laser stripe; no frame-column opportunity switches to a separated ridge. The integer raw-intensity maximum can occupy an adjacent row on this broad, textured stripe and is therefore shown as context rather than used as a subpixel ridge-identity gate.

The old H10 ridge lies only about 3.5 px above the lower crop boundary. Extending the band moves that boundary about 32 px away, removing a Hessian-convolution boundary influence while retaining the same physical stripe.

The -21.07% H10 sigma_pixel P95 change is therefore accepted as a real repeatability improvement from the band fix, not an alternate-ridge switch.

`phase_a_extraction_chain_frozen = true`

## Outputs

- `B05_A10_band_candidate_audit.csv`: frame summaries, per-column summaries, and all frame-column pairs.
- `B05_A10_band_candidate_audit.png`: largest-difference raw-image overlays.
- `B05_A10_band_candidate_audit.md`: this audit report.
