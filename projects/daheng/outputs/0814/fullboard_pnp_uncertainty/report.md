# Task 6D — Full-board PnP truth uncertainty audit

`FULLBOARD_TRUTH_UNCERTAINTY = B. MODERATE`

## Scope and boundary

- FIT-only frames: `001–018`, `025–036` (30 frames); 027 retained and reported separately. Only explicit FIT files were opened; Validation 019–024 and 037–040 were not read.
- Formal intrinsics/distortion and current `SOLVEPNP_ITERATIVE` + `solvePnPRefineLM` were retained. No formal intrinsics, solver settings, laser surface, frame selection, or correction were changed.
- Frozen provenance SHA-256: `4cd60c8f77ee2358329a9f844b2f8861b1f53c13c40698ec0361f3fb05a8dc66`; formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac`. Cone is only an observed-residual reference.

## Full-board Monte Carlo

- Each frame uses all 88 detected corners. Corner perturbations are zero-mean Gaussian draws from that frame's empirical reprojection residual covariance (centered du/dv), with **1000** PnP re-solves; perturbation size is independent of Cone residual.
- Truth pixels are all extracted centers with valid full-board ray-plane intersections; Cone z-range validity is not used to select them.
- Typical MC point-field P95 |delta lambda|: **0.0255 mm**; worst frame P95: **0.0327 mm**.
- Full-span random balanced bootstrap uses four board-corner anchors plus random interior corners, **100** replicates/frame; checkerboard A/B and uniform sparse designs are fixed full-span diagnostics.

| frame | full PnP RMSE px | MC sigma P95 mm | MC delta P95 P95 mm | random balanced delta P95 P95 mm | Cone RMSE mm | sigma/Cone ratio | tilt deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 0.1633 | 0.0127 | 0.0253 | 0.0244 | 0.0769 | 0.165 | 24.766 |
| 002 | 0.1480 | 0.0118 | 0.0233 | 0.0213 | 0.0652 | 0.182 | 21.011 |
| 003 | 0.1508 | 0.0146 | 0.0284 | 0.0199 | 0.0854 | 0.171 | 18.949 |
| 004 | 0.1009 | 0.0110 | 0.0219 | 0.0307 | 0.1207 | 0.091 | 3.122 |
| 005 | 0.1030 | 0.0137 | 0.0257 | 0.0270 | 0.0633 | 0.216 | 2.879 |
| 006 | 0.0962 | 0.0108 | 0.0210 | 0.0312 | 0.0655 | 0.165 | 2.534 |
| 007 | 0.1162 | 0.0164 | 0.0317 | 0.0280 | 0.0807 | 0.204 | 2.587 |
| 008 | 0.0955 | 0.0114 | 0.0221 | 0.0256 | 0.0676 | 0.169 | 4.578 |
| 009 | 0.1097 | 0.0158 | 0.0313 | 0.0331 | 0.0849 | 0.186 | 3.814 |
| 010 | 0.1190 | 0.0133 | 0.0261 | 0.0234 | 0.0686 | 0.193 | 7.374 |
| 011 | 0.1176 | 0.0116 | 0.0226 | 0.0241 | 0.0739 | 0.158 | 12.548 |
| 012 | 0.0883 | 0.0106 | 0.0207 | 0.0221 | 0.0494 | 0.214 | 4.508 |
| 013 | 0.1072 | 0.0135 | 0.0269 | 0.0238 | 0.0595 | 0.227 | 2.642 |
| 014 | 0.0959 | 0.0133 | 0.0257 | 0.0277 | 0.0758 | 0.175 | 2.656 |
| 015 | 0.1107 | 0.0105 | 0.0205 | 0.0208 | 0.0851 | 0.123 | 14.249 |
| 016 | 0.1245 | 0.0120 | 0.0235 | 0.0200 | 0.0607 | 0.198 | 17.942 |
| 017 | 0.1183 | 0.0105 | 0.0204 | 0.0195 | 0.0715 | 0.146 | 17.088 |
| 018 | 0.1216 | 0.0163 | 0.0327 | 0.0369 | 0.0932 | 0.175 | 3.036 |
| 025 | 0.1562 | 0.0140 | 0.0276 | 0.0186 | 0.0729 | 0.192 | 21.633 |
| 026 | 0.1653 | 0.0165 | 0.0326 | 0.0277 | 0.0634 | 0.260 | 17.986 |
| 027 (027) | 0.1741 | 0.0092 | 0.0175 | 0.0199 | 0.3688 | 0.025 | 26.648 |
| 028 | 0.1836 | 0.0133 | 0.0261 | 0.0283 | 0.0859 | 0.155 | 21.111 |
| 029 | 0.1052 | 0.0131 | 0.0256 | 0.0241 | 0.0575 | 0.227 | 2.997 |
| 030 | 0.1310 | 0.0128 | 0.0246 | 0.0172 | 0.0498 | 0.256 | 15.601 |
| 031 | 0.1137 | 0.0156 | 0.0302 | 0.0290 | 0.0473 | 0.329 | 11.222 |
| 032 | 0.1294 | 0.0126 | 0.0250 | 0.0330 | 0.0628 | 0.201 | 11.750 |
| 033 | 0.1221 | 0.0104 | 0.0208 | 0.0211 | 0.0496 | 0.210 | 16.035 |
| 034 | 0.1003 | 0.0127 | 0.0250 | 0.0269 | 0.0492 | 0.258 | 3.828 |
| 035 | 0.1332 | 0.0159 | 0.0304 | 0.0417 | 0.1107 | 0.143 | 12.055 |
| 036 | 0.0908 | 0.0132 | 0.0257 | 0.0264 | 0.0815 | 0.162 | 2.624 |

## Balanced full-span bootstrap

| design | frame-design aggregate count | median lambda RMSE mm | P95 lambda RMSE mm | P95-of-P95 mm | P95 normal angle deg | P95 plane distance mm |
|---|---:|---:|---:|---:|---:|---:|
| checkerboard_A | 30 | 0.0063 | 0.0191 | 0.0206 | 0.0115 | 0.0275 |
| checkerboard_B | 30 | 0.0066 | 0.0202 | 0.0218 | 0.0112 | 0.0286 |
| uniform_sparse | 30 | 0.0108 | 0.0174 | 0.0250 | 0.0106 | 0.0347 |
| full_span_random | 30 | 0.0081 | 0.0265 | 0.0352 | 0.0175 | 0.0465 |

## Spatial/pose dependence

- The complete `sigma_lambda(u,v)` map is in `lambda_uncertainty_map.csv`; per-frame Spearman correlations with u, v, lambda and edge/center ratios are in `frame_truth_uncertainty.csv`.
- Across frames, edge/center sigma ratio median **1.254**, maximum **1.621**; board-tilt versus MC sigma P95 Spearman rho **-0.205** (p=0.277). This is a modest, non-universal edge effect rather than a stable tilt law.
- Frame 027: MC sigma P95 **0.0092 mm**, MC delta P95 P95 **0.0175 mm**, random balanced delta P95 P95 **0.0199 mm**, Cone RMSE **0.3688 mm**.

## Answers

1. Task 6C subset instability is mainly geometric degeneration: **True** (full-board MC worst P95 / Task 6C subset worst P95 = 0.154).
2. Full-board truth typically stabilizes to about **0.0255 mm** point-field P95; worst frame is **0.0327 mm**.
3. PnP uncertainty explains most observed residual: **False**; median MC sigma/Cone RMSE ratio is **0.184**.
4. Sensor-edge/pose increase: edge/center median **1.254×**, max **1.621×**, tilt rho **-0.205** (p=0.277); no universal pose law.
5. Next step: **improve PnP / corner / intrinsics**.

## Conclusion

`FULLBOARD_TRUTH_UNCERTAINTY = B. MODERATE`.
The gates are descriptive: LOW requires full-board MC and balanced full-span variation to remain well below the observed Cone residual; MODERATE permits a smaller but non-negligible fraction; HIGH means full-board truth uncertainty reaches the residual scale or exceeds the declared margins.

Generated figures: `p95_delta_lambda_truth_by_frame.png`, `truth_uncertainty_vs_cone_residual.png`, `lambda_uncertainty_map.png`, and `balanced_bootstrap_plane_consistency.png`.
