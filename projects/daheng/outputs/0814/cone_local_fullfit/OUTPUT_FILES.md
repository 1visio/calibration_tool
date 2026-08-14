# Task 3B-2 output files

| file | meaning | boundary |
|---|---|---|
| report.md | local Full-FIT/SVD/jackknife conclusions | FIT-only, no validation claim |
| local_fullfit_result.json | M0-local initialization and M_local_fullfit result | diagnostic only |
| local_jacobian_svd.csv | local scaled Jacobian SVD and step stability | no regularization |
| local_frame_jackknife.csv | 30 leave-one-FIT-frame-out local fits | validation not opened |
| local_jackknife_prediction_vs_v.csv | local jackknife lambda drift by v | fixed evaluation grid |
| legacy_comparison.json | Task 3A legacy condition/jackknife comparison | reads prior FIT-only artifacts only |
| legacy_vs_local_singular_spectrum.png | singular spectrum comparison | presentation aid |
| legacy_vs_local_jackknife_prediction_vs_v.png | prediction stability comparison | presentation aid |
| local_weakest_direction_composition.png | local weakest loading | local coordinate interpretation |
| provenance.json | split, hash and no-writeback provenance | no deployment authorization |
