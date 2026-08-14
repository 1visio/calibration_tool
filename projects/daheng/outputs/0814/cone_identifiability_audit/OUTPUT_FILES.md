# Task 3A output files

| file | meaning | boundary |
|---|---|---|
| dataset_split.yaml | explicit FIT/VALIDATION registry | validation is not opened |
| provenance.json | hashes, quality provenance and invalid summary | no model selection from validation |
| fullfit_diagnostic_result.json | in-memory M_diag_fullfit result | not production |
| parameter_comparison.csv | M0 vs full-FIT parameters | normalized interpretation only |
| frame_jackknife.csv | leave-one-FIT-frame-out fits | omitted frame remains diagnostic FIT data |
| jackknife_prediction_vs_v.csv | prediction drift versus v | fixed FIT-derived grid |
| jacobian_svd.csv | scaled SVD, step stability and loadings | local, not global identifiability proof |
| parameter_coupling.csv | column/covariance coupling | local linear diagnostic |
| weak_direction_profile.csv | nuisance-refit weak-direction objective/profile | no validation |
| weak_direction_profile_v.csv | profile prediction drift by v | no validation |
| apex_alpha_profile.csv | two-dimensional apex/alpha profile with nuisance refit | FIT-only diagnostic |
| local_parameter_sensitivity.csv | M0/full-fit d(lambda)/d(theta) versus v | invalids retained in counts |
| m0_derivative_invalid_audit.csv | M0 finite-difference failures | no silent deletion |
| *.png | required diagnostic plots, including apex_alpha_profile.png | presentation aids, not extra evidence |
| report.md | Task 3A conclusions and Q1–Q7 | does not authorize deployment |
