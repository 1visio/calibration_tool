# Task 4A output files

| file | meaning | boundary |
|---|---|---|
| residual_points.csv | every FIT truth point, M0/local lambda and e_lambda | invalid rows retained |
| residual_decomposition_vs_v.csv | fixed 30/60/100 px b(v)+delta_g(v) diagnostics | not a correction model |
| frame_bootstrap.csv | frame-level bootstrap samples for b and delta_g | no point bootstrap |
| report.md | offset/gain/u/v/frame/top-bottom diagnosis and A/B/C/D choice | FIT-only |
| residual_vs_v.png | residual distribution versus v | diagnostic plot |
| offset_b_vs_v.png | b(v) for three fixed bin widths | diagnostic plot |
| gain_delta_g_vs_v.png | delta_g(v) for three fixed bin widths | diagnostic plot |
| residual_region_distribution.png | top/middle/bottom distributions | diagnostic plot |
| m0_vs_local_residual.png | M0/local residual comparison | diagnostic plot |
| provenance.json | split isolation, hash and bootstrap provenance | no validation claim |
