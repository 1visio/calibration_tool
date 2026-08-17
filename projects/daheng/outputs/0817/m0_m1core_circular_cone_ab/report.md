# Task 6I - M0 vs M1-core Circular Cone end-to-end A/B

`CAMERA_CALIBRATION_EDGE_CAUSAL_EFFECT = C. WEAK`

FIT candidates were fitted independently with the same production CircularConeModel path. M0/M1-core are diagnostic candidates only; the formal K/D and Cone files were not overwritten.
Validation was opened only after both FIT candidates completed. No 0815 laser/nolaser data were used.
Residual convention: `e_lambda = lambda_truth - lambda_model`.

## FIT Cone candidates

| candidate | cost | objective MSE | selected points | alpha deg | apex delta vs M0 | axis delta deg |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 1.37482 | 0.000458272 | 3000 | 88.8748 | 0 | 0 |
| M1-core | 1.3501 | 0.000450034 | 3000 | 88.8808 | 0.453085 | 0.0116528 |

## FIT region metrics

| candidate | region | bias mm | RMSE mm | P95 mm | max mm |
|---|---|---:|---:|---:|---:|
| M0 | global | 0.00267811 | 0.0981152 | 0.161491 | 0.616774 |
| M0 | top | 0.195987 | 0.231999 | 0.471987 | 0.5388 |
| M0 | middle | 0.00251022 | 0.0975335 | 0.160858 | 0.616774 |
| M0 | bottom | -0.0692534 | 0.0771893 | 0.112625 | 0.154235 |
| M1-core | global | 0.0026405 | 0.0972732 | 0.161707 | 0.613142 |
| M1-core | top | 0.197953 | 0.232661 | 0.47131 | 0.538135 |
| M1-core | middle | 0.00248516 | 0.0966619 | 0.159959 | 0.613142 |
| M1-core | bottom | -0.0724542 | 0.0796995 | 0.115299 | 0.161722 |

## Frozen Validation A/B

| candidate | region | bias mm | MAE mm | RMSE mm | P95 mm | max mm |
|---|---|---:|---:|---:|---:|---:|
| M0 | global | -0.0292164 | 0.0724098 | 0.0877603 | 0.162853 | 0.397889 |
| M0 | top | 0.0920976 | 0.0921182 | 0.101529 | 0.152929 | 0.161733 |
| M0 | middle | -0.0293795 | 0.0722517 | 0.087593 | 0.162784 | 0.397889 |
| M0 | bottom | -0.13402 | 0.13402 | 0.144512 | 0.209411 | 0.21191 |
| M1-core | global | -0.0285403 | 0.0723103 | 0.0876237 | 0.162304 | 0.396681 |
| M1-core | top | 0.0944032 | 0.0944032 | 0.103607 | 0.155688 | 0.164488 |
| M1-core | middle | -0.0287042 | 0.0721428 | 0.0874439 | 0.162126 | 0.396681 |
| M1-core | bottom | -0.135524 | 0.135524 | 0.14625 | 0.212211 | 0.214661 |

## Camera-change attribution

- global RMSE improvement: `0.00155609`
- top RMSE improvement: `-0.0204623`
- bottom RMSE improvement: `-0.0120272`
- edge/middle ratio: `1.40446` -> `1.42867`
- top-bottom bias asymmetry: `0.226118` -> `0.229927` mm

## 027 diagnostic

027 remains in FIT and is reported separately. See `frame027_camera_ab.csv`; no frame was deleted or reweighted after seeing Validation.

## Controls

- Same Steger extraction settings and frozen UV per frame for both K/D candidates.
- Same full-board PnP solver, Circular Cone parameterization, frame balancing, soft_l1 loss, bounds, optimizer and formal v-domain.
- No Elliptical Cone, quadric, correction, LUT, v compensation, or production writeback.

## Outputs

- `m0_m1core_truth_difference.csv`
- `m0_m1core_cone_fit_summary.csv`
- `m0_m1core_fit_region_metrics.csv`
- `m0_m1core_validation_metrics.csv`
- `m0_m1core_validation_by_frame.csv`
- `m0_m1core_validation_by_region.csv`
- `frame027_camera_ab.csv`
- `Cone_M0_AB.yaml`
- `Cone_M1core_AB.yaml`
