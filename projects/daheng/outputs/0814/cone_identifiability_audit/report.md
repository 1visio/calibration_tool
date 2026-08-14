# Task 3A — Circular Cone FIT-only identifiability audit

**FIT_ONLY = TRUE**
**FORMAL_CONE_UNCHANGED = TRUE**
**NEXT_OPTION = C**

## Scope and split isolation

- FIT used for every optimization/diagnostic: `001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 025, 026, 027, 028, 029, 030, 031, 032, 033, 034, 035, 036` (30 frames).
- Validation frozen and not opened: `019, 020, 021, 022, 023, 024, 037, 038, 039, 040`.
- Split role is an explicit registry; acquisition manifest split tags are not authoritative.
- Formal working domain: v=[241.998, 2731.978] px.
- Evaluation grid: u=[1770.0, 2160.0] (41 samples), v formal domain (101 samples).
- Formal Cone SHA-256: `478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac` (before/after identical).

## Full-FIT diagnostic baseline

`M_diag_fullfit` is an in-memory diagnostic model produced by the existing CircularConeModel.fit path. It is not a production artifact.

| region | M0 RMSE / mm | M_diag_fullfit RMSE / mm | M0 P95 / mm | M_diag_fullfit P95 / mm |
|---|---:|---:|---:|---:|
| global | 0.0304076 | 0.0302431 | 0.0503903 | 0.0502615 |
| top_formal_edge | 0.0675328 | 0.0734944 | 0.142859 | 0.150562 |
| middle_formal | 0.0301623 | 0.0300172 | 0.0500951 | 0.0497612 |
| bottom_formal_edge | 0.0356602 | 0.023169 | 0.049748 | 0.0336433 |

- Formal optimizer cost: `1.38595`; selected points: `3000`; status: `success`.
- Objective is the existing frame-balanced sampled Circular Cone residual with `soft_l1`, `f_scale_mm=0.10`, negative-axial penalty and existing bounds; the frozen M0 axis/apex/alpha are explicit solver starts, and no validation-derived weight/threshold was introduced.

## Parameter comparison

See `parameter_comparison.csv`; normalized delta uses the predeclared interpretation scales (1 degree for axis angles, 10 mm for apex, 0.1 degree for alpha).

## Frame jackknife

- Leave-one-frame-out count: `30`; every omitted frame is a complete FIT frame and is never removed permanently.
- Jackknife grid P95 |delta lambda| median/max across omitted frames: `0.00658091` / `0.0173586` mm.
- Formal-middle training RMSE median/max: `0.0302282` / `0.0304209` mm.
- `jackknife_prediction_vs_v.csv` separates top, middle and bottom; edge-only growth with stable middle indicates edge prediction instability rather than uniform surface movement.

## Jacobian / SVD

- Condition number: `133578`; effective rank: `6/6`; weakest/strongest ratio: `7.486e-06`.
- Weakest normalized loading: `theta_axis=-0.0246, phi_axis=-0.0014, A_x=-0.2875, A_y=+0.0488, A_z=-0.9246, alpha=-0.2438`.
- Combined apex/alpha loading norm in weakest direction: `0.9997`.
- SVD is performed on robust-weighted, frame-balanced Jacobian columns after explicit physical-unit scaling; raw mm/rad column magnitudes are not compared.

## Weak-direction profile

- Profile uses FIT-only selected points and nuisance-parameter refit in the five-dimensional complement of the weakest normalized singular direction; no validation result sets the displacement range.
- Objective cost max/min over the scanned feasible profile (reported as max relative cost to t=0): `1.000132`.
- Weak-profile maximum P95 |delta lambda|: top=`0.00550701` mm, middle=`0.00335605` mm, bottom=`0.00685017` mm; valid grid count is preserved in the CSV.
- The objective is nearly flat while the physical parameters move substantially; this demonstrates parameter non-identifiability, not automatic evidence that the edge residual is explained.
- `weak_direction_profile.csv` / `weak_direction_profile_v.csv` contain the one-dimensional nuisance-refit profile; `apex_alpha_profile.csv` is the requested two-dimensional apex–alpha profile with the remaining four coordinates re-optimized.
- Apex–alpha 2D profile: minimum off-origin relative cost=`1.001868`, maximum=`2.238914`; the shallow off-origin valley is aligned with compensating apex/alpha changes, while the corners are not a flat fit-equivalent solution.

## Local sensitivity

`local_parameter_sensitivity.csv` reports M0 and M_diag_fullfit d(lambda)/d(theta_i) along v, plus d(lambda)/d(theta_weak). Positive/negative finite differences use the recorded physical step; invalid intersections remain counted.

## M0 derivative invalid audit

- Invalid M0 ±0.5px derivative rows: `243`; these rows are not silently deleted from the audit.
- The CSV records frame, u/v, reason, formal-domain membership and edge region; counts by frame/bin/region are summarized in this report and provenance JSON.

## Quality provenance

- FIT extension frames retain PnP RMSE, Steger point count and dynamic-range warnings. Influential frames are reported by jackknife metrics but are not automatically deleted.
- Validation metadata is registry-only (`opened_in_task3a=false`); no validation residual, profile, Jacobian or model choice is produced in this task.

## Answers to required questions (FIT-only)

- Q1 six-parameter stability: `not stably identifiable as individual physical parameters`; surface prediction stability is reported separately.
- Q2 weak direction: `YES`.
- Q3 apex/alpha coupling: `YES / material loading`.
- Q4 jackknife: edge prediction P95 max=`0.0173586` mm; middle training RMSE max=`0.0304209` mm; see v-resolved CSV for asymmetry.
- Q5 top-edge gain mismatch: `WEAK for the top-edge mismatch: the weak apex/alpha valley is real, but its scanned surface drift is only a few microns while the full-fit diagnostic worsens top RMSE.`
- Q6 top/bottom asymmetry: full-fit top RMSE change=`0.00596153` mm, bottom change=`-0.0124911` mm; this is an asymmetric FIT diagnostic, not a validation claim.
- Most influential leave-one-frame-out fold by grid P95: omitted `004` with `0.0173586` mm.
- Q7 next action: `C` — 先解决 Circular Cone 参数弱可辨识/参数化问题。

## Limits

- This is a diagnostic audit. M_diag_fullfit and jackknife models must not be deployed or written back to the formal Cone file.
- Full-sensor regions outside the formal v domain are not used to claim identifiability.
