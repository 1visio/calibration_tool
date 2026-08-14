# Task 3B-1 output files

| file | meaning | conclusion boundary |
|---|---|---|
| report.md | final equivalence result and next-step gate | no top-edge conclusion, no deployment authorization |
| local_parameterization_definition.md | exact local coordinates and conversion equations | coordinate change only |
| reference_anchor.json | FIT-only P_ref and per-frame centroid provenance | validation not used |
| roundtrip_equivalence.csv | M0/M_diag legacy→local→legacy and grid lambda checks | no optimization |
| objective_equivalence.json | objective, ray-intersection, axis and nappe equivalence | same physical cone only |
| provenance.json | hash, split isolation and no-optimizer provenance | no validation model selection |
| circular_cone_local_parameterization.py | conversion implementation | not production runtime |
| test_circular_cone_local_parameterization.py | unit tests for inverse mapping/basis/axis sign | does not test Full-FIT |
