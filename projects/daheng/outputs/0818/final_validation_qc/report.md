# Frozen Full-36 Quadratic / Circular Cone independent Validation

`FINAL_C0_STATUS = QUADRATIC`

## 结论

- 模型选择只使用 Primary Validation 055–060；Historical 与 All Validation 不会回写或改变 Primary 结论。
- Primary 选择结果：`FINAL_C0_STATUS = QUADRATIC`。判定规则：Primary 055-060 only: at least 4/6 lower-error metrics and no fewer primary pose RMSE wins; Historical/pooled rows are not inputs.
- Primary 数据：5400 points / 6 poses；全 Validation：14400 points / 16 poses。
- Q/C 参数均从冻结 Full-36 YAML 读取；没有根据 Validation 重新拟合、调参或训练 C1。

## Primary 055–060

| model | bias / mm | RMSE / mm | P95 / mm | Max / mm | worst v-bin | worst RMSE / mm | worst P95 / mm | pose RMSE wins |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Quadratic | -0.014469 | 0.094036 | 0.196720 | 0.461213 | v_0000_0100 | 0.219383 | 0.262694 | 3/6 |
| Cone | -0.028302 | 0.098009 | 0.197712 | 0.552490 | v_0000_0100 | 0.180442 | 0.275984 | 3/6 |

Primary same-point paired RMSE delta (Q−C) = -0.003973 mm; P95 delta = -0.000992 mm; Q absolute-error-better fraction = 0.592.
Primary metric winners: {"abs_bias": "Quadratic", "global_rmse": "Quadratic", "global_p95": "Quadratic", "global_max": "Quadratic", "worst_rmse": "Cone", "worst_p95": "Quadratic"}; metric votes: {"quadratic_graph": 5, "circular_cone": 1}.

## Historical regression / pooled consistency

| scope | model | bias / mm | RMSE / mm | P95 / mm | Max / mm | worst v-bin | worst RMSE / mm |
|---|---|---:|---:|---:|---:|---|---:|
| 019–024 | Quadratic | -0.005182 | 0.077971 | 0.145219 | 0.259251 | v_0300_0400 | 0.132999 |
| 019–024 | Cone | -0.006625 | 0.081179 | 0.148741 | 0.262116 | v_2400_2500 | 0.136895 |
| 037–040 | Quadratic | -0.060924 | 0.099032 | 0.190544 | 0.385406 | v_0700_0800 | 0.188522 |
| 037–040 | Cone | -0.069277 | 0.103475 | 0.206741 | 0.425742 | v_0700_0800 | 0.199170 |
| Historical pooled | Quadratic | -0.027479 | 0.087009 | 0.167192 | 0.385406 | v_0700_0800 | 0.151210 |
| Historical pooled | Cone | -0.031685 | 0.090757 | 0.179879 | 0.425742 | v_2900_3000 | 0.197642 |
| All Validation | Quadratic | -0.022600 | 0.089709 | 0.181801 | 0.461213 | v_0000_0100 | 0.186847 |
| All Validation | Cone | -0.030417 | 0.093543 | 0.186812 | 0.552490 | v_2900_3000 | 0.184488 |

Historical/pooled 数值只用于检查一致性，不参与 `FINAL_C0_STATUS` 的决策。

## Pose-level paired comparison

Primary pose RMSE wins：Quadratic=3, Cone=3。完整逐 pose 同点比较见 `validation_pose_comparison.csv`。

## v-bin 与 edge coverage

- v-bin 固定为 100 px，范围 [0,3000)。每个 bin 的 Bias/RMSE/P95/Max 与有效率见 `validation_v_bin_metrics.csv`。
- 严格 v 域外点数：1；它们保留在 global/pose 指标，并在 v-bin 表中单列为 `out_of_range`，没有静默丢弃。
- 图中分别显示 Primary、Historical pooled、All Validation 的有符号 bias 与 RMSE；第四 panel 显示 Primary 的 Q−C RMSE/P95。

## Provenance / reuse audit

- Intrinsics：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml`，SHA-256 `d162d581ffd12df510b15e4edd42536a97abb4dc7d883352b0be76cb8c65f9b0`。
- Formal extraction config：`D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml`，SHA-256 `2241737d68276dbdfb226f5285b7ae77dad07be2eaef02ed66966d6b6206cebf`。
- Full-36 source metadata：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\candidate_models\full_fit\model_parameters.json`，source=FIT 001-018,025-036,049-054；mask=full_board_physical；36 poses/32400 points。
- 019–024：复用 0811 正式 calibration_points.csv validation rows；037–040、055–060：仅补取缺失 points，三联图和配置不修改。
- 每批三联图均与对应 frames.csv 做 SHA-256 校验；PnP truth 为同 pose chessboard solvePnP 平面与 camera ray 的交点。

| artifact                               | path                                                                                                                                           | role                                        | scope                                         | model                                        |   point_count |   pose_count | mask                                                 | extraction                                                              | weighting                   | cv_protocol                                | action                                     | provenance_status   | notes                                                                                                                               |
|:---------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|:----------------------------------------------|:---------------------------------------------|--------------:|-------------:|:-----------------------------------------------------|:------------------------------------------------------------------------|:----------------------------|:-------------------------------------------|:-------------------------------------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Full-36 quadratic_graph.yaml           | D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\candidate_models\full_fit\quadratic_graph.yaml | frozen Full-36 Quadratic candidate          | FIT 001-018,025-036,049-054                   | quadratic_graph                              |         32400 |           36 | full_board_physical; inset=0 mm                      | FIT artifact; unchanged for evaluation                                  | Full-36 frozen model source | Full-36 FIT reference; no Validation refit | reused; instantiated only                  | CONFIRMED           | sha256=113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27; fit() not called                                           |
| Full-36 circular_cone.yaml             | D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\grouped_cv_model_comparison\candidate_models\full_fit\circular_cone.yaml   | frozen Full-36 Circular Cone candidate      | FIT 001-018,025-036,049-054                   | circular_cone                                |         32400 |           36 | full_board_physical; inset=0 mm                      | FIT artifact; unchanged for evaluation                                  | Full-36 frozen model source | Full-36 FIT reference; no Validation refit | reused; instantiated only                  | CONFIRMED           | sha256=cc2d93d751a9c2d02ee69ed65305b3d92defcffdbc03db33bbe6635161759b20; fit() not called                                           |
| calibration_result.yaml                | D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\intrinsics\calibration_result.yaml                                         | camera intrinsics/distortion                | all Validation                                | Q/C                                          |               |           16 | n/a                                                  | undistortPoints input                                                   | n/a                         | n/a                                        | reused                                     | CONFIRMED           | sha256=d162d581ffd12df510b15e4edd42536a97abb4dc7d883352b0be76cb8c65f9b0                                                             |
| laser_model_fit_config.daheng.yaml     | D:\Docs\linelaserscan\calibration_tool\configs\laser_model_fit_config.daheng.yaml                                                              | formal Validation extraction protocol       | 037-040,055-060; provenance for 019-024 reuse | Q/C                                          |               |           16 | full_board_physical; inset=0 mm                      | Steger; sigma=1.5; continuity degree=2; threshold=2.0 px; max=900/frame | frame-balanced evaluation   | frozen prediction only                     | reused; protocol asserted                  | CONFIRMED           | sha256=2241737d68276dbdfb226f5285b7ae77dad07be2eaef02ed66966d6b6206cebf                                                             |
| calibration_points.csv validation rows | D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\laser_model\calibration_points.csv                                         | formal extracted rays/PnP truth for 019-024 | historical 019-024                            | Q/C prediction recomputed from reused points |          5400 |            6 | full_board_physical per 0811 stage/config provenance | reused; no re-extraction                                                | frame-balanced evaluation   | same pointwise domain; frozen prediction   | reused                                     | CONFIRMED           | 900 points/pose; image hashes checked against frames.csv                                                                            |
| 019_024 triplet images + frames.csv    | D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\validation                                                             | Validation triplet input                    | 019_024                                       | Q/C                                          |          5400 |            6 | full_board_physical; inset=0 mm                      | reused formal protocol                                                  | frame-balanced evaluation   | frozen prediction only                     | reused points and images                   | CONFIRMED           | image_count=18; manifest=D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\frames.csv                         |
| 037_040 triplet images + frames.csv    | D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\validation_edge_holdout\validation                                     | Validation triplet input                    | 037_040                                       | Q/C                                          |          3600 |            4 | full_board_physical; inset=0 mm                      | new formal protocol extraction                                          | frame-balanced evaluation   | frozen prediction only                     | reused images; extract only missing points | CONFIRMED           | image_count=12; manifest=D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane\validation_edge_holdout\frames.csv |
| 055_060 triplet images + frames.csv    | D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\validation                                                        | Validation triplet input                    | 055_060                                       | Q/C                                          |          5400 |            6 | full_board_physical; inset=0 mm                      | new formal protocol extraction                                          | frame-balanced evaluation   | frozen prediction only                     | reused images; extract only missing points | CONFIRMED           | image_count=18; manifest=D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\laser_plane_0817\frames.csv                    |
| Validation 0817 C1 artifacts           | D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\c1_independent_validation                                                  | historical reference only                   | 019-024,037-040                               | C1/Cone legacy                               |               |           10 | not used as current Q/C source                       | not used for current metrics                                            | not used                    | not same frozen Full-36 Q/C comparison     | excluded from model metrics                | EXCLUDED            | read only as provenance context; no C1 training or residual-based pose selection                                                    |

## Constraints

- 不修改 FIT、原始 Validation 图像、mask、sampling、weighting 或正式配置。
- 不重新拟合 Quadratic/Circular Cone；只加载 frozen YAML 并计算 intersection/prediction。
- 不训练 C1；不以 residual 删除 pose；所有 16 个 Validation pose 均保留。
- 0817 C1 artifact 仅作历史 provenance context，不作为当前 Q/C 指标来源。
