# Frozen Full-36 Quadratic residual observability and tail audit

QUADRATIC_RESIDUAL_OBSERVABILITY = MODERATE_1D
TAIL_STRUCTURE = FRAME_DOMINATED
C1_NEXT_STEP = TRY_1D_FS

## 结论

- 使用完整 Full-36 FIT：36 poses、32,400 points；Validation 未读取。
- residual 定义为 r = lambda_truth - lambda_quadratic。冻结 YAML 只加载并计算 ray-surface intersection，没有调用 fit。
- raw bias=0.004187 mm，RMSE=0.101429 mm，P95(|r|)=0.180098 mm，Max(|r|)=0.722866 mm。
- frame median 范围=-0.105802–0.149254 mm，跨度=0.255057 mm；去 frame median 后 P05–P95=-0.138782–0.128277 mm。
- observability=MODERATE_1D；next step=TRY_1D_FS。这只是进入 C1 实验的门控，不是已经拟合 C1。

## Artifact provenance / reuse audit

- 直接复用 Full-36 calibration points、camera rays 和 PnP truth；未重复 Steger 提点。
- 30 帧的 board-local mask boundary 坐标复用既有 frame geometry；049–054 只读取 chess 图补充 PnP 几何，不读取 laser 图。

| artifact | action | status | notes |
|---|---|---|---|
| Full-36 calibration points/ray/PnP truth | REUSED_EXISTING | CONFIRMED | 32400 points; no re-extraction |
| Frozen Full-36 Quadratic YAML | LOADED_ONLY | CONFIRMED | sha256=113d3c1b8f92d5a734a2bf612b82a4bd59c0436a89664b5e565e7dd1034bab27; fit() not called |
| Full-36 metadata | REUSED_EXISTING | CONFIRMED | source/mask/point-count assertions passed |
| Formal extraction config | READ_ONLY_ASSERTED | CONFIRMED | full_board_physical; inset=0; unchanged Steger/continuity |
| Intrinsics | REUSED_EXISTING | CONFIRMED | sha256=d162d581ffd12df510b15e4edd42536a97abb4dc7d883352b0be76cb8c65f9b0 |
| Frame board geometry summary | REUSED_EXISTING | CONFIRMED | 30 frame PnP poses for board-local boundary coordinates |
| FIT chess 049-054 | READ_CHESS_ONLY | SUPPLEMENTAL | PnP boundary geometry only; laser points not re-extracted |
| Validation datasets | NOT_READ | EXCLUDED | FIT-only audit |
| Old Cone observability script | REFERENCE_ONLY | REFERENCE | method reference only; old output excluded 049-054 |

## Residual observability

- PCA center=(-0.0099487926, -0.013651736); explained variance s=0.9864, t=0.0136; sqrt eigenvalue ratio=8.527.
- Robust span s=0.326690, t=0.034848, t/s=0.1067.
- Per-frame t/s median=0.0129; fraction >=0.10=0.222.

| predictor | centered low-frequency amplitude / mm | binned EV | frame same-sign | frame EV median |
|---|---:|---:|---:|---:|
| s | 0.186971 | 0.3341 | 0.444 | 0.7439 |
| t | 0.083314 | 0.0528 | 0.611 | 0.6766 |
| v | 0.186774 | 0.3335 | 0.444 | 0.7439 |

诊断量级参考沿用旧方法：PnP uncertainty 0.025–0.033 mm；strong 需要幅度至少 0.099 mm、同号至少 0.60、frame EV 至少 0.10；moderate 需要幅度至少 0.0495 mm、同号至少 0.40、frame EV 至少 0.03。

## Tail audit

| threshold | count | fraction | unique frames | top frame | top frame share | top v-bin | top s-bin | boundary edge share | response low20 share |
|---|---:|---:|---:|---|---:|---|---|---:|---:|
| abs_gt_p95 | 1620 | 0.0500 | 35 | 027 | 0.275 | v_0100_0200 | s_-0.19237_-0.15828 | 0.219 | 0.228 |
| abs_gt_0.30 | 459 | 0.0142 | 13 | 027 | 0.865 | v_1700_1800 | s_-0.02195_+0.01213 | 0.013 | 0.227 |
| abs_gt_0.40 | 359 | 0.0111 | 7 | 027 | 0.964 | v_1300_1400 | s_-0.02195_+0.01213 | 0.003 | 0.206 |

tail_points.csv 为长表，按三个 threshold 保留所有 tail points；quadratic_residual_points.csv 保留全部点，不因 residual 大而删除。

## Classification evidence

- clear 2D ray support=False；t actionable=True。
- s evidence amplitude=0.209060 mm；t evidence amplitude=0.206124 mm。
- tail gates: frame_dominated=True; repeated_cross_frame=False; edge_or_low_response=False; spatially_concentrated=True。

## Scope exclusions

- 未读取 019–024、037–040、055–060；未使用 Validation residual 调参。
- 未重新拟合 Quadratic C0；未拟合 C1；未修改 Steger、mask、weighting。
- 未因 residual 大而删除 pose 或点。
