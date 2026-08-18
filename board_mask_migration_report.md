# Laser-plane 提点 mask 迁移报告

## 结论

正式三联图激光点提取已经统一到 `full_board_physical` mask：

- 棋盘内角点：`11 × 8`
- 格距：`20 mm`
- 完整方格范围：`X=[-20, 220] mm`、`Y=[-20, 160] mm`
- inset：`0.0 mm`
- 不做额外像素腐蚀或膨胀
- 不改变 Steger、每扫描线单点选择或 continuity 参数

正式 FIT 与 Validation 的三联图都经过同一个 `process_dataset()`，因此现在共用同一个 `board_mask_for_pose()` 和 `full_board_physical_mask()`；后续 `laser_surface_models` stage 仍通过同一 `fit_laser_models_from_triplets.py` 入口执行。

本次只修改代码、配置和合成单元测试，没有读取 FIT/Validation 图像，没有重新拟合任何激光模型。

## 相关脚本梳理

| 环节 | 文件/函数 | 作用 | 本次处理 |
|---|---|---|---|
| PnP | `scripts/fit_laser_models_from_triplets.py:199-255` | 检测 11×8 内角点并求当前棋盘 pose | 保持不变 |
| mask helper | `calibration_tool/board_mask.py` | 构造完整物理棋盘四角、PnP 投影和 polygon mask | 新增统一 helper |
| mask 选择 | `scripts/fit_laser_models_from_triplets.py:274-313` | 根据 `extraction.board_mask_mode` 选择正式 mask | 新增；默认完整物理边界 |
| 候选提取 | `scripts/fit_laser_models_from_triplets.py:404-432` | 正差分、Steger、每扫描线选一点、continuity | 保持不变 |
| FIT/Validation 入口 | `scripts/fit_laser_models_from_triplets.py:899-996` | 读取三联图并生成标定点 | 改为调用统一 mask 选择器 |
| 模型标定 | `scripts/fit_laser_models_from_triplets.py:1202-1237` | 使用同一提点结果拟合 global plane/quadratic/cone | 模型代码未修改 |
| 审计兼容层 | `scripts/audit_board_mask_exclusion_049_054.py:136-153` | 旧审计中的完整边界投影函数 | 保留函数名，改为委托正式 helper |

以下旧 mask 调用仍保留在历史审计脚本中，用于“修改前后”比较；它们不是正式重新标定入口，也没有被语义悄悄替换。`board_inner_mask()` 本身的 convex-hull 与 `margin_px` 语义保持不变。

仓库文档中另有历史兼容的 `laser_plane_shared_steger` stage，它走独立的 `calibration/src` 提点链，并不调用本次迁移的三联图 `board_inner_mask()`；本次没有扩大范围修改该 stage。

## 新 mask 语义

`calibration_tool/board_mask.py` 根据内角点数量推导完整方格边界：

```text
x_min = -square_size_mm + inset_mm
x_max = pattern_cols * square_size_mm - inset_mm
y_min = -square_size_mm + inset_mm
y_max = pattern_rows * square_size_mm - inset_mm
```

因此 11×8、20 mm、0 mm inset 时正好是 12×9 方格的外边界。边界四角经当前帧 PnP pose、相机内参和畸变参数投影后，再由 `fillConvexPoly` 生成图像 mask。mask 只覆盖该物理四边形，不会通过固定像素外扩到棋盘外白边或铝框。

配置模式：

- `full_board_physical`：正式默认模式；使用完整物理边界，`board_mask_inset_mm` 默认 `0.0`，忽略 `board_mask_margin_px`。
- `inner_corner_hull`：兼容模式；显式使用原有 `board_inner_mask()` 和 `board_mask_margin_px`。
- 未配置 `board_mask_mode` 时，代码默认 `full_board_physical`，避免新配置遗漏时回退到旧 inner-corner hull。

已更新配置：

- `configs/laser_model_fit_config.daheng.yaml`
- `configs/laser_model_fit_config.daheng-0812.yaml`
- `scripts/laser_model_fit_config.yaml`

## 测试

新增 `tests/test_board_mask.py`，使用合成相机和 PnP pose 验证：

1. 11×8、20 mm 的边界严格为 `[-20,-20]`、`[220,-20]`、`[220,160]`、`[-20,160]`；
2. 边界按 PnP 投影，而不是固定像素外扩；
3. 四周外圈方格代表点在 mask 内；
4. 物理边界外的点不在 mask 内；
5. 默认模式是完整物理边界，旧模式必须显式指定，且旧 `board_inner_mask` 结果保持一致。

验证命令及结果：

```text
python -m py_compile calibration_tool/board_mask.py scripts/fit_laser_models_from_triplets.py scripts/audit_board_mask_exclusion_049_054.py tests/test_board_mask.py
python -m unittest tests.test_board_mask tests.test_laser_orientation -v
Ran 6 tests ... OK
```

本轮没有执行正式标定命令，因其会读取数据并拟合激光模型；下次正式标定时，使用上述三个配置之一即可采用完整棋盘物理边界。
