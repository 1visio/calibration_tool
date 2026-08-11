# LaserSearchRegion 第 1 阶段重构

## 1. 范围与不变量

本阶段只统一 Steger 搜索域的数据结构、坐标语义和调用入口，不改变 calibration_tool 或 online point-cloud tool 的上层调用方式。

以下内容保持不变：

- `_detect_steger_band()` 的 seed、active-row、margin、max-height 和 auto-band 公式；
- `sigma`、`threshold`、`deriv_thresh`、`roi_margin`、`roi_max_height` 的值和含义；
- Gaussian 导数、Hessian、亚像素中心和 candidate selection；
- auto region 与 additional region 取最小包络并集的既有策略；
- 上层 continuity、RANSAC、质量分析和 geometry gate。

## 2. 旧 API

| API | 既有语义 | 限制 |
| --- | --- | --- |
| `extract_steger_columns(gray, options, additional_band_bounds=...)` | 在逐列扫描结果上返回结构化数据；额外范围与 auto band 取包络 | 仅支持 `scan_axis=column`；名称隐含 top/bottom 语义 |
| `steger_backend(gray, options)` | 返回 `(u, v)` 点数组；row 模式由内部转置实现 | 没有 row/column 通用的额外搜索域参数 |
| `additional_band_bounds=(start, end)` | 工作图 axis-0 上的半开区间 | 只暴露在 column API |

`additional_band_bounds` 本阶段不删除，继续作为 backward-compatible alias。

## 3. 新 API

### 3.1 `LaserSearchRegion`

```python
LaserSearchRegion(
    start_px: int,
    end_px: int,
    source: str,
)
```

`[start_px, end_px)` 始终是原图中沿激光条纹法向轴的半开搜索区间。公共语义不使用 `top`、`bottom`、`y_min` 或 `y_max`。

校验规则：

- `start_px`、`end_px` 必须是整数；
- 必须满足 `start_px < end_px`；
- `source` 必须是非空字符串；
- 区间会裁到原图法向轴长度，裁后为空时明确抛出 `ValueError`。

### 3.2 统一入口

```python
extract_steger(
    gray,
    options=None,
    *,
    search_region=None,
    diagnostic=False,
)
```

该入口统一处理：

- `scan_axis=column` / `scan_axis=row`；
- row 模式的内部转置；
- 原图法向轴 search region 到工作图 axis-0 的映射；
- row 模式结果恢复为原图 `(u, v)` 坐标；
- 通用 normal-axis 元数据。

这里的 `search_region` 延续既有 `additional_band_bounds` 策略，是 additional/reference region，不是强制 crop：auto region 和 additional region 同时存在时，最终区域仍取两者的最小包络并集。

`extract_steger_columns()` 新增 `additional_search_region`；`steger_backend()` 新增可选的 `search_region`。两者均保留原调用形式。

## 4. normal-axis 坐标定义与转换

| `scan_axis` | 激光延伸/扫描方向 | 原图法向轴 | 内部工作图 | search region 映射 | 输出恢复 |
| --- | --- | --- | --- | --- | --- |
| `column` | 原图 `u/x` | 原图 `v/y` | `gray` | 原图 `v` 直接对应工作图 axis-0 | 无需交换 |
| `row` | 原图 `v/y` | 原图 `u/x` | `gray.T` | 原图 `u` 在转置后直接对应工作图 axis-0 | 交换 `u/v`，并把 corrected signal 转回原图形状 |

因此调用者始终使用原图坐标：row 模式不会要求调用者先自行 transpose，也不会要求把 `[u_start, u_end)` 改写为工作图的 top/bottom。

内部 `_detect_steger_band()` 仍是 columnwise 实现，其返回 tuple 和旧 `final_band_top_px` 等 metadata 暂时保留；统一入口额外发布 `normal_axis`、`final_search_region_start_px`、`final_search_region_end_px` 和 `valid_scanline_count`。这些新名称只做语义归一化，不参与候选筛选。

## 5. 兼容策略

- `extract_steger_columns(..., additional_band_bounds=(a, b))` 仍可用，并转换为 `LaserSearchRegion(a, b, source="additional_band_bounds")`。
- 同时传入 `additional_search_region` 和 `additional_band_bounds` 会明确报错，避免静默覆盖。
- `steger_backend(gray, options)` 的无 search-region 路径保留旧轻量实现，避免在线逐帧路径因结构化 metadata 包装产生额外运行开销。
- `steger_backend(..., search_region=region)` 和正式 `extract_steger(...)` 使用统一的 row/column 坐标转换。
- calibration_tool 与 online 上层文件均未修改；它们未传新参数时继续沿用原策略。

## 6. 修改文件

- `calibration/src/realtime_steger.py`
  - 新增 `LaserSearchRegion`；
  - 新增正式 `extract_steger()`；
  - 为 column/row 统一 additional normal-axis region；
  - 保留 `extract_steger_columns()`、`additional_band_bounds` 和 `steger_backend()` 兼容入口。
- `calibration_tool/tests/test_laser_search_region.py`
  - 覆盖 column/row、有无额外区域、转置对称性以及非法/空区域。
- `calibration_tool/docs/laser_search_region_refactor.md`
  - 本设计和验证记录。

## 7. 测试结果

### 定向测试

```text
calibration_tool> python -m pytest -q tests/test_laser_search_region.py
6 passed, 21 subtests passed

0704line-laser-3d-scanner> python -m pytest -q \
  laser_measurement_tool/tests/test_online_core.py::OnlineCoreTests::test_controller_throttles_ui_signals_without_throttling_processing \
  laser_measurement_tool/tests/test_backends.py
16 passed
```

六类新增验证均通过：

1. column、无 extra region：新旧结果逐数组完全一致；
2. column、有 extra region：新接口与 `additional_band_bounds` 完全一致；
3. row、无 extra region：与旧 transpose backend 点结果完全一致；
4. row、有 extra region：范围作用于原图 `u` 方向；
5. synthetic horizontal/vertical stripe：转置后的中心坐标严格对称；
6. 非法、空或冲突 search region：明确报错。

### 现有完整 pytest

```text
calibration_tool: 101 passed, 21 subtests passed
online tool:      189 passed, 7 subtests passed, 1 failed
```

用户确认当前配置为最终版后，已通过正式 `golden-build` 入口刷新 baseline 和快照；`golden-check` 为 `matches: true / change_count: 0`，calibration_tool 完整套件现已全部通过。

- online 的 `test_controller_throttles_ui_signals_without_throttling_processing` 是 0.65 秒墙钟节流断言；在完整套件中本次处理 6 帧而失败，和 `tests/test_backends.py` 一起单独复跑时通过（合计 16 passed）。

两个 Git 工作区均执行 `git diff --check`，结果通过；online 工作区保持 clean。

## 8. 阶段结论

```text
behavior_changed = false
steger_math_changed = false
search_region_api_unified = true
```
