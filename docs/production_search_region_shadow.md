# Stage 3-2：Production Search Region Shadow Mode

## 1. 阶段边界

本阶段建立 pre-Steger `DetectorSummary` 和 `resolve_production_search_region()`，但 resolver 只运行在 shadow mode：

```text
current auto/additional final region ────────────> formal band -> Hessian
                         │
                         └─> shadow resolver -> proposed region -> metadata only
```

正式 Gaussian/Hessian 仍只使用 Stage 3-2 前的 `band_bounds`。shadow proposal 不回写 `band_bounds`，不触发第二次 Steger，不改变 candidate selection、centerline、PASS/FAIL 或保存门禁。

本阶段修改共享实现 `calibration/src/realtime_steger.py`，但 GUI 只会像以前一样消费正式 extraction 和既有 health 字段；新增 shadow metadata 没有接入 GUI 显示或控制逻辑。在线 `steger_backend()` 也不消费 proposal，没有加入前帧状态或 tracking。

## 2. `DetectorSummary`

`DetectorSummary` 是 `_detect_steger_band()` 在 Gaussian/Hessian 前构建的不可替换正式结果的旁路摘要。它保存：

| 字段 | 含义 |
|---|---|
| `normal_axis` | column=`v`，row=`u` |
| `normal_axis_extent` | 原图 normal axis 长度 |
| `row_peak` | 工作图 axis-0 每个位置沿 scan axis 的最大 DN |
| `row_sum` | 工作图 axis-0 每个位置沿 scan axis 的总 DN |
| `threshold` | 正式 intensity threshold |
| `seed` | `argmax(row_sum)`，坐标为原图 normal axis |
| `adaptive_threshold` | `max(threshold, 0.3 * row_peak[seed])` |
| `active_mask` | 完整 `row_peak >= adaptive_threshold` 一维 mask |
| `active_intervals` | `active_mask` 的全部半开连续区间，不再只保留 seed 分量 |
| `seed_active_interval` | 包含 seed 的 active interval；仍用于现有 auto region |
| `roi_margin` / `roi_max_height` | detector 的正式参数 |
| `margin_before_clip` / `margin_after_clip` | seed interval 加 margin 前后范围 |
| `margin_clamped_start/end` | margin 是否越过原图 normal-axis 边界 |
| `roi_max_height_applied` | 现有 max-height cap 是否生效 |
| `auto_search_region` | 完全沿用旧公式得到的正式 auto region |

`StegerExtraction.detector_summary` 保留完整数组和 intervals，供只读 debug/研究使用。metadata 只保存 compact 标量和 interval 列表，避免复制 `row_peak/row_sum/active_mask` 大数组。

`_detect_steger_band()` 的返回类型和 `(start,end)` 数值没有改变。实现只是把原有局部量结构化；seed component、margin、clamp、max-height 的计算顺序保持原样。

## 3. normal-axis 语义

### column

```text
原图 shape = (H,W)
normal axis = 原图 v/row
normal_axis_extent = H
row_peak[v] = max_u(image[v,u])
```

### row

`extract_steger()` 仍先转置图像；summary 在转置工作图上计算，但公开语义恢复为原图：

```text
原图 shape = (H,W)
working image = image.T, shape (W,H)
normal axis = 原图 u/column
normal_axis_extent = W
row_peak[u] = max_v(image[v,u])
```

因此 `seed`、所有 intervals、auto/proposed region 在 row 模式中都直接使用原图 `u`，调用者不需要转置坐标。单元测试使用互为转置的 horizontal/vertical synthetic bands，断言两者 summary 数组和 interval 数值完全一致，仅 `normal_axis` 分别为 `v/u`。

## 4. pre-Steger outside-region evidence

`_outside_region_evidence()` 在 final region 确定后、裁 band 和 Hessian 之前执行。它只读取 `DetectorSummary` 和正式 intensity threshold：

- `outside_active_intervals`：final region 外仍为 adaptive-active 的 normal-axis 区间；
- `outside_peak_intervals`：final region 外满足 `row_peak >= threshold` 的区间；
- active/peak position count；
- region 外最大 peak DN。

该 evidence 不使用 derivative、Hessian、subpixel center、response 或 continuity。它与 Stage 2A 的 post-call diagnostics 目的相近，但现在能够在同一次 formal Steger 前形成 normal-axis 坐标证据。

metadata 字段：

```text
outside_region_active_intervals_px
outside_region_peak_intervals_px
outside_region_active_position_count
outside_region_peak_position_count
outside_region_peak_max_intensity_dn
```

## 5. Shadow `resolve_production_search_region()`

公共函数：

```python
resolve_production_search_region(
    detector_summary,
    current_search_region=None,
    *,
    minimum_safe_clearance_px=14,
) -> ProductionSearchRegionResolution
```

第一版返回对象严格包含：

- `proposed_search_region`
- `would_expand`
- `reason`

`14 px` 来自 Stage 2B 对当前 `sigma=1.5` 真实数据的整数 minimum-safe-clearance 建议；它目前只是 shadow policy 常量，不写入 YAML，也不会改变正式运行。

第一版 proposal 规则：

1. current region 默认为 summary 的现有 auto region；调用方传入 final auto/additional region 时以 final 为准。
2. 从完整 `row_peak` 找到所有 `row_peak >= threshold` 的 normal-axis intervals。
3. 对这些 intervals 两边各保留 `minimum_safe_clearance_px`，与 current region 取包络后裁到图像范围。
4. 包络比 current 更大时 `would_expand=true`；否则只返回 current。
5. proposal 无论多大，都不替换正式 region。

当前 reason code：

| reason | 含义 |
|---|---|
| `no_current_search_region` | auto/additional 均无 region |
| `no_significant_intensity_evidence` | 没有超过正式 threshold 的 peak interval |
| `significant_intensity_outside_current_region` | current 外存在明显强度证据 |
| `significant_intensity_near_current_boundary` | 证据仍在 current 内，但达不到 shadow clearance |
| `expansion_limited_by_image_boundary` | 有 near/outside 证据，但已无法越过原图边界扩展 |
| `current_region_has_safe_intensity_clearance` | 当前范围已覆盖全部显著强度并满足 clearance |

## 6. Shadow metadata 与正式 region 隔离

每次结构化 extraction 增加：

```text
detector_summary_available
detector_normal_axis
detector_normal_axis_extent_px
detector_seed_px
detector_threshold_dn
detector_adaptive_threshold_dn
detector_active_intervals_px
detector_seed_active_interval_px
detector_roi_margin_px
detector_roi_max_height_px
detector_margin_clamped_start/end
detector_roi_max_height_applied

shadow_resolver_available
shadow_minimum_safe_clearance_px
shadow_proposed_search_region_start/end_px
shadow_proposed_search_region_source
shadow_would_expand
shadow_reason
```

正式 region 继续由以下既有字段表示：

```text
original_band_top/bottom_exclusive_px
reference_envelope_top/bottom_exclusive_px
final_band_top/bottom_exclusive_px
final_search_region_start/end_px
```

实现先固定 `final_region/band_bounds`，再计算 shadow metadata，最后仍用原 `band_bounds` 执行 `band = image[top:bottom]`。测试还会 mock resolver 强制提出全幅 region，并逐数组断言 `u/v/valid/response/offset/normal/corrected_signal` 与未 mock 时完全相同。

## 7. 未接入的内容

本阶段没有接入：

- GUI active behavior、warning 或采集门禁；
- 棋盘 polygon / chess boundary mask；
- hardware ROI offset 或全传感器坐标；
- online previous-frame tracking/reacquire；
- expanded-band audit 或第二次 Steger；
- 正式 search-region 配置或 auto/additional 合并策略。

## 8. 修改文件与测试

- `calibration/src/realtime_steger.py`
  - 新增 `DetectorSummary`、`OutsideRegionEvidence`、`ProductionSearchRegionResolution`；
  - 结构化既有 detector 中间量；
  - 新增纯强度 outside evidence；
  - 新增 shadow `resolve_production_search_region()` 和 metadata；
  - 正式 `band_bounds` 保持不变。
- `calibration_tool/tests/test_production_search_region_shadow.py`
  - column 全部 active components/seed/margin 状态；
  - row 原图 `u` normal-axis 语义；
  - outside evidence 和 proposed envelope；
  - safe interval 不扩展；
  - margin clamp 与 `roi_max_height` 状态；
  - 强制 shadow proposal 不能改变正式 extraction 数组。

定向测试：

```text
python -m pytest -q tests/test_production_search_region_shadow.py
6 passed, 7 subtests passed

python -m pytest -q tests/test_laser_search_region.py tests/test_geometry_experiment.py
29 passed, 21 subtests passed
```

完整回归：

```text
python -m pytest -q
142 passed, 28 subtests passed

online compatibility smoke:
python -m pytest -q tests/test_backends.py
13 passed
```

## 9. 阶段结论

```text
behavior_changed = false
formal_steger_result_changed = false
shadow_resolver_available = true
```
