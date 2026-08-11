# Stage 3-1：Production Search Region Inputs 审计

## 1. 范围和时间边界

本轮只审计当前 `calibration_tool` 及其实际加载的共享实现 `../calibration/src/realtime_steger.py`，不修改算法、GUI、配置或正式输出。

本文把“正式 Steger”定义为：对最终裁剪 band 计算 Gaussian 一、二阶导数、Hessian 特征值/法向、亚像素 offset 和候选中心。当前代码的准确边界是：

```text
输入图像 / 参数 / optional LaserSearchRegion
  -> scan_axis=row 时转置
  -> _detect_steger_band                    pre-Steger
  -> auto 与 additional region 合并         pre-Steger
  -> image[start:end]                       pre-Steger
  ------------------------------------------------------
  -> gaussian_filter 五组导数                formal Steger begins
  -> Hessian / normal / subpixel offset
  -> 每 scanline 最强 candidate
```

对应代码：`calibration/src/realtime_steger.py:669-694,388-468`。因此 `_detect_steger_band()` 虽然位于正式 extraction 调用内部，但它本身是 pre-Steger 强度分析，不依赖 Hessian。

还需要区分两种“可用”：

1. **计算上 pre-Steger 可用**：在 Gaussian/Hessian 前已经能从图像或配置得到；适合放进共享 resolver。
2. **当前上层在调用前可见**：calibration/GUI 调用者无需先执行一次 extraction 就已经持有。当前 auto detector 的部分结果虽然计算上属于 pre-Steger，但公共 API 只在 extraction 返回后通过 metadata 暴露，上层不能直接拿它们决定同一次 extraction 的 region。

## 2. 当前正式调用链

### 2.1 GUI 实时 health 路径

```text
CameraConfig hardware ROI
  -> CameraSession.get_frame() -> CapturedFrame.image
  -> PreviewThread._emit_frame()
       -> analyze_frame()                         既有轻量质量
       -> RealtimeStegerQualityAnalyzer.analyze()
            -> extract_steger(image, options, diagnostic=True)
            -> auto region / final region
            -> formal Steger
       -> analyze_search_region_health()          post-Steger health
```

证据：`calibration_tool/gui/workers.py:118-160`；`calibration_tool/camera/steger_quality.py:122-156`。

GUI analyzer 在构造时读取共享 Steger profile，并用项目/采集配置的 `laser.orientation` 覆盖 `scan_axis`：horizontal→column，vertical→row。当前调用只传 `image/options/diagnostic=True`，**没有传 explicit/additional LaserSearchRegion**。

`analyze_frame()` 在时间顺序上早于 Steger。它能得到 coverage、FWHM、饱和度等聚合量，但不保留逐 scanline peak 的 normal-axis 坐标，因此当前 `FrameQuality` 不能直接生成 search interval。参见 `camera/quality.py:13-49,104-147`。

### 2.2 离线标定/重建 shared 路径

- 激光平面 V2：先生成 background-subtracted `corrected`、棋盘 pose、`roi_polygon` 和 `chess_boundary_mask`，然后对完整 `corrected` 调用 `extract_steger_columns()`；polygon/mask 只在中心产生后门控。参见 `calibrate_laser_plane_core.py:382-427`、`calibrate_laser_plane_core_v2.py:212-259`。
- ground extrinsics V2：读取完整灰度图并校验内参尺寸，直接调用 `extract_steger_columns()`，之后才做 continuity、RANSAC 和 laser-plane/depth gate。参见 `calibrate_ground_extrinsics_steger_v2.py:113-193`。
- reconstruction V4 shared：完整图直接 extraction，之后才做 segment continuity、可选 RANSAC 和三维 geometry。参见 `reconstruct_ground_pointcloud_cloudcompare_v4.py:137-224`。

这些活动的 calibration 生产入口当前也都没有传 additional region。Phase-A `geometry_experiment.py`、Stage 2B characterization 和相关 audit 脚本会显式传 region，但属于实验/诊断路径，不是生产策略。

三联图 laser-surface model fitter 使用另一条全图 Hessian 路径，没有 auto band、`roi_margin`、`roi_max_height` 或 `LaserSearchRegion`；本审计不把它误写成 shared resolver 的现有调用者。

## 3. auto search band 如何得到

实现位于 `calibration/src/realtime_steger.py:256-313`。以下变量都在 Gaussian/Hessian 前产生。

### 3.1 column 模式

输入图像 shape 为 `(H,W)`，normal axis 是原图 `v/row`：

1. `row_peak[v] = max_u(gray[v,u])`。
2. 若 `max(row_peak) < threshold`，auto region 为 `None`。
3. `row_sum[v] = sum_u(gray[v,u])`，`seed = argmax(row_sum)`。
4. `adaptive_threshold = max(threshold, 0.3 * row_peak[seed])`。
5. `active[v] = row_peak[v] >= adaptive_threshold`。
6. 从 seed 向两侧扩展，只取**包含 seed 的连续 active 区间** `[raw_top, raw_bottom)`。
7. 两边各加 `roi_margin` 并裁到 `[0,H)`。
8. 若结果高度超过 `roi_max_height`，改为以 seed 为中心、最多 `roi_max_height` 的区间，并再次贴合图像边界。

这里的 “active component” 不是二维 connected-component 或 Steger ridge；它只是 `active` 一维布尔数组中包含 seed 的连续分量。其它 active 分量的位置在计算时已经存在，但当前算法不枚举、不保留，也不参与 auto band。

seed 由整行亮度总和决定，而不是由 `row_peak` 最大值决定。因此一条更长或总能量更高的亮结构可能赢得 seed，即使另一条局部峰更强。

### 3.2 row 模式

`extract_steger()` 在 detector 前先把原图 `(H,W)` 转为 `gray.T` `(W,H)`。所以相同代码中的：

- `row_peak` 实际是逐原图 column 的最大强度；
- `row_sum/seed_row/active component` 的坐标实际是原图 `u/column`；
- `roi_margin` 沿原图 `u` 生效；
- `roi_max_height` 实际限制原图 normal-axis 宽度。

调用者的 `LaserSearchRegion` 始终使用原图 normal-axis 坐标；row 模式为原图 `u`。转置后该坐标数值直接对应工作图 axis-0，无需调用者交换 start/end。最终 `(u,v)` 输出再由 `_restore_row_axis()` 恢复。

旧 metadata 中的 `seed_row/raw_candidate_top/raw_candidate_bottom` 仍是内部工作域命名；row 模式下应解释为原图 `u`，不能按原图 `v` 理解。统一字段 `normal_axis` 和 `final_search_region_start/end_px` 才是对上层无歧义的语义。

### 3.3 detector 中产生但保留程度不同的信息

| 信息 | pre-Hessian 已存在 | 是否写入 extraction metadata | 当前上层能否在同一次调用前读取 |
|---|---|---|---|
| 原图/工作图 shape | 是 | 间接体现在 region/数组 shape | 是，调用者本来就有 image |
| `row_peak[]` | 是 | 否 | 否；除非重新计算或调用 private detector |
| `row_sum[]` | 是 | 否 | 否 |
| `seed` | 是 | `seed_row` | 否；只在 extraction 返回后看到 |
| `adaptive_threshold` | 是 | 是 | 否；只在返回后看到 |
| 完整 `active[]` | 是 | 否 | 否 |
| seed 所属 raw active interval | 是 | `raw_candidate_top/bottom` | 否；只在返回后看到 |
| margin 前/后的 interval | 是 | `margin_before_clip/after_clip` | 否；只在返回后看到 |
| `roi_max_height_applied` | 是 | 是 | 否；只在返回后看到 |
| auto final interval | 是 | `original_band_*` | 否；只在返回后看到 |

`_extract_columnwise()` 即使 `diagnostic=False` 也创建并填充 `auto_band_trace`，所以 seed/bounds/cap 状态会进入正式 extraction metadata；但这不等于上层能在 formal Steger 前利用它们。要让 conditional resolver 使用它们，应在共享模块内部把 detector summary 作为 pre-Hessian 对象传给 resolver，而不是先完整跑一次 Steger。

## 4. `roi_margin` 与 `roi_max_height`

当前正式 profile `calibration/config/realtime_steger.yaml` 为：

```yaml
sigma: 1.5
threshold: 30.0
deriv_thresh: 0.5
roi_margin: 48
roi_max_height: 512
scan_axis: column
```

- `roi_margin` 是 seed active interval 两侧的固定 normal-axis padding，不是中心到最终边界的保证；如果 seed 选错分量，它不能覆盖另一条真实激光。
- `roi_max_height` 只限制 **auto region**。它在 margin 之后应用，并以 seed 为中心重新裁剪。
- explicit/additional region 是在 auto cap 之后合并的，因此可以把最终 region 扩到 `roi_max_height` 之外。合并后不会再次套用 `roi_max_height`。
- 图像边界 clamp 可能使单侧实际 margin 小于配置值；`margin_before_clip/after_clip` 可以在 Hessian 前识别这种情况。
- `sigma` 在 region 决定前已知，因此 Gaussian kernel 所需上下文（当前 health 用 `ceil(4*sigma)`）也属于 pre-Steger 可用约束；但真正的中心到边界 clearance 仍需中心产生后才能计算。

## 5. explicit/additional `LaserSearchRegion`

`LaserSearchRegion(start_px,end_px,source)` 表示原图 normal axis 上的半开区间。它在调用前即可由配置、reference、棋盘几何或其它上层先验构造，属于明确的 pre-Steger 输入。

当前处理顺序（`realtime_steger.py:360-385,388-460`）：

1. detector 生成 auto region；
2. additional region 裁到当前工作图 normal-axis extent，裁后为空时报错；
3. 两者取包络并集：`[min(start), max(end))`；
4. auto 和 additional 都不存在时返回空 extraction；
5. 只存在其中一个时直接使用该 region；
6. 最后才裁出 band 并执行 Hessian。

所以 `search_region` 当前是 **additional envelope/hint**，不是强制 crop，也不是与 auto 的 intersection。`source` 只用于 provenance，不影响合并策略。

兼容 API `additional_band_bounds=(start,end)` 会转换为 `source="additional_band_bounds"` 的 `LaserSearchRegion`；`additional_search_region` 与旧 alias 不能同时传。新统一 `extract_steger()` 和 `steger_backend(...,search_region=...)` 均支持 column/row 原图语义。

当前 GUI analyzer 没有 search-region 参数；激光平面、ground、reconstruction 的活动 shared 调用也没有传 additional region。也就是说，生产结果目前仍是 auto-only。

## 6. calibration ROI：三个不同概念

“ROI”在当前系统中至少有三种含义，不能混为一个 search region。

### 6.1 相机 hardware ROI

`CameraConfig` 在取帧前就有 `offset_x/offset_y/width/height`；海康和大恒 session 会把它写入相机，并返回对应 shape 的局部图及 `CapturedFrame.offset_x/offset_y`。参见 `camera/models.py:31-74`、`camera/mvs.py:130-137`、`camera/daheng.py:408-438`。

它是严格的 pre-Steger crop。GUI analyzer 当前只接收 `frame.image`，没有接收 `CapturedFrame` 或 offset，因此 shared detector 的所有坐标都是 hardware-ROI 局部坐标。若未来上层提供全传感器坐标的 region：

- column 模式需减去 `offset_y`；
- row 模式需减去 `offset_x`；
- 然后再裁到局部 image extent。

当前 GUI 默认全幅时 offset 为零；任务切换若改变 hardware ROI，会先重配相机，再对新局部图独立 auto-detect。

### 6.2 棋盘标定 polygon / boundary mask

激光平面标定的 `prepare_frame()` 在 formal Steger 前已经完成：

- chessboard corners 与 pose；
- 投影后的 `pose.roi_polygon`；
- `chess_boundary_mask`；
- laser−nolaser 的正差分 `corrected`。

这些都是计算上、也是上层对象中真正可见的 pre-Steger 信息。但当前 `extract_centres()` 把完整 corrected image 送入 Steger，之后才用 polygon/mask 把中心标为 `rejected_roi/rejected_chess_boundary`。因此它们目前**不决定 Hessian crop**。

polygon 可以在未来 calibration 上层投影成 normal-axis envelope，作为 additional/reference hint；boundary mask 更适合继续保留为 post-center 的棋盘边缘排除规则，不宜直接作为公共一维 region 的核心语义。

### 6.3 reconstruction/geometry ROI

内参尺寸、相机模型、laser plane、深度范围、ground 假设等在调用前可能已加载，但当前 image polygon、continuity、RANSAC、射线求交和 depth gate 都消费 Steger 中心，属于 post-Steger 筛选。它们不会保护 Hessian 免受错误 search crop 影响。

## 7. A：pre-Steger 可用信息

### A1. 当前 shared extractor 内部在 Hessian 前已经持有

- 原始二维灰度/差分图、dtype、shape 和 normal-axis extent；
- `scan_axis` 及 row 模式转置后的坐标映射；
- `sigma/threshold/deriv_thresh/roi_margin/roi_max_height`；
- `row_peak[]`、`row_sum[]`、seed；
- adaptive threshold、完整一维 active mask、seed 所属 active interval；
- margin 后 interval、图像边界 clamp 状态、max-height cap 状态；
- optional explicit/additional `LaserSearchRegion` 及 source；
- auto/additional 合并后的 final region；
- final band 内和全图的纯强度统计。

最后一项值得特别说明：`full_image_max_intensity`、`band_max_intensity`、`intensity_peak_outside_detected_band` 不需要 Hessian，理论上都能在 region 决定后立即计算。当前实现为了构造统一 `StegerColumnDiagnostics`，在 Hessian 之后才执行这些语句并随 extraction 返回；这是**调度/封装位置是 post，信息依赖是 pre**。

### A2. calibration/GUI 上层在调用前已经持有，但当前未全部传入 core

- 项目 `laser.orientation`，可确定 `scan_axis` 和 normal axis；
- CameraConfig hardware ROI 和 CapturedFrame offset；
- 当前任务 role、quality mode、exposure/gain、sensor full scale、task tags；
- GUI `analyze_frame()` 已产生的 coverage/FWHM/饱和聚合量；
- 激光平面标定的 corrected difference image、board pose、roi polygon、boundary mask；
- offline ground/reconstruction 的 image shape、intrinsics 和已加载几何模型；
- 实验路径显式构造的 reference envelope。

其中 exposure、FWHM、coverage 可以作为“信号是否可信”的辅助置信度，但不包含 normal-axis interval 坐标，不能单独解决 region 定位。

当前 PreviewThread/analyzer 是逐帧无状态的；同 pose 的 chess/nolaser/laser 虽在 capture plan 中顺序存在，但没有把上一任务图像或上一帧中心作为本帧 resolver 输入。

## 8. B：必须 formal Steger 后才能得到的信息

以下信息依赖 Gaussian/Hessian、亚像素求解或中心集合，不能用于决定**同一次** formal Steger 的 region，除非付出第二次 Steger；生产 resolver 不应这样做。

- Hessian eigenvalue、normal、first/second derivative；
- 每像素 derivative/response/offset gate 状态；
- 每 scanline 被选择的最强 ridge candidate；
- `StegerExtraction.u_px/v_px/valid/response/offset_px/normal_y_abs`；
- `derivative_condition_passed`、`ridge_response_passed`、`subpixel_offset_passed`、`accepted` 和 rejection reason；
- valid fraction、response statistics、same-candidate fraction；
- 中心到 final region 的 boundary clearance min/P05/median、inside-kernel fraction；
- continuity segment、line RANSAC inlier/residual；
- board polygon/mask 对**中心**的接受结果；
- 去畸变后的线模型、laser/board plane intersection、正深度和其它 geometry gate；
- Stage 2B 的 paired center shift，以及 expanded-region 与 formal 的中心差。

例外是 GUI 的 `outside_search_region_peak_fraction`：该 health 字段当前在 extraction 后消费 diagnostics，但其底层只比较强度峰，逻辑上可前移到 resolver；它不属于“必须 Steger 后”。相反，`center_near_search_boundary` 必须有正式中心，因此是真正的 post-Steger health。

## 9. pre/post 总表

| 数据 | 依赖 Hessian/中心 | 适合决定同一次 region | 当前生产是否使用 |
|---|---:|---:|---|
| image shape / normal-axis extent | 否 | 是 | 用于 clamp/transpose |
| orientation / scan_axis | 否 | 是 | 是 |
| `row_peak/row_sum/seed/active` | 否 | 是 | auto detector 使用 seed component |
| 其它 active components | 否 | 是 | 当前丢弃 |
| margin/max-height/clamp 状态 | 否 | 是 | 当前按固定公式使用 |
| explicit/reference region | 否 | 是 | API 支持；生产未传 |
| hardware ROI offset | 否 | 是，若 region 是全幅坐标 | core 当前不可见 |
| board roi polygon envelope | 否 | 是，calibration-specific | 当前仅 post-center gate |
| full-vs-band intensity peak risk | 否 | 是 | 仅 post-call diagnostics 展示 |
| GUI coverage/FWHM | 否 | 只能作置信度辅助 | 当前不进入 resolver |
| formal center/response/offset | 是 | 否 | post extraction |
| boundary clearance | 是 | 否 | post health |
| continuity/RANSAC/geometry | 是 | 否 | post filters |

## 10. 对 conditional safe-region resolver 的简短建议

后续最小、低成本且不需要第二次 Steger 的 resolver，最适合优先使用：

1. **共享 detector summary**：normal-axis extent、`row_peak/row_sum`、seed、完整 active mask/全部 active intervals、seed component、margin/max-height/clamp 状态。不要只保留包含 seed 的一个分量。
2. **纯强度 outside evidence**：在 Hessian 前比较 final region 内外的逐 scanline peak，复用现有 `intensity_peak_outside_detected_band` 逻辑；它比 post-Steger valid count 更早且开销很低。
3. **已知 kernel/safety context**：使用 pre-known `sigma` 和经 characterization 定型的 clearance policy 检查候选 region 是否有足够 normal-axis padding；不要用本次中心 clearance 反过来触发同帧第二次 Steger。
4. **calibration-specific explicit hint**：棋盘 `roi_polygon`、reference surface/envelope 或项目静态范围由 calibration 上层生成原图 normal-axis `LaserSearchRegion`，共享层只负责验证、坐标转换和合并。
5. **hardware ROI provenance**：resolver 明确接收局部 image extent；需要全幅先验时，由上层用 `offset_y`（column）或 `offset_x`（row）转换后再传入。

不建议把 Hessian response、formal centerline、continuity/RANSAC 或三维 geometry 用作同次 resolver 输入，因为这会形成循环依赖并要求第二次 Steger。也不建议仅凭 GUI coverage/FWHM 决定坐标范围：它们适合置信度判断，但没有保留条纹的 normal-axis 位置。

本轮结论：

```text
behavior_changed = false
steger_math_changed = false
```
