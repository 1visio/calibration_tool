# Steger 搜索域调用图审计

> 审计性质：只读代码审计，不修改运行行为。

> 当前标定工具：`calibration_tool` 分支 `feature/daheng-camera-support`，HEAD `5d0d01e`。  
> Phase-A 对照：`feature/phase-a-baseline-distance-laser-angle`，HEAD `f747bd6`。  
> 在线点云工具：`0704line-laser-3d-scanner/laser_measurement_tool`，分支 `feature/scan-stage1-simulated-pitch`。

## 1. 范围与结论

本审计覆盖标定 workflow 及其实际加载的 `../calibration/src`、在线点云项目的实时/离线扫描/单图 GUI/性能诊断入口，以及 Phase-A 分支验证过的 `auto band ∪ reference band` 调用方式。

核心结论：

1. **标定兼容路径和在线生产路径最终共用同一文件**：`D:/Docs/linelaserscan/calibration/src/realtime_steger.py`。
2. 当前共享核心总会先运行 auto band；`additional_band_bounds` 与 auto band 取包络并集，不是限制范围。
3. Gaussian 导数、Hessian 和亚像素计算只在最终 band 的裁剪图上进行。真实线在 band 外会消失；线距 band 边缘过近时，卷积边界会影响 Hessian/亚像素值。
4. 在线生产链没有使用上一帧中心或上一帧 band。每个实际处理帧独立 auto-detect；相机硬件 ROI 只是输入坐标系。
5. `scan_axis=row` 时共享实现先转置，核心中的 `top/bottom` 实际对应原图列坐标 `u/x`。
6. API 当前不对称：`steger_backend()` 支持 `column/row`，但 `additional_band_bounds` 只暴露在仅接受 `scan_axis=column` 的 `extract_steger_columns()` 上。
7. `calibration_tool/camera/quality.py` 不调用 Steger；其 peak/coverage/FWHM 不能证明 Steger band 未截断真实激光。

## 2. 统一共享核心

### 2.1 实际模块归一化

```text
calibration_tool workflow
  -> calibration_tool/calibration_tool/stages.py
  -> ../calibration/src/<stage module>.py
  -> ../calibration/src/steger_laser_center.py（部分入口兼容 API）
  -> ../calibration/src/realtime_steger.py

online / offline point-cloud tool
  -> laser/laser_extractor.py
  -> laser/backends.py::steger_backend 或 shared_steger_backend
  -> 动态加入 ../../../calibration/src
  -> ../calibration/src/realtime_steger.py
```

证据：

- `calibration/src/steger_laser_center.py:484-512` 在文件末尾覆盖旧入口，调用 `realtime_steger.extract_steger_columns()`。
- 在线 `laser/backends.py:399-411` 的公开 `steger_backend()` 动态导入同一个 shared 文件；`:449-463` 的 `shared_steger_backend()` 也委托它。
- 在线 `laser/backends.py:277-396` 的本地 `_detect_steger_band()` / `_extract_steger_columnwise()` 是保留的历史实现，公开 backend 不调用。

### 2.2 auto band 与 Gaussian/Hessian

共享实现 `calibration/src/realtime_steger.py` 的关键位置：

- `StegerParams`：`:60-83`；类默认 `roi_margin=120`、`roi_max_height=512`、`scan_axis=column`。
- YAML/CLI：`:148-215`；当前 `calibration/config/realtime_steger.yaml` 为 `roi_margin=48`、`roi_max_height=512`、`scan_axis=column`。
- `_detect_steger_band()`：`:218-275`。
- `_extract_columnwise()`：`:322-538`。
- `extract_steger_columns()`：`:541-564`。
- `steger_backend()`：`:567-584`。

auto band 算法：

1. `row_peak = max(gray, axis=1)`；全图最大值低于 `threshold` 时无 band。
2. `seed = argmax(sum(gray, axis=1))`。
3. 阈值为 `max(threshold, 0.3 * row_peak[seed])`，从 seed 向两侧扩展连续 active 行。
4. 两侧加 `roi_margin`，裁到图像范围。
5. 超过 `roi_max_height` 时，以 seed 为中心裁成最大高度。
6. 返回半开区间 `[top, bottom)`。

最终范围规则：

```text
auto 无 + additional 无 -> 空结果
auto 有 + additional 无 -> auto
auto 无 + additional 有 -> additional
auto 有 + additional 有 -> [min(top), max(bottom))
```

`additional_band_bounds` 会先裁到当前输入图 axis-0 范围，裁后为空则报错。它是 reference envelope，不是 intersection。

`realtime_steger.py:391-446` 先执行 `band = image[top:bottom]`，再计算 `ry/rx/ryy/rxx/rxy`、Hessian 主特征值、法向和亚像素 offset。SciPy 默认边界模式作用于裁剪后的 band 边界，因此边缘距离会影响结果。

候选需满足：法向有效、法向二阶导 `< -deriv_thresh`、两个 offset 分量均不超过 `0.6 px`、band 灰度不低于 `threshold`；每个内部 column 保留最强负曲率候选，输出 `v` 加回 `top`。

### 2.3 diagnostics 不进入正式结果

`realtime_steger.py:108-124,460-520` 在 `diagnostic=True` 时可记录全图/band 强度峰、`intensity_peak_outside_detected_band`、各门限状态和 rejection reason。这些字段被标记为 diagnostic-only。当前标定 stage 和在线生产入口均未启用它，正式质量指标无法直接看到 band 截断原因。

## 3. calibration_tool 调用入口

### 3.1 workflow 总调用图

```text
WorkflowRunner / ComputationService.run
  -> stages.py::STAGES[module]
  -> laser_plane_shared_steger
       -> calibrate_laser_plane_core_v2.run
       -> realtime_steger.extract_steger_columns
       -> board ROI/chess boundary -> plane fit
  -> laser_surface_models
       -> calibrate_laser_surface_models.main
       -> scripts/fit_laser_models_from_triplets.py
       -> independent full-image Steger Hessian -> model fitting
  -> ground_extrinsics_shared_steger
       -> calibrate_ground_extrinsics_steger_v2.run
       -> steger_laser_center.extract_steger_columns
       -> realtime_steger.extract_steger_columns
       -> continuity / optional RANSAC / depth gate
  -> reconstruct_shared_steger
       -> reconstruct_ground_pointcloud_cloudcompare_v4.main
       -> steger_laser_center.extract_steger_columns
       -> realtime_steger.extract_steger_columns
       -> continuity / optional RANSAC / reconstruction gates
```

stage 注册在 `calibration_tool/calibration_tool/stages.py:26-76`；`ComputationService.run()` 于 `:88-145` 动态导入并执行。

### 3.2 活动调用点

#### laser_plane_shared_steger

- 位置：`calibration/src/calibrate_laser_plane_core_v2.py:212-259`。
- image shape/crop：`corrected.shape == (H,W)`，来自 `prepared.corrected`，否则 `background_subtracted`；有背景处理，无空间 crop。
- scan_axis/config：`StegerSettings` + `realtime.options_from_args()`；CLI 虽允许 row，本入口调用仅-column API。
- band：auto 是；`additional_band_bounds` 否。
- 后处理/消费者：image bounds、`pose.roi_polygon`、`chess_boundary_mask`；再进入棋盘平面交点、激光平面拟合、validation RMSE/P95。

#### laser_surface_models

- 位置：`calibration_tool/scripts/fit_laser_models_from_triplets.py:899-996`。
- image shape/crop：laser/background/chess 均为 `(H,W)`；同尺寸正差分，无空间 crop；board hull mask 只门控候选。
- scan_axis/config：项目 `laser.orientation` 映射 horizontal/vertical，不使用 shared `scan_axis`。
- band：无 auto band，也无 additional。
- 后处理/消费者：每 column/row 最强响应、多项式 continuity、均匀下采样、射线-棋盘平面正深度；进入三种光片模型拟合。

#### ground_extrinsics_shared_steger

- 位置：`calibration/src/calibrate_ground_extrinsics_steger_v2.py:113-193`。
- image shape/crop：完整灰度 `(intrinsics_height,intrinsics_width)`，精确匹配内参，无 crop。
- scan_axis/config：`steger.settings_from_args()`；CLI 允许 row，但最终调用仅-column API。
- band：auto 是；additional 否。
- 后处理/消费者：continuity；flat-ground 时 undistorted line RANSAC；激光平面求交/depth；至少 30 点；counts 进入 ground quality。

#### reconstruct_shared_steger

- 位置：`calibration/src/reconstruct_ground_pointcloud_cloudcompare_v4.py:137-224,227-285`。
- image shape/crop：`read_mono8()` 得 `(H,W)`；声明标定尺寸时精确匹配；无 crop。
- scan_axis/config：stage 强制 `--steger-extractor shared`；profile/CLI；调用仅-column API。
- band：auto 是；additional 否。
- 后处理/消费者：none 或 segment continuity；可选 undistorted line RANSAC；随后求交稳定、finite、positive/depth、障碍线拟合。

V4 的 `measurement-tool` 诊断分支在同文件 `:107-159`，输入相同且无 crop；它读取 measurement tool profile/inline `steger`，调用支持 column/row 的公开 backend。backend 本身无 continuity/RANSAC，随后仍走 V4 geometry。

#### 曲率论文诊断

- 位置：`calibration/src/plot_laser_line_curvature_paper.py:198-264`。
- image shape/crop：每幅原始灰度 `(H,W)`，无 crop。
- config：`measurement-tool` 读工具配置；`shared` 使用默认 column。
- band：auto 是；additional 否。
- 后处理：不做 continuity/RANSAC/geometry；全部中心进入曲率分析。`:380-445` 的 ROI 只用于绘图。

上述 laser plane、ground extrinsics、V4 shared 和曲率 shared 调用 `extract_steger_columns()`。`realtime_steger.py:557-558` 在 `scan_axis != column` 时抛错。它们的 CLI 虽声明 row，当前并未形成可运行的 row 路径；在线生产的 `steger_backend()` 才支持 row。

### 3.3 三联图独立 Hessian 路径

- `positive_difference()`：`fit_laser_models_from_triplets.py:273-281`，同尺寸正差分。
- `steger_candidates()`：`:284-332`，完整 diff 上以 `mode=nearest` 计算五个 Gaussian/Hessian 数组。
- `board_inner_mask()`：`:258-270`，只是候选布尔门控，不是预先 crop。
- `select_one_per_scanline()`：`:335-380`；horizontal 每 column 最强并拟合 `y=f(x)`；vertical 每 row 最强并拟合 `x=f(y)`。
- `extract_laser_centers()`：`:404-432`。

该路径没有 auto band、`roi_margin`、`roi_max_height` 或 `additional_band_bounds`。未来统一 search region 时，不应修改其 Hessian 核心公式。

### 3.4 质量分析与 Steger

`calibration_tool/calibration_tool/camera/quality.py:13-151` 不调用 Steger。horizontal 在原图逐 column 做 peak/coverage、沿 row 算 FWHM；vertical 转置后复用。它检查原图可见性与线宽，不检查 extractor 最终 band。

所以可能出现原图 quality 正常，但 auto band 选了另一亮带或把真实线压在边缘；stage 只看到 `rejected_steger`、有效点减少或拟合样本变化。`stages.py:174-290` 的门禁读取最终 RMSE/P95、flatness 等，也不读取 band diagnostics。

### 3.5 历史/非活动实现

- `calibration/src/steger_laser_center.py:90-250`：历史实现，文件末尾入口覆盖并委托 realtime。
- `calibration/src/calibrate_laser_plane_core_v2.py:173-210`：旧 import helper，当前 `extract_centres()` 不调用。
- `calibration/src/calibrate_ground_extrinsics_steger.py:136-240`：V1 独立 Hessian，不在 stage registry。
- `calibration/src/reconstruct_ground_pointcloud_cloudcompare_v3.py:93-220`：V3 独立 Hessian，不在 stage registry。

## 4. online point-cloud tool 调用入口

### 4.1 实时逐帧生产链

```text
MvsCameraSession / DahengCameraSession.get_frame
  -> CapturedFrame(image=hardware ROI, offset_x, offset_y)
  -> OnlineController._acquire_loop -> latest-frame slot
  -> OnlineController._process_loop
  -> FramePipeline.run_frame
  -> extract_laser_center(frame.image, params)
  -> laser.backends.steger_backend
  -> calibration/src/realtime_steger.steger_backend
  -> centers_local + hardware ROI offset
  -> reconstruct_uv_to_ground
```

- `online/controller.py:157-176` 持续采集；`:183-198` 对 latest-frame slot 取出的每个实际处理帧调用一次 `run_frame()`。处理跟不上时 slot 可覆盖积压帧，所以不保证每个相机帧都处理。
- `_last_camera_frame` 只统计 frame number gap；`_last_result` 只用于显示/统计，均不进入下一帧 search region。
- `online/pipeline.py:44-61` 将 `frame.image` 原样送入 extractor；`:51-55` 在 Steger 后才加硬件 ROI offset。
- `online/mvs_camera.py:232-253,318-352` 与 `online/daheng_camera.py:268-288,409-474` 在 SDK 上配置 Width/Height/Offset，返回 shape 为 `(config.height, config.width)` 的局部图。
- `online/models.py:25-45` 当前默认硬件 ROI 为 `width=2448, height=300, offset=(0,880)`；GUI 可覆盖。

### 4.2 调用点逐项记录

#### 实时 OnlineController -> FramePipeline.run_frame

- 是否每帧：每个实际处理帧一次。
- image shape/crop：`(camera ROI height, camera ROI width)`；硬件已裁 ROI，pipeline 无软件 crop。
- scan_axis/config：`measure_tool.yaml -> extraction.profile -> realtime_steger.yaml`，inline `extraction.steger` 覆盖；当前 column。
- auto band：是。additional：否，公开 backend 无此参数。上一帧信息：无。
- 后处理：backend 无 continuity/RANSAC；重建时 image ROI polygon、求交稳定/finite/positive/depth gates。

#### OfflineScanRunner

- 位置：`scan/offline_scan.py:109-199,210-289`。
- 是否每帧：每个离线输入/角度一次。
- image shape/crop：文件原 shape，可为全幅或带 offset 的 ROI；不再 crop。
- scan_axis/config：复用同一个 `FramePipeline`。auto 是，additional 否，上一帧信息无；repeat-one 只是重复相同图。
- 后处理：与实时相同，再做扫描坐标变换。

#### 单图 GUI

- 位置：`gui/main_window.py:586-618`。
- 调用频率：用户每次点击提取一次。
- image shape/crop：`_image.shape`，可为全幅或用户提供 offset 的 ROI；提取前无软件 crop。
- config：`AppConfig.extraction_options_by_method`；auto 是，additional 否，上一帧信息无。
- 后处理：ROI manager 在提取后筛 baseline/obstacle 点；重建和 measurement 再做 geometry/outlier gates。

#### 性能/AOI 诊断

- 位置：`tools/bench_pipeline.py:460-546`。
- 它先对全图提取，再由同一次结果生成 band，`crop_band()` 后再次调用 backend，并把坐标 offset 加回。
- 内外两次 backend 仍各自 auto-band；没有传 additional。
- 这只是“用已有结果模拟 AOI”的 benchmark，不是生产帧间状态。

### 4.3 配置解析

`configs/measure_tool.yaml:19-36` 当前为：

```yaml
extraction:
  method: steger
  profile: ../../../calibration/config/realtime_steger.yaml
  steger: {}
  shared_steger: {}
```

`app_config.py:169-224` 读取 profile 的 `steger` 映射，再以 inline `extraction.steger` 覆盖；旧名称 `shared_steger` 被归一化为相同实时参数。`laser/backends.py:474-496` 校验并绑定 backend。

### 4.4 在线 Steger 后处理

共享 `steger_backend()` 只返回每个 scanline 的最强有效 Steger 点，不调用 `continuity_filter_columns()`、`points_from_valid_columns()` 或 line RANSAC。在线后续门限在 `reconstruction/reconstructor.py:500-599`：

- 可选 `image_roi_polygon`，是 Steger 后像素 gate；
- 相机去畸变；
- 光片求交稳定性/有效根；
- finite、正深度、工作距离；
- 转地面坐标后再次 finite。

GUI baseline/obstacle ROI 与 measurement outlier/line fit 均发生在 Steger 后，不能防止 search band 内错误 ridge 形成连续伪点。

## 5. `scan_axis=row` 的转置坐标语义

设原图 shape 为 `(H,W)`，像素为 `(u,v)=(column,row)`。

### 5.1 column

```text
input: (H,W)
axis-0 band: 原图 row/v
内部逐 column: 每个 u 选择一个 v
输出: (u,v)
```

此时 `top/bottom` 与 `additional_band_bounds` 都是原图 `v/y` 半开区间。

### 5.2 row

`realtime_steger.py:581-584` 的实际变换：

```text
原图 image              shape (H,W)
image.T                 shape (W,H)
转置域 axis-0 band       原图 column/u
转置域逐 column          内部 u' = 原图 v，内部 v' = 原图 u
内部结果 (u',v')         = (原图 v, 原图 u)
pixels[:, ::-1]          -> (原图 u, 原图 v)
```

因此 row 模式中：

- `_detect_steger_band()` 的 `row_peak` 实际逐原图 column 取峰；
- `seed_row` 实际是原图 seed column；
- `roi_margin` 沿原图 `u/x` 生效；
- `roi_max_height` 实际限制原图 search-band 宽度；
- `band_top_px/bottom_px` 若原样暴露，实际表示原图 `u`，变量名会误导调用方。

公开 `steger_backend()` 没有 `additional_band_bounds`；`extract_steger_columns()` 虽有该参数，却在转置前拒绝非-column。因此当前没有受支持的 API 能把原图 `u` envelope 交给 row 模式。

`tools/bench_pipeline.py:82-100` 的 `AxisSpec` 已正确表达 column crop axis=0、row crop axis=1，可作为轴语义参考，但它仍是 benchmark 局部模型。

## 6. Phase-A 已验证改进

Phase-A 没有把 policy 接进标定 workflow 或在线生产链；它在几何实验上层构造 reference envelope，再调用共享 extractor：

```text
scripts/geometry_experiment.py::_extract_reference_stacks
  -> 读取完整灰度图
  -> realtime.extract_steger_columns(
       image, steger_options, additional_band_bounds=reference_envelope)
  -> 保存 original/reference/final band metadata
  -> 用正式 Steger valid 输出计算实验统计
```

关键位置为该分支 `scripts/geometry_experiment.py:1307-1382,2081-2094`。未传 envelope 的 original stacks 与传入 envelope 的正式 stacks 分开计算；reference envelope 来自冻结 reference curve 与人工确认 ROI。diagnostic 输出不回写正式结果。

已验证证据：

- `B05_A10` H10：auto `[793,895)`，reference `[886,927)`，final `[793,927)`。
- 原 band 中心距下边界约 `3.5 px`；扩展后约 `35.6 px`，消除 Hessian 卷积边界影响。
- 1850/1850 配对点保留；最大中心差 `0.193985 px`；95.6757% 保持同一整数 candidate row；扩展后响应 100% 更强。
- 7 个原 H1 不可用配置中 6 个恢复；剩余 1 个由 ROI trim sensitivity 规则导致，不是 band detection 失败。
- 可比 H10/H30 主敏感度最大变化 `+0.3574%`，低于 1%；`B05_A10` H10 repeatability P95 改善 `-21.0730%`。

来源：Phase-A 分支 `experiments/geometry_baseline_angle/results/B05_A10_band_candidate_audit.md`、`band_fix_regression.md`，以及 `tests/test_geometry_experiment.py:713-780` 的 band 外峰/additional envelope synthetic tests。

## 7. 建议的最小重构边界

### 7.1 进入共享 `LaserSearchRegion`

共享对象只负责在当前图像坐标系中决定 Gaussian/Hessian 可安全计算的 axis-neutral 区域：

- 明确 `scan_axis` / stripe orientation；
- 原图 shape 与 search-axis extent；
- auto candidate interval、可选 reference/predicted interval、合并 policy；
- kernel context margin：不仅要求峰在 band 内，还要保证最小 Gaussian/Hessian 上下文；
- max extent、clamp、原图 interval 与 columnwise 工作域 interval 的转换；
- provenance/metadata：auto、reference、final、来源、clamp/max-size 状态；
- 公共字段使用 `search_axis/low/high_exclusive`，不要继续用跨方向含混的 top/bottom/max_height。

建议共享 API 形态：

```text
resolve_search_region(image, params, hint) -> resolved region
extract_steger(image, params, resolved region, diagnostic=False)
  -> points + metadata
```

也可为 `steger_backend()` 增加 axis-neutral hint；关键是 transpose 只在共享边界内完成，上层不自行交换 `u/v`。

### 7.2 保留在 calibration 上层

- reference envelope 来源：棋盘/reference surface、冻结 curve、pose/task ROI；
- auto-only fallback 与缺少 reference 时的失败策略；
- board polygon、chess boundary、训练/验证拆分；
- continuity、平面假设 RANSAC、射线-棋盘/激光面 geometry gates；
- search-region diagnostics 如何进入质量门禁，区分 line absent、outside auto band、near edge；
- Phase-A 正式/诊断产物隔离与配置 hash/provenance。

标定上层不应直接传 `(top,bottom)` 或在 row 模式手工 transpose；应提交原图坐标的 reference hint。

### 7.3 保留在 realtime/online 上层

- 硬件 ROI Width/Height/Offset 与全幅坐标恢复；
- latest-frame slot、帧丢弃/吞吐策略；
- 可选上一帧状态、置信度、扩张速度、丢失后全局 reacquire；
- 在线预算和 tracked-region 失败 fallback；
- 三维重建 image ROI、深度、求交、工作距离 gates；
- telemetry：region 来源、边界距离、reacquire 次数。

上一帧信息若启用，只生成原图坐标 hint；auto/reference/prediction 合并与 row/column 转换仍由 shared `LaserSearchRegion` 完成。

### 7.4 最小落地顺序

1. 建立 axis-neutral region 数据结构和 row/column 坐标测试，不改变默认 auto policy。
2. 将现有 `additional_band_bounds` 适配成 reference hint，保持 column 数值逐点回归。
3. 让公开 backend 和结构化 extraction API 都支持 column/row + hint。
4. 标定 workflow 接入 Phase-A envelope，并让 near-edge/outside-band diagnostics 进入质量报告。
5. 在线先接静态配置 region；上一帧 tracking 作为独立上层策略。

## 8. 检索项覆盖清单

- `_detect_steger_band`：shared `realtime_steger.py:218-275`；online `laser/backends.py:277-303` 另有不被公开 backend 调用的历史副本。
- `extract_steger_columns`：shared 结构化、仅-column API；多个 calibration 兼容入口；online 兼容模块末尾也委托 shared。
- `steger_backend`：online 生产公开入口，委托 shared；shared backend 支持 row transpose。
- `additional_band_bounds`：shared 仅-column API；当前生产 calibration/online 均未传；Phase-A geometry experiment 传入。
- `roi_margin` / `roi_max_height`：shared params/profile/CLI；row 模式实际沿原图 u 生效。
- `scan_axis`：online backend 可用；多个 calibration CLI 虽可配置 row，但其仅-column API 会拒绝。
- Gaussian/Hessian：shared 最终 band 上计算；triplet stage 有独立全图实现；其余 V1/V3/兼容代码是历史实现。
- Steger 前 crop/ROI：calibration 活动 shared 入口无空间 crop；triplet 只有 board mask；online 生产只有相机硬件 ROI；bench 有显式软件 AOI；measurement/geometry polygon 均为 Steger 后 gate。
- continuity/RANSAC/geometry：已在第 3、4 节逐入口列出；online backend 本身没有 continuity/RANSAC。
- 前一帧信息：在线生产没有；bench 只用同一次全图结果模拟 AOI。

## 9. 集成前必须保留的回归约束

- column/horizontal 且不提供 hint 时，最终 band 与中心点必须和当前 shared extractor 逐点一致。
- reference hint 继续采用 Phase-A 验证的 envelope-union 语义，除非另开行为变更评审。
- row/vertical 测试同时断言：原图 u 区间映射到转置域 axis-0、输出 `(u,v)` 交换正确、metadata 不再误称 top/bottom。
- 合成条纹分别覆盖“完全在 auto band 外”和“峰仍在 band 内但距边缘过近”。
- 标定 quality 报告直接暴露 search-region 原因，不能只依赖最终点数或 RMSE。
- 不修改 Steger 二维 Hessian 亚像素公式、相机内参模型或激光三维几何模型。
