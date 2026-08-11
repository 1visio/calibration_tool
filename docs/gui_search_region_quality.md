# GUI Search Region Quality（Stage 2A）

## 1. 阶段边界

Stage 2A 把正式 Steger extraction 的 search-region health 加入现有 GUI 实时采集链路，不建立独立离线质量工作流，也不修改 Steger centerline、现有 `FrameQuality` PASS/FAIL 或操作员保存按钮门禁。

代码审计发现：Stage 2A 前 GUI 的激光质量只执行 `camera.quality.analyze_frame()` 中的 peak/coverage/FWHM，并未调用正式 Steger。Steger 原本只在采集后的激光模型拟合阶段运行。为满足实时 search-region health，本阶段在 `PreviewThread` 的 laser 帧处理中加入一次共享正式 extraction；health 指标直接消费这次 extraction，不执行第二次 Steger。

## 2. 现有 GUI 工作流与调用链

### 2.1 三联图计划与逐任务预览

```text
capture recipe / capture YAML
  -> camera.plan_builder.build_capture_plan_from_recipe()
     生成 chess / nolaser / laser 等同一 pose 的顺序 CaptureTask
  -> gui.pages.CapturePage.preview_selected_task()
  -> gui.pages.CameraPage.request_preview_task()
  -> gui.workers.PreviewThread.request_task_config()
  -> PreviewThread.run() 在同一个 camera session 中切换曝光和 quality_mode
```

三联图在 GUI 中是同一 pose 下顺序执行的多个 `CaptureTask`，不是先把三张图合成为一个新的实时图像对象。涉及的数据结构为：

- `CaptureRecipe` / `CaptureRecipeItem`：GUI 可编辑配方；
- `CapturePlan` / `CaptureTask`：capture YAML round-trip 后的正式任务；
- `CapturedFrame`：相机帧及时间戳；
- `FrameQuality`：既有正式质量指标和 `warnings/passed`；
- `SearchRegionHealth`：新增旁路 search-region 指标，不进入 `FrameQuality`。

### 2.2 单帧质量、线程与 signal/slot

```text
camera session.get_frame()
  [PreviewThread / QThread]
  -> camera.quality.analyze_frame()              既有正式 GUI 质量
  -> realtime_steger.extract_steger(..., diagnostic=True)  每个 laser 帧一次
  -> camera.steger_quality.analyze_search_region_health()
  -> PreviewThread.frame_ready(frame, payload)
       [queued signal 到 GUI 主线程]
  -> CameraPage._on_frame()
       更新相机页质量并发出 CameraPage.frame_ready
  -> CapturePage._on_camera_frame()
       更新批量采集页实时画面和质量显示
```

`PreviewThread` 是最合理的 diagnostics 接入层：图像仍在线程内，正式 extraction 只需运行一次，GUI 主线程只接收低体积字典，不接触 Hessian 数组。

保存链路为：

```text
操作员点击“稳定后保存当前帧”
  -> CapturePage.capture_current_task_frame()
  -> CameraPage.capture_after_settle(callback)
  -> CameraPage._on_frame() 在稳定帧到达后调用 callback(frame, quality)
  -> CapturePage._save_guided_frame(..., preview_quality)
```

保存复用同一帧已计算的 `preview_quality/search_region_health`，不会再次执行 Steger。`settling` 和帧处理耗时等瞬态字段不会写入保存后的正式 quality record。

## 3. 正式 extractor 与配置来源

GUI 通过 `calibration_src/realtime_steger.py` 加载 Stage 1 的共享正式入口：

```python
extract_steger(image, options, diagnostic=True)
```

参数读取共享 `calibration/config/realtime_steger.yaml`；仅根据项目/采集计划中的 `laser.orientation` 映射 `scan_axis`：

- `horizontal` -> `scan_axis=column`；
- `vertical` -> `scan_axis=row`。

本阶段没有传 additional search region，没有改动 auto-band、Gaussian/Hessian、candidate selection 或输出 centerline。

## 4. 新增指标

所有指标沿 Stage 1 定义的 LaserSearchRegion normal axis 计算：

- `scan_axis=column`：normal axis 为原图 `v`；
- `scan_axis=row`：normal axis 为原图 `u`。

指标如下：

| 字段 | 定义 |
| --- | --- |
| `search_region_start_px` | `StegerExtraction.metadata.final_search_region_start_px` |
| `search_region_end_px` | `StegerExtraction.metadata.final_search_region_end_px`，半开区间终点 |
| `search_region_size_px` | `end - start` |
| `boundary_clearance_min_px` | 所有正式有效中心到最近 search-region 边界距离的最小值 |
| `boundary_clearance_p05_px` | 上述距离的 P05 |
| `boundary_clearance_median_px` | 上述距离的中位数 |
| `kernel_support_px` | `ceil(4 * sigma)` |
| `boundary_inside_kernel_fraction` | 边界距离小于 `kernel_support_px` 的有效中心比例 |
| `outside_search_region_peak_fraction` | `StegerColumnDiagnostics.intensity_peak_outside_detected_band` 在全部 scanline 中的比例 |

health 统计仅包含一维坐标筛选、percentile 和 bool mean。`outside_search_region_peak_fraction` 直接复用 diagnostics 已有的 full-image peak 与 band peak 比较，不运行 expanded-band audit，不新增全图 Hessian。

## 5. WARNING 逻辑

Search Region Quality 与正式 laser quality 完全隔离：

- `boundary_clearance_p05_px < kernel_support_px`：
  `center_near_search_boundary`；
- `outside_search_region_peak_fraction > 1%`：
  `possible_signal_outside_search_region`；
- 没有上述原因：`GOOD`；否则：`WARNING`。

加载或辅助分析异常时显示 `search_region_health_unavailable`，但预览继续运行，正式 `FrameQuality.warnings/passed` 不变。

## 6. GUI 显示位置

相机页面原“质量”字段下新增独立的 `Search Region Quality` 行；批量采集页面“任务实时画面”中的原质量文本下新增相同信息。

```text
Search region: WARNING
Boundary P05: 3.5 px · Kernel support: 6 px · Outside-band risk: 28.2%
Reasons: center_near_search_boundary, possible_signal_outside_search_region
```

正常情况显示 `Search region: GOOD`。warning 原因保留 machine-readable code，现场操作员能够直接看到原因。

原质量文本仍只读取 `quality["warnings"]`；保存按钮仍只由任务配置匹配、settling 状态和采集进行状态控制，不读取 `search_region_health.status`。

## 7. 只读 replay/debug 入口

现有 GUI 不支持把历史文件注入相机 `PreviewThread`。新增只读 CLI：

```text
python -m calibration_tool search-region-replay IMAGE \
  --calibration-src ../calibration/src \
  --laser-orientation horizontal
```

该入口只读取图像并输出同一个 `RealtimeStegerQualityAnalyzer` 的 YAML 结果，不修改原图、capture plan 或数据集 manifest。

## 8. 真实案例验证

验证数据从 `feature/phase-a-baseline-distance-laser-angle` 的 Git 对象只读加载；每个 case 使用 5 个真实 TIFF 帧。

| case | 序列 | 结果 | Boundary P05 | Outside-band risk | 原因 |
| --- | --- | --- | --- | --- | --- |
| `B05_A10` | `multiheight/0001..0005.tif` | 5/5 WARNING | 3.504～3.547 px | 28.064%～28.227% | 5/5 `center_near_search_boundary`；5/5 `possible_signal_outside_search_region` |
| `B12p5_A10` | `reference/0001..0005.tif` | 5/5 GOOD | 32.621～33.166 px | 0% | 无 |

因此 B05_A10 能稳定显示 center-near-boundary warning；正常 B12p5 reference case 没有误报 boundary warning。

## 9. GUI 帧处理性能

测量对象为同一个真实 `B05_A10 multiheight/0001.tif`（2048×2448 Mono8），预热后各运行 30 次；before 是 Stage 2A 前既有 `analyze_frame()`，after 是相同正式质量加一次共享 Steger 和 health。

| 测量 | P50 | P95 |
| --- | ---: | ---: |
| GUI quality before | 301.5 ms | 367.3 ms |
| GUI quality after | 369.9 ms | 462.6 ms |
| health metrics 自身 | 0.47 ms | 0.74 ms |

新增一维 health 统计低于 1 ms。after 的主要增量来自代码审计中确认原 GUI 缺失、而 Stage 2A 为实时 health 必须加入的单次正式 Steger；没有第二次 Steger、expanded-band audit 或额外 Hessian。

## 10. 修改文件与测试

- `calibration_tool/camera/steger_quality.py`：共享 extractor 适配和 health metrics；
- `calibration_tool/gui/workers.py`：laser 帧单次 extraction、旁路 payload 和耗时；
- `calibration_tool/gui/pages.py`：两个 GUI 页面显示及保存时复用 payload；
- `calibration_tool/cli.py`：只读 replay/debug 入口；
- `tests/test_gui_search_region_quality.py`：normal-axis、metrics、warning 文本、一次调用和正式质量隔离；
- `tests/test_camera_cli.py`：replay 入口只读 smoke test。

验证结果：

```text
python -m pytest -q
107 passed, 21 subtests passed
```

## 11. 阶段结论

```text
behavior_changed = false
steger_math_changed = false
gui_search_region_health_visible = true
formal_laser_quality_changed = false
capture_acceptance_changed = false
```
