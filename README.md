# 线激光统一标定工具：阶段 0 + 阶段 1 + 阶段 2 + 阶段 3 + 阶段 4

本目录是独立于现有离线、在线 GUI 的标定核心。当前版本完成：

- 将两套工具当前使用的 `measure_tool.yaml` 及其引用标定文件固化为 golden baseline；
- 记录精确文件指纹、语义配置指纹、算法 profile 和代表性回归指标；
- 检查 config、manifest、激光中心提取算法和标定产物是否匹配；
- 对内参、激光平面、外参和地面补偿执行统一质量门禁；
- 通过统一 stage API 调用现有计算脚本，不复制标定数学实现；
- 支持一个 YAML workflow 顺序完成多个计算阶段；
- 只有质量通过的 profile 才能默认发布为不可拆分标定包。
- 通过统一 camera protocol 管理 MVS 真机与 synthetic 模拟相机；
- 支持相机枚举、参数设置、预览质量分析、曝光序列和 YAML 批量采集；
- 每个采集数据集带逐帧 manifest、CSV、文件哈希，并支持异常后的任务级续采。
- 提供独立的 PySide6 五步标定向导 MVP：项目、相机、采集、标定、结果与残差。
- 汇总 workflow、补偿前后指标、golden 漂移、运行 profile 和产物哈希，生成可发布或拒绝的验收报告。

当前版本仍是独立 CLI/核心库，没有修改或接入现有离线、在线 GUI。

完整的新项目采集、标定、验收和发布操作见
[`docs/线激光标定工具用户手册.md`](docs/线激光标定工具用户手册.md)。

## 快速使用

在本目录执行：

```powershell
python -m calibration_tool golden-build
python -m calibration_tool golden-check
python -m calibration_tool audit
python -m calibration_tool list-stages
```

审计输出默认写入 `reports/golden_audit.yaml`。当前 baseline 是“现状快照”，不是“全部质量通过”的样板；已知问题会作为 fail 保留，避免后续重构意外掩盖它们。

查看单个运行配置：

```powershell
python -m calibration_tool profile `
  "D:\Docs\linelaserscan\linelaser_tool\laser_measurement_tool\configs\measure_tool.yaml" `
  --expected-extractor shared_steger
```

运行单个阶段时，`--` 后参数传给原算法入口：

```powershell
python -m calibration_tool run intrinsics -- `
  --fit-dir <fit目录> `
  --test-dir <validation目录> `
  --output <输出目录> `
  --fit-pattern "*.tif" `
  --test-pattern "*.tif" `
  --pattern-cols 11 --pattern-rows 8 --square-size-mm 20
```

也可复制并填写 `configs/workflow.example.yaml`，随后执行：

```powershell
python -m calibration_tool workflow <workflow.yaml>
```

## Golden baseline 定义

`configs/golden_sources.yaml` 当前包含：

- 离线工具：`linelaser_tool/laser_measurement_tool/configs/measure_tool.yaml`
- 在线工具：`0704line-laser-3d-scanner/laser_measurement_tool/configs/measure_tool.yaml`
- 当前内参、激光平面、三联图运动、两种地面外参和补偿结果的代表性指标

`golden-build` 会生成：

- `golden/baseline.yaml`：路径、规范化 SHA-256、算法 profile、回归指标；
- `golden/snapshots/`：两套运行 config 和它们引用的标定文件快照。

哈希会先把 CRLF 规范化为 LF，因此 Windows/Git 换行转换不会制造虚假变化；同时仍记录 raw hash 供诊断。

## 阶段 2：相机与采集管理

`configs/camera.example.yaml` 默认使用 synthetic 相机，可在无硬件环境完成全链路测试；
`configs/camera.mvs.example.yaml` 是海康 MV-CS050-60GM 真机模板。两者的 2448 × 2048
采集几何来自当前离线、在线 golden config 引用的相机内参。

常用命令：

```powershell
# 验证模拟相机和质量分析
python -m calibration_tool camera-list --config configs/camera.example.yaml
python -m calibration_tool camera-preview --config configs/camera.example.yaml `
  --quality-mode laser --frames 20 --snapshot preview.png

# 枚举 MVS 真机
python -m calibration_tool camera-list --config configs/camera.mvs.example.yaml

# 执行批量计划；涉及换姿态时加 --interactive
python -m calibration_tool capture-plan configs/capture_plan.example.yaml --interactive

# 失败后使用原计划续采；计划内容变化时工具会拒绝续采
python -m calibration_tool capture-plan configs/capture_plan.example.yaml --resume --interactive

# 快速采集同一姿态的曝光序列
python -m calibration_tool capture-exposure-series `
  --config configs/camera.mvs.example.yaml `
  --output data/chessboard_pose_01 --dataset-id chessboard_pose_01 --pose-id pose_01 `
  --exposures-us 400 800 1200 2000 --frames-per-exposure 2
```

完整数据集包含 `dataset_manifest.yaml` 和 `frames.csv`。manifest 会保存设备身份、请求值与
相机回读值、帧号连续性、相机/主机时间戳、SHA-256，以及过曝、欠曝、动态范围、清晰度、
棋盘检测或激光覆盖率等质量信息。质量告警当前用于提示和记录，不会擅自丢弃原始图像。

## 阶段 3：PySide6 标定向导 MVP

启动真实 MVS 相机模式：

```powershell
cd D:\Docs\linelaserscan\calibration_tool
python -m calibration_tool gui --project configs\wizard_project.example.yaml
```

没有相机时可使用 synthetic 后端检查界面：

```powershell
python -m calibration_tool gui --simulate
```

向导包含五个页面：

1. **项目**：保存工作目录、相机配置、workflow 和棋盘参数；
2. **相机与曝光**：后台枚举设备，设置曝光、增益、PixelFormat、ROI，实时预览质量；
3. **批量采集**：输入一个或多个曝光值，按姿态批量采集，支持 `.inprogress` 续采；
4. **一键标定**：后台执行阶段 1 workflow，并逐阶段显示运行状态；
5. **结果与残差**：展开 metrics/quality gates，发现残差与诊断文件，表格查看并绘制 CSV 数值曲线。

MVP 暂时不修改现有离线、在线 GUI。后续接入时可直接复用当前 camera protocol、
`run_capture_plan`、`run_workflow` 和结果查看组件。

## 真机验证记录

阶段 2/3 已在 `MV-CS050-60GM`（SN `DA7711077`，GigE 地址 `169.254.124.26`）上验证：

- Mono12、2448 × 2048、曝光 1200 μs 连续取流 10 帧，全部质量通过；
- PySide6 预览线程运行约 3 秒取得 12 帧，无线程或 SDK 错误；
- 1000/1200 μs 两档真实曝光批量采集成功，请求值与相机回读值一致；
- 真机采集 manifest 位于 `reports/hardware_capture_smoke_20260803/dataset_manifest.yaml`。

### Mono8、Mono12 与显示亮度

完整逻辑和采集建议见 [`docs/camera_exposure_and_quality.md`](docs/camera_exposure_and_quality.md)。

- 曝光时间控制的是传感器积分时间；Mono8/Mono12 控制输出量化位深，不应把二者的画面显示亮度直接比较；
- 固定量程显示时，Mono8 使用 0–255，Mono12 使用 0–4095 映射到屏幕 0–255；
- “自动拉伸预览”会按当前帧分位数增强暗部，只改变屏幕显示，不改变原始 DN、曝光或保存图像；
- 若正式测量使用 Mono8，建议内参、激光平面、地面外参和验证也使用相同的 Mono8、ROI、分辨率和运行时提取链；
- Mono12 可用于曝光摸底或诊断，但不要把 Mono12 标定图与 Mono8 运行图混成一套正式数据。

曝光和增益可在取流期间点击“应用曝光/增益”在线写入，界面随后显示相机回读值。
PixelFormat、Offset 和 Width/Height 会改变数据格式或成像几何，取流期间会被禁用，需停止后修改并重新开始。

### 三种质量模式

- `通用曝光`：检查过曝比例、暗像素比例和全局动态范围，同时报告 Laplacian 清晰度；
- `激光线`：允许大面积暗背景，检查激光对比度、逐列横向覆盖率、过曝和动态范围；
- `棋盘格`：在通用曝光检查之外，要求找到配置数量的完整内角点。内参棋盘图应关闭激光；失败时会区分
  内角点配置不匹配、激光遮挡以及普通的未检测到。

## 阶段 4：报告、补偿和验收闭环

完整设计与验收规则见 [`docs/acceptance_closeout.md`](docs/acceptance_closeout.md)。

生成当前 golden 状态的验收报告：

```powershell
cd D:\Docs\linelaserscan\calibration_tool
python -m calibration_tool acceptance-report configs\acceptance_plan.example.yaml --overwrite
```

输出包括：

- `acceptance_report.yaml`：机器可读结论、全部门禁和补偿前后指标；
- `acceptance_report.html`：可直接交付的自包含报告；
- `acceptance_metrics.csv`：可筛选的门禁表；
- 输入配置、标定文件、补偿表和诊断产物的路径、大小及 SHA-256。

向导第 5 页已改为“报告、补偿与验收”，可选择验收计划、后台生成报告并打开 HTML。
若 `release.enabled: true`，只有 `decision: accepted` 才会创建标定发布包；被拒绝时发布状态为
`blocked`，不会复制或覆盖运行标定文件。

## 统一 stage

| stage | 旧算法入口 | 用途 |
|---|---|---|
| `intrinsics` | `calibrate_chessboard_opencv_reusable.py` | 内参拟合和独立验证 |
| `laser_plane_shared_steger` | `calibrate_laser_plane_core_v2.py` | shared Steger 激光平面 |
| `ground_extrinsics_board_only` | `calibrate_ground_extrinsics_board_only.py` | 棋盘基准面外参 |
| `ground_extrinsics_shared_steger` | `calibrate_ground_extrinsics_steger_v2.py` | 混合式地面外参 |
| `ground_bias` | `generate_ground_bias_compensation.py` | 偏差表和 holdout 验证 |
| `reconstruct_shared_steger` | `reconstruct_ground_pointcloud_cloudcompare_v4.py` | 端到端三维验证 |

三维重建 stage 强制 `--steger-extractor shared`，防止标定时和运行时混用中心提取器。

## 发布规则

`bundle-build` 默认拒绝以下情况：

- 运行 config 的提取器和标定链不一致；
- 标定文件缺失或 manifest 哈希错误；
- 提供的质量报告为 `fail`；
- 目标目录非空。

`--allow-failed` 仅用于诊断快照，生成的 manifest 会记录 `release_override: true`，不能当作正式生产标定包。
