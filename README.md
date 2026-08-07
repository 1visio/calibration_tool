# 线激光统一标定工具

线激光统一标定工具是当前实时测量系统的独立标定、采集、质量审计和发布组件。它负责把相机内参、激光表面模型、地面外参和可选地面补偿组织成可追溯的 calibration bundle，供离线/在线测量工具加载。

当前版本完成阶段 0～4：

- 固化离线、在线运行配置和标定文件的 golden baseline；
- 检查配置、manifest、提取算法、标定产物和文件哈希是否匹配；
- 统一相机协议，支持 MVS 真机和 synthetic 模拟相机；
- 支持相机枚举、预览质量分析、曝光序列和 YAML 批量采集；
- 为每个数据集写入逐帧 manifest、CSV、SHA-256 和质量指标，并支持中断续采；
- 提供 PySide6 五步标定向导：项目、相机、采集、标定、报告/补偿/验收；
- 生成 workflow、补偿前后指标、golden 漂移、运行 profile 和产物哈希报告；
- 只有通过质量门禁和正式验收的配置，才能发布为不可拆分的标定包。

标定工具与实时 GUI 目前是两个独立入口。标定工具不会自动修改实时窗口；完成标定后，应把通过验收的 calibration bundle 和运行配置明确交给实时工具。

完整的现场操作步骤见：[线激光标定工具用户手册](docs/线激光标定工具用户手册.md)。

## 1. 快速开始

### 1.1 创建环境与安装依赖

本目录当前没有单独的 `requirements.txt`，建议在标定工具目录创建虚拟环境并安装核心依赖：

```powershell
cd D:\Docs\linelaserscan\calibration_tool
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install `
  numpy scipy opencv-python PyYAML PySide6 matplotlib pandas
```

如果使用真实 MVS 相机，还需要安装海康 MVS SDK，并确保 SDK 的 Python/运行库位数与当前解释器一致。没有相机时可以使用 synthetic 后端完成界面和采集链路演练。

### 1.2 检查命令行入口

```powershell
python -m calibration_tool --help
python -m calibration_tool list-stages
```

建议先在当前快照上执行：

```powershell
python -m calibration_tool golden-build
python -m calibration_tool golden-check
python -m calibration_tool audit
```

`golden-build` 记录当前状态，`golden-check` 检查后续是否发生漂移，`audit` 执行质量门禁。当前仓库中的历史 baseline 是“现状快照”，其中可能保留已知失败项；看到 `fail` 不代表新项目不能通过验收，而是工具没有掩盖旧问题。

当前快照还保留了旧版 `linelaser_tool` 的路径和旧标定文件哈希。若 `golden-check` 报告 `camera_intrinsics.yaml` 不存在或哈希变化，应先确认新标定包、运行配置和项目归属，再有意执行 `golden-build` 更新 baseline；不要为了让测试变绿而直接覆盖历史证据。

### 1.3 启动五步向导

真实 MVS 相机：

```powershell
python -m calibration_tool gui --project configs\wizard_project.example.yaml
```

无相机演练：

```powershell
python -m calibration_tool gui --simulate
```

向导五页分别为：

1. 项目；
2. 相机与曝光；
3. 批量采集；
4. 一键标定 workflow；
5. 报告、补偿与验收。

## 2. 目录与产物

```text
calibration_tool/
├─ calibration_tool/          Python 包与 CLI/GUI 实现
├─ configs/                   相机、采集、workflow、质量和验收模板
├─ docs/                     用户手册、验收规则和曝光说明
│  └─ images/user_manual/     向导页面截图
├─ data/                      示例/现场采集数据集
├─ golden/                    golden baseline 与快照
├─ reports/                   审计、workflow、验收报告
├─ releases/                 通过验收后发布的标定包
└─ runs/                     中间运行目录（默认不提交）
```

`.gitignore` 默认忽略 `runs/`、`.venv/`、Python 缓存和编译文件；正式发布时应根据数据保密要求决定哪些原始图像可以提交。标定结果、manifest、报告和配置必须保持同一版本可追溯。

### 2.1 典型项目目录

建议每套设备创建独立项目，不要把多个相机或多个机械姿态混在同一目录：

```text
projects/<project-id>/
├─ config/
│  ├─ camera.yaml
│  ├─ workflow.yaml
│  ├─ acceptance_plan.yaml
│  └─ runtime.yaml
├─ data/
│  ├─ intrinsics/fit/
│  ├─ intrinsics/validation/
│  ├─ laser_plane/fit/
│  ├─ laser_plane/validation/
│  ├─ ground/fit/
│  ├─ ground/validation/
│  ├─ ground_bias/raw/
│  └─ obstacle_validation/
├─ runs/
├─ reports/
└─ releases/
```

## 3. CLI 命令速查

| 命令 | 用途 |
|---|---|
| `golden-build` | 从 registry 生成 baseline 和快照 |
| `golden-check` | 检查当前配置/标定是否偏离 baseline |
| `audit` | 对 baseline 执行质量门禁 |
| `profile <config>` | 解析运行配置并检查提取器/标定引用 |
| `list-stages` | 列出统一计算阶段 |
| `run <stage> -- ...` | 执行一个阶段，`--` 后参数传给原算法入口 |
| `workflow <plan.yaml>` | 按 YAML 顺序执行多个阶段 |
| `bundle-build` | 发布不可拆分的 calibration bundle |
| `camera-list` | 枚举 MVS 或 synthetic 相机 |
| `camera-preview` | 取流并输出曝光、清晰度和激光覆盖指标 |
| `capture-plan` | 按 YAML 计划批量采集，支持交互和续采 |
| `capture-exposure-series` | 同一姿态按多个曝光值快速采集 |
| `gui` | 启动 PySide6 标定向导 |
| `acceptance-report` | 生成补偿前后对比和正式验收报告 |

示例：

```powershell
# 解析实时工具配置，确认运行时提取器
python -m calibration_tool profile `
  ..\0704line-laser-3d-scanner\laser_measurement_tool\configs\measure_tool.yaml `
  --expected-extractor steger

# 模拟相机预览 20 帧并保存一张快照
python -m calibration_tool camera-preview `
  --config configs\camera.example.yaml `
  --quality-mode laser --frames 20 --snapshot preview.png

# 枚举真实 MVS 相机
python -m calibration_tool camera-list `
  --config configs\camera.mvs.example.yaml

# 执行批量计划；姿态切换前要求人工确认
python -m calibration_tool capture-plan configs\capture_plan.example.yaml `
  --interactive --preview-window

# 从中断的 .inprogress 数据集续采
python -m calibration_tool capture-plan configs\capture_plan.example.yaml `
  --resume --interactive

# 一次采集多个曝光档位
python -m calibration_tool capture-exposure-series `
  --config configs\camera.mvs.example.yaml `
  --output data\exposure_series_01 `
  --dataset-id exposure_series_01 --pose-id pose_01 `
  --exposures-us 400 600 800 1200 --frames-per-exposure 2
```

## 4. 相机与采集配置

### 4.1 相机配置

复制以下模板后再填写现场值：

- `configs/camera.example.yaml`：默认 synthetic；
- `configs/camera.mvs.example.yaml`：真实 MV-CS050-60GM 模板。

关键字段：

```yaml
backend: mvs
serial_number: ""
calibration_src: ../../calibration/src

camera:
  exposure_us: 1200.0
  gain_db: 0.0
  pixel_format: Mono8
  offset_x: 0
  offset_y: 0
  width: 2448
  height: 2048
  timeout_ms: 2000

board:
  pattern_cols: 11
  pattern_rows: 8
```

当前模板中的 `1200 μs` 是标定采集起点，不是实时工具的 GUI 默认值。实时窗口当前默认曝光为 `600 μs`；正式标定、验证和运行应根据现场光学条件选择一致的像素格式、ROI 和曝光策略，并把实际值写入数据集 manifest。

`quality` 段控制过曝、欠曝、动态范围、激光横向覆盖和棋盘检测门限。质量告警会被记录到 manifest，不会擅自删除原始图像。

### 4.2 批量采集计划

`configs/capture_plan.example.yaml` 定义数据集目录、基础相机参数、质量阈值、棋盘参数和 `tasks`。每个任务可以覆盖曝光、增益、ROI、文件格式、质量模式、稳定帧数和保存帧数。

激光平面三联图的角色必须保持：

```text
chess 001.tif     高曝光、激光关闭；棋盘位姿和运动参考
nolaser 001.tif   低曝光、激光关闭；背景扣除
laser 001.tif     相同低曝光、激光开启；激光中心提取
```

三张图之间不得移动棋盘、相机或支架。`nolaser` 和 `laser` 应使用相同短曝光；短曝光图看不清棋盘并不等于发生了运动。

### 4.3 数据集追溯文件

每个成功数据集应至少包含：

- 原始图像；
- `dataset_manifest.yaml`；
- `frames.csv`；
- 设备身份、请求参数和相机实际回读参数；
- 相机帧号、相机/主机时间戳和 SHA-256；
- 每帧曝光、清晰度、棋盘检测或激光覆盖质量。

如果采集被取消，工具会保留同级 `.inprogress` 状态。只有在计划 hash 不变时才能续采；修改计划后必须新建数据集，不能覆盖旧数据。

## 5. PySide6 向导操作

### 5.1 第 1 页：项目

![项目页面](docs/images/user_manual/01_project.png)

填写项目 ID、工作目录、相机配置、workflow、验收计划、棋盘内角点列/行数和方格尺寸。点击“保存项目…”后再点击“应用到向导”。棋盘填写的是内角点数量，不是黑白方格数量。

### 5.2 第 2 页：相机与曝光

![相机与曝光页面](docs/images/user_manual/02_camera.png)

操作顺序：

1. 加载相机 YAML；
2. 枚举设备并确认型号、序列号；
3. 设置 Mono8/Mono12、曝光、增益和 ROI；
4. 选择“通用曝光/棋盘格/激光线”质量模式；
5. 开始取流并观察质量指标；
6. 取流期间可以在线应用曝光/增益，PixelFormat、Offset、宽高需停止后修改。

建议增益保持 `0 dB`，优先用曝光、遮光和光学调节改善质量。正式运行使用 Mono8 时，内参、激光模型、地面外参和验证数据也尽量使用 Mono8。

### 5.3 第 3 页：批量采集

![批量采集页面](docs/images/user_manual/03_capture.png)

填写输出数据集、数据集 ID、曝光序列/姿态任务、保存格式和质量模式，点击“生成并检查计划”确认任务预览，再点击“开始采集”。正式三联图应按 `chess → nolaser → laser` 完成一组后再移动棋盘。

若中断，保持计划不变，勾选“续采对应的 .inprogress 数据集”。页面会记录计划、数据集根目录、fit/validation 目录、manifest 和 `frames.csv`。

### 5.4 第 4 页：一键标定

![一键标定页面](docs/images/user_manual/04_calibration.png)

`configs/workflow.example.yaml` 是结构示例。复制后必须按项目目录填写路径，并确认实际要运行的 stage 为 `enabled: true`。建议先“检查 Workflow 阶段”，确认输入目录、输出目录、拟合/验证数据和算法 profile，再点击“一键运行完整标定”。

### 5.5 第 5 页：报告、补偿与验收

结果页可查看 stage、fit/validation、诊断图、残差 CSV 和补偿前后指标。验收报告的 `decision` 与 workflow 是否完成不是同一个概念：

| 结果 | 含义 |
|---|---|
| `Workflow: completed` | 已启用阶段完成且阶段级门禁通过 |
| `decision: accepted` | workflow、独立验证、补偿、运行 profile、golden 和产物全部通过正式验收 |
| `decision: rejected` | 报告生成成功，但至少有一个门禁未关闭，禁止发布 |

完整的现场按钮说明、曝光摸底和采集数量建议见[用户手册第 4～9 节](docs/线激光标定工具用户手册.md)。

## 6. Workflow 与统一计算阶段

执行一个工作流：

```powershell
python -m calibration_tool workflow projects\<project-id>\config\workflow.yaml
```

当前统一 stage：

| stage | 用途 |
|---|---|
| `intrinsics` | 相机内参拟合和独立验证 |
| `laser_surface_models` | 同时拟合 global plane、quadratic graph、circular cone，并选择正式模型 |
| `laser_plane_shared_steger` | 历史全局平面/shared Steger 兼容 stage |
| `ground_extrinsics_board_only` | 棋盘基准面地面外参 |
| `ground_extrinsics_shared_steger` | 混合式地面外参兼容 stage |
| `ground_bias` | 地面逐列偏差表和 holdout 验证 |
| `reconstruct_shared_steger` | 端到端地面点云/障碍物验证 |

三维重建 stage 强制使用 `shared`/统一实时 Steger 提取器，避免标定和运行链路混用不同中心算法。新项目默认推荐 `laser_surface_models` 的 `circular_cone`；旧的 `global_plane` 文件仍可加载，但需要在报告中明确模型类型。

## 7. 报告、补偿与发布

### 7.1 生成验收报告

```powershell
python -m calibration_tool acceptance-report `
  configs\acceptance_plan.example.yaml --overwrite
```

常见输出：

```text
reports/<report-id>/
├─ acceptance_report.yaml
├─ acceptance_report.html
└─ acceptance_metrics.csv
```

报告会记录 workflow、质量报告、runtime profile、golden baseline、补偿前后指标、输入文件路径/大小/SHA-256 和诊断产物。

正式补偿必须满足：

```text
evaluation_frame_count > 0
build_frame_count + evaluation_frame_count == loaded_frame_count
```

把 `validation_count` 设为 0 会被正式验收拒绝，即使补偿后误差看起来很小。

### 7.2 发布 calibration bundle

只有验收 `decision: accepted` 时才允许发布。示例：

```powershell
python -m calibration_tool bundle-build `
  --config ..\0704line-laser-3d-scanner\laser_measurement_tool\configs\measure_tool.yaml `
  --output releases\mv-cs050-60gm-accepted-v1 `
  --package-id mv-cs050-60gm-accepted-v1 `
  --expected-extractor steger `
  --quality-report reports\current_acceptance\acceptance_report.yaml
```

默认发布门禁包括：

- 运行配置的提取器与标定链一致；
- 标定文件存在且 manifest 哈希正确；
- 质量报告不是 `fail`；
- 目标目录为空，避免覆盖已有发布包。

`--allow-failed` 只允许生成诊断快照，manifest 会记录 `release_override: true`，不能作为生产标定包。

## 8. 与实时工具的集成约定

实时工具的 `laser_measurement_tool/configs/measure_tool.yaml` 至少需要指向同一套：

```yaml
calibration:
  manifest: calibration/manifest.yaml
  intrinsics: calibration/calibration_result.yaml
  laser_model: calibration/circular_cone.yaml
  extrinsics: calibration/camera_ground_extrinsics.yaml
  ground_u_compensation: null
```

集成时必须同时核对：

1. 相机型号、全幅尺寸和 ROI；
2. Mono8/Mono12；
3. `OffsetX/OffsetY`；
4. 激光中心提取器和共享 profile；
5. 标定模型单位、坐标方向和工作深度；
6. 地面补偿是否在 manifest 中启用；
7. runtime config、manifest 和结果 JSON 中的 package ID/SHA-256。

当前实时配置的 `extraction.profile` 指向同级 `D:\Docs\linelaserscan\calibration\config\realtime_steger.yaml`。如果把项目迁移为单一 GitHub 仓库，必须把该 profile 一并纳入版本控制，或者复制到仓库内并更新相对路径；不能让 GitHub 上的 `measure_tool.yaml` 依赖开发机的绝对路径。

实时工具当前默认曝光是 `600 μs`，而本目录的相机模板默认值是 `1200 μs`。二者是不同入口的默认值，不能把模板中的数值误认为实时 GUI 的实际回读值；正式发布时应在文档、采集 manifest 和运行配置中记录最终采用的曝光。

## 9. 质量与验收建议

一套正式标定建议至少包含独立的：

| 数据组 | 建议数量 | 用途 |
|---|---:|---|
| 内参拟合集 | 15～25 个姿态 | 拟合内参和畸变 |
| 内参验证集 | ≥ 5 个姿态 | 独立重投影验证 |
| 激光三联图拟合集 | 12～18 组 | 激光表面拟合 |
| 激光三联图验证集 | 4～6 组 | 独立激光误差验证 |
| 地面外参拟合集 | ≥ 6 张 | 相机到地面坐标转换 |
| 地面外参验证集 | ≥ 2 张 | 独立平面验证 |
| 平地补偿数据 | 建表 ≥ 8，建议 20；验证 ≥ 2，建议 5 | 逐列补偿与 holdout |
| 障碍物验收数据 | ≥ 3 张 | 端到端高度/长度验证 |

数量不能替代姿态多样性。模糊、反光、裁边或相互重复的图像不应为了凑数量而保留。

## 10. 测试与故障排查

### 10.1 运行测试

```powershell
cd D:\Docs\linelaserscan\calibration_tool
python -m unittest discover -s tests -v
```

GUI 测试需要 PySide6 和可用的 Qt/offscreen 环境；真机测试还需要 SDK 和相机设备。

### 10.2 常见问题

| 现象 | 优先检查 |
|---|---|
| `camera-list` 没有设备 | MVS SDK、相机占用、网络、序列号和 Python 位数 |
| synthetic 正常、MVS 失败 | SDK 加载、网卡/USB、设备权限和 `backend: mvs` |
| 棋盘检测失败 | 内角点数量、激光是否关闭、曝光/失焦/反光、棋盘是否完整入镜 |
| 激光覆盖率低 | 激光方向、曝光、遮光、线宽和 `quality.min_laser_coverage` |
| workflow 找不到输入 | 所有相对路径以 workflow YAML 所在目录解析；检查 fit/test/validation 目录 |
| 续采被拒绝 | 原计划 hash 已改变；应恢复原计划或新建数据集 |
| 验收 rejected | 打开 HTML/YAML 报告，逐项处理失败门禁；不要用 `--allow-failed` 伪装生产发布 |
| 实时工具无法加载标定 | 检查 manifest 哈希、runtime config 和共享 Steger profile 是否来自同一版本 |

## 11. GitHub 第一阶段提交清单

- [ ] 清理 README、示例 YAML 和报告中的个人绝对路径，改为相对路径或占位符；
- [ ] 确认 `calibration/config/realtime_steger.yaml` 的归属和提交位置；
- [ ] 提交 `configs/`、`docs/`、向导截图、测试和必要的样例标定文件；
- [ ] 不提交 `.venv/`、`runs/`、临时日志、相机凭证和未经授权的原始数据；
- [ ] 每套标定数据保留 `dataset_manifest.yaml`、`frames.csv` 和 SHA-256；
- [ ] 运行 `golden-check`、`audit`、完整单元测试；
- [ ] 用 `--simulate` 启动向导并完成一条采集/报告 smoke test；
- [ ] 真实设备完成 `camera-list`、预览和最小采集验证；
- [ ] 生成并审阅 `acceptance_report.html`，只有 `accepted` 才发布 bundle；
- [ ] 在实时工具中加载发布包，确认单帧重建、硬件 ROI 偏移和结果导出正常。

## 12. 相关文档

- [线激光标定工具用户手册](docs/线激光标定工具用户手册.md)
- [相机曝光与质量说明](docs/camera_exposure_and_quality.md)
- [验收闭环说明](docs/acceptance_closeout.md)
- [实时工具用户手册](../0704line-laser-3d-scanner/laser_measurement_tool/docs/ONLINE_USER_MANUAL.md)
- [实时工具配置说明](../0704line-laser-3d-scanner/laser_measurement_tool/docs/USAGE_CONFIG.md)
