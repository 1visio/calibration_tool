# 线激光统一标定工具

线激光统一标定工具是当前实时测量系统的独立标定、采集、质量审计和发布组件。它负责把相机内参、激光表面模型、地面外参和可选地面补偿组织成可追溯的 calibration bundle，供离线/在线测量工具加载。

当前版本完成阶段 0～4：

- 固化离线、在线运行配置和标定文件的 golden baseline；
- 检查配置、manifest、提取算法、标定产物和文件哈希是否匹配；
- 统一相机协议，支持海康 MVS、大恒 Galaxy 和 synthetic 模拟相机；
- 通过单一 `camera channel` 注册表切换相机配置、激光方向和推荐 workflow；
- 支持相机枚举、预览质量分析、曝光序列和 YAML 批量采集；
- 为每个数据集写入逐帧 manifest、CSV、SHA-256 和质量指标，并支持中断续采；
- 提供 PySide6 五步标定向导：项目、相机、采集、标定、报告/补偿/验收；
- 生成 workflow、补偿前后指标、golden 漂移、运行 profile 和产物哈希报告；
- 只有通过质量门禁和正式验收的配置，才能发布为不可拆分的标定包。

标定工具与实时 GUI 目前是两个独立入口。标定工具不会自动修改实时窗口；完成标定后，应把通过验收的 calibration bundle 和运行配置明确交给实时工具。

完整的现场操作步骤见：[线激光标定工具用户手册](docs/线激光标定工具用户手册.md)。

## 当前支持范围与健康状态（2026-08-13）

当前仓库可以正常启动 CLI、运行 synthetic 相机链路并执行测试；但现有历史标定快照尚未达到“全部验收通过”，不能把启动成功等同于可以发布生产标定包。

| 检查项 | 当前结果 | 说明 |
|---|---|---|
| `python -m calibration_tool --help`、`list-stages` | 通过 | CLI 入口和 7 个统一 stage 可加载 |
| `camera-list --config configs/camera.example.yaml` | 通过 | synthetic 设备 `SIMULATED-MV-CS050-60GM` 可枚举 |
| `camera-preview`（synthetic） | 可运行但有告警 | 示例帧返回 `dynamic_range_low`，这是质量提示，不是采集异常 |
| `python -m pytest -q` | 见第 11 节的当前验证结果 | 软件测试与历史 golden 漂移是两类结果，不能互相替代 |
| `golden-check` | `matches: false` | 发现 9 项配置/标定哈希变化或文件缺失 |
| `audit` | `overall: fail` | 当前历史数据仍缺少独立内参测试、补偿 holdout 等正式门禁 |

因此，当前可确认的是“软件链路可启动、synthetic 演练可用、绝大多数单元测试通过”；海康 MVS 与大恒 Galaxy 后端均需在安装对应 SDK、接入设备后完成真机验证。

## 1. 快速开始

### 1.1 创建环境与安装依赖

本目录当前没有单独的 `requirements.txt`，推荐直接按 `pyproject.toml` 安装本包。下面的 `scipy`、`matplotlib`、`pandas` 仅供 `scripts/` 下的模型比较脚本使用，不是 CLI/GUI 的最小运行依赖：

```powershell
cd D:\Docs\linelaserscan\calibration_tool
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
# 运行测试时安装：
.\.venv\Scripts\python.exe -m pip install pytest
# 需要运行 scripts/ 下的模型比较脚本时再安装：
.\.venv\Scripts\python.exe -m pip install scipy matplotlib pandas
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

### 1.3 从统一相机通道启动五步向导

海康 MVS：

```powershell
python -m calibration_tool gui --channel hikrobot
```

大恒 Galaxy：

```powershell
python -m calibration_tool gui --channel daheng
```

无相机演练：

```powershell
python -m calibration_tool gui --simulate
```

三个入口都读取同一个 `configs/camera_channels.example.yaml`。如果使用
`--project` 打开已保存项目，项目中的 `camera_channel` 优先；不需要再到第 2 页
重复选择一份相机 YAML。

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
├─ configs/                   统一相机通道、采集、workflow、质量和验收模板
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
│  ├─ camera_channels.yaml
│  ├─ wizard_project.yaml
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
| `workflow <plan.yaml>` / `workflow --project <project.yaml>` | 执行显式 workflow，或直接使用项目所选通道的 workflow |
| `bundle-build` | 发布不可拆分的 calibration bundle |
| `camera-list` | 按统一通道枚举海康、大恒或 synthetic 相机 |
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

# 从同一入口枚举海康或大恒相机
python -m calibration_tool camera-list --channel hikrobot
python -m calibration_tool camera-list --channel daheng

# 执行批量计划；姿态切换前要求人工确认
python -m calibration_tool capture-plan configs\capture_plan.example.yaml `
  --config configs\camera_channels.example.yaml --channel hikrobot `
  --interactive --preview-window

# 从中断的 .inprogress 数据集续采
python -m calibration_tool capture-plan configs\capture_plan.example.yaml `
  --resume --interactive

# 一次采集多个曝光档位
python -m calibration_tool capture-exposure-series `
  --config configs\camera_channels.example.yaml --channel hikrobot `
  --output data\exposure_series_01 `
  --dataset-id exposure_series_01 --pose-id pose_01 `
  --exposures-us 400 600 800 1200 --frames-per-exposure 2
```

## 4. 相机与采集配置

### 4.1 统一相机通道

`configs/camera_channels.example.yaml` 是海康、大恒和模拟相机的单一入口：

```yaml
default_channel: hikrobot
channels:
  hikrobot:
    label: 海康 MVS（MV-CS050-60GM）
    config: camera.mvs.example.yaml
    workflow_plan: workflow.example.yaml
  daheng:
    label: 大恒 Galaxy（USB3 Vision）
    config: camera.daheng.example.yaml
    workflow_plan: workflow_daheng.yaml
  synthetic:
    label: 模拟相机（无硬件演练）
    config: camera.example.yaml
    workflow_plan: workflow.example.yaml
```

通道只负责选择，不把设备参数复制进多个位置。各相机 YAML 仍分别保存序列号、
ROI、像素格式、曝光、质量阈值和 SDK 选项；项目 YAML 保存 `camera_channel`；生成
采集计划时再把实际 backend、序列号和相机参数固化到计划与 manifest。这样切换
方便，同时不会让旧数据被新通道静默复用。旧版直接传入单相机 YAML 的命令仍兼容。

| channel | backend | 激光方向默认值 | 模板 / 推荐 workflow |
|---|---|---|---|
| `hikrobot` | `mvs` | `horizontal` | `camera.mvs.example.yaml` / `workflow.example.yaml` |
| `daheng` | `daheng` | `vertical` | `camera.daheng.example.yaml` / `workflow_daheng.yaml` |
| `synthetic` | `synthetic` | `horizontal` | `camera.example.yaml` / `workflow.example.yaml` |

`quality` 段控制过曝、欠曝、动态范围、激光覆盖和棋盘检测门限。质量告警会写入
manifest，不会擅自删除原始图像。

### 4.2 海康 MVS 系统：采集与分析

1. 安装海康 MVS SDK，确认 Python、SDK 和运行库位数一致；确认
   `calibration/src/capture_chessboard_exposure_series.py` 可用。
2. 复制并编辑 `camera.mvs.example.yaml`：多相机现场必须填写
   `serial_number`；核对 GigE 网卡、`2448×2048`、ROI、Mono8/Mono12 和超时。
3. 做连接与曝光预检：

   ```powershell
   python -m calibration_tool camera-list --channel hikrobot
   python -m calibration_tool camera-preview --channel hikrobot `
     --quality-mode chessboard --frames 20
   ```

4. 启动 `python -m calibration_tool gui --channel hikrobot`。第 1 页填写独立的
   项目 ID/工作目录，选择“海康 MVS”，点击“应用到向导”；第 2 页确认自动枚举的
   型号和序列号，再调曝光、增益和 ROI。
5. 第 3 页生成计划并按 `chess → laser → nolaser` 采集。默认方向是
   `horizontal`；模板 `1200 μs`、向导配方中的曝光都只是起点，必须按现场回读
   调整。常见三联图只切换曝光/增益时会复用同一个 MVS session，不再关闭重开。
6. 把 `workflow.example.yaml` 复制到项目目录，在第 1 页指向该副本并应用；第 4 页
   检查输入/输出路径，把最近采集结果更新到 workflow 后执行。CLI 可在保存项目后
   直接运行：

   ```powershell
   python -m calibration_tool workflow --project projects\hikrobot-01\wizard_project.yaml
   ```

### 4.3 大恒 Galaxy 系统：采集与分析

1. 安装完整 Galaxy SDK 和随附的 `gxipy`。默认搜索
   `C:\Program Files\Daheng Imaging\GalaxySDK`；非默认位置设置
   `DAHENG_GALAXY_ROOT`，或用 `DAHENG_GALAXY_PYTHON_PATH` 指向包含 `gxipy`
   的目录。当前后端枚举 USB3 Vision 设备。
2. 复制并编辑 `camera.daheng.example.yaml`：多相机现场填写 `serial_number`；核对
   `4096×3000`、ROI、Mono8/Mono12、USB3 链路和 `discovery_timeout_ms`。
3. 做连接与曝光预检：

   ```powershell
   python -m calibration_tool camera-list --channel daheng
   python -m calibration_tool camera-preview --channel daheng `
     --quality-mode chessboard --frames 20
   ```

4. 新项目启动 `python -m calibration_tool gui --channel daheng`；继续当前仓库的
   大恒项目可启动 `python -m calibration_tool gui --project projects\daheng\project.yaml`。
   第 1 页选择“大恒 Galaxy”并应用后，工具会同时切换到 `daheng` backend、
   `vertical` 激光方向和推荐的 `workflow_daheng.yaml`，第 2 页自动枚举设备。
5. 第 3 页完成三联图采集。模板 `120000 μs` 是当前现场样例起点，不应直接套用
   到另一套光学系统；特别要重新确认短曝光 `nolaser/laser` 是否仍有足够覆盖率。
6. 新项目先复制 `workflow_daheng.yaml` 到项目目录，修改 data/output 与模型配置
   路径，并在第 1 页把 workflow 指向该副本后重新应用、保存；第 4 页再更新最近
   采集输入并运行。仓库现有 `projects/daheng/project.yaml` 已显式链接当前大恒
   workflow，因此也可直接使用：

   ```powershell
   python -m calibration_tool workflow --project projects\daheng\project.yaml
   ```

两套系统必须使用不同项目目录和数据集。不能因为棋盘规格相同就混用内参、激光
模型或外参；发布前要核对 manifest 中的设备序列号、分辨率、ROI、像素格式和通道。

### 4.4 批量采集计划

`configs/capture_plan.example.yaml` 定义数据集目录、基础相机参数、质量阈值、棋盘参数和 `tasks`。每个任务可以覆盖曝光、增益、ROI、文件格式、质量模式、稳定帧数和保存帧数。

激光平面三联图的角色必须保持：

```text
chess 001.tif     高曝光、激光关闭；棋盘位姿和运动参考
laser 001.tif     低曝光、激光开启；激光中心提取
nolaser 001.tif   相同低曝光、激光关闭；背景扣除
```

三张图之间不得移动棋盘、相机或支架。应按 `chess → laser → nolaser` 采集；`nolaser` 和 `laser` 使用相同短曝光，短曝光图看不清棋盘并不等于发生了运动。

### 4.5 数据集追溯文件

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

填写项目 ID 和独立工作目录，从“相机通道”下拉框选择海康、大恒或模拟相机。
通道会带出相机配置、激光方向和推荐 workflow；仍可在项目页把 workflow 换成
项目副本。填写验收计划和棋盘参数后点击“应用到向导”，工具会加载通道并在后台
枚举设备。需要长期复用时再“保存项目…”。棋盘填写的是内角点数量，不是黑白方格数量。

### 5.2 第 2 页：相机与曝光

![相机与曝光页面](docs/images/user_manual/02_camera.png)

相机 YAML 已由第 1 页统一加载，不需要在这里重复选文件。操作顺序：

1. 确认当前通道/backend 和自动枚举出的型号、序列号；枚举失败时修复 SDK/连接后点击“枚举相机”；
2. 设置 Mono8/Mono12、曝光、增益和 ROI；
3. 选择“通用曝光/棋盘格/激光线”质量模式；
4. 开始取流并观察质量指标；
5. 取流期间可以在线应用曝光/增益，PixelFormat、Offset、宽高需停止后修改。

建议增益保持 `0 dB`，优先用曝光、遮光和光学调节改善质量。正式运行使用 Mono8 时，内参、激光模型、地面外参和验证数据也尽量使用 Mono8。

### 5.3 第 3 页：批量采集

![批量采集页面](docs/images/user_manual/03_capture.png)

填写输出数据集、数据集 ID、曝光序列/姿态任务、保存格式和质量模式，点击“生成并检查计划”确认任务预览，再点击“开始采集”。正式三联图应按 `chess → laser → nolaser` 完成一组后再移动棋盘。

若中断，保持计划不变，勾选“续采对应的 .inprogress 数据集”。页面会记录计划、数据集根目录、fit/validation 目录、manifest 和 `frames.csv`。

### 5.4 第 4 页：一键标定

![一键标定页面](docs/images/user_manual/04_calibration.png)

页面默认显示项目/通道选定的 workflow。`configs/workflow.example.yaml` 与
`configs/workflow_daheng.yaml` 都是结构或现场样例，复制后必须按项目目录填写
路径，并确认实际要运行的 stage 为 `enabled: true`。建议先“从最近采集结果更新
Workflow 输入”，再“检查 Workflow 阶段”，确认拟合/验证数据、输出目录和算法
profile 后点击“一键运行完整标定”。

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
# 已保存项目可省略 workflow 路径，直接使用项目所选通道的 workflow_plan：
python -m calibration_tool workflow --project projects\<project-id>\wizard_project.yaml
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

实时工具当前默认曝光是 `600 μs`，海康模板是 `1200 μs`，大恒现场模板是
`120000 μs`。这些都是不同入口的起点，不能当成相机实际回读值；正式发布时应在
文档、采集 manifest 和运行配置中记录最终采用的曝光。

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

## 10. 功能冗余与效率审查

本次审查覆盖相机适配、项目/相机/采集 GUI、workflow、验收和发布链路。没有发现
可以直接删除且不影响兼容性的核心生产死代码；主要问题是状态重复、重复 I/O 和
少数有意保留的兼容入口。

| 项目 | 结论 / 当前处理 |
|---|---|
| 项目页、相机页、采集计划分别维护 backend/config | 属于操作冗余；已改为项目页单一 channel，页面 2 自动加载，计划只保存可追溯快照 |
| 项目页和标定页都可改 workflow 路径 | 已收敛到项目页；第 4 页只读显示当前项目 workflow，避免分析时使用另一份路径 |
| 海康和大恒各有 SDK adapter | 不是冗余；两者复用同一 `CameraProvider/CameraSession` 协议，厂商差异必须隔离 |
| GUI 旧 `discover_result_artifacts` 包装函数 | 是兼容层，仍有测试/旧插件契约，本次不删除 |
| 多模块的小型 `_mapping` / `_relative_path` helper | 有少量重复，但收益低且异常语义略有差异；不为“去重”扩大改动 |
| MVS 每次 task 参数变化都完整 `configure` | 已优化：仅曝光/增益变化走在线更新；像素格式、ROI、尺寸、timeout 变化才停流重配 |
| SHA-256 对图像/点云整文件 `read_bytes()` | 已改为 1 MiB 分块哈希，并保持 CRLF 规范化结果兼容 |
| 采集 manifest/CSV 持久化 | manifest 每帧写入是断电续采事实来源；`frames.csv` 不按每帧重写、主要在 task 边界刷新，不应删除 |
| 验收报告多次写 YAML/HTML、复制并重复哈希 | 确认是后续高收益优化点；涉及 release manifest 的最终哈希，需单独重构和回归，不在本次相机切换改动中混做 |
| workflow 重复运行没有结果缓存 | 可按输入/配置哈希增加 stage cache；算法输出存在覆盖和副作用风险，需另行设计 |
| 每任务人工确认、settle frames | 是现场安全/稳定性步骤，不是无条件冗余；CLI 可不启用 `--interactive`，稳定帧应通过真机试验调小 |

因此，本次删除的是“重复选择和重复重连”，没有删除采集追溯、人工姿态确认或厂商
adapter。后续若继续优化，优先处理验收发布阶段的重复写盘/哈希，再评估 workflow
cache；不要以牺牲 manifest 一致性为代价追求表面速度。

## 11. 测试与故障排查

### 11.1 运行测试

```powershell
cd D:\Docs\linelaserscan\calibration_tool
python -m pytest -q
# 也可使用标准库测试发现器：
python -m unittest discover -s tests -v
```

本次代码验证结果为 `169 passed, 28 subtests passed, 1 failed`；唯一失败仍是
`tests/test_golden_snapshot.py::test_generated_baseline_matches_sources`，原因是仓库中既有
offline/online 配置与已确认 golden baseline 不一致。若该项失败，应先确认漂移是否
有意，再由项目负责人通过 `golden-build` 正式刷新 baseline，不要仅为让测试变绿而
手工覆盖哈希。

GUI 测试需要 PySide6 和可用的 Qt/offscreen 环境；真机测试还需要 SDK 和相机设备。`camera-preview` 返回 `dynamic_range_low` 时应调曝光、光源或质量阈值，并保留原始告警，不要把它当成程序崩溃。

### 11.2 常见问题

| 现象 | 优先检查 |
|---|---|
| `camera-list --channel hikrobot` 没有设备 | MVS SDK、相机占用、GigE 网络、序列号和 Python 位数 |
| synthetic 正常、MVS 失败 | SDK 加载、网卡/USB、设备权限和 `backend: mvs` |
| 大恒相机无法枚举 | 确认完整 Galaxy SDK、Python/SDK 位数、`DAHENG_GALAXY_ROOT`、USB3 连接和设备是否被其他程序占用 |
| 加载计划提示“相机通道不匹配” | 返回第 1 页选择生成该计划的通道并应用，或为当前通道重新生成新计划；不要改 plan backend 伪装兼容 |
| 棋盘检测失败 | 内角点数量、激光是否关闭、曝光/失焦/反光、棋盘是否完整入镜 |
| 激光覆盖率低 | 激光方向、曝光、遮光、线宽和 `quality.min_laser_coverage` |
| workflow 找不到输入 | 所有相对路径以 workflow YAML 所在目录解析；检查 fit/test/validation 目录 |
| 续采被拒绝 | 原计划 hash 已改变；应恢复原计划或新建数据集 |
| 验收 rejected | 打开 HTML/YAML 报告，逐项处理失败门禁；不要用 `--allow-failed` 伪装生产发布 |
| 实时工具无法加载标定 | 检查 manifest 哈希、runtime config 和共享 Steger profile 是否来自同一版本 |

## 12. 发布前检查清单

- 海康/大恒真机最小验收范围：设备枚举、序列号/型号读取、Mono8/Mono12 与 ROI 回读、曝光/增益在线更新、帧号/时间戳、超时/断流、停止与关闭生命周期。
- [ ] 清理 README、示例 YAML 和报告中的个人绝对路径，改为相对路径或占位符；
- [ ] 确认 `calibration/config/realtime_steger.yaml` 的归属和提交位置；
- [ ] 提交 `configs/`、`docs/`、向导截图、测试和必要的样例标定文件；
- [ ] 不提交 `.venv/`、`runs/`、临时日志、相机凭证和未经授权的原始数据；
- [ ] 每套标定数据保留 `dataset_manifest.yaml`、`frames.csv` 和 SHA-256；
- [ ] 运行 `golden-check`、`audit`、完整单元测试；
- [ ] 用 `--simulate` 启动向导并完成一条采集/报告 smoke test；
- [ ] 两个真实通道分别完成 `camera-list --channel ...`、预览和最小采集验证；
- [ ] 生成并审阅 `acceptance_report.html`，只有 `accepted` 才发布 bundle；
- [ ] 在实时工具中加载发布包，确认单帧重建、硬件 ROI 偏移和结果导出正常。

## 13. 相关文档

- [线激光标定工具用户手册](docs/线激光标定工具用户手册.md)
- [相机曝光与质量说明](docs/camera_exposure_and_quality.md)
- [验收闭环说明](docs/acceptance_closeout.md)
- [实时工具用户手册](../0704line-laser-3d-scanner/laser_measurement_tool/docs/ONLINE_USER_MANUAL.md)
- [实时工具配置说明](../0704line-laser-3d-scanner/laser_measurement_tool/docs/USAGE_CONFIG.md)
