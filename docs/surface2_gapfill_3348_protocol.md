# Surface-2 q2 gap-fill：33/38/43/48 mm 采集与接入协议

本协议只增加四个采集高度，不改变 Frozen C0、Frozen C1、q1/q2 定义、GUI 或生产测高链路。

## 已落地的采集计划

| height | plan | 帧数 | provisional exposure | 依据 |
|---:|---|---:|---:|---|
| 33 mm | `configs/obs_33mm.yaml` | 5 positions × 5 repeats = 25 | 350 µs | 30/46 mm 邻近采集 |
| 38 mm | `configs/obs_38mm.yaml` | 25 | 380 µs | 36/40 mm 邻近采集 |
| 43 mm | `configs/obs_43mm.yaml` | 25 | 350 µs | 30/46 mm 邻近采集 |
| 48 mm | `configs/obs_48mm.yaml` | 25 | 300 µs | 50 mm 邻近采集 |

四份 plan 已用现有 `load_capture_plan()` 做 schema round-trip 校验：每份均为 5 个 position task、每 task 5 帧、gain=0、Mono8、4096×3000、offset=(0,0)、serial=`GCB26060793`。

### 现场确认门槛

现有数据没有单一统一 exposure：30/46 mm 的实际帧分别出现 350 µs，36/40 mm 为 380 µs，50 mm 为 300 µs。因此上表是邻近高度的可审计 provisional 值，不是静默冻结的生产参数。每个 task 开始前必须确认相机实际 read-back、gain、laser power、机械状态与 plan 一致；若现场确认不同，记录 manifest 的 requested/actual 值后再继续。laser power 不在现有 plan/frames schema 中，必须由操作员单独确认并写入采集记录。

capture plan 不设置分析 ROI；使用现有全幅采集和相同 detector/Steger 配置。采集后人工 ROI 只按 median image、五帧 Steger overlay 中的物理 ground plane 几何冻结，不能查看 height error/residual。

## 采集命令

在有大恒相机的现场终端执行；本命令会真实采集，不能用作离线验证：

```text
cd D:\Docs\linelaserscan\calibration_tool
python -m calibration_tool capture-plan configs\obs_33mm.yaml --interactive --preview-window
python -m calibration_tool capture-plan configs\obs_38mm.yaml --interactive --preview-window
python -m calibration_tool capture-plan configs\obs_43mm.yaml --interactive --preview-window
python -m calibration_tool capture-plan configs\obs_48mm.yaml --interactive --preview-window
```

每个 plan 应产生 `projects/daheng/data/obs_<height>mm`，每个目录 25 张 TIFF、25 行 `frames.csv` 与 `dataset_manifest.yaml`。姿态顺序只表示该次采集的 task provenance；跨高度 position 对齐在后处理阶段按 Frozen-C0 q1 排序，不能把 `pose_id` 当作统一 position。

## 离线接入与人工 ROI review

采集完成后，在 scanner 仓库运行：

```text
cd D:\Docs\linelaserscan\0704line-laser-3d-scanner
python tools\prepare_surface2_gapfill_roi_review.py
```

入口复用现有 `prepare_surface2_roi_review.py`、Daheng gauge evaluator 与 median/overlay 生成器，输出到独立目录：

`outputs/daheng_c1_gauge_blocks_20260819_ground4a/surface2_gapfill_3348_review/`

它会做 100 帧完整性/尺寸/hash/manifest 审计、每 TIFF 一次 Steger、20 组五帧 median、centerline、geometry-only overlay 与 `surface2_roi_registry_manual_draft.json`。warning 帧会保留，不按质量告警或 residual 自动删除。

人工确认只能修改/冻结 `height_v_range`、`baseline_v_ranges` 等图像几何选区。完成后保留 draft provenance，并将最终 20 个 entry 和顶层 `manual_confirmed`/`manual_confirmed_count=20` 写入 `surface2_roi_registry_manual.json`；每个 entry 也必须 `manual_confirmed=true`。在这一步之前不得运行 q-domain 分析。

## 9 高度 q-domain 审计

ROI 冻结后运行：

```text
python tools\analyze_surface2_gapfill_domain.py
```

该入口严格要求新数据 integrity=100/100、Steger call count=100、20/20 geometry-only frozen ROI，并复用 canonical Surface-2B 的 30/36/40/46/50 mm formal CSV；不会覆盖旧 `surface2`、`surface2b` 或 `surface2br2` 输出。它生成 9 高度 q1/q2 statistics、相邻 P05–P95 gap/overlap、grouped-CV q2 support、q1-q2 coverage、q2-vs-height、raw residual-vs-q2，以及单独的 `C1_PARAMETER_SHA` 和 `C1_FILE_SHA256` provenance。

50 mm 仅用于本阶段 domain/残差审计，保持 strict held-out，不参与后续 B2/S0 模型选择。`Q2_VS_Q1Q2_MODEL_SELECTION_ALLOWED=YES` 只在 q2 gap 全部满足、各高度五个 condition 完整且 held-out development height 的 q2 full hull 不需 extrapolation 时出现；本脚本不拟合任何 correction。

## Provenance 边界

- 新增计算：四个新高度的 100 帧审计、一次/帧 Steger、20 组 median/overlay、20-entry ROI review 后的 Frozen C0/C1 重建和 q-domain 汇总。
- 复用结果：30/36/40/46/50 mm 的 canonical Surface-2B samples、Surface-1A Frozen q definition、Frozen C0/C1 文件与配置。
- `C1_PARAMETER_SHA` 是 canonical runtime C1_4k 参数 payload 的 SHA；`C1_FILE_SHA256` 是 Frozen C1 JSON 原始字节 SHA。二者在报告和 summary 中分开记录。
- 不重新拟合 C0/C1，不重新定义 q1/q2，不用 residual 驱动 ROI，不拟合 S1/spline/LUT/Δh/Δlambda，不做 random point split。
