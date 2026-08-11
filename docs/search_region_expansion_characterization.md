# Stage 2B Search Region Expansion Characterization

## 结论

在本次真实数据、冻结的 `sigma=1.5` 配置下：

```text
recommended_minimum_safe_clearance_px = 14
minimum_stable_expansion_each_side_px = 32
minimum_observed_boundary_clearance_p05_px = 13.349572311341763
```

`14 px` 是把稳定档中观测到的最小 Boundary P05（13.3496 px）向上取整得到的整数建议。它是本次数据对当前 `sigma=1.5` 的定型结果，不应直接外推到其他 `sigma`、相机分辨率或光斑尺度；这些条件变化后应重新 characterization。

这里的 `+N px` 表示把每帧 formal search region 的两个 normal-axis 边界分别向外扩展 N px，因此 region 总尺寸最多增加 `2N px`。该 expansion 只用于一次性离线实验，不是生产配置建议，也没有接入 GUI。

## 实验边界与数据

实验脚本为 `scripts/search_region_expansion_characterization.py`。它从只读 Git ref `feature/phase-a-baseline-distance-laser-angle` 读取 TIFF，不切换当前工作分支、不修改原始数据。

使用 50 帧/序列：

- `B05_A10_full_scanlines`：B05_A10 multiheight 完整 scanlines，Stage 2A 正样本；formal Boundary P05 = 3.5279 px。
- `B05_A10_boundary_sensitive_h10`：同一批图像的 H10 区域 `u=[1857, 1899]`，隔离已有的 near-boundary 亚像素偏移。
- `B12p5_A10_normal_reference`：B12p5_A10 reference 完整 scanlines，正常 negative control；formal Boundary P05 = 32.9026 px。
- `B05_A10_h1_truncation`：同一批 B05 图像的 H1 区域 `u=[1800, 1833]`；formal 无有效中心，是已有 truncation case。

正式 Steger 参数全部冻结：

| 参数 | 值 |
|---|---:|
| `sigma` | 1.5 |
| `threshold` | 30.0 |
| `deriv_thresh` | 0.5 |
| `roi_margin` | 48 |
| `roi_max_height` | 512 |
| `scan_axis` | `column` |

本数据为 `scan_axis=column`，normal axis 是原图 `v/row`。每帧先执行一次 formal extraction 取得正式 `LaserSearchRegion`，随后仅把这个 normal-axis region 扩展为 `+8/+16/+32/+48`；Gaussian/Hessian、candidate selection 与其余参数均未改变。

## 指标定义

- pairing：同一帧、同一 scanline/column 且两档都有效的中心配对。
- center shift：配对中心在 normal axis 上的绝对位移。
- same candidate：`floor(normal-axis subpixel center)` 相同，沿用 Phase-A 定义。
- valid fraction：有效中心数 / 可用 scanline 数。
- response statistics：有效中心 response 的 mean/P50/P95。
- Boundary P05：有效中心到 search region 最近边界距离的第 5 百分位。

稳定判据应用于候选档之后的所有可用相邻档（至少一档）：center shift P95 ≤ 0.01 px、valid fraction 绝对变化 ≤ 0.01、same candidate fraction ≥ 0.99。候选档依次检查 `+8/+16/+32`。

## 每档结果

下面列出关键数值。完整的 paired center shift P50/P95/max、same candidate fraction、valid fraction，以及 response mean/P50/P95 均保存在 [per_level_metrics.csv](../experiments/search_region_expansion_characterization/per_level_metrics.csv)。`—` 表示 formal 没有有效 H1 中心，无法与 formal 配对。

| case | expansion/side | Boundary P05 px | valid % | shift vs formal P50/P95/max px | same candidate % | response P50/P95 |
|---|---:|---:|---:|---:|---:|---:|
| B05 full | formal | 3.5279 | 25.947 | 0 / 0 / 0 | 100.000 | 11.887 / 20.970 |
| B05 full | +8 | 0.4771 | 38.974 | 0 / 0.05538 / 2.30497 | 99.074 | 10.468 / 20.305 |
| B05 full | +16 | 4.5568 | 47.410 | 0 / 0.05538 / 2.30497 | 99.074 | 11.325 / 19.983 |
| B05 full | +32 | 13.3496 | 53.874 | 0 / 0.05538 / 2.30497 | 99.074 | 11.516 / 21.708 |
| B05 full | +48 | 29.3340 | 53.992 | 0 / 0.05538 / 2.30497 | 99.074 | 11.503 / 21.702 |
| B05 H10 | formal | 2.9889 | 100.000 | 0 / 0 / 0 | 100.000 | 3.783 / 5.675 |
| B05 H10 | +8 | 11.1337 | 100.000 | 0.06119 / 0.15316 / 0.20282 | 91.628 | 4.157 / 6.077 |
| B05 H10 | +16 | 19.1337 | 100.000 | 0.06119 / 0.15316 / 0.20282 | 91.628 | 4.157 / 6.077 |
| B05 H10 | +32 | 35.1337 | 100.000 | 0.06119 / 0.15316 / 0.20282 | 91.628 | 4.157 / 6.077 |
| B05 H10 | +48 | 51.1337 | 100.000 | 0.06119 / 0.15316 / 0.20282 | 91.628 | 4.157 / 6.077 |
| B12p5 | formal | 32.9026 | 43.526 | 0 / 0 / 0 | 100.000 | 15.102 / 24.760 |
| B12p5 | +8 | 40.9026 | 43.526 | 0 / 0 / 0 | 100.000 | 15.102 / 24.760 |
| B12p5 | +16 | 48.9026 | 43.526 | 0 / 0 / 0 | 100.000 | 15.102 / 24.760 |
| B12p5 | +32 | 64.9026 | 43.526 | 0 / 0 / 0 | 100.000 | 15.102 / 24.760 |
| B12p5 | +48 | 80.9026 | 43.526 | 0 / 0 / 0 | 100.000 | 15.102 / 24.760 |
| B05 H1 | formal | — | 0.000 | — | — | — |
| B05 H1 | +8 | 0.4668 | 90.294 | — | — | 4.085 / 4.842 |
| B05 H1 | +16 | 8.0681 | 97.471 | — | — | 4.391 / 5.386 |
| B05 H1 | +32 | 24.0681 | 97.471 | — | — | 4.391 / 5.386 |
| B05 H1 | +48 | 40.0681 | 97.471 | — | — | 4.391 / 5.386 |

H10 的 “vs formal” 位移在 +8 后保持不变，说明 formal region 的边界确实改变了亚像素中心，而不是扩展越大中心持续漂移。H1 formal 没有中心，因此对该案例应看相邻档，而不是 “vs formal”。

## 相邻档稳定性

完整结果见 [adjacent_shift_metrics.csv](../experiments/search_region_expansion_characterization/adjacent_shift_metrics.csv)。

| case | transition | center shift P95 px | valid fraction delta | same candidate % |
|---|---:|---:|---:|---:|
| B05 full | 8→16 | 1.18428 | +0.08436 | 80.127 |
| B05 full | 16→32 | 0.000277 | +0.06464 | 99.309 |
| B05 full | 32→48 | 0 | +0.00118 | 100.000 |
| B05 H10 | 8→16 | 0 | 0 | 100.000 |
| B05 H10 | 16→32 | 0 | 0 | 100.000 |
| B05 H10 | 32→48 | 0 | 0 | 100.000 |
| B12p5 | 8→16 | 0 | 0 | 100.000 |
| B12p5 | 16→32 | 0 | 0 | 100.000 |
| B12p5 | 32→48 | 0 | 0 | 100.000 |
| B05 H1 | 8→16 | 0.39120 | +0.07176 | 100.000 |
| B05 H1 | 16→32 | 0 | 0 | 100.000 |
| B05 H1 | 32→48 | 0 | 0 | 100.000 |

`+8` 明显不够：完整 B05 的 8→16 P95 为 1.1843 px，H1 为 0.3912 px。H1 在 `+16` 后稳定，但完整 B05 的 16→32 仍新增 6.46% 有效点，表明 region 仍在截断可提取信号。只有 `+32→+48` 在全部案例中同时满足中心、candidate 和 valid fraction 稳定判据。

在 `+32` 档，四个证据案例的 Boundary P05 分别为 13.3496、35.1337、64.9026、24.0681 px；限制值来自完整 B05。因此建议最小 Boundary P05 clearance 为向上取整后的 **14 px**。这不是建议生产时固定扩大 32 px；32 px 只是本数据从原 formal region 到首个稳定实验档的 expansion。

## 产物与可复现性

- [per_level_metrics.csv](../experiments/search_region_expansion_characterization/per_level_metrics.csv)：全部 per-level 指标。
- [adjacent_shift_metrics.csv](../experiments/search_region_expansion_characterization/adjacent_shift_metrics.csv)：全部相邻档指标。
- [center_shift_p95_by_expansion.png](../experiments/search_region_expansion_characterization/center_shift_p95_by_expansion.png)：中心位移曲线。
- [summary.json](../experiments/search_region_expansion_characterization/summary.json)：冻结参数、数据文件、判据和机器可读结论。

复现实验：

```bash
python scripts/search_region_expansion_characterization.py
```

本实验没有修改或替换 `realtime_steger.py`，没有写入 GUI/PreviewThread、formal PASS/FAIL、保存门禁或 online point-cloud 路径。

```text
realtime_gui_behavior_changed = false
formal_steger_changed = false
production_expansion_sweep_added = false
characterization_complete = true
```
