# Stage 3-3：Production Search Region Shadow Replay

## 1. 结论

Stage 3-2 shadow proposal 在 B05 boundary-sensitive 和 H1 truncation 上都恢复到 Stage 2B stable reference，且没有选择 alternate ridge；但它对正常 B12p5 仍提出了不必要的 9 px 扩展。因此当前策略只完成了 problem-case recovery，尚不能判定为可直接切换生产行为。

```text
shadow_strategy_validated = false
problem_cases_recovered = true
normal_cases_unchanged = true
normal_unnecessary_expansion_avoided = false
alternate_ridge_detected = false

behavior_changed = false
formal_steger_result_changed = false
```

`normal_cases_unchanged=true` 表示 B12p5 的 valid、中心、candidate 和 response 数值完全不变；它不代表 resolver 避免了无效计算范围。后者单独记录为 `normal_unnecessary_expansion_avoided=false`。

## 2. 实验方法

一次性脚本 [production_search_region_replay.py](../scripts/production_search_region_replay.py) 从只读 Git ref `feature/phase-a-baseline-distance-laser-angle` 加载真实 TIFF，不切换分支、不修改数据。

每个数据序列使用 50 帧，冻结参数：

```yaml
sigma: 1.5
threshold: 30.0
deriv_thresh: 0.5
roi_margin: 48
roi_max_height: 512
scan_axis: column
```

每帧独立执行三次离线 extraction：

1. `current`：不传 additional region，得到当前正式 auto region；
2. `proposed`：把 shadow resolver 的 proposal 作为 additional `LaserSearchRegion`；
3. `stable_reference`：把 current region 每侧扩展 48 px，即 Stage 2B 最大、已稳定的 reference 档。

三次运行只用于 replay 对比。正式 calibration/GUI 路径仍使用 current region，proposal 没有回写生产行为。

pairing 使用同一帧、同一 scanline，same candidate 沿用 `floor(normal-axis subpixel center)` 定义。

## 3. Resolver 输入与 proposal

下表 interval 均为 normal-axis 半开区间。active intervals 显示第 1 帧代表值；全部 50 帧的精确 interval 和少量 1 px active 波动保存在 [resolver_frames.csv](../experiments/production_search_region_replay/resolver_frames.csv)。current/proposed region、seed interval 和 outside evidence 在各 case 的 50 帧中保持一致。

| case | current | proposed | would expand | reason | outside active | outside threshold peak | seed interval |
|---|---:|---:|---:|---|---|---|---:|
| B05 H10 boundary | `[793,895)` | `[793,942)` | 50/50 | `significant_intensity_outside_current_region` | `[899,908)`, `[911,916)` | `[895,896)`, `[898,909)`, `[910,918)`, `[926,928)` | `[841,847)` |
| B05 H1 truncation | `[793,895)` | `[793,942)` | 50/50 | 同上 | 同上 | 同上 | `[841,847)` |
| B05 full scanlines | `[793,895)` | `[793,942)` | 50/50 | 同上 | 同上 | 同上 | `[841,847)` |
| B12p5 normal | `[908,1024)` | `[899,1024)` | 50/50 | `significant_intensity_near_current_boundary` | 无 | 无 | `[956,976)` |

代表 active intervals：

```text
B05:
  [841,847), [862,864), [890,893), [899,908), [911,916)

B12p5:
  [914,918), [941,942), [943,945), [947,948), [949,950), [956,976)
```

B05 的 current region 外存在明确 active 和 threshold-peak interval，proposal 的扩展方向与 H1/H10 真实信号位置一致。

B12p5 的 current region 外没有 active 或 threshold-peak evidence。它仍扩展，是因为 `[914,918)` 距 current 下边界只有 6 px，小于 shadow policy 的 14 px；但正式 Steger 中心的 Boundary P05 实际为 32.90 px。说明“任意 threshold interval 靠近边界”会把与正式 ridge 无关的亮结构当成扩展理由。

## 4. Extraction 对比

Stable reference 为 current 每侧 `+48 px`。`—` 表示 current H1 没有有效中心，无法 pairing。

| case | mode | valid % | Boundary P05 px | shift vs stable P50/P95/max px | same candidate % | response mean/P50/P95 |
|---|---|---:|---:|---:|---:|---:|
| B05 H10 | current | 100.000 | 2.9889 | 0.06119 / 0.15316 / 0.20282 | 91.628 | 3.910 / 3.783 / 5.675 |
| B05 H10 | proposed | 100.000 | 50.1337 | 0 / 0 / 0 | 100.000 | 4.428 / 4.157 / 6.077 |
| B05 H10 | stable | 100.000 | 51.1337 | 0 / 0 / 0 | 100.000 | 4.428 / 4.157 / 6.077 |
| B05 H1 | current | 0.000 | — | — | — | — |
| B05 H1 | proposed | 97.471 | 39.0681 | 0 / 0 / 0 | 100.000 | 4.363 / 4.391 / 5.386 |
| B05 H1 | stable | 97.471 | 40.0681 | 0 / 0 / 0 | 100.000 | 4.363 / 4.391 / 5.386 |
| B05 full | current | 25.947 | 3.5279 | 0 / 0.05538 / 2.30497 | 99.074 | 11.420 / 11.887 / 20.970 |
| B05 full | proposed | 53.992 | 28.3340 | 0 / 0 / 0 | 100.000 | 11.592 / 11.503 / 21.702 |
| B05 full | stable | 53.992 | 29.3340 | 0 / 0 / 0 | 100.000 | 11.592 / 11.503 / 21.702 |
| B12p5 | current | 43.526 | 32.9026 | 0 / 0 / 0 | 100.000 | 15.203 / 15.102 / 24.760 |
| B12p5 | proposed | 43.526 | 40.8725 | 0 / 0 / 0 | 100.000 | 15.203 / 15.102 / 24.760 |
| B12p5 | stable | 43.526 | 80.9026 | 0 / 0 / 0 | 100.000 | 15.203 / 15.102 / 24.760 |

完整机器可读结果见 [extraction_metrics.csv](../experiments/production_search_region_replay/extraction_metrics.csv) 和 [case_comparisons.csv](../experiments/production_search_region_replay/case_comparisons.csv)。

## 5. A：问题 case 是否修复

### B05 boundary-sensitive H10

- 50/50 帧 `would_expand=true`；
- current 相对 stable 的中心 P95 偏移为 0.1532 px，same candidate 仅 91.63%；
- proposed 相对 stable 的 P50/P95/max 全为 0，same candidate 100%；
- response statistics 与 stable 完全一致。

因此 proposal 消除了 current crop boundary 对 H10 亚像素中心的影响。

### H1 truncation

- current valid fraction 为 0；
- proposed 恢复至 97.47%，与 stable 完全一致；
- proposed 与 stable 的中心差为 0、same candidate 100%、response 完全一致。

因此 H1 truncation 被 proposal 恢复。

综合结论：`problem_cases_recovered=true`。

## 6. B：normal case 是否避免不必要 expansion

B12p5 的 current、proposed、stable 提取结果完全相同：valid fraction、所有配对中心、integer candidate 和 response statistics 均无差异。因此：

```text
normal_cases_unchanged = true
```

但 shadow proposal 仍把 `[908,1024)` 扩成 `[899,1024)`，50/50 帧都触发 `significant_intensity_near_current_boundary`。扩出的 9 px 没有改变任何正式中心，属于不必要 expansion：

```text
normal_unnecessary_expansion_avoided = false
```

这也是当前 `shadow_strategy_validated=false` 的原因。后续 policy 应优先信任真正的 outside-region evidence；对仅 near-boundary 的 threshold interval，需要增加持续长度、与 dominant/seed component 的关系或其它 pre-Steger ridge plausibility 条件，不能直接扩展。

## 7. C：是否引入 alternate ridge

判据：proposal 与 stable 的 paired center shift P95 ≥ 0.5 px 且不同 floor candidate ≥ 1%，或 proposal-only candidate ≥ 1%。

四个 case 中：

- proposed 与 stable 的 P95/max 均为 0；
- same candidate 均为 100%；
- proposed-only 和 stable-only candidate 都为 0。

因此：

```text
alternate_ridge_detected = false
```

本数据没有证据表明 `[793,942)` 的 B05 proposal 或 `[899,1024)` 的 B12p5 proposal 引入 alternate ridge。

## 8. 输出与复现

- [resolver_frames.csv](../experiments/production_search_region_replay/resolver_frames.csv)：每帧 current/proposed、reason、outside evidence、active/seed intervals。
- [extraction_metrics.csv](../experiments/production_search_region_replay/extraction_metrics.csv)：current/proposed/stable 的全部 extraction 指标。
- [case_comparisons.csv](../experiments/production_search_region_replay/case_comparisons.csv)：case 级恢复与 alternate-ridge 指标。
- [summary.json](../experiments/production_search_region_replay/summary.json)：数据来源、判据和最终布尔结论。

复现：

```bash
python scripts/production_search_region_replay.py
```

本脚本只读真实数据并写实验产物，没有把 proposed region 接入 GUI、正式 calibration workflow 或 online。

验证：

```text
python -m py_compile scripts/production_search_region_replay.py
passed

python -m pytest -q
142 passed, 28 subtests passed

git diff --check
passed
```
