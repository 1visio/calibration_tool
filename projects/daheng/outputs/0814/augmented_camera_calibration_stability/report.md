# Task 6H-1 — Augmented camera calibration stability A/B

`EDGE_EXTENSION_CAMERA_GAIN = B. MODERATE`

推荐作为下一阶段诊断 candidate：`M1-core`（正式 K/D 文件未改写）。

## Scope and controls

- M0：chess `001–018`；M1-core：M0 + `026,027,028,035`；M1-full：M1-core + `031,033,032,030`。
- 未使用 extension `025,029,034,036`；`027` 保留并在 extension leverage 表中单独可见。激光传播只使用 FIT `001–018,025–036`，未打开 Validation。
- 三个候选均使用同一 formal corner pipeline、`CALIB_FIX_K3`、`SOLVEPNP_ITERATIVE + solvePnPRefineLM`；未更换 distortion model。
- Circular Cone 仅从冻结 provenance 读取，未重新拟合；Steger、正式 K/D 均未修改。
- Fixed-coverage corner-noise MC：每个候选 100 次，seed `20260817`；每帧 88 个角点均保留。扰动协方差由 formal K/D 的该帧 PnP reprojection residual 居中估计，三候选保持同一噪声定义。

## Candidate calibration

| candidate | poses | global RMSE (px) | Δfx (px) | Δfy (px) | Δcx (px) | Δcy (px) | Δk1 | Δk2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 18 | 0.118739 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 |
| M1-core | 22 | 0.128318 | +0.930 | +0.908 | +1.903 | −2.775 | +0.000291 | −0.006312 |
| M1-full | 26 | 0.127724 | +1.382 | +1.397 | +1.285 | −2.661 | +0.000172 | −0.007369 |

M1 的 global training RMSE 比 M0 高约 7.6–8.1%，所以没有把 RMSE 最低作为选择依据；但上述 K/D 位移说明 extension 不是对 M0 的无变化冗余复制。

## Camera-side LOO stability

表中 LOO 数值是每次删去一个 calibration pose 后，在固定 laser FIT 上得到的逐 frame P95 `|Δlambda|` 的分布；它相对于 formal M0 truth，因而包含候选整体 K/D 位移。比较 frame-dependence 时主要看 P95/max 尾部。

| candidate | LOO median (mm) | P90 (mm) | P95 (mm) | max (mm) |
|---|---:|---:|---:|---:|
| M0 | 0.038707 | 0.263770 | 0.337410 | 0.399967 |
| M1-core | 0.117254 | 0.181689 | 0.246740 | 0.306194 |
| M1-full | 0.154313 | 0.204742 | 0.265623 | 0.326453 |

原 M0 高 leverage pose 的逐次删除结果（`candidate_frame_p95_median_mm`）：

| omitted pose | M0 | M1-core | M1-full |
|---:|---:|---:|---:|
| 001 | 0.326 | 0.138 | 0.058 |
| 002 | 0.400 | 0.306 | 0.326 |
| 003 | 0.190 | 0.069 | 0.118 |
| 010 | 0.237 | 0.250 | 0.284 |
| 017 | 0.089 | 0.060 | 0.108 |

因此 M1-core 明显收窄了 M0 的总体 LOO 尾部和大多数高-leverage pose 的影响；`002` 仍是主要敏感 pose，`010` 没有改善。M1-full 并未在 LOO 尾部继续改善 core。

## Extension-frame leverage

| omitted extension | M1-core P95 (mm) | M1-full P95 (mm) |
|---:|---:|---:|
| 026 | 0.042 | 0.050 |
| 027 | 0.148 | 0.200 |
| 028 | 0.085 | 0.129 |
| 035 | 0.155 | 0.190 |
| 031 | — | 0.169 |
| 033 | — | 0.171 |
| 032 | — | 0.114 |
| 030 | — | 0.129 |

`027` 是新增 extension 中 leverage 最大的 pose；加入 full 后其删除影响反而上升到约 0.200 mm，说明 M1-full 没有把新增高 tilt 约束变成充分冗余。

## Fixed-coverage corner-noise MC

Raw 列相对于 formal M0 truth，包含候选 K/D 的固定整体位移；centered 列相对于该候选自己的全量解，是 corner-noise stability 的主比较量。

| candidate | MC success | raw global P95 median (mm) | centered global P95 median (mm) | centered P95 tail (mm) | centered max (mm) | fx std (px) | k2 std |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 100 | 0.100595 | 0.100595 | 0.293773 | 0.379678 | 1.54221 | 0.006394 |
| M1-core | 100 | 0.128315 | 0.108490 | 0.290329 | 0.379010 | 1.52154 | 0.006553 |
| M1-full | 100 | 0.151282 | 0.081014 | 0.244329 | 0.317481 | 1.21823 | 0.005919 |

M1-core 的中心化 MC 与 M0 接近但略差；M1-full 的中心化 MC 更好，说明增加 full poses 对纯角点噪声传播有收益。但该收益没有同步转化为更低的 LOO frame-leverage 尾部，不能据此单独推荐 M1-full。

## Coverage and coupling

| candidate | tilt range (deg) | depth range (mm) | apparent-size range | sensor-u range | Spearman(tilt,depth) |
|---|---:|---:|---:|---:|---:|
| M0 | 22.232 | 63.601 | 0.258762 | 0.334687 | −0.5769 |
| M1-core | 24.116 | 63.601 | 0.298219 | 0.364055 | −0.6160 |
| M1-full | 24.116 | 63.601 | 0.298219 | 0.364055 | −0.6315 |

M1 增加了高 tilt、较大 apparent size 和低 sensor-u 支撑，但没有增加 depth range，也没有改善 tilt-depth coupling；coupling 反而略变强。因此 observability 是部分增加，不是完整的 pose-parameter decoupling。

## Conclusions

1. **M1-core 是否降低 M0 高-leverage frame dependence？** 是。LOO P95/max 从 `0.337/0.400 mm` 降到 `0.247/0.306 mm`，但 pose `002` 与 `010` 仍需关注。
2. **M1-full 是否进一步提高稳定性？** 不全面。它的 centered MC 最好，但 LOO P95/max (`0.266/0.326 mm`) 比 M1-core 差，新增 `027` 也变得更有 leverage。
3. **extension 是否提高 observability？** 部分提高：tilt/size/u 覆盖扩展、LOO 尾部收窄；但 K/D 整体移动、depth 未扩展、tilt-depth coupling 变差，尚不能称为强增益。
4. **tilt-depth coupling 是否仍是主要薄弱点？** 是。M0→M1 的 Spearman 绝对值由 `0.577` 增至 `0.616/0.631`。
5. **推荐 candidate：** `M1-core` 作为下一阶段诊断冻结对象；M1-full 保留为 MC 对照，不冻结为正式模型。若必须优先优化纯角点噪声而非 LOO leverage，可单独记录 M1-full 的 MC 优势。

最终结论：`EDGE_EXTENSION_CAMERA_GAIN = B. MODERATE`。这是 stability/observability 的局部改善，不是对正式相机模型的自动替换授权。

## Outputs

- `m0_m1_intrinsics_comparison.csv`
- `m0_m1_loo_stability.csv`
- `m0_m1_corner_mc.csv`
- `extension_frame_leverage.csv`
- `m0_m1_coverage_comparison.csv`
- `provenance.json`
