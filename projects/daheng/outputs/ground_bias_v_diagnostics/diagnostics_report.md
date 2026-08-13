# 大恒 ground-bias v residual 诊断报告

诊断日期：2026-08-12  
输入：`ground_bias_v_experiment_0812/input/laser 001.csv` 至 `laser 031.csv`  
帧数：31  
补偿：未应用  
平滑：未应用  

## 1. residual 定义与计算口径

每帧独立使用现有 ground-bias 平面 residual 实现：

```text
稳健拟合：Zg = a_i*Xg + b_i*Yg + c_i
r_i = Zg - (a_i*Xg + b_i*Yg + c_i)
r_i(v) = 同一 image row v 内 signed vertical residual 的中位数
```

平面拟合使用 iterative MAD rejection，阈值 3.5，最多 8 次迭代。本报告中的 `signed residual` 是沿 `Zg` 方向的有符号残差，而不是正交距离；这与运行时 `Z_corrected = Z_raw - b(v)` 的补偿量纲一致。

诊断统计没有做跨帧 MAD rejection，目的是保留真实的帧间分歧、符号翻转和边缘异常；仅在每帧拟合参考平面时剔除平面拟合离群点。所有数据都保持未补偿状态。

31 帧的 v 覆盖不一致：没有任何一个 image row 同时被全部 31 帧覆盖。相关系数因此按每一对 frame 的实际共同支持区计算，CSV 同时记录 `common_sample_count`。报告中的相关性汇总仅采用共同样本数不少于 100 的 frame pair。

### 三种 reference plane 严格对照

新增诊断同时计算三种 reference plane，既有 heatmap、逐 v 统计和后续 A–E 结论仍以
`self_fitted` 为 baseline：

| 模式 | residual 定义 | MAE / mm | RMS / mm | P95 abs / mm | median pair correlation | median profile explained energy |
|---|---|---:|---:|---:|---:|---:|
| `self_fitted` | `Zg-(a_i Xg+b_i Yg+c_i)` | 0.04700 | 0.06248 | 0.13033 | 0.4705 | 0.4839 |
| `fixed_normal_per_frame_offset` | `Zg-median(Zg_i)` | 0.10064 | 0.15676 | 0.38610 | 0.2978 | 0.2896 |
| `fixed_ground_plane` | `Zg-Z0` | 0.12301 | 0.16417 | 0.33617 | 0.2978 | 0.3271 |

`fixed_ground_plane` 的 `Z0=0.0 mm` 来自冻结的 0811 ground extrinsics，而不是任意设为零：
该文件明确声明棋盘图案表面是 ground 坐标的 zero surface；实现还把 camera-frame 平面经
`T_ground_from_camera` 变换到 ground frame，数值验证结果为水平的 `Zg=0` 平面。若这些
语义或数值检查不成立，诊断会停止。

逐帧明细见 `reference_plane_mode_comparison_per_frame.csv`。其中：

```text
apparent_tilt_deg = degrees(atan(sqrt(self_fit_a^2 + self_fit_b^2)))
```

**`apparent_tilt_deg` 只是“窄带自拟合得到的表观倾角”，不能当作棋盘真实机械倾角。**
本数据的窄带 self-fit condition number 中位数约 3,024、最大约 9,875，因此其 `a,b`
尤其不能替代独立棋盘 PnP 给出的真实姿态。

## 2. 数据覆盖与总体统计

- v 范围：0–2999 px；3000 行均至少被一帧覆盖。
- residual 矩阵有效 bin 数：40,924。
- 至少 5 帧覆盖：2,981 行。
- 至少 10 帧覆盖：2,631 行。
- 至少 20 帧覆盖：107 行。
- 单行最大覆盖：22 帧；全 31 帧共同覆盖：0 行。
- 逐帧平面拟合中位 inlier 比例：90.88%；最低：84.64%。
- 平面 design condition number 中位数：3,024；最大值：9,875。

最后一项说明单帧激光点在 XY 上仍接近窄带，二维平面参数有一定病态性。所有拟合均保持 rank 3，但 residual 结果仍应结合这一几何限制解释，不能把拟合平面当成独立测得的棋盘真值。

在 `sample_count >= 10` 的 2,631 行中：

- residual 跨帧 std 中位数：0.03769 mm；
- residual MAD 中位数：0.01985 mm；
- `|median b(v)|` 中位数：0.02850 mm；
- residual 符号一致率中位数：78.95%；
- 49.90% 的行符号一致率至少 80%；
- 13.76% 的行符号一致率不超过 60%。

## 3. A — 是否存在稳定重复的 residual(v) 波形

**存在部分稳定重复波形，但不是全 v 范围稳定，也不是所有 frame 同幅同相。**

证据：

- frame pair 在共同支持不少于 100 行时，相关系数中位数为 **0.470**；P10 为 **-0.123**，P90 为 **0.812**。
- 85.05% 的 pair correlation 为正，但只有 46.37% 达到 0.5；7.69% 低于 -0.2。
- 用跨帧 median profile `b(v)` 解释所有已观测 residual energy，解释比例为 **48.39%**。
- heatmap 中可以看到若干重复的宽带红/蓝结构，但也有明显的 frame-dependent 幅值、局部反相和覆盖断裂。

因此，数据支持“有一个共同的一维成分”，但该成分大约只能解释一半 residual energy；剩余约一半仍是帧相关变化或噪声。

## 4. B — 一致性较高的 v 区域

按 `sample_count >= 10`、符号一致率以及跨帧 std 综合判断，较可信的共同波形主要位于：

| v 区域 | 主要表现 |
|---|---|
| 100–399 | residual 以正值为主，符号一致率约 95–100%；但最上部支持通常只有约 10–12 帧。 |
| 700–999 | 负 residual 宽带最稳定，符号一致率接近 100%；std 约 0.038–0.054 mm。 |
| 1200–1399 | 正 residual 重复性较好，符号一致率约 85–94%。 |
| 2400–2599 | 正 residual 再次出现，符号一致率约 80–92%，std 约 0.031–0.035 mm。 |

采用更严格的定义——`sample_count >= 10`、符号一致率至少 80%，且 std 位于所有充分覆盖行的最低四分位（不高于 0.02977 mm）——连续至少 20 行的核心区只有：

- v = 1174–1198；
- v = 2410–2432。

这说明“看起来同号”的宽区间并不都具有低离散度；可用于精确 LUT 的高置信核心区域比视觉波形范围窄。

## 5. C — residual 符号随 frame 改变的区域

将 `sample_count >= 10` 且正 residual 比例处于 35%–65% 定义为明显 sign mixing。符号混杂集中在：

| v 区域 | sign-mixing 特征 |
|---|---|
| 400–499 | 充分覆盖行中约 43.5% 出现明显正负混杂。 |
| 1100–1199 | 约 32%；位于负波形向正波形过渡处。 |
| 1400–1599 | 约 39%–47%；median bias 很小，frame 间符号不稳定。 |
| 1600–1799 | 约 18%–27%，且 std 升至约 0.073–0.077 mm。 |
| 1800–2099 | 最明显，100 px block 的 sign-mixing 比例约 63%、77%、67%。 |
| 2200–2299 | 约 73%。 |
| 2600–2799 | 约 23%–28%。 |

sign mixing 通常不是完全连续的整段，而是以 5–19 行的小簇交错出现；完整簇列表保存在 `diagnostics_summary.json`。这也是不能仅凭一条平滑 median 曲线判断系统误差的原因。

## 6. D — 上边缘和下边缘是否更不稳定

固定将图像高度的前后 10% 定义为边缘：

| 区域 | sample_count 中位数 | residual std 中位数 | residual MAD 中位数 | 符号一致率中位数 | `|median b|` 中位数 |
|---|---:|---:|---:|---:|---:|
| 上边缘 v=0–299 | 9 | 0.03749 mm | 0.02116 mm | 100% | 0.10416 mm |
| 中部 v=300–2699 | 14 | 0.03774 mm | 0.02028 mm | 77.78% | 0.02806 mm |
| 下边缘 v=2700–2999 | 13 | 0.04526 mm | 0.01864 mm | 75% | 0.01774 mm |

结论不是“上下两端都同样恶化”：

- **上边缘的主要问题是覆盖不足**。其 std 与中部接近，且观察到的帧大多同号，但 sample count 中位数只有 9；强正 residual 的结论基于较少 frame，置信度低于中部。
- **下边缘确实更不稳定**。总体 std 比中部高约 20%；其中 v=2800–2899 的 std 中位数达到 0.06480 mm，比中部约高 72%。

## 7. E — 一维 b(v) 是否足以描述主要系统误差

**不足以作为完整误差模型；最多是一个中等强度的共同分量。**

支持这一判断的主要证据：

1. common median profile 仅解释 48.39% residual energy。
2. pairwise correlation 中位数只有 0.470，且只有 46.37% 的 frame pair 达到 0.5。
3. 多个宽区域存在显著 sign mixing；同一个 v 在不同 frame 上可能需要相反补偿。
4. 下边缘离散度升高，上边缘覆盖不足。
5. 每帧二维平面拟合本身 condition number 较高，说明单条激光窄带不能提供非常强的二维平面约束。

因此当前数据不支持直接把全范围 median `b(v)` 发布为正式补偿表。它可能在 700–999、1200–1399、2400–2599 等区间消除一部分稳定波形，但也可能在 sign-mixing 区域对部分 frame 产生反向修正。若下一步继续评估，应仍使用严格独立 holdout，并按区域同时检查改善率、sign consistency 和支持帧数；本报告本身未执行任何补偿。

## 8. 输出文件

- `residual_frame_v_heatmap.png`：31 帧逐 v signed residual；白色为该帧无有效点。
- `residual_v_statistics.csv`：逐 v 的样本数、median、mean、std、MAD 和 P95 absolute residual。
- `residual_v_median_sigma.png`：median `b(v)`、±1 sigma 和 sample count；未平滑。
- `frame_residual_correlation.csv`：所有 frame pair 的共同支持数和相关系数。
- `per_frame_plane_fit_diagnostics.csv`：各帧稳健平面系数、条件数和 inlier 数。
- `reference_plane_mode_comparison_per_frame.csv`：三模式所需的逐帧参考参数和误差。
- `reference_plane_mode_comparison_summary.csv`：三种 reference mode 的总体 residual 严格对照。
- `diagnostics_summary.json`：报告使用的结构化汇总、分区和 sign-mixing 范围。
- `generate_diagnostics.py`：本次诊断的可复核生成脚本。

## 最终结论

```text
A. 稳定重复波形：部分存在，但全局一致性中等。
B. 高一致性区域：主要为 100–399、700–999、1200–1399、2400–2599；严格低离散核心更窄。
C. 符号翻转：主要集中在 400–499、1100–1199、1400–1699、1800–2099、2200–2299、2600–2799。
D. 边缘：下边缘明显更不稳定；上边缘主要是覆盖不足，sigma 未明显恶化。
E. 一维 b(v)：不足以独立描述主要系统误差，只能描述约一半的共同 residual 成分。
```
