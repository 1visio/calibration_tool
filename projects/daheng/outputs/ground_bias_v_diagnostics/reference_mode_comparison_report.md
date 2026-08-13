# Reference-plane mode comparison report

## 范围与统一口径

- 输入：现有 laser 001–031 重建点；三种模式使用同一批 frame 和 image-row `v` bin。
- 未做 compensation；未使用 smooth window。
- residual 均为 ground-frame `Zg` 方向 signed vertical residual，不是正交点面距离。
- pair correlation 只统计共同支持不少于 100 个 `v` 的唯一 frame pair。
- `median_residual_std_mm`、`median_abs_bias_mm` 和 sign mixing 使用 `sample_count >= 10` 的行。
- top/middle/bottom 分别为 `v=0–299`、`300–2699`、`2700–2999`，区域 std 使用至少 2 帧覆盖的行。

## 统一结果

| mode | median pair corr | pair corr >=0.5 | explained energy | median residual std / mm | median abs(bias) / mm | sign mixing |
|---|---:|---:|---:|---:|---:|---:|
| A `self_fitted` | 0.4705 | 46.37% | 0.4839 | 0.03769 | 0.02850 | 21.89% |
| B `fixed_normal_per_frame_offset` | 0.2978 | 26.57% | 0.2896 | 0.11545 | 0.03546 | 27.88% |
| C `fixed_ground_plane` | 0.2978 | 26.57% | 0.3271 | 0.11839 | 0.06266 | 15.49% |

## A. B 相对 A 是否提高跨帧一致性

没有明显提高，反而降低：median pair correlation 变化 -0.1727，explained residual energy 变化 -0.1943，median residual std 增加 +206.36%。
因此现有数据不支持“自由平面拟合此前吸收了某个稳定的一维系统 residual，而固定法向后可恢复它”这一解释。

## B. C 相对 B 的恶化与重新摆放 offset

C 相对 B 的 median pair correlation 变化 +0.0000，explained residual energy 变化 +0.0375，median residual std 变化 +2.55%，median abs(bias) 增加 +76.74%。
区域 std 的变化分别为：top +1.36%、middle +4.08%、bottom +4.78%。
固定全局 Z0 会保留每帧整体高度变化，而 B 会移除该变化。31 帧 `median(Zg)-Z0` 的范围为
0.14838 mm，样本标准差 0.03055 mm，P95 absolute offset
0.10832 mm。这说明重新摆放与重建链共同表现出的逐帧高度 offset 在绝对 residual 中不可忽略；
但没有独立 PnP，不能把它全部归因于棋盘物理重新摆放。
同时，C 与 B 的相关系数理论上对逐帧常数平移不敏感，所以不能仅靠 correlation 判断 offset 大小。
同理，C 的 sign-mixing fraction 下降会受到整体 residual 符号偏移影响，不能单独解释为波形一致性提高。

## C. apparent tilt 与 condition number

- raw Pearson r：0.4579
- `log10(condition number)` Pearson r：0.4358
- Spearman r：0.1423

按 `|r| >= 0.5` 作为明显关系的描述阈值，两者关系不明显。无论相关性强弱，
`apparent_tilt_deg` 都来自激光重建点的窄带自拟合，不能解释为棋盘真实机械倾角。

apparent tilt：median 32.908°，P95
62.300°，max 75.868°。

## D. 棋盘真实倾角变化是否可忽略

**INCONCLUSIVE**

这 31 帧没有逐帧独立棋盘 PnP 平面；A 的 `a,b` 来自同一批窄带激光重建点，B/C 又是水平法向假设，
三者都不能独立观测棋盘真实姿态。因此现有数据既不足以支持真实倾角变化可忽略，也不足以证明它不可忽略。

## Z0 来源

`fixed_ground_plane` 使用 `Z0=0.0 mm`。来源为 `D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0811\ground_extrinsics\camera_ground_extrinsics.yaml`：外参明确规定
`checkerboard pattern surface` 为 ground zero surface；camera-frame 平面经 `T_ground_from_camera`
变换后也数值验证为 `Zg=0`。这不是任意把 Z0 设为零。

## 重要限制

`apparent_tilt_deg = degrees(atan(sqrt(a^2+b^2)))` 只是“窄带自拟合得到的表观倾角”，
不允许把它直接解释成棋盘真实机械倾角。
