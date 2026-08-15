# Task 6F — Camera calibration fixed-coverage stability audit

`CAMERA_CALIBRATION_STABILITY_SOURCE = C. BOTH`

本报告只读取正式 camera FIT chess 001–018，以及激光 FIT 001–018、025–036。Validation 未打开；正式 K/D、畸变模型、Steger、Cone 均未修改/重拟合。

## 结论摘要

全 18 帧角点噪声 MC（2 次）典型 candidate-global P95 = 0.0816637 mm；按 frame 的 P95 中位数 = nan mm，95% 尾部 = nan mm。
普通 frame bootstrap（Task 6E）对应的 frame-P95 中位数为 0.352331 mm；full-18 与普通 bootstrap 的典型比值为 nan。

## 方法比较

| method | candidates | unique frames (min/median/max) | frame-P95 median (mm) | frame-P95 95% (mm) | global-P95 median (mm) | P95/Cone ratio median |
|---|---:|---:|---:|---:|---:|---:|
| naive_frame_bootstrap | 500 | 8/12/15 | 0.352331 | 1.09439 | nan | 15.2762 |
| loo | 18 | 17/17/17 | 0.0391471 | 0.338801 | nan | nan |
| full18_corner_noise_mc | 2 | 18/18/18 | 0.0744933 | 0.121171 | 0.0816637 | 0.798582 |
| coverage_matched_16 | 2 | 16/16/16 | 0.144819 | 0.223831 | 0.153106 | 1.73573 |
| coverage_matched_14 | 2 | 14/14/14 | 0.505166 | 0.54527 | 0.530022 | 7.14188 |
| coverage_matched_12 | 2 | 12/12/12 | 0.365166 | 0.617969 | 0.394724 | 3.72132 |

## Coverage-preserving subsets

- 12/18: selected 2 subsets; frame-P95 median across candidates/frames = 0.365166 mm; global-P95 median = 0.394724 mm; minimum selected span ratio = 0.8286.
- 14/18: selected 2 subsets; frame-P95 median across candidates/frames = 0.505166 mm; global-P95 median = 0.530022 mm; minimum selected span ratio = 0.9976.
- 16/18: selected 2 subsets; frame-P95 median across candidates/frames = 0.144819 mm; global-P95 median = 0.153106 mm; minimum selected span ratio = 0.9976.

## Coverage sensitivity

- board_center_v_norm: Spearman rho=0.488, p=0.3262, n=6; predictor is subset coverage loss/range loss.
- board_tilt_deg: Spearman rho=0.1234, p=0.8158, n=6; predictor is subset coverage loss/range loss.

## 027

027 的 full-18 corner-noise MC frame-P95 中位数 = 0.0751654 mm，95% candidate tail = 0.115205 mm；正式冻结 Cone RMSE = 0.3688 mm，truth-uncertainty / Cone RMSE = 0.203811。

## 判断

全 18 个 unique pose 均保留时，角点噪声实验代表 formal calibration 本身的 uncertainty；而普通有放回 frame bootstrap 的 unique-pose 数量明显下降，代表 coverage degeneration。若 full-18 远小于普通 bootstrap、且 16/14/12 的 coverage-matched 结果仍接近 full-18，则 6E 的大 variation 主要来自 coverage loss，而不是 corner noise。

## 文件

- `full18_corner_mc_intrinsics.csv`
- `full18_corner_mc_truth.csv`
- `coverage_matched_subsets.csv`
- `coverage_truth_sensitivity.csv`
- `bootstrap_method_comparison.csv`
- `provenance.json`
