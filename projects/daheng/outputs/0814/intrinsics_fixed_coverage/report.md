# Task 6F — Camera calibration fixed-coverage stability audit

`CAMERA_CALIBRATION_STABILITY_SOURCE = B. COVERAGE_DOMINANT`

本报告只读取正式 camera FIT chess 001–018，以及激光 FIT 001–018、025–036。Validation 未打开；正式 K/D、畸变模型、Steger、Cone 均未修改/重拟合。

## 结论摘要

全 18 帧角点噪声 MC（1000 次）典型 candidate-global P95 = 0.122387 mm；按 frame 的 P95 中位数 = 0.117624 mm，95% 尾部 = 0.31553 mm。
普通 frame bootstrap（Task 6E）对应的 frame-P95 中位数为 0.352331 mm；full-18 与普通 bootstrap 的典型比值为 2.995。

## 方法比较

| method | candidates | unique frames (min/median/max) | frame-P95 median (mm) | frame-P95 95% (mm) | global-P95 median (mm) | P95/Cone ratio median |
|---|---:|---:|---:|---:|---:|---:|
| naive_frame_bootstrap | 500 | 8/12/15 | 0.352331 | 1.09439 | nan | 15.2762 |
| loo | 18 | 17/17/17 | 0.0391471 | 0.338801 | nan | nan |
| full18_corner_noise_mc | 1000 | 18/18/18 | 0.117624 | 0.31553 | 0.122387 | 1.59913 |
| coverage_matched_16 | 8 | 16/16/16 | 0.092735 | 0.704689 | 0.101241 | 1.31136 |
| coverage_matched_14 | 8 | 14/14/14 | 0.438759 | 0.535892 | 0.458942 | 5.88866 |
| coverage_matched_12 | 8 | 12/12/12 | 0.399651 | 0.801887 | 0.418066 | 5.89699 |

## Coverage-preserving subsets

- 12/18: selected 8 subsets; frame-P95 median across candidates/frames = 0.399651 mm; global-P95 median = 0.418066 mm; minimum selected span ratio = 0.8286.
- 14/18: selected 8 subsets; frame-P95 median across candidates/frames = 0.438759 mm; global-P95 median = 0.458942 mm; minimum selected span ratio = 0.9383.
- 16/18: selected 8 subsets; frame-P95 median across candidates/frames = 0.092735 mm; global-P95 median = 0.101241 mm; minimum selected span ratio = 0.831.

## Coverage sensitivity

- board_center_z_mm: Spearman rho=0.09915, p=0.6448, n=24; predictor is subset coverage loss/range loss.
- apparent_bbox_area_fraction: Spearman rho=-0.07531, p=0.7265, n=24; predictor is subset coverage loss/range loss.
- board_tilt_deg: Spearman rho=0.07428, p=0.7301, n=24; predictor is subset coverage loss/range loss.
- board_center_v_norm: Spearman rho=-0.005214, p=0.9807, n=24; predictor is subset coverage loss/range loss.
- board_center_u_norm: Spearman rho=-0.003287, p=0.9878, n=24; predictor is subset coverage loss/range loss.

## Full-18 pose dependence

Full-18 MC 的 frame-level median P95 在 18 个正式姿态间约为 0.112–0.123 mm；按现有 18 帧 coverage 做探索性 Spearman 检查，board-center depth 与 uncertainty 的相关最强（rho≈0.963, p≈1.6e-10），其次是 tilt（rho≈-0.610, p≈0.007）和 apparent board size（rho≈-0.527, p≈0.025）。这些变量彼此耦合，不能解释为独立因果；sensor-u/v 没有同等级的稳定关系。Top/Middle/Bottom 的 candidate-P95 中位数约为 0.115/0.115/0.112 mm，未见明显 sensor-edge amplification。

## 下一步

当前 18 帧对 typical full-board truth 已足够稳定，不需要因为 6E 的 naive bootstrap 结果立即推翻正式 K/D 或重采全部标定。若要降低 coverage-loss tail，应补充具有不同 depth/tilt/board-size、同时覆盖 sensor 中心与边缘的独立 calibration poses；这属于降低 coverage uncertainty 的后续实验，不是本轮调参。

## 027

027 的 full-18 corner-noise MC frame-P95 中位数 = 0.113811 mm，95% candidate tail = 0.301495 mm；正式冻结 Cone RMSE = 0.3688 mm，truth-uncertainty / Cone RMSE = 0.308597。

## 判断

全 18 个 unique pose 均保留时，角点噪声实验代表 formal calibration 本身的 uncertainty；而普通有放回 frame bootstrap 的 unique-pose 数量明显下降，代表 coverage degeneration。实际 coverage-matched 结果为 12/18=0.418066 mm；14/18=0.458942 mm；16/18=0.101241 mm，其中 14/18 与 12/18 已明显高于 full-18，说明删减 unique pose 即使保持大范围 coverage，仍会引入显著 calibration variation。因此 6E 的大 variation 主要来自 coverage/pose 数量，而不是 full-18 corner noise。

## 文件

- `full18_corner_mc_intrinsics.csv`
- `full18_corner_mc_truth.csv`
- `coverage_matched_subsets.csv`
- `coverage_truth_sensitivity.csv`
- `bootstrap_method_comparison.csv`
- `provenance.json`
