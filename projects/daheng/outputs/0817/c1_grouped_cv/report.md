# C1 grouped cross-validation — 1D ray-domain residual correction

`C1_FEASIBILITY = MODERATE`

是否值得 freeze C1 进入独立 Validation：**YES**。即使为 YES，本报告只冻结候选定义供独立 Validation 检验，不代表已经部署或通过 Validation。

## Scope and frozen boundary

- 输入逐点表：`D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0817\spatial_residual_observability\fit_ray_residual_points.csv`；共 **26,663** 个有效 FIT ray，frame 集严格为 001–018、025–036。
- 上一轮 PCA 的 `pca_s` 原样使用；本轮 s domain 固定为 **[-0.16412256362, 0.194958484593]**，只使用 predictor support，不使用 held-out residual 选择 knot。
- C0 = Frozen Circular Cone；C1 = `lambda_cone + F(s)`，残差定义为 `lambda_truth - lambda_prediction`。
- 027 保留在训练候选和 held-out folds 中，不删除、不重加权。
- 未读取 Validation 019–024、037–040；未重新拟合 K/D 或 Cone；未做 C2/C3；没有 point-wise random split。

## C1 model and CV protocol

- 每个 fold 留出一个完整 frame，共 **30** 个 leave-one-frame-out folds；训练点按 frame 赋权，每帧总权重为 1。
- C1 使用 cubic B-spline，比较 interior knots = **3/4/5/6**（对应 basis count = 7/8/9/10），二阶差分 penalty = **0.1**（moderate_second_difference）。
- 每个 fold 的 F(s) 仅用训练 frames 拟合；Top/Middle/Bottom 为 `v∈[0,300) / [300,2700) / [2700,3000)`。
- 选型同时看 worst-region RMSE、edge/middle ratio、bias range 和逐 frame 改善比例；不以 training 或 pooled global RMSE 单独选型。

## Model comparison

| model | knots | CV global RMSE median (mm) | global improvement median | worst-region RMSE median (mm) | worst-region improvement median | edge/middle ratio median | global frame improve fraction | worst-region frame improve fraction | selection gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0 | 0.0700603 | nan% | 0.0780901 | nan% | 1.4246 | nan | nan | False |
| C1_3k | 3 | 0.0666763 | 2.796% | 0.0667891 | 6.931% | 0.9386 | 0.633 | 0.800 | False |
| C1_4k | 4 | 0.0637839 | 7.933% | 0.0648698 | 10.934% | 1.0134 | 0.833 | 0.867 | True |
| C1_5k | 5 | 0.0634918 | 8.937% | 0.0645205 | 12.305% | 1.0454 | 0.800 | 0.900 | True |
| C1_6k | 6 | 0.0594076 | 10.299% | 0.0621826 | 15.364% | 1.0259 | 0.833 | 0.900 | True |

选中的最简单稳定候选：**C1_4k**（interior knots = 4）。其 CV global RMSE 中位数改善 **7.933%**，worst-region RMSE 中位数改善 **10.934%**；逐 frame global / worst-region 改善比例为 **0.833 / 0.867**。
C1_3k 的 global median improvement 为 **2.796%**，低于全视场门槛 **5.0%**，因此未被选中。
选中候选的 held-out global RMSE improvement 百分位：P05 **-2.539%**，median **7.933%**，P95 **27.426%**；worst-region 的对应 median improvement 为 **10.934%**。
C0 的 CV worst-region RMSE median/P95 = **0.0780901/0.288562 mm**；C1 = **0.0648698/0.253932 mm**。
C0/C1 global bias range = **0.391952/0.40072 mm**；worst-region bias range = **0.57876/0.482394 mm**；edge/middle median ratio = **1.4246/1.0134**。
frame 027（保留）global RMSE 为 **0.3688 → 0.366687 mm**，改善 **0.573%**；worst-region 改善 **0.573%**，因此该帧没有显示出 material improvement。

## Feasibility gates

- 基本稳定门槛：global frame improvement ≥ 0.60 且 global median improvement ≥ 5.0%；worst-region frame improvement ≥ 0.50；worst-region median improvement ≥ 5.0%；worst-region P95 不恶化超过 2.0%；edge/middle ratio 不恶化超过 2.0%。
- 当前选择：`C1_4k`；training-to-CV worst-region improvement gap = **32.134 percentage points**，作为泛化风险提示，不参与单独选型；held-out gates 仍是主要判断依据。
- `C1_FEASIBILITY = MODERATE`：这是 FIT-only grouped-CV 结论，不是 Validation 结论。

## Per-frame result

逐 frame 的 C0/C1 global、Top/Middle/Bottom RMSE、P95、bias 及改善百分比见 `c1_per_frame_improvement.csv`；全部候选的 fold-level 明细见 `c1_grouped_cv_metrics.csv`。

## Provenance

- points SHA-256: `ea68251e05e1d472db7e25bb2090b2094f2e813f04dd5475fea1c06e4af01f8f`
- PCA summary SHA-256: `a01dca9079cf8985c4a7d3e97235a6e4d6249751ef633d3d2928ec3ab6a51c83`
- 由于本轮没有打开 Validation，是否 freeze C1 的 YES 仅表示值得送入下一轮独立 Validation，不表示可直接生产部署。
