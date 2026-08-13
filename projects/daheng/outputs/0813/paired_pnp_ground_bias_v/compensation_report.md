# Experimental paired-PnP ground bias v compensation

## 最终判断

**B_V_COMPENSATION = PARTIAL**

- 3个独立 holdout 中，RMSE/P95 同时改善：3/3。
- 明显新局部过补偿 block：1。
- 聚合 holdout RMSE：0.158674 → 0.070008 mm；P95：0.263984 → 0.134587 mm。

PASS 要求至少2/3 holdout帧的 RMSE 与 P95 同时改善、聚合 RMSE/P95 同时改善，且没有明显新局部过补偿。局部过补偿判据在查看 holdout 前固定为：100 px block、至少30点，且 RMSE 增加同时达到25%和0.03 mm，或 P95增加同时达到25%和0.05 mm。

工程价值判断：可信工作区内的一维 b(v) 具有显著但有条件的工程价值——三帧整体RMSE/P95均改善，说明主要系统分量确实可被一维模型消除；但013在v=300–399出现新局部恶化，且012的全局P-V轻微增加，因此当前只适合作为support-gated实验方案，还不能作为正式生产补偿。

## 严格数据隔离与建表

- BUILD：001–010；INDEPENDENT HOLDOUT：011–013。
- holdout 未参与 LUT估计、support threshold、平滑或任何判据选择。
- `b(v)` 是每帧同一整数 v 内 residual median 的跨 BUILD 帧 median；无跨帧 outlier threshold，`smooth_window=1`。
- 高精度工作区固定为 `300≤v≤2699`；可靠支持固定为 `BUILD sample_count≥5`。支持覆盖 2046/2400 (85.25%)。
- 工作区外、上下边缘和内部 support 缺口完全不补偿；不插值、不跨缺口、不外推。

## Holdout逐帧结果（只在可靠支持样本上比较）

| frame | samples | Bias before→after | MAE before→after | RMSE before→after | P95 before→after | P-V before→after | std before→after | sign mixing before→after | local overcomp blocks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 011 | 1149 | -0.138757→-0.043149 | 0.138980→0.047070 | 0.153228→0.057940 | 0.254296→0.109717 | 0.384652→0.297078 | 0.065031→0.038684 | 0.013925→0.233246 | 0 |
| 012 | 1806 | -0.126398→-0.037390 | 0.126744→0.046683 | 0.142380→0.059391 | 0.246415→0.113667 | 0.488290→0.500267 | 0.065559→0.046157 | 0.003322→0.420819 | 0 |
| 013 | 1059 | -0.176436→-0.081483 | 0.176693→0.084176 | 0.188110→0.094442 | 0.290446→0.157015 | 0.411058→0.366022 | 0.065267→0.047770 | 0.013220→0.062323 | 1 |

## 明显新局部过补偿

- frame 013，v=300–399，n=54：RMSE 0.085011→0.122460 mm，P95 0.139566→0.165643 mm；trigger=RMSE。

`sign_mixing_index = 2*min(positive_fraction, 1-positive_fraction)`，0表示单一符号，1表示正负各半。另在 JSON 中报告同一 v 跨3个holdout帧的 mixed-sign row fraction。

## 工程边界

- 这是 experimental table，不是正式 runtime LUT。CSV/NPY 显式包含 support mask；现有运行时 `np.interp` 会跨内部缺口插值，因此不得直接把本表接入该路径。
- 实验应用使用整数 v 的 exact lookup；只有该行 support=true 才减去 bias。
- 未补偿点保持原始 Z，图中白色区域不参与 before/after 精度宣称。
- 补偿后 sign mixing 上升主要反映 residual 从单侧负偏置移向零点两侧；它与Bias/MAE/RMSE/P95/std及局部过补偿需要联合解释，不能单独作为失败指标。
