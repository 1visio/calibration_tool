# Circular Cone 优化审计

## 审计范围与结论

本报告只做静态审计与既有产物核对，没有运行任何拟合、重优化或参数写回。事实范围为：

- 标定编排项目：`calibration_tool`；
- workflow 实际入口：`calibration/src/calibrate_laser_surface_models.py`；
- Circular Cone 实际拟合实现：`calibration_tool/scripts/fit_laser_models_from_triplets.py`；
- 0811 标定产物：`calibration_tool/projects/daheng/outputs/0811/laser_model`；
- 0813 paired PnP 诊断实际复用的线上重建实现：`linelaser_tool/laser_measurement_tool/reconstruction/reconstructor.py`。

最重要的审计结论如下。

1. 当前 Circular Cone 是 6 自由度模型：顶点 3、单位轴 2、半顶角 1。
2. 0811 拟合并不最小化 paired PnP 的重建点到真值平面 residual；它最小化训练三维点相对圆锥母线的一个加权、鲁棒 residual。重建点到棋盘平面的误差只在拟合完成后用于评价。
3. 0811 完整训练产物含 16,102 点，但 Cone 优化实际使用 2,988 点，即 18 帧各 166 点。该次运行中入选点的数值权重均为 1；一般实现则是“每帧总权重相等”，不是全局逐点恒等权。
4. 没有 image-region/edge 权重。每帧沿纵向 `v` 均匀抽样会保留线段两端，缓解中部原始点密度问题；但窄边缘区域仍按其样本数量占比进入 loss，而且 `soft_l1` 会降低大 residual 的影响。因此边缘系统误差仍可能被更大范围的中部数据压低影响力。
5. 0811 标定点实际覆盖约 `v=[241.998, 2731.978] px`，全幅图像为 3000 行；更靠近上下传感器边缘的区域没有进入当前 Cone 拟合，属于外推区域。
6. 后续 sensitivity / paired-PnP reoptimization 应直接调用公开的 `reconstruct_uv_to_ground()`，不得复制拟合脚本中的另一套 ray/cone 求交公式。

## 0. 实际调用链与数据边界

workflow 把 `laser_surface_models` 注册为模块 `calibrate_laser_surface_models`（`calibration_tool/calibration_tool/stages.py:41`），并把配置的 `calibration_src` 加到 `sys.path` 后导入、调用其 `main(argv)`（`stages.py:80-117`）。

`calibration/src/calibrate_laser_surface_models.py` 是薄入口；它通过文件路径动态加载 `calibration_tool/scripts/fit_laser_models_from_triplets.py`，然后直接转发到后者的 `main(argv)`（`calibration/src/calibrate_laser_surface_models.py:16-37`）。因此 Cone 的参数化、objective 和 optimizer 全部以该拟合脚本为准，`calibration/src` 中没有另一套实现。

0811 的 `stage_run.yaml` 记录了实际参数（`projects/daheng/outputs/0811/laser_model/stage_run.yaml:6-16`）：

```text
--config  calibration_tool/configs/laser_model_fit_config.daheng.yaml
--output-dir  calibration_tool/projects/daheng/outputs/0811/laser_model
--model  circular_cone
--laser-orientation  vertical
```

当前 0811 Cone 的拟合/验证数据不是 `extrinsics0813`：

- 0811 拟合：`projects/daheng/data/laser_plane/fit` 的 001–018；
- 0811 验证：`projects/daheng/data/laser_plane/validation` 的 019–024；
- 0813 paired PnP：`projects/daheng/data/extrinsics0813` 的 fit 001–010、validation 011–013，只用于当前冻结 Cone 的独立真值诊断，没有参与 0811 参数估计。

依据分别见 `configs/laser_model_fit_config.daheng.yaml:29-35` 与 `scripts/generate_paired_pnp_residual_diagnostics.py:46-80,813-840`。

## 1. Circular Cone 数学参数化

### 1.1 几何参数与方程

令：

- 顶点 `A = (A_x, A_y, A_z)`，单位 mm；
- 单位轴方向 `d = (d_x, d_y, d_z)`，满足 `||d||=1`；
- 半顶角 `theta_c`，满足 `0 < theta_c < 90°`；代码字段名为 `half_apex_angle_deg`，优化变量名为 `alpha`。

对相机坐标点 `X`，记 `q=X-A`、`a=q·d`。当前求交使用的圆锥隐式方程是：

```text
F(X; A,d,theta_c)
  = ((X-A)·d)^2 - cos(theta_c)^2 ||X-A||^2
  = 0
```

该方程本身描述双叶圆锥；线上求交再用 `a >= 0` 选择由 `d` 指向的物理单叶（`reconstructor.py:256-264,385-399`）。拟合完成后如果样本的中位轴向坐标为负，代码会翻转 `d`，使已输出的轴指向物理叶（`fit_laser_models_from_triplets.py:782-792`）。`sheet_sign` 没有写入参数文件，物理叶由轴的方向本身编码。

### 1.2 自由度

| 参数 | 存储标量数 | 约束 | 自由度 |
|---|---:|---|---:|
| apex `A` | 3 | 无内部等式约束 | 3 |
| axis `d` | 3 | `||d||=1` | 2 |
| half angle `theta_c` | 1 | 区间约束 | 1 |
| 合计 | 7 | 一个单位范数约束 | **6** |

内部 solver 不直接优化三个轴分量，而是优化轴的球坐标：

```text
d(theta_axis, phi_axis) = [
  sin(theta_axis) cos(phi_axis),
  sin(theta_axis) sin(phi_axis),
  cos(theta_axis)
]
```

所以内部参数向量为 6 维：

```text
Theta_solver = [theta_axis, phi_axis, A_x, A_y, A_z, alpha]
```

实现见 `fit_laser_models_from_triplets.py:665-677,695-699`。其中 `theta_axis`、`phi_axis`、`alpha` 均以 rad 传给 solver，顶点三分量以 mm 表示；输出时 `alpha` 转为 degree。

### 1.3 axis 是否显式归一化

是，但不同层的方式不同：

- 拟合时：`angles_to_vector()` 由三角函数生成单位向量，单位范数由参数化保证，不需要额外的单位范数 residual（`fit_laser_models_from_triplets.py:672-677`）。
- 初值转换时：`vector_to_angles()` 会先执行 `axis / ||axis||`（`fit_laser_models_from_triplets.py:665-669`）。
- 标定工具通用校验器：只校验 `axis_unit_camera` 非零，没有把文件内数值写回归一化（`calibration_tool/laser_models.py:223-243`）。
- 正式运行加载器：显式返回 `axis / ||axis||`（`linelaser_tool/laser_measurement_tool/calibration/config_loader.py:295-329`）。
- 正式求交函数：在使用前再次显式执行 `axis = axis / axis_norm`（`reconstructor.py:365-379`）。

0811 文件中轴范数为 `0.99999999999999989`，数值上等于 1。

## 2. 当前 calibration objective

### 2.1 标定点如何形成

每幅训练三联图先由棋盘角点 PnP 得到该帧棋盘平面，再提取激光中心像素 `(u,v)`。像素经 `cv2.undistortPoints()` 变为射线 `[x_n,y_n,1]`，然后与该帧棋盘平面求交，得到训练三维点 `P_i`（`fit_laser_models_from_triplets.py:435-449,929-978`）。

因此这些三维点不依赖旧激光模型，但依赖 0811 三联图自身的棋盘 PnP。它们不是 0813 strict paired PnP 数据。

### 2.2 实际最小化的 residual

对入选训练点 `P_i`：

```text
q_i      = P_i - A
a_i      = q_i · d                         # axial
r_i      = sqrt(max(||q_i||^2-a_i^2, 0))  # radial
e_i      = r_i / tan(alpha) - a_i
p_i      = gamma * min(a_i, 0)
gamma    = 2.0                             # negative_axial_penalty
```

传给 SciPy 的 residual vector 是：

```text
z(Theta) = concat(
  sqrt(w_i) * e_i,
  sqrt(w_i) * p_i
)
```

实现见 `fit_laser_models_from_triplets.py:695-708`。

`e_i=0` 等价于单叶圆锥母线关系 `r_i=a_i tan(alpha)`。它也可写成：

```text
e_i = (r_i cos(alpha) - a_i sin(alpha)) / sin(alpha)
```

即母线截面有符号正交距离再除以 `sin(alpha)`。由于缩放项也随待优化的 `alpha` 变化，它不是一个与参数无关缩放的固定 Euclidean point-to-surface loss。

特别注意，以下量都不是当前 Cone 拟合 objective：

- ray-cone 交点到棋盘真值平面的 residual；
- `evaluate_model()` 计算的 `board_error_mm`；
- `surface_distance()` 返回的隐式面一阶距离 `F/||grad F||`；
- 0813 paired PnP 的边缘 residual；
- validation 019–024 的任何指标。

这些量只用于拟合后的评价（`fit_laser_models_from_triplets.py:796-803,1006-1043,1239-1260`）。因此 0811 的 `optimizer_cost=0.759047...` 与报告中的 `train/validation board_rmse_mm` 不能互换解释。

### 2.3 robust loss

存在 robust loss。0811 配置是：

```yaml
loss: soft_l1
f_scale_mm: 0.10
```

见 `configs/laser_model_fit_config.daheng.yaml:64-68`。SciPy 对上面的每个 residual 分量应用：

```text
Cost(Theta) = 1/2 * C^2 * sum_j rho((z_j/C)^2)
C = 0.10
rho(s) = 2 * (sqrt(1+s) - 1)    # soft_l1
```

因此普通 Cone residual 和负轴向惩罚都经过 `soft_l1`；超过约 `0.10` 的分量会逐渐降低影响力。

### 2.4 point / frame / region weighting

一般实现的基础权重是：

```text
w_i_raw = 1 / n_frame(i)
w_i     = w_i_raw / mean(w_raw)
```

即每帧的总基础权重相同，同一帧内点等权（`fit_laser_models_from_triplets.py:492-496`）。提取响应 `response` 虽写入 `calibration_points.csv`，但没有用于拟合权重。

在进入 optimizer 前还有两级抽样：

1. 单帧提取最多 900 点；对于纵向激光，点先按 `v` 排序，再沿 `v` 均匀抽样（`fit_laser_models_from_triplets.py:335-380,404-432`）。
2. Cone 总拟合上限为 3000；对每帧再次按有序索引均匀抽样，配额为 `max(20, 3000 // frame_count)`（`fit_laser_models_from_triplets.py:716-727`）。

0811 的实际情况为：

| 项目 | 数值 |
|---|---:|
| 完整训练点 | 16,102 |
| 训练帧 | 18 |
| 各帧完整点数 | 822–900 |
| Cone 每帧入选点 | 166 |
| Cone optimizer 实际点数 | 2,988 |
| 实际 `w_i` | 全部为 1 |

所以对“每个 calibration point 是否等权”的准确回答是：

- 对 16,102 个产物点：否，其中 13,114 个根本没有进入 Cone optimizer；
- 对 2,988 个实际入选点：0811 本次是等权，因为每帧恰好都是 166 点；
- 对通用实现：是“帧等权”，如果某帧入选点更少，则该帧每个点权重更大。

没有按 `u`、`v`、图像边缘、中部、深度区间或 board region 做显式权重，也没有 edge oversampling/edge loss。每帧沿 `v` 均匀抽样能防止某一局部因原始采样更密而简单占满配额，并保留检测线段端点；但它没有保证“边缘区间”和“中部区间”总权重相等。窄边缘带天然比宽中部区域样本少，且边缘大 residual 又会被 `soft_l1` 降权，因此仍有被中部整体趋势掩盖的风险。

此外，0811 全部训练点的实际范围约为：

```text
u = [1836.559, 2131.671] px
v = [ 241.998, 2731.978] px
```

Cone 二次抽样保留了上述端点，但 `v<约242` 与 `v>约2732` 没有训练约束。若 0813 residual 的“边缘”落在这些区域，当前模型是在外推，不只是区域权重较小。

## 3. 当前 optimizer

### 3.1 初值来源与多起点

先在全部训练点上拟合 `PlaneModel`。该平面使用帧等权并执行 8 次基于 MAD scale 的迭代降权 SVD（`fit_laser_models_from_triplets.py:507-525`），然后作为 Cone 初值提示。

Cone 的初值组合为：

- axis：初始平面法向 `plane.normal` 及其反向，共 2 个；
- apex：
  - 原点到初始平面的垂足 `-plane.d * plane.normal`；
  - 相机原点 `[0,0,0]`；
  - 平面垂足沿训练点第一主方向 `+100 mm`；
  - 平面垂足沿同一方向 `-100 mm`；
- half angle：配置值 `89.0°`、固定 `85.0°`、固定 `89.8°`，共 3 个。

0811 没有配置 `apex_initial_mm`，所以总计 `2 × 4 × 3 = 24` 个起点。若未来设置 `apex_initial_mm`，会再增加一组 apex 起点。实现见 `fit_laser_models_from_triplets.py:729-761`。

### 3.2 solver

调用 `scipy.optimize.least_squares()`（`fit_laser_models_from_triplets.py:763-772`）。代码没有显式传入 `method` 或 `jac`：

- solver method：SciPy 默认 `trf`（Trust Region Reflective，支持 bounds）；
- Jacobian：默认 `2-point` 数值差分；
- 多起点选择：取 `result.cost` 最小者，而不是按 validation 指标选择；
- 当前产物：`fit_success: true`，`optimizer_cost: 0.7590472395097801`。

代码对候选的过滤条件较宽：只有 `not result.success and result.cost <= 0` 才跳过；因此一般情况下，一个 `success=false` 但正 cost 的结果仍可能成为 `best`。0811 最终结果本身为 `success=true`（`fit_laser_models_from_triplets.py:773-794`）。

### 3.3 bounds

0811 的 solver bounds 为：

```text
theta_axis ∈ [0, pi]
phi_axis   ∈ [-pi, pi]
A_x        ∈ [-1000, 1000] mm
A_y        ∈ [-1000, 1000] mm
A_z        ∈ [ -500,  500] mm
alpha      ∈ [60.0°, 89.95°]
```

见 `fit_laser_models_from_triplets.py:733-739` 与 `configs/laser_model_fit_config.daheng.yaml:54-68`。apex 初值会先裁剪到 bounds 内部。0811 输出 apex 没有落在 bounds 上，半顶角 `89.07255°` 也没有达到 `89.95°` 上界；但它已接近圆锥趋向平面的几何极限，apex/axis/angle 的相关性值得后续 sensitivity 单独检查。

### 3.4 normalization 与 regularization

- axis：单位范数由球坐标参数化严格保证。
- frame weights：归一化到平均权重为 1。
- point coordinates：没有中心化或尺度归一化。
- optimizer parameters：mm 与 rad 混合；没有显式 `x_scale`，当前审计环境 SciPy 1.15.3 的默认值为 `1.0`。
- regularization：没有 apex、axis、angle 的先验或正则项。
- `negative_axial_penalty=2.0` 是物理单叶软惩罚，不是参数正则化。
- 没有解析 Jacobian、协方差、Hessian/condition number 或参数不确定度计算。

### 3.5 convergence criteria 与可追溯性

代码只显式设置：

```text
max_nfev = 3000    # 每个起点
verbose = 0
```

没有显式设置 `ftol`、`xtol`、`gtol`。本次审计环境的 SciPy 1.15.3 默认值均为 `1e-8`，对应 cost change、step change 和一阶 optimality 的停止条件；默认还有 `method='trf'`、`jac='2-point'`、`x_scale=1.0`。

但是 0811 产物没有记录 SciPy 版本、`status`、`message`、`nfev`、`njev`、`optimality`、active bounds 或最终 Jacobian，因此只能确认代码采用默认 tolerance 和最终 `success=true`，不能从现有产物还原究竟是哪一条停止条件触发，也不能据此判断参数是否充分可辨识。

## 4. 正式 pixel → ray → cone intersection → XYZ 函数

### 4.1 必须复用的公开入口

正式公开函数是：

```python
reconstruction.reconstructor.reconstruct_uv_to_ground(
    pixels_uv,
    calibration,
    params=None,
) -> ReconstructionResult
```

定义于 `linelaser_tool/laser_measurement_tool/reconstruction/reconstructor.py:448-531`，并由 `reconstruction/__init__.py:3-20` 公开导出。后续 sensitivity / reoptimization 应直接复用这个函数；需要相机坐标 XYZ 时读取返回值的 `points_camera`，需要 paired PnP 地面系 residual 时读取 `points_ground`。

完整正式调用链为：

```text
pixels_uv (u,v)
  -> cv2.undistortPoints(K,D)
  -> rays = [x_n, y_n, 1]
  -> _intersect_laser_surface(...)
  -> _intersect_circular_cone(...)
  -> _solve_quadratic_all(...) + _choose_roots(...)
  -> lambda
  -> points_camera = rays * lambda
  -> T_ground_from_camera
  -> points_ground
  -> optional ground compensation
```

行号依据：

- 像素去畸变和 ray：`reconstructor.py:478-487`；
- 模型分派：`reconstructor.py:423-445`；
- Circular Cone 二次方程：`reconstructor.py:360-401`；
- 稳定二次求根：`reconstructor.py:178-215`；
- 物理选根：`reconstructor.py:218-265`；
- 相机/地面 XYZ：`reconstructor.py:489-529`。

这里的 ray 没有做 Euclidean 单位归一化，而是固定 `ray_z=1`，所以 `lambda` 正好是相机深度 `Zc`（`reconstructor.py:226-229`）。这属于正式接口语义，后续不能把 `lambda` 错当成沿单位射线的欧氏距离。

0811 正式运行配置还施加：

```text
min_camera_depth_mm = 630
max_camera_depth_mm = 715
model_range_margin_mm = 2
z_valid_range_mm = [631.5503788651798, 713.6828851676036]
ground_u_compensation = null
```

见 `configs/measure_tool_daheng_0811.yaml:7-35`。求根还要求正深度、落在工作距离/模型范围内、属于 axis 的正向物理叶；若有多个候选，选择最接近模型 `z_valid_range` 中点的根。

0813 paired PnP 诊断已经按正式方式复用：它从 `linelaser_tool/laser_measurement_tool` 导入 `load_calibration_files()` 和 `reconstruct_uv_to_ground()`，并在每帧调用后用 `points_ground` 计算 PnP 平面 residual（`generate_paired_pnp_residual_diagnostics.py:30-43,813-846`）。

### 4.2 不应作为后续正式复用入口的重复实现

拟合脚本还存在：

- `pixels_to_rays()`（`fit_laser_models_from_triplets.py:435-438`）；
- `CircularConeModel.intersect_rays()`（`fit_laser_models_from_triplets.py:805-816`）；
- `choose_roots()`（`fit_laser_models_from_triplets.py:852-879`）。

它们是当前拟合/比较脚本内部实现，不是线上公开 API，而且选根规则与正式运行路径不同。例如拟合评价使用初始平面的逐点 `lambda_hint`，并至少外扩 `z_valid_range` 50 mm；正式运行使用运行配置的工作距离、2 mm margin 和模型范围中点。拟合版本在没有正向叶根时还可能保留反向候选，而正式版本会判为无有效交点。

因此后续研究若复制或直接调用拟合脚本内部求交，得到的 edge residual 可能与实际线上链路不一致。最小正式复用单元必须是公开的 `reconstruct_uv_to_ground()`；其下划线函数只用于说明内部链路，不应作为稳定外部接口导入。

## 5. 现有 Cone 参数文件与字段

### 5.1 0811 原始拟合产物

文件：

```text
calibration_tool/projects/daheng/outputs/0811/laser_model/models/circular_cone.yaml
```

字段：

| 字段 | 当前值/含义 |
|---|---|
| `model_type` | `circular_cone` |
| `description` | 一般刚体变换后的标准圆锥说明 |
| `axis_unit_camera` | `[0.9596376594680045, 0.009625633081993221, -0.28107456256043695]` |
| `apex_camera_mm` | `[-115.59970742084231, 1.7418894155446691, 327.03307376521605]` mm |
| `half_apex_angle_deg` | `89.07255025246596` degree |
| `fit_success` | `true` |
| `optimizer_cost` | `0.7590472395097801`，SciPy robust cost，不是 RMSE |
| `z_valid_range_mm` | `[631.5503788651798, 713.6828851676036]` |

字段由 `CircularConeModel.to_dict()` 写出（`fit_laser_models_from_triplets.py:818-828`）。

### 5.2 0811 stage 选中产物

文件：

```text
calibration_tool/projects/daheng/outputs/0811/laser_model/laser_model.yaml
calibration_tool/projects/daheng/outputs/0811/laser_model/laser_plane.yaml
```

两者是相同的选中模型文档；`laser_plane.yaml` 只是历史文件名兼容，内容仍是 Circular Cone。它们包含上述核心字段，并额外包含：

- `laser.orientation`；
- `model_selection.default_model/supported_models/source`；
- `metrics.train`；
- `metrics.validation`。

`laser_model_selection.yaml` 只记录选中类型和文件名，不包含几何参数。

### 5.3 当前 0813 诊断/测量实际加载的部署文件

部署 Cone 文件：

```text
linelaser_tool/laser_measurement_tool/configs/calibration_daheng_0811/circular_cone.yaml
```

选择它的应用配置：

```text
linelaser_tool/laser_measurement_tool/configs/measure_tool_daheng_0811.yaml
```

部署文件的核心几何数值与 0811 原始产物相同，额外加入：

- `schema_version`；
- `source_calibration_run`；
- `coordinate_system: camera`；
- `units: mm`；
- `quality.validation_*`。

部署文件保留 `fit_success` 和 `z_valid_range_mm`，但不含 `optimizer_cost`。正式 loader 必需并使用的几何字段是 `model_type`、`axis_unit_camera`、`apex_camera_mm`、`half_apex_angle_deg`；`fit_success=false` 会被拒绝，`z_valid_range_mm` 若存在则参与选根（`calibration/config_loader.py:287-335`）。加载后还会增加内存字段 `source_path`，并把 axis 归一化。

## 6. 对下一阶段研究的直接含义

当前审计不能仅凭静态代码判定“参数估计不佳”还是“Circular Cone 形式不足”，但已确认后续实验必须控制以下混杂因素：

1. 现有参数不是在 0813 paired PnP objective 上估计的；先固定模型形式，仅在 paired PnP fit 001–010 上重估 6 个参数，才能与原参数公平比较。
2. sensitivity 和 reoptimization 的 loss 必须明确选择：若研究的是实际测量 residual，应最小化正式重建输出到 paired PnP 平面的 residual，而不是继续使用 `r/tan(alpha)-a` 的训练点表面 residual。
3. 需固定 validation 011–013，不能参与参数、权重、bounds、robust scale 或停止规则选择。
4. 需分别报告全局等权、帧等权、region 等权/edge-aware weighting；否则“参数是否可改善边缘”会与 weighting 混在一起。
5. 必须在内存中替换候选 `calibration["laser_model"]` 并调用 `reconstruct_uv_to_ground()`，候选参数输出到新的实验目录；不得覆盖 0811 原始产物或当前部署 YAML。
6. 当前半顶角接近 90°，且 optimizer 没有参数尺度归一化、解析 Jacobian或 condition/covariance 输出；参数 sensitivity 至少应检查局部 Jacobian/SVD、profile loss 和多起点一致性。

## 7. 最终清单

- **当前参数向量 `Theta`**

  - 几何形式：`Theta_geom = [A_x, A_y, A_z, d_x, d_y, d_z, theta_c]`，带 `||d||=1`，所以是 6 DoF。
  - solver 形式：`Theta_solver = [theta_axis, phi_axis, A_x, A_y, A_z, alpha]`，6 个无等式约束变量。
  - 当前存储值：

    ```text
    A = [-115.59970742084231, 1.7418894155446691, 327.03307376521605] mm
    d = [0.9596376594680045, 0.009625633081993221, -0.28107456256043695]
    theta_c = 89.07255025246596 degree
    ```

  - 由存储值等价转换的 solver 坐标：

    ```text
    [1.8557099550137575,
     0.01003015064354381,
     -115.59970742084231,
     1.7418894155446691,
     327.03307376521605,
     1.5546092750536373]
    # angle units: rad; apex units: mm
    ```

- **当前 loss**

  ```text
  z = concat(sqrt(w_i)*(r_i/tan(alpha)-a_i),
             sqrt(w_i)*2*min(a_i,0))
  least_squares(loss="soft_l1", f_scale=0.10 mm)
  ```

  该 loss 在 0811 train 001–018 的 3D 标定点上拟合，不是 paired PnP 重建平面 residual。

- **当前 weighting**

  通用规则是每帧总基础权重相等、帧内等权；0811 实际先从 16,102 点均匀抽成 2,988 点（18 帧各 166），所以入选点全部 `w_i=1`。无 response、image-region 或 edge 权重；另有 `soft_l1` residual 降权。

- **最小可复用函数**

  `linelaser_tool/laser_measurement_tool/reconstruction/reconstructor.py` 的公开函数 `reconstruct_uv_to_ground()`；相机 XYZ 取 `ReconstructionResult.points_camera`，地面 XYZ 取 `points_ground`。不得另写 pixel-to-ray / ray-cone 公式，也不应把私有 `_intersect_circular_cone()` 当外部稳定 API。

- **后续 sensitivity / reoptimization 需要改哪些文件**

  - 必须新增实验入口（建议新文件而不是覆盖既有诊断）：`calibration_tool/scripts/circular_cone_sensitivity.py` 和/或 `calibration_tool/scripts/reoptimize_circular_cone_paired_pnp.py`，加载 fit 001–010、冻结 validation 011–013，并直接调用正式 `reconstruct_uv_to_ground()`。
  - 必须新增独立实验配置，例如 `calibration_tool/configs/circular_cone_reoptimization.daheng-0813.yaml`，显式记录 bounds、loss、f_scale、weighting、solver tolerances、seed/多起点和输出目录；不要改写用于复现 0811 的 `laser_model_fit_config.daheng.yaml`。
  - 若决定把新 objective 合并回现有标定器，需修改 `calibration_tool/scripts/fit_laser_models_from_triplets.py` 中 `CircularConeModel._residual()`、`fit()` 和数据入口；`calibration/src/calibrate_laser_surface_models.py` 仍只是转发器，除非 CLI/入口协议变化，否则无需修改。
  - 新增对应测试到 `calibration_tool/tests/`，至少覆盖参数 round-trip、固定 residual 长度、无效求交惩罚、frame/region weighting、fit/validation 隔离和不写回 0811 产物。
  - `linelaser_tool/laser_measurement_tool/reconstruction/reconstructor.py` 与 `calibration/config_loader.py` 在“只做参数 sensitivity/reoptimization”时应保持不变并被复用；只有正式 API 本身确需扩展时，才连同 `linelaser_tool/laser_measurement_tool/tests/test_reconstructor.py` 一起修改。
  - 所有候选参数和报告必须写入新的实验输出目录；不得修改 `projects/daheng/outputs/0811`、`configs/calibration_daheng_0811/circular_cone.yaml` 或任何现有正式标定结果。
