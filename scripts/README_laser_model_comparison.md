# 基于激光三联图的三种激光表面模型拟合比较

## 1. 脚本做什么

脚本直接读取每个姿态的三张图：

1. `chess 001.tif`：正常曝光、关闭激光，用于检测棋盘格角点和计算该姿态的棋盘格平面；
2. `nolaser 001.tif`：低曝光、关闭激光，作为背景；
3. `laser 001.tif`：低曝光、开启激光。

脚本先计算 `laser - nolaser` 正差分，再用二维 Hessian/Steger 方法提取亚像素激光中心。每一个激光像素对应一条相机射线，射线与当前棋盘格平面求交后得到三维标定点。此步骤不使用旧的 `laser_plane.yaml`，因此不会把旧模型误差带入新模型。

然后拟合并比较：

- `global_plane`：传统全局平面；
- `quadratic_graph`：稳定二次曲面，自动选择最适合写成因变量的坐标轴；
- `circular_cone`：论文标准圆锥二次曲面的相机坐标一般形式，不强制论文中特定的单轴安装假设。

## 2. 安装依赖

建议在项目虚拟环境中执行：

```powershell
python -m pip install numpy opencv-python scipy pandas matplotlib pyyaml tabulate
```

`tabulate` 仅用于生成 Markdown 表格。

## 3. 修改配置

复制并编辑 `laser_model_fit_config.yaml`，至少确认：

- `intrinsics` 指向当前相机内参；
- `pattern_cols`、`pattern_rows` 是内角点数；
- `square_size_mm` 是相邻内角点的真实距离；
- 三联图命名模式；
- 训练集和验证集目录、编号。

训练集和验证集必须按完整图像姿态分开，不能将同一幅图中的点随机拆分。

配置中的 `default_model`（默认 `circular_cone`）决定正式选用哪一套参数；三种
模型始终都会拟合。命令行可用 `--model global_plane|quadratic_graph|circular_cone`
覆盖，旧别名 `plane_abcd` 等价于 `global_plane`。

## 4. 运行

```powershell
python fit_laser_models_from_triplets.py --config laser_model_fit_config.yaml
```

## 5. 主要输出

当前示例配置的输出目录是：

```text
../runs/0804/laser_model_comparison/
```

其中：

- `calibration_points.csv`：由棋盘格平面求得的真实三维激光点；
- `model_comparison.csv`：训练集和验证集总指标；
- `per_image_metrics.csv`：每幅图像的误差；
- `pointwise_model_errors.csv`：逐点误差；
- `models/*.yaml`：三个模型参数；
- `laser_model.yaml`：默认选用模型（当前默认是 `circular_cone`）；
- `laser_plane.yaml`：历史文件名兼容副本，内容与 `laser_model.yaml` 相同；
- `laser_model_selection.yaml`：模型选择和支持列表；
- `validation_error_vs_u.png`：误差随横向像素的变化；
- `validation_error_vs_v.png`：误差随纵向像素的变化；
- `validation_error_vs_depth.png`：误差随真实深度的变化；
- `previews/`：角点和激光中心提取预览；
- `comparison_report.md`：自动生成的汇总报告。

## 6. 最重要的评价指标

优先查看独立验证集中的：

```text
board_rmse_mm
board_p95_abs_mm
board_max_abs_mm
valid_rate
```

这里的 `board_error_mm` 不是单纯的“点到拟合曲面距离”，而是：

1. 使用待比较模型从激光像素反算三维点；
2. 计算该点到该幅图真实棋盘格平面的距离。

因此它最接近实际三维测量误差。

## 7. 如何判断结果

- 二次图曲面验证误差显著低于全局平面，且 `error_vs_u` 中的 U 形趋势消失：说明全视场存在稳定二次弯曲，二次曲面有效。
- 圆锥模型进一步优于二次图曲面：说明论文的圆锥光面假设更符合当前激光器。
- 圆锥半顶角顶到上限、顶点落到边界、有效求交率低：说明圆锥模型发生退化，不应使用。
- 训练误差降低但验证误差升高：模型过拟合或训练姿态覆盖不足。
- 三种模型都随深度出现同方向趋势：优先检查棋盘格实际格距、相机内参的深度尺度和三联图之间是否移动。

## 8. 与量块实验衔接

选出独立验证集最优模型后，应修改测量工具的射线求交模块，再用量块做独立验证。建议：

1. 所有量块放在相同横向位置；
2. 使用未参与模型拟合的数据；
3. 同时报告绝对高度误差与相邻高度差误差；
4. 检查误差对真实高度的线性斜率是否显著下降；
5. 量块左右两侧仍应拟合局部基准线，避免平台倾斜和零点偏移进入高度结果。
