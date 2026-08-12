# Steger 2D Candidate / Scan-Axis 解耦审计

> 审计范围：`calibration_tool` `main`，HEAD `9f0f337`；共享实现为
> `D:/Docs/linelaserscan/calibration/src/realtime_steger.py`。
>
> 本轮只新增本文档，不修改 Steger、calibration、online 或 conditional
> search-region resolver 的代码与行为。

## 1. 结论

| 问题 | 结论 |
| --- | --- |
| Gaussian/Hessian 是否二维 | 是。五组一、二阶 Gaussian 导数覆盖 `x/y` 和混合项，Hessian 特征分解、法向导数与亚像素位移均在二维像素网格上计算。|
| `scan_axis` 从哪里开始影响 | 数值行为第一次受影响是在公共入口把 `row` 图像转置为 columnwise 工作域；因此它先影响 auto detector 和 band crop，随后才决定按哪个原图方向做每 scanline 压缩。|
| 最强 candidate 与 Hessian 能否拆开 | 可以。准确切点在二维 `valid/response/offset` 场生成之后、`argmax(axis=0)` 之前。|
| 能否新增 `extract_steger_candidates_2d()` | 可以，且适合做共享数学接口。它应返回全部通过强度、负曲率/response 和二维 subpixel-offset gate 的稀疏候选，不执行 scanline 选择。|
| calibration 新数据流是否可行 | 可行。建议为 `rectangular ROI + halo -> 2D candidates -> polygon/mask -> continuity/association -> final line`。|
| online 是否可保留快速路径 | 可以。online 继续使用 dense field 直接逐 scanline `argmax`，不物化全部稀疏候选，也不引入 polygon/continuity。|

关键判断是：**二维 Steger 数学与 scan-axis 无内在耦合；当前耦合发生在 search-band
坐标适配和候选降维阶段。** 因此无需复制或改写 Hessian 公式。

## 2. 当前实现的准确边界

当前正式路径为：

```text
original image + params
  -> scan_axis=row: transpose to columnwise working image
  -> 1D normal-axis intensity detector
  -> auto/additional region envelope merge
  -> contiguous rectangular band crop
  -> five 2D Gaussian derivatives
  -> 2D Hessian eigenvalue / normal
  -> directional first/second derivative
  -> 2D subpixel offset and per-pixel gates
  -> argmax(axis=0): one strongest candidate per working-domain column
  -> scan_axis=row: restore original (u, v)
```

对应证据：

- 参数与输入检查：`realtime_steger.py:888-896`；
- `row` 转置：`realtime_steger.py:950-975`；
- detector、region merge、band crop：`realtime_steger.py:658-738`；
- 2D derivative/Hessian/subpixel：`realtime_steger.py:739-786`；
- 每列最强候选：`realtime_steger.py:787-805`；
- row 坐标恢复：`realtime_steger.py:933-947`；
- online 兼容快速入口：`realtime_steger.py:1010-1032`。

conditional resolver 当前只生成 metadata proposal；正式 Gaussian/Hessian 仍只使用
`band_bounds`。这一边界由 `realtime_steger.py:724-727` 明确保证，本审计不建议在本轮
改变它。

## 3. Gaussian/Hessian 是完整二维计算

### 3.1 导数覆盖

`realtime_steger.py:739-746` 对同一个二维 band 计算：

```text
ry  = order (1, 0)    rx  = order (0, 1)
ryy = order (2, 0)    rxx = order (0, 2)
rxy = order (1, 1)
```

`scipy.ndimage.gaussian_filter` 内部可以使用可分离卷积实现，但其语义仍是二维 Gaussian
导数，不是“每 scanline 上的一维 Steger”。特别是 `rxy` 明确保留了两个图像轴的耦合。

`realtime_steger.py:748-779` 随后在每个二维像素上：

1. 从 `rxx/rxy/ryy` 求两个 Hessian 特征值；
2. 选择绝对值较大的主特征值及其法向；
3. 计算法向一阶导和法向二阶导；
4. 计算 `offset = -first_derivative / second_derivative`；
5. 投影为二维 `offset_x/offset_y`。

因此 ridge response 和 subpixel center 在 scan-axis 选择之前已经是二维量。

### 3.2 当前 per-pixel gate

`realtime_steger.py:780-786` 的正式候选条件是：

```text
normal is numerically valid
second_derivative < -deriv_thresh
abs(offset_x) <= 0.6 px
abs(offset_y) <= 0.6 px
band intensity >= threshold
```

可将其更清楚地拆成：

- derivative/ridge-sign：法向有效且法向二阶导为负；
- ridge-response：`-second_derivative > deriv_thresh`；
- subpixel：两个坐标分量均有限且不超过 `0.6 px`；
- intensity：中心锚点灰度达到 `threshold`。

当前没有额外的二维 NMS 或局部极大值检查。一个 scanline 可以先产生多个合法像素候选，
只是它们马上被后续 `argmax` 压成一个。

### 3.3 仍需注意的 ROI 边界效应

二维不等于“与 crop 无关”。实现先执行 `band = image[top:bottom]`，再在 band 上调用
Gaussian filter。默认边界模式因此作用于裁剪边界；真实 ridge 距边界太近时，Hessian、法向
和 offset 都可能变化。

所以未来 board bounding ROI 必须区分：

- **compute ROI**：供 Gaussian/Hessian 使用，包含足够 halo；
- **accept ROI**：真正允许输出候选的 board bounding rectangle/polygon。

不能先把 polygon 外像素清零再做 Hessian；硬 mask 边缘本身会产生强导数和伪 ridge。

## 4. `scan_axis` 的实际耦合位置

### 4.1 首次影响

`extract_steger()` 在 `realtime_steger.py:962-966` 执行：

```text
column -> working_image = image
row    -> working_image = contiguous(image.T)
```

这是 `scan_axis` 第一次改变数值输入。之后 `_extract_columnwise()` 始终认为：

- axis 0 是 search/normal axis；
- axis 1 是 scanline index；
- 在每个 axis-1 位置选择一个 axis-0 候选。

因此 `row` 从 formal Hessian 之前就已经影响 detector、region extent 和 band crop；但这只是
坐标适配，不是另一套 Hessian 数学。`_build_detector_summary()` 中 `normal_axis` 的 `u/v`
选择（`realtime_steger.py:314-320`）只影响 metadata 命名，真正的数值变化来自上述转置。

### 4.2 候选压缩才是 scan-axis 算法策略

`realtime_steger.py:787-792`：

```text
strength = valid ? -second_derivative : -inf
best_row = argmax(strength, axis=0)
```

这一步才把二维候选场解释为“一条激光线”：column 模式每个原图 `u` 留一个 `v`；row
模式在转置域做同样操作，恢复后等价于每个原图 `v` 留一个 `u`。

该选择：

- 不参与 Gaussian/Hessian；
- 不反馈 derivative、response 或 offset gate；
- 不使用相邻 scanline continuity；
- 响应并列时由 `np.argmax` 保留第一个位置。

因此拆分不会改变数学，只会把候选场的消费策略从核心中显式分离出来。

## 5. 建议的共享 2D candidate 接口

### 5.1 对外契约

建议新增 axis-free、opt-in 接口：

```python
@dataclass(frozen=True, slots=True)
class StegerCandidates2D:
    u_px: np.ndarray                 # float64, shape (K,), original-image coordinates
    v_px: np.ndarray                 # float64, shape (K,)
    anchor_u_px: np.ndarray          # int32, Hessian sample pixel
    anchor_v_px: np.ndarray          # int32
    response: np.ndarray             # float32, -second_derivative
    first_derivative: np.ndarray     # float32, along normal
    second_derivative: np.ndarray    # float32, along normal
    offset_u_px: np.ndarray          # float32
    offset_v_px: np.ndarray          # float32
    normal_u: np.ndarray             # float32
    normal_v: np.ndarray             # float32
    metadata: dict[str, Any]

def extract_steger_candidates_2d(
    gray: np.ndarray,
    options: Mapping[str, Any] | StegerKernelParams,
    *,
    compute_roi: ImageRect | None = None,
    accept_roi: ImageRect | None = None,
) -> StegerCandidates2D:
    ...
```

接口语义：

1. 不接受、不解释 `scan_axis`；二维候选不存在 column/row 方向。
2. 返回全部通过当前 intensity、derivative/ridge-response、subpixel gate 的候选，不做
   per-scanline argmax、continuity、RANSAC 或 geometry gate。
3. 返回坐标始终是原输入图 `(u, v)`，即便内部只计算局部矩形。
4. `anchor_*` 用于无歧义地索引离散 mask；`u_px/v_px` 用于 polygon point test 和后续几何。
5. 不返回 dense Hessian/derivative 图，避免让大数组逃逸到调用者；只返回 `K` 个稀疏记录。
6. `compute_roi` 必须覆盖 `accept_roi + Gaussian halo`；metadata 记录二者、clamp 和 halo。

参数也应拆开：二维 kernel 只需要 `sigma`、intensity threshold、minimum ridge response 和
maximum subpixel offset。当前名为 `deriv_thresh` 的参数实际门控负法向二阶导幅值，可在兼容
层映射为 `min_ridge_response`；`roi_margin/roi_max_height/scan_axis` 属于 region/selection
策略，不属于 2D kernel。

### 5.2 内部复用边界

建议的内部结构是“一次 dense 计算、两个消费者”：

```text
_compute_steger_fields_2d(rectangular band, kernel params)
  -> valid, response, first/second derivative, normal, offset_u/v
       |
       +-> sparse pack -> extract_steger_candidates_2d()       [calibration]
       |
       +-> argmax on dense response per scanline               [online/current API]
```

这样既不会复制 Hessian 公式，也不会让 online 为全部候选执行 `nonzero`、稀疏数组分配、
分组、排序或 continuity。两条路径都只能调用一次 formal Gaussian/Hessian。

不建议让 `extract_steger_candidates_2d()` 自己调用 conditional resolver。它应消费调用方已经
决定的矩形计算域；resolver policy 与二维数学保持正交。现有 auto-band 路径也可继续只供
online adapter 使用。

## 6. Calibration 数据流可行性

当前 laser-plane 路径对完整 `corrected` 调用逐列 extractor，随后才执行
`pose.roi_polygon` 和 `chess_boundary_mask`（
`calibrate_laser_plane_core_v2.py:212-259`）。这会在 polygon gate 之前丢掉同一列的其它二维
候选：如果 polygon 外的假 ridge 更强，polygon 拒绝它后已经没有次强真 ridge 可恢复。

建议的数据流为：

```text
corrected large ROI / board bounding ROI
  -> expand rectangular compute ROI by Gaussian/safety halo
  -> extract_steger_candidates_2d() once
  -> keep subpixel centres inside board polygon
  -> reject anchor pixels in chess_boundary_mask
  -> continuity / candidate association
  -> at most one final point per chosen scanline
  -> existing board-plane intersection and laser-plane fitting
```

可行性判断：**可行，而且比“先每列最强、再 polygon/mask”更符合 calibration 的先验顺序。**

落地时需要另行确定的 policy（本审计不替其做算法选择）：

- continuity 是局部 segment、稳健多项式还是动态规划；
- 多条连续 ridge 并存时的选择准则；
- board polygon 边缘需要多大 accept margin；
- 无连续路径时是失败、降级为 per-scanline strongest，还是扩大 ROI 重试。

仓库已有一条概念验证：
`scripts/fit_laser_models_from_triplets.py:284-332` 先产生全体二维候选，
`:335-380` 再按 orientation 做 per-scanline 选择与连续性。但它是独立 float64 全图实现，
不应成为第二套共享数学；应只借鉴数据流，将公式统一到 `realtime_steger.py` 的 float32
kernel。

Ground/reconstruction 当前则在逐列压缩后才做 continuity/RANSAC，例如
`calibrate_ground_extrinsics_steger_v2.py:127-168`。它们可继续使用兼容路径；无需为了新增
calibration 2D 接口同步改动所有消费者。

## 7. Online 快速路径可以原样保留

`steger_backend()` 在无 explicit region 时已有专门轻量分支（
`realtime_steger.py:1018-1027`）：column 直接调用 `_extract_columnwise()`，row 只做转置、
columnwise extraction 和坐标交换。

建议的兼容约束：

1. `steger_backend()` 的公开签名与返回 `N x 2` 点数组不变；
2. auto detector、band merge 和 per-scanline dense `argmax` 的顺序不变；
3. online 不调用 `extract_steger_candidates_2d()`，只复用内部 2D field kernel；
4. row 继续由 adapter 转置/恢复，2D kernel 不认识 `scan_axis`；
5. 不增加第二次 Hessian；
6. 无 hint 的 column/row 输出保持逐点一致。

现有 `tests/test_laser_search_region.py:57-126` 已覆盖 unified/legacy/backend 的 column 数组
一致性、row 转置对称性和原图 normal-axis region 语义，可作为未来拆分的首要回归基线。

## 8. 计算量与内存粗评

### 8.1 复杂度模型

令输入为 `H x W`，正式二维计算矩形为 `Hc x Wc`，`N = Hc*Wc`，Gaussian kernel
半径与 `sigma` 成正比：

- 强度 detector：当前为 `O(HW)`；即使 formal band 很窄，也要扫描完整输入；
- 五组二维 Gaussian 导数：约 `O(5 * N * kernel_width)`，是主要耗时；
- Hessian algebra、gate、dense argmax：`O(N)`；
- 稀疏候选打包：`O(N + K)`，`K` 为 gate 后候选数；
- polygon/mask：`O(K)`；
- continuity：取决于实现，合理分桶后可做到近似 `O(K)` 或 `O(K log K)`。

因此真正降低 formal Steger 成本的是**在卷积前缩小矩形 compute ROI**。polygon/mask 若只在
候选产生后执行，只降低后处理和 `K`，不会降低 Gaussian/Hessian 成本。

### 8.2 三种输入域

以仓库真实全幅 `2048 x 2448` 和在线默认 hardware ROI `300 x 2448` 为参照。board
bounding ROI 没有固定尺寸，表中用 `512 x 2448` 作为保守示例，只用于展示线性比例，不代表
某块棋盘的实测 bbox。

| 场景 | 像素数 | 相对 full image | 适用判断 |
| --- | ---: | ---: | --- |
| full image `2048 x 2448` | 5,013,504 | 100% | 最大覆盖；2D candidate 全图卷积最贵。当前 auto 路径虽读全图 detector，formal Hessian 通常只在最多 512 高的 band。|
| board bbox 示例 `512 x 2448` | 1,253,376 | 25.0% | formal kernel 约为全图四分之一；实际收益按真实 bbox 面积线性变化。必须加 halo。|
| hardware ROI `300 x 2448` | 734,400 | 14.6% | 输入本身已由相机裁剪；当前 online formal band 还可能小于 300。坐标在 Steger 后再加 hardware offset。|

对 2D candidate calibration 路径，若只对 board bbox 工作，必须在转换为 float32 **之前**裁出
局部 compute ROI，才能同时节省转换、卷积和内存。当前 `_extract_columnwise()` 是先把完整
输入转为 float32，再做 normal-axis band crop（`realtime_steger.py:670-738`），这一点不能
直接视为任意二维 bbox 已经优化。

### 8.3 内存

一张 float32 图每像素 4 bytes。五个可复用 derivative workspace（
`realtime_steger.py:28-58,739-746`）需要 `20*N` bytes；再加 float32 working image 和
contiguous band，当前最低常驻量约为 `28*N` bytes（full-domain 情况），尚未包含 Hessian
algebra 临时数组。

当前 `root/midpoint/eigenvalue/normal/derivative/offset/strength` 等大量 dense float32 数组在
同一函数作用域内同时存活，实际峰值可粗看作每像素约 100～130 bytes，再加 SciPy 临时缓冲、
原始输入和 Python/NumPy 对象开销：

| 场景 | 单张 float32 plane | 5 derivative buffers | 最低内部常驻量 | 粗略 practical peak |
| --- | ---: | ---: | ---: | ---: |
| full `2048 x 2448` | 19.1 MiB | 95.6 MiB | 133.9 MiB | 约 0.5～0.7 GiB |
| board 示例 `512 x 2448` | 4.8 MiB | 23.9 MiB | 33.5 MiB | 约 125～175 MiB |
| hardware `300 x 2448` | 2.8 MiB | 14.0 MiB | 19.6 MiB | 约 73～103 MiB |

这些是量级估算，不是本轮新增 benchmark；NumPy/SciPy 版本、临时数组释放时机和是否启用
diagnostics 会改变峰值。`row` 公共结构化路径还需要 contiguous transpose；online 的轻量
backend 不恢复整张 `corrected_signal`，因此额外成本较低。

稀疏 `StegerCandidates2D` 本身建议控制在约 48～64 bytes/candidate。典型单条线的 `K`
远小于 ROI 像素数，开销很小；棋盘边缘或反射很多时 `K` 会增大，极端最坏情况可接近 `N`。
所以接口不得把全部 dense Hessian planes 一并返回。

已有 GUI 测量可作为时间量级旁证：真实 `2048 x 2448` Mono8 图像，预热 30 次后，加入一次
共享 Steger 的 GUI quality P50/P95 从 `301.5/367.3 ms` 变为 `369.9/462.6 ms`；health
统计自身仅 `0.47/0.74 ms`（`docs/gui_search_region_quality.md:148-158`）。这不是纯 extractor
benchmark，不能直接按面积换算绝对毫秒，但确认主要成本在 formal Steger，而不在候选后的
一维统计。

## 9. 建议的后续实现边界（本轮不实施）

若后续恢复实现，最小充分顺序是：

1. 提取共享、axis-free 的 dense 2D field kernel，先用现有 column/row 测试证明输出逐点不变；
2. 增加 `extract_steger_candidates_2d()` 稀疏适配器和 synthetic 多 ridge 测试；
3. 增加 compute ROI/accept ROI/halo 坐标测试，覆盖图像边缘 clamp；
4. 仅在 laser-plane calibration 接入 `bbox -> candidates -> polygon/mask -> continuity`；
5. 保持 online backend 走 dense argmax 快速路径，并做吞吐/峰值内存回归；
6. conditional search-region resolver 继续作为独立议题，不与本次数学拆分捆绑。

本轮审计结论：

```text
steger_math_is_2d = true
scan_axis_is_coordinate_and_selection_policy = true
candidate_interface_feasible = true
calibration_2d_flow_feasible = true
online_fast_path_can_remain = true
source_code_changed = false
```
