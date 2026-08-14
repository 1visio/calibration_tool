# Circular Cone 等价局部参数化

## 正式 legacy 模型

正式模型仍为

`Theta_legacy = [theta_axis, phi_axis, A_x, A_y, A_z, alpha]`

`d = spherical(theta_axis, phi_axis)` 为单位轴方向，`A` 为 apex，`alpha` 为 half-apex angle。圆锥方程保持正式实现：

`||(X-A) - ((X-A)·d)d|| / tan(alpha) - (X-A)·d = 0`

求交仍使用正式的 ray-cone quadratic intersection、正深度/工作距离筛选和 `axial >= 0` 的物理 nappe 选择；本参数化没有重新定义模型。

## 固定参考锚点

`P_ref = [-5.400870389961511, -7.002832713450796, 681.3566154039942] mm`，由 `30` 个 FIT frame 的 ray-plane truth 逐 frame 计算 3D centroid，再对 30 个 centroid 等权平均。Validation 不参与锚点计算。

对每个轴 `d`，用确定性的相机坐标基构造 `e1,e2`：优先投影 camera-Z 到 `d` 的正交平面；若近似平行则使用 camera-Y；`e2 = d × e1`。

令 `C_ref` 为 cone axis 与过 `P_ref` 的法向截面相交点，定义：

`C_ref = P_ref + c1*e1 + c2*e2`

`s_ref = (P_ref-A)·d`

`rho_ref = s_ref*tan(alpha)`

`q = cot(alpha) = 1/tan(alpha)`

因此局部向量为：

`Theta_local = [theta_axis, phi_axis, c1, c2, rho_ref, q]`

## 严格双向转换

`legacy_to_local`：由 `A,d,alpha,P_ref` 计算 `C_ref,s_ref,rho_ref,q`。

`local_to_legacy`：

`s_ref = rho_ref*q`

`A = P_ref + c1*e1 + c2*e2 - s_ref*d`

`alpha = atan2(1,q)`

这个映射在 `0 < alpha < 90°`、`q > 0` 下是严格可逆的。轴向量不取反，因而 nappe convention 不改变。

## 为什么是局部坐标

原始 apex 位于观测区域之外，且 `alpha≈90°` 时 apex displacement 与 alpha 变化会形成很长的弱谷。新坐标把几何量改写为观测区域附近的 axis-line 横向位置、参考截面半径和局部斜率；它只改变坐标，不保证条件数已经改善。条件数和 Full-FIT 稳定性留给 Task 3B-2。
