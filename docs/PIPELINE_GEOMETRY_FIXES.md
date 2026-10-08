---
name: pipeline-geometry-fixes
description: "Key pipeline fixes — image center through H, relaxed bounds, color annotations, trajectory visualization"
metadata:
  node_type: memory
  type: project
  originSessionId: 917f67dd-a1b0-4200-bdf6-4eabaf5de24b
---

## Pipeline 几何修正记录

### 1. 图像中心过 H 替代角点质心

**问题**: `compute_center(dst)` 对 4 个变换后角点取质心，由于单应性变换中不同点的齐次归一化分母 w 不同，导致 `centroid(H(corners)) ≠ H(centroid(corners))`，即使在数学上也不等价。

**修复**: 直接将无人机图像中心点 `(w/2, h/2)` 通过 H 矩阵变换：
```python
center_pt = np.float32([[[w / 2, h / 2]]])
denormalized_center = cv2.perspectiveTransform(center_pt, H)[0][0]
```

**影响**: 近正下视场景效果接近（H ≈ 仿射），但数学上更正确。有显著透视效应时差异可达 56px。

### 2. 放宽 center 越界检查

**问题**: `center[0] < 0 or center[0] > 1` 硬边界检查，000009 帧 center x=-0.005（仅溢出 0.5%）被拒绝，导致选择了匹配数少但 center 在界内的错误瓦片，误差从 33.7m 飙到 161.9m。

**修复**: 放宽到 ±10%：
```python
if center[0] < -0.1 or center[0] > 1.1 or center[1] < -0.1 or center[1] > 1.1:
    continue
```

**效果**:
- MAE: 32.8m → 23.4m (-29%)
- 最大误差: 161.9m → 57.1m (-65%)
- 匹配率: 80% → 93.3%

### 3. 彩色标记点修复

**问题**: `draw_center` 在灰度图上用 BGR 颜色画圆，OpenCV 只取颜色元组第一个值：`(255,0,255)` → 白色，`(0,255,0)` → 黑色。

**修复**: 先用原始灰度图调 `make_matching_plot_fast` 生成 BGR 画布，再在画布上直接画彩色圆：
- 品红 `(255,0,255)`: 预测 look-at 点
- 绿色 `(0,255,0)`: GT 真值

### 4. 新增方法

**`gps_to_pixel(gps_coordinate, satellite_image)`** (base.py):
GPS 坐标 → 卫星图像素坐标，利用瓦片 top_left/bottom_right 线性插值。

**`draw_center(image, center, color=(255,0,255))`** (base.py):
新增 `color` 参数，默认品红色，向后兼容。

### 5. 轨迹可视化脚本

**`scripts/plot_trajectory_on_mosaic.py`**:
- 运行完整 pipeline 收集预测和 GT 坐标
- 在 mosaic.tiff 上绘制两条轨迹：
  - 绿色线：GT 真值轨迹
  - 品红线：预测轨迹
- 参数: `--image-folder`, `--map-db`, `--mosaic`, `--output`, `--device`, `--max-images`

**Why**: 直观对比预测轨迹与真值轨迹的空间偏移模式。
**How to apply**: `python scripts/plot_trajectory_on_mosaic.py --image-folder data/my_experiment/query_test --map-db data/my_experiment/map/tiles --mosaic data/my_experiment/map/mosaic.tiff --output data/my_experiment/trajectory.jpg --device cuda`

### 待完成: BA_loop 姿态修正

当前 pipeline 输出的是光轴 ground look-at 点，不等于无人机自身地面投影。需要：
1. 图像时间戳 → BA_loop 位姿 (timestamp matching)
2. 用 `R_body_world × R_cam × [0,0,1]ᵀ` 计算光轴方向
3. 用高度 H 计算水平偏移量: `Δ = H × (d_x, d_y) / |d_z|`
4. `drone_GPS = look_at_GPS - offset`
