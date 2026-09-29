# 轨迹滤波模块 (Trajectory Filter)

这个模块提供了多种轨迹滤波器，用于对 xyzrpy 格式的轨迹进行滤波处理，减少噪声并提高轨迹的稳定性。

## 文件结构

```
filter/
├── __init__.py                    # 模块初始化文件
├── README.md                      # 说明文档
├── simple_trajectory_filter.py    # 简化版轨迹滤波器
├── trajectory_filter.py           # 完整版轨迹滤波器
├── rotation_filter.py             # 高级旋转滤波器 (推荐)
├── advanced_rotation_filter.py    # 完整版高级旋转滤波器
├── offline_rotation_filter.py     # 离线旋转滤波器 (n,6) 格式
├── vive_tracker_with_filter.py    # Vive Tracker 集成版
├── filter_example.py              # 使用示例
├── rotation_filter_demo.py        # 旋转滤波演示
├── test_filter.py                 # 测试脚本
└── test_offline_filter.py         # 离线滤波器测试
```

## 快速开始

### 1. 离线滤波 (推荐)

```python
from filter import OfflineRotationFilter
import numpy as np

# 创建离线旋转滤波器
filter_obj = OfflineRotationFilter(
    filter_type='savgol',  # Savitzky-Golay 滤波
    window_size=7,         # 窗口大小
    polyorder=2            # 多项式阶数
)

# 准备轨迹数据 (n, 6) 格式
trajectory = np.random.randn(100, 6)  # 100个点，6维位姿

# 滤波整个轨迹
filtered_trajectory = filter_obj.filter_trajectory(trajectory)
print(f"输入形状: {trajectory.shape}")
print(f"输出形状: {filtered_trajectory.shape}")
```

### 2. 实时滤波

```python
from filter import RotationFilter

# 创建实时旋转滤波器
filter_obj = RotationFilter(
    filter_type='savgol',  # Savitzky-Golay 滤波
    window_size=7,         # 窗口大小
    polyorder=2            # 多项式阶数
)

# 滤波单个位姿
pose_xyzrpy = [x, y, z, roll, pitch, yaw]
filtered_pose = filter_obj.filter_pose(pose_xyzrpy)
```

### 3. 传统方法

```python
from filter import SimpleTrajectoryFilter

# 创建简化滤波器
filter_obj = SimpleTrajectoryFilter(
    filter_type='moving_average',  # 滤波类型
    window_size=5                  # 窗口大小
)

# 滤波单个位姿
pose_xyzrpy = [x, y, z, roll, pitch, yaw]
filtered_pose = filter_obj.filter_pose(pose_xyzrpy)
```

### 4. 与 Vive Tracker 集成

```python
from filter import ViveTrackerWithFilter

# 创建带滤波的 tracker
tracker = ViveTrackerWithFilter(
    pub=True,                      # 启用 ROS2 TF
    filter_type='moving_average',  # 滤波类型
    filter_window_size=5           # 滤波窗口大小
)

# 获取滤波后的位姿
filtered_pose = tracker.get_filtered_pose()
```

## 支持的滤波类型

### SimpleTrajectoryFilter

1. **移动平均** (`moving_average`)
   - 简单有效，适合大多数场景
   - 参数：`window_size` - 窗口大小

2. **指数平滑** (`exponential`)
   - 响应快，适合实时应用
   - 参数：`alpha` - 平滑系数 (0-1)

3. **低通滤波** (`lowpass`)
   - 专业滤波，需要 scipy
   - 参数：`cutoff_freq`, `sampling_rate`

### TrajectoryFilter

1. **低通滤波** (`lowpass`)
2. **卡尔曼滤波** (`kalman`)
3. **移动平均** (`moving_average`)
4. **巴特沃斯滤波** (`butterworth`)

### OfflineRotationFilter (推荐)

1. **Savitzky-Golay 滤波** (`savgol`)
   - 保持数据特征，适合平滑轨迹
   - 参数：`window_size`, `polyorder`
   - 输入：`(n, 6)` numpy array
   - 输出：`(n, 6)` numpy array

2. **Butterworth 滤波** (`butterworth`)
   - 低通滤波，去除高频噪声
   - 参数：`cutoff_freq`, `sampling_rate`, `order`
   - 输入：`(n, 6)` numpy array
   - 输出：`(n, 6)` numpy array

3. **移动平均** (`moving_average`)
   - 简单有效，适合离线处理
   - 参数：`window_size`
   - 输入：`(n, 6)` numpy array
   - 输出：`(n, 6)` numpy array

### RotationFilter (实时)

1. **Savitzky-Golay 滤波** (`savgol`)
   - 保持数据特征，适合平滑轨迹
   - 参数：`window_size`, `polyorder`

2. **Butterworth 滤波** (`butterworth`)
   - 低通滤波，去除高频噪声
   - 参数：`cutoff_freq`, `sampling_rate`, `order`

3. **移动平均** (`moving_average`)
   - 简单有效，适合实时应用
   - 参数：`window_size`

## 运行示例

```bash
# 运行离线旋转滤波器演示 (推荐)
python filter/offline_rotation_filter.py

# 运行高级旋转滤波器演示
python filter/rotation_filter.py

# 运行完整演示
python filter/advanced_rotation_filter.py

# 运行简化示例
python filter/filter_example.py

# 运行旋转滤波对比演示
python filter/rotation_filter_demo.py

# 运行集成版 Vive tracker
python filter/vive_tracker_with_filter.py
```

## 依赖项

- numpy
- scipy (用于低通滤波)
- matplotlib (用于可视化)
- scipy.spatial.transform (用于四元数转换)

## 重要说明：旋转滤波

### 为什么不能直接对 RPY 进行滤波？

1. **角度不连续性**：RPY 角度在 ±π 处有跳跃，直接滤波会导致错误结果
2. **万向锁问题**：某些 RPY 组合会导致奇异点
3. **角度插值问题**：两个相近的角度可能相差 2π，直接平均会得到错误结果

### 正确的旋转滤波方法

本模块使用以下方法正确处理旋转：

1. **四元数转换**：将 RPY 转换为四元数进行滤波
2. **球面平均**：对四元数使用球面平均而非线性平均
3. **球面插值**：使用 SLERP 进行四元数插值
4. **归一化**：确保四元数保持单位长度

### 演示

运行 `rotation_filter_demo.py` 可以看到错误和正确滤波方法的对比：

```bash
python filter/rotation_filter_demo.py
```

## 注意事项

1. 旋转信息在四元数空间进行滤波，避免角度不连续问题
2. 位置信息使用线性滤波方法
3. 滤波器状态会在每次调用 `filter_pose()` 时更新
4. 使用 `reset()` 方法可以重置滤波器状态
5. 实时应用建议使用 `SimpleTrajectoryFilter`

## 性能建议

- **实时应用**: 使用 `moving_average` 或 `exponential` 滤波
- **离线处理**: 使用 `lowpass` 或 `kalman` 滤波
- **窗口大小**: 3-7 通常效果较好，过大会延迟响应
- **平滑系数**: 0.3-0.7 通常效果较好
