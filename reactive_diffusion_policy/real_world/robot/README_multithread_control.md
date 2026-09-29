# 多线程控制功能说明

## 概述

在 `FlexivController` 类中添加了多线程控制功能，允许机械臂在后台线程中持续执行 `absolute_control`，并支持实时修改目标位姿。

## 主要特性

1. **后台控制线程**: 机械臂在独立线程中持续执行控制命令
2. **实时位姿更新**: 支持通过 `tcp_move_v3` 实时修改目标位姿
3. **线程安全**: 使用锁机制确保多线程环境下的数据安全
4. **优雅关闭**: 支持通过 `stop_event` 安全停止控制线程
5. **自动重启**: `reset_to_home` 操作会自动停止并重启控制线程

## 新增方法

### 核心控制方法

#### `tcp_move_v3(goal_pose: List[float])`
实时修改目标位姿，通过控制线程持续执行。

**参数:**
- `goal_pose`: 目标位姿 `[x, y, z, w, x, y, z]` (位置 + 四元数)

**示例:**
```python
# 设置目标位姿
target_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
controller.tcp_move_v3(target_pose)
```

### 线程管理方法

#### `_start_control_thread()`
启动控制线程（内部方法）

#### `_stop_control_thread()`
停止控制线程（内部方法）

#### `is_control_active() -> bool`
检查控制线程是否正在运行

#### `pause_control()`
暂停控制（设置目标位姿为None）

#### `resume_control(target_pose: List[float])`
恢复控制（设置新的目标位姿）

### 状态查询方法

#### `get_control_status() -> dict`
获取控制状态信息

**返回:**
```python
{
    "is_running": bool,           # 线程是否运行
    "is_active": bool,            # 控制是否激活
    "has_target": bool,           # 是否有目标位姿
    "target_pose": List[float],   # 当前目标位姿
    "control_frequency": int      # 控制频率
}
```

#### `get_target_pose() -> Optional[List[float]]`
获取当前目标位姿（线程安全）

#### `set_target_pose(target_pose: List[float])`
设置目标位姿（线程安全）

## 使用流程

### 1. 基本使用

```python
# 创建控制器（自动启动控制线程）
controller = FlexivController()

# 设置目标位姿，控制线程会自动执行
controller.tcp_move_v3([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])

# 实时更新目标位姿
controller.tcp_move_v3([0.6, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0])

# 暂停控制
controller.pause_control()

# 恢复控制
controller.resume_control([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
```

### 2. 重置操作

```python
# reset_to_home 会自动停止并重启控制线程
controller.reset_to_home()
```

### 3. 关闭控制器

```python
# 关闭时会自动停止控制线程
controller.close()
```

## 线程安全

- 所有对 `target_pose` 的访问都通过 `control_thread_lock` 保护
- 支持多线程同时调用 `tcp_move_v3`
- 使用 `threading.Event` 进行线程间通信

## 控制频率

控制频率由 `control_frequency` 参数决定，默认为 24Hz。可以在创建控制器时修改：

```python
controller = FlexivController(control_frequency=50)  # 50Hz控制频率
```

## 错误处理

- 控制线程中的错误会被捕获并记录到日志
- 出错时会短暂等待后继续执行
- 支持优雅的错误恢复

## 测试

使用 `test_multithread_control.py` 脚本测试功能：

```bash
python test_multithread_control.py
```

选择测试模式：
1. 基本功能测试
2. 线程安全性测试

## 注意事项

1. **初始化**: 控制器创建时会自动启动控制线程
2. **资源管理**: 确保在程序结束时调用 `close()` 方法
3. **位姿格式**: `tcp_move_v3` 使用 `[x, y, z, w, x, y, z]` 格式
4. **线程依赖**: 控制线程依赖于机器人接口，确保接口正常初始化
5. **性能**: 高频率控制可能会影响系统性能，根据需要调整控制频率

## 与现有功能的兼容性

- 新功能与现有的 `tcp_move`、`tcp_move_v2` 等方法完全兼容
- 不影响原有的单次运动控制功能
- 可以同时使用多线程控制和传统控制方法

