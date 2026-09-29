# DexUMI 采集 PKL 数据说明（回放开发用）

> 本文档供**灵巧手回放**与**机械臂轨迹回放**实现时引用。  
> 数据来源：`teleop/collect_data.py`，经 PICO 手部追踪 + dex-retargeting 录制。

---

## 一、文件格式总览

```python
import pickle
with open("data/demo_xxx.pkl", "rb") as f:
    data = pickle.load(f)

assert isinstance(data, dict) and "messages" in data
messages: list[dict] = data["messages"]   # 按时间顺序排列的帧列表
```

每一帧 `message` 是一个 `dict`，字段如下：

| 字段 | 类型 | 当前是否有效 | 用途 |
|------|------|-------------|------|
| `timestamp` | `float` | 是 | 本机墙钟时间（`time.time()`），秒，UTC 浮点 |
| `mainClockMonotonicNs` | `int` | 是 | PICO SDK 单调时钟，纳秒 |
| `handCommand` | `np.ndarray(6,) uint8` 或 `None` | 是 | Linker O6 六关节命令，0~255 |
| `trajectoryPose` | `np.ndarray(7,) float32` 或 `None` | 是 | 手掌中心 Palm 在世界系的位姿 |
| `rgbImage` | `np.ndarray` / `str` / `None` | **暂空** | 预留 RGB |
| `rgbFrameId` | `int` / `None` | **暂空** | 预留帧号 |
| `rgbCaptureNs` | `int` / `None` | **暂空** | 预留采集时刻 |
| `rgbAlignResidualNs` | `int` / `None` | **固定 None** | 预留与手部时间对齐残差 |

默认采集：**仅 `active==1` 的有效帧**会写入；无效帧被丢弃。  
若录制时加了 `--keep-invalid`，无效帧会保留，但 `handCommand` / `trajectoryPose` 为 `None`。

典型采样率：**30 Hz**（`--hz 30`，默认），帧间隔约 33 ms，**不保证严格等间隔**（受 PICO/重定向耗时影响）。

---

## 二、坐标系说明（必读）

采集数据里存在**两套互不相同的坐标语义**，回放时必须分开处理。

### 2.1 `trajectoryPose` —— PICO 追踪世界系（空间轨迹）

| 属性 | 说明 |
|------|------|
| **参考点** | PICO 26 关键点中的 **Palm（手掌中心）**，数组索引 **0** |
| **坐标系名称** | **PICO SDK 世界追踪坐标系**（下文简称 **PICO World**） |
| **是否绝对坐标** | **是**。每一帧都是该点在 PICO World 下的绝对位姿，**不是**相对手腕、不是相对头显、不是 MANO 局部系 |
| **位置单位** | **米 (m)**，`[x, y, z]` |
| **姿态表示** | 四元数 **`[qx, qy, qz, qw]`**（向量部分在前，标量 `w` 在最后） |
| **数组形状** | `(7,)`，`dtype=float32` |
| **数据来源** | `xrobotoolkit_sdk` → `get_*_hand_tracking_state()` 的原始输出，**未经**手腕居中、SVD、MANO 变换 |

#### PICO World 的物理含义

- 原点、X/Y/Z 轴方向由 **PICO 运行时 / XRoboToolkit 追踪系统**定义，通常与**房间级 SLAM 或头显建立的追踪空间**一致。
- **不是**机器人 `base_link` 坐标系。
- **不是** URDF / MuJoCo 模型坐标系。
- **不是** dex-retargeting 使用的 MANO 手部局部系（MANO 只用于算 `handCommand`，不会写入 `trajectoryPose`）。

#### 四元数约定

```
trajectoryPose = [x, y, z, qx, qy, qz, qw]
```

- 旋转顺序：**qx, qy, qz, qw**（与 ROS `geometry_msgs/Quaternion`、PICO SDK 原始顺序一致）。
- 使用时建议先 `np.linalg.norm(q) ≈ 1` 归一化，再转旋转矩阵或欧拉角。
- Python 示例（转 3×3 旋转矩阵，右手系）：

```python
def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    import numpy as np
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
```

#### 机械臂回放时必须做的坐标变换

`trajectoryPose` **不能直接**当作机械臂末端在机器人基座下的目标位姿。需要外参标定：

```
T_robot_palm(t) = T_robot_pico @ T_pico_palm(t)
```

其中：

- `T_pico_palm(t)`：由 `trajectoryPose` 构造的 4×4 齐次变换（PICO World 下 Palm 位姿）。
- `T_robot_pico`：PICO World → 机器人基座（或机械臂 `{base}`）的**固定外参**，需标定或手眼标定得到。
- 可选：再乘工具坐标系 `T_ee_palm`（若末端法兰不是对准 Palm 中心）。

若只做**位置跟踪**、忽略手掌朝向，可仅用 `trajectoryPose[:3]` 做路径，但仍需把位置从 PICO World 变换到机器人工作空间。

#### 与「相对轨迹」的区别

当前 PKL **没有**存储：

- 相对第一帧的位移
- 相对头显 / 桌面的坐标
- 已滤波或插值后的轨迹

回放程序如需「以录制起点为原点」，应在加载后自行：

```python
T0 = pose_to_matrix(messages[0]["trajectoryPose"])
T_i = pose_to_matrix(messages[i]["trajectoryPose"])
T_rel = np.linalg.inv(T0) @ T_i
```

---

### 2.2 `handCommand` —— 无空间坐标系（关节命令空间）

| 属性 | 说明 |
|------|------|
| **物理含义** | Linker O6 右手 **6 个主动关节** 的离散控制量 |
| **数值范围** | `uint8`，**0 ~ 255** |
| **与 URDF 关系** | 由重定向弧度经线性映射得到，**不是**笛卡尔空间位姿 |

#### 关节顺序（固定，不可打乱）

```python
LINKER_JOINTS = [
    "thumb_cmc_pitch",   # 索引 0
    "thumb_cmc_yaw",     # 索引 1
    "index_mcp_pitch",   # 索引 2
    "middle_mcp_pitch",  # 索引 3
    "ring_mcp_pitch",    # 索引 4
    "pinky_mcp_pitch",   # 索引 5
]
```

#### 命令语义（与真机 SDK 一致）

| `handCommand[i]` | 关节状态 |
|------------------|----------|
| **250** | 该关节 **张开**（URDF `qpos = 0`） |
| **0** | 该关节 **屈曲到上限**（URDF `qpos = upper`） |
| 中间值 | 线性插值 |

映射公式（采集时使用，回放逆解可选）：

```
cmd = round( 250 * (1 - q_rad / upper) ),  clip 到 [0, 255]
```

各关节 `upper`（弧度）：

| 关节 | upper (rad) |
|------|-------------|
| thumb_cmc_pitch | 0.58 |
| thumb_cmc_yaw | 1.36 |
| index/middle/ring/pinky_mcp_pitch | 1.60 |

#### 从动关节（mimic）

PKL 中**不存储** `thumb_ip`、`index_dip` 等从动关节。真机与 MuJoCo 仿真中它们由机械/URDF mimic 关系自动跟随主控关节。

---

### 2.3 重定向内部用过的 MANO 系（未写入 PKL）

采集时为了计算 `handCommand`，内部曾将手部关键点变换到 **MANO 局部坐标系**（手腕为原点、消除全局旋转）。该中间结果**没有**存入 PKL。

| 对比项 | MANO 局部系（内部） | `trajectoryPose`（PKL） |
|--------|---------------------|-------------------------|
| 原点 | 手腕 | Palm 中心 |
| 是否绝对坐标 | 否（每帧手腕在原点） | 是（PICO World） |
| 用途 | dex-retargeting 输入 | 机械臂轨迹 |
| 是否在 PKL | **否** | **是** |

**切勿**把 MANO 系约定套用到 `trajectoryPose` 上。

---

## 三、时间戳说明

每帧有两个时间字段，用途不同：

| 字段 | 类型 | 含义 | 推荐用途 |
|------|------|------|----------|
| `timestamp` | `float` | `time.time()`，本机墙钟秒 | 与外部日志、ROS bag 墙钟对齐 |
| `mainClockMonotonicNs` | `int` | PICO SDK 单调纳秒 | **手部与 PICO 数据对齐**；计算帧间隔 |

回放节拍建议：

```python
# 用 SDK 时间算帧间隔（更稳定）
t_ns = np.array([m["mainClockMonotonicNs"] for m in messages if m["handCommand"] is not None])
dt = np.diff(t_ns) * 1e-9   # 秒
# 或固定按录制 hz（若已知为 30）
period = 1.0 / 30.0
```

---

## 四、回放任务拆分

### 4.1 灵巧手回放

**输入**：每帧 `handCommand`（6,）`uint8`  
**输出**：Linker O6 CAN 指令或 MuJoCo 关节角

真机（与 `teleop_real.py` 一致）：

```python
from o6_right_hand.controller import O6RightHand
hand = O6RightHand(can_channel="can0")
hand.move(cmd.tolist())   # cmd: list[int] 长度 6
```

MuJoCo（需弧度 + mimic，见 `hand_control_example.kinematic_control`）：

```python
# 可选：cmd -> rad 逆映射
q = upper * (1 - cmd / 250.0)
target = {name: q[i] for i, name in enumerate(LINKER_JOINTS)}
kinematic_control(model, data, target)
```

注意：从动关节 `*_dip`、`*_ip` 由 mimic 自动跟随，无需从 PKL 读取。

---

### 4.2 机械臂轨迹回放

**输入**：每帧 `trajectoryPose`（7,）PICO World 下 Palm 位姿  
**输出**：机械臂末端（或工具中心）在 **机器人基座系** 下的 `T_robot_ee(t)`

步骤：

1. 读取 `trajectoryPose` → 构造 `T_pico_palm(t)`（4×4）。
2. 左乘标定矩阵 `T_robot_pico`（及可选 `T_ee_palm`）。
3. 按 `mainClockMonotonicNs` 或固定 `hz` 发送给机械臂控制器（MoveIt / 笛卡尔伺服等）。
4. 与灵巧手回放**时间对齐**：同一 `message` 索引的 `handCommand` 与 `trajectoryPose` 为同一时刻采样。

---

## 五、实现回放程序时的提示词（可直接复制）

```
请基于 DexUMI 项目采集的 PKL 文件实现回放，要求同时支持：
1）Linker O6 灵巧手关节命令回放；
2）机械臂跟随手掌中心轨迹回放。

【PKL 结构】
- 顶层 dict，键 "messages" 为 list[dict]。
- 每帧字段：timestamp (float 墙钟秒)、mainClockMonotonicNs (int PICO纳秒)、
  handCommand (np.uint8 shape(6,))、trajectoryPose (np.float32 shape(7,))、
  rgb* 字段当前均为 None。

【坐标系 - 极其重要】
- trajectoryPose 是 PICO SDK 世界追踪系下的绝对位姿，单位米，参考点为 Palm（手掌中心，PICO 26点中的 index 0）。
- 格式 [x, y, z, qx, qy, qz, qw]，四元数为 qx,qy,qz,qw 顺序。
- 不是机器人 base_link 系，不是 MANO 局部系，不是相对第一帧的位移。
- 机械臂回放前必须乘以标定外参 T_robot_pico（PICO World → 机器人基座），可选 T_ee_palm。
- handCommand 是关节空间 0~255 命令，无笛卡尔坐标系；顺序为：
  thumb_cmc_pitch, thumb_cmc_yaw, index_mcp_pitch, middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch。
- 250=张开，0=完全屈曲；与 teleop_real.py / collect_data.py 一致。

【时间】
- 用 mainClockMonotonicNs 做帧间对齐；默认约 30Hz。
- 跳过 handCommand 或 trajectoryPose 为 None 的帧（若存在）。

【灵巧手】
- 真机：O6RightHand.move(cmd_list)。
- 仿真：cmd 逆映射为弧度后 kinematic_control，mimic 关节自动跟随。

【机械臂】
- trajectoryPose → 4x4 位姿 → T_robot_pico @ T_pico_palm → 发送给机械臂。
- 与 handCommand 按 message 索引同步回放。

【依赖参考】
- 采集：teleop/collect_data.py
- 真机控制：o6_right_hand/controller.py、teleop/teleop_real.py
- 完整 pipeline 说明：teleop/RETARGETING_PIPELINE.md
```

---

## 六、加载示例代码

```python
import pickle
import numpy as np

def load_episode(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    msgs = data["messages"]
    valid = [m for m in msgs if m.get("handCommand") is not None]
    return valid

def trajectory_positions(msgs):
  """PICO World 下 Palm 位置序列 (N, 3)，单位米。"""
    return np.stack([m["trajectoryPose"][:3] for m in msgs])

def hand_commands(msgs):
    """(N, 6) uint8"""
    return np.stack([m["handCommand"] for m in msgs])
```

---

## 七、常见问题

**Q：能否把 trajectoryPose 直接当机械臂末端位姿？**  
A：不能。必须做 PICO World → 机器人基的标定变换。

**Q：PICO World 的 X/Y/Z 朝哪？**  
A：由 PICO 运行时定义，与具体房间/头显追踪初始化有关；实现时以外参标定为准，不要假设与机器人 URDF 轴对齐。

**Q：handCommand 和 trajectoryPose 是否同一时刻？**  
A：是。同一 `messages[i]` 内二者来自同一帧 `read_raw()` + 重定向。

**Q：左手数据？**  
A：当前采集脚本默认 `--hand right`；若用左手，PICO 仍输出 PICO World 下的 Palm，但 handCommand 对应左手需换 yml/URDF（本 PKL 规范未覆盖左手命令语义）。

---

*文档版本：与 `collect_data.py`（Palm 作为 trajectoryPose）一致。*
