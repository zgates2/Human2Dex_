# PICO → Linker O6 实时重定向 Pipeline

> 模块：`teleop/`
> 关键文件：`pico_hand.py`、`retargeter.py`、`teleop_mujoco.py`、`teleop_real.py`
> 依赖：`dex-retargeting`（DexPilot 优化器）、`load_in_mujoco.py`、`hand_control_example.py`、`o6_right_hand` SDK
> 单位规范：内部一律使用 **米 (m)** 与 **弧度 (rad)**；只在 ↔ 真机命令时换算为 0~255 整数。

## 1. 数据流总览

```
PICO 头戴 (右手)
  │  26 × (x,y,z, qx,qy,qz,qw)，世界系，单位 m
  ▼
PicoHandReader.read_raw()
  │  保留 (26, 7) 原始
  ▼
① 26 → 21 拓扑映射         （pico_hand.py: PICO2MEDIAPIPE）
  │  得到 MediaPipe 21 顺序的 (21, 3)
  ▼
② 平移居中                 pts21 -= pts21[0] 绝对得到相对
  │  wrist 设为原点
  ▼
③ SVD 估计手部局部 frame   _estimate_wrist_frame(centered)
  │  用 [wrist, index_mcp, middle_mcp] 三点 + SVD 求 normal/x/z
  ▼
④ 对齐到 MANO 局部系       joint_pos = centered @ rot @ OPERATOR2MANO_RIGHT
  │  (21, 3) MANO 关键点：+z = wrist→middle finger，+x = palm 法向 内侧
  ▼
LinkerO6Retargeter.retarget(joint_pos)
  │
  ├─ ⑤ MANO → robot base_link 系   pts_base = joint_pos @ R_mano_to_base
  │     (R 在 __init__ 时由 robot zero-pose forward kinematics 自动校准)
  │
  ├─ ⑥ 构造 DexPilot ref_value
  │     ref[k] = pts_base[task_idx[k]] - pts_base[origin_idx[k]]
  │     task / origin 索引由 DexPilot 默认按 MediaPipe 21 生成 (15 对 vector)
  │
  └─ ⑦ NLOPT 优化 → 6 个主动关节弧度
        thumb_cmc_pitch, thumb_cmc_yaw,
        index_mcp_pitch, middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch
  ▼
[MuJoCo 路径]                         [真机路径]
kinematic_control(model, data, q)     _angles_rad_to_cmd(q)
  ① 主控关节 qpos[..] = q[i]            ① cmd = clip(250 * (1 - q / upper), 0, 250)
  ② 手动求解 equality (mimic)           ② O6RightHand.move(cmd)   (CAN 总线)
  ③ mj_forward
```

## 2. 各阶段细节

### 阶段 ①：PICO 26 → MediaPipe 21 拓扑映射

PICO 给出 26 个 bone-origin 点（含 `Palm`、各指 `Metacarpal`），而 `dex-retargeting` 期望的是 **MediaPipe Hands 21** 关键点（`wrist + 5 指 × 4 节`）。映射表写死在 `pico_hand.py: PICO2MEDIAPIPE`，规则：

- `Palm` 与 4 个非拇指 `*_Metacarpal` **丢弃**（拇指无 metacarpal，所以 thumb 的 4 个点全部保留）。
- 其他 21 点按 PICO 关节物理位置一一对应到 MediaPipe。

### 阶段 ②：平移居中

```
pts21 = pts21 - pts21[0]   # wrist 设为原点
```

PICO 输出是**绝对世界系坐标**（基于某个全局基准的轨迹）。SVD 估 frame 前必须先减去 wrist。

### 阶段 ③：SVD 估手部局部 frame

完全复刻 `dex-retargeting/example/vector_retargeting/single_hand_detector.py::estimate_frame_from_hand_points`：

1. 取 3 个点：`[wrist, index_mcp(5), middle_mcp(9)]`。
2. `x_vector = wrist - middle_mcp`（沿手掌竖向）。
3. 3 点共面，用 SVD 找平面 normal（=`vh[2,:]`）。
4. Gram-Schmidt 在平面内重正交化 `x`，再 `z = x × normal`。
5. **极性校正**：若 `z · (index_mcp - middle_mcp) < 0`，把 normal 和 z 各乘 -1，保证 z 大致指向 index 一侧。
6. `frame = stack([x, normal, z], axis=1)` 是一个 3×3 旋转矩阵（列向量 = operator 系基），`det = +1`。

### 阶段 ④：MANO 局部系对齐

```python
OPERATOR2MANO_RIGHT = [[0,0,-1],[-1,0,0],[0,1,0]]
joint_pos = centered @ rot @ OPERATOR2MANO_RIGHT
```

MANO 局部系的物理约定：


| 轴    | 物理意义                        |
| ---- | --------------------------- |
| `+x` | palm 内侧法向（朝向掌心）             |
| `+y` | middle → index 方向           |
| `+z` | wrist → middle fingertip 方向 |


至此 `(21, 3)` 关键点已经**与全局位姿无关**，只剩相对手形信息。

### 阶段 ⑤：MANO → robot base_link 系（最大的坑，见 §3.4）

`dex-retargeting` 内部对 robot 端的 fingertip 用 pinocchio 做 forward kinematics，结果在 **pinocchio model root frame = URDF 的根 link** 下。对 Linker O6，URDF 根是 `base_link`，它和实际手部 frame `hand_base_link` 之间有一个 `rpy="1.57 3.14 0"` 的固定旋转，导致 fingertip 在 `base_link` 系下沿 `**-y`** 伸出，而不是直觉上的 `+z`。

修复：`LinkerO6Retargeter.__init__()` 自动跑一次 `qpos = 0` 的 forward kinematics，取 `(hand_base_link, index_proximal, middle_proximal)` 三点位置，按同一套 SVD+OPERATOR2MANO 逻辑估出 **robot 自身的"虚拟 MANO frame"**，再用其转置作为 `R_mano_to_base`，保证：

- `mano +z (手指方向) → base_link 系下 ≈ (0, -1, 0)`，与 robot fingertip 朝向一致。
- 行列式 `det = +1`，不引入镜像。

### 阶段 ⑥：DexPilot ref_value 构造

DexPilot 优化器要求传入一组**距离向量** `ref_value`（不是关键点本身）。当 `target_link_human_indices is None` 时由 `DexPilotOptimizer.generate_link_indices(5)` 自动生成，对应 5 指共 `4+3+2+1 + 5 = 15` 条 vector：

- 前 10 条是手指之间的两两距离（index-thumb, middle-thumb, …, pinky-ring）。
- 后 5 条是 wrist→fingertip。

索引（默认）：

```
origin = [8,12,16,20, 12,16,20, 16,20, 20,  0, 0, 0, 0, 0]   # mediapipe ids
task   = [4, 4, 4, 4,  8, 8, 8, 12,12, 16,  4, 8,12,16,20]
```

`ref[k] = pts_base[task[k]] - pts_base[origin[k]]`。

**方向必须与 robot 端一致**（`optimizer.py` 第 528–530 行 `robot_vec = task_link_pos - origin_link_pos`）。我们传 `task - origin` ✓。

### 阶段 ⑦：弧度 → 6 主动关节

经过 NLOPT 与 mimic adaptor，`SeqRetargeting.retarget(ref)` 返回**全部主动+从动关节**的 qpos。`LinkerO6Retargeter` 通过 `ret2linker` 索引取出 6 个主控关节弧度，顺序与 `LINKER_JOINTS` 一致。

### 阶段 ⑧A：MuJoCo 仿真

`hand_control_example.kinematic_control` 现在做三件事：

1. 把 6 个主控关节 `clip` 到 URDF limit 后写入 `data.qpos`。
2. **手动求解 mjEQ_JOINT 约束**（`_apply_mimic_equalities`），让所有 `*_ip / *_dip` 跟随。
3. 调用 `mujoco.mj_forward` 更新空间位姿。

### 阶段 ⑧B：真机 CAN 控制

`teleop_real.py::_angles_rad_to_cmd` 把弧度映射为 0~250 整数：

```
cmd_i = clip( int(round( 250 * (1 - q_i / upper_i) )), 0, 250 )
```

约定：

- `cmd = 250` ↔ **完全张开**（关节 qpos = 0）
- `cmd =   0` ↔ **完全屈曲**（关节 qpos = upper limit）

`upper` 取自 URDF `<limit upper=...>`，写死在 `URDF_UPPER` 字典里。真机的 mimic 是机械结构内置（齿轮/弹簧）自动驱动，无需软件下发。

## 3. 踩过的坑（按时间顺序，重要性 ★ 越多越关键）

### 3.1 ★★★ MANO 系与 robot URDF base_link 系不对齐

**症状**：真实手张开 → MuJoCo 显示完全屈曲（4 指 mcp_pitch 顶到 upper=1.6）；真实手握拳 → MuJoCo 部分张开。整个映射方向"反"了。

**根因**：`dex-retargeting` 假定我们传入的 ref 与 robot forward kinematics 输出在**同一坐标系**下，但实际上：

- MANO 系 `+z` = 手指方向。
- Linker O6 `base_link` 系下手指方向 ≈ `(0, -1, 0)` = `-y`（因为 URDF `base` joint 有 `rpy="1.57 3.14 0"`）。

两套主轴差 90°，DexPilot 把人手的 `+z` 直接当成 robot 的 `+z`，于是优化器找一个让 fingertip 朝 `+z_base` 的姿态——而 `+z_base` 是 base_link 的"上方"，唯一能让 fingertip 朝上的方式就是 4 指完全折回去屈曲。

**修复**：在 `LinkerO6Retargeter.__init_`_ 里自动校准 `R_mano_to_base`（见阶段 ⑤）。debug 输出形如：

```
[Retargeter] R_mano_to_base 自动校准:
    [-0.994  -0.002  +0.107]
    [-0.107  +0.063  -0.992]
    [-0.005  -0.998  -0.062]
  det=+1.0000  (应为 +1)
  说明：mano +z (手指方向) -> base_link 系下 (-0.01, -1.00, -0.06)
```

`det = +1` 且 `mano +z → 接近某个单轴` 就说明校准成功。换其他 URDF 时**这一步会自动适配**，不用改代码。

### 3.2 ★★★ MuJoCo `mj_forward` 不驱动 mimic equality

**症状**：6 个主控关节（CMC、MCP）正确移动，但每根手指最远端的 `thumb_ip`、`*_dip` 永远停在 0，手指看上去只弯了一半。

**根因**：URDF `<mimic>` 在 `load_in_mujoco.py` 中已经转成 MuJoCo `<equality>` 约束（`polycoef = [offset, multiplier, 0, 0, 0]`），但 `**mj_forward` 只算 forward kinematics，不求解任何 constraint**；equality 只有在 `mj_step` 物理仿真里才被纳入 LCP/Cone 求解。所以 kinematic 模式（直接写 qpos + `mj_forward`）下 mimic 关节是被"冻结"的。

**修复**：在 `hand_control_example.kinematic_control` 里加 `_apply_mimic_equalities(model, data)`，自己遍历 `model.eq` 计算 `q_slave = c0 + c1*q_master + ...` 并写回 `data.qpos`。URDF 是唯一真相，新增/修改 mimic 不需要动 Python 代码。

> 真机路径下不存在这个坑：mimic 是机械齿轮/弹簧实现的，CAN 命令只控制主控关节。

### 3.3 ★★ `linker_o6.yml` 含 `dex-retargeting` 不识别的字段

**症状**：`RetargetingConfig.from_dict()` 抛 `TypeError: unexpected keyword argument 'target_link_human_indices_dexpilot'`。

**根因**：YAML 里的字段名是 `target_link_human_indices_dexpilot`（多一个后缀），且其中的索引 `[0, 4, 9, 14, 19, 24]` 是 **PICO 26 编号**，不是 MediaPipe 21 编号——即便能识别也会越界。

**修复**：`retargeter._load_section` 用白名单 `_ALLOWED_CFG_KEYS` 过滤未知字段，把这个键悄悄丢掉。一旦 `target_link_human_indices = None`，DexPilot 会自动生成正确的 MediaPipe 21 索引。日志中会打印：

```
[Retargeter] 忽略 YAML 中不支持的字段: ['target_link_human_indices_dexpilot']
```

### 3.4 ★★ URDF 相对路径与 `set_default_urdf_dir` 拼接出错

**症状**：`ValueError: URDF path .../linker_o6/linker_o6/linker_o6/right/linkerhand_o6_right.urdf does not exist`，多了一段 `linker_o6`。

**根因**：YAML 写的是 `urdf_path: linker_o6/linker_o6/right/linkerhand_o6_right.urdf`，而 `RetargetingConfig.set_default_urdf_dir("/.../linker_o6")` 会在它前面再拼一段。

**修复**：`retargeter._resolve_urdf_path` 在传给 `RetargetingConfig.from_dict` 之前就把 `urdf_path` 解析成**绝对路径**，依次尝试 `yaml_dir / yaml_dir.parent / yaml_dir.parent.parent / 仓库根`，命中第一个存在的文件即可。

### 3.5 ★★ YAML 顶层键可能是 `retargeting:` 也可能是 `left:` / `right:`

`dex-retargeting` 自带 sample 用 `retargeting:` 顶层；项目里早期版本用 `left:` / `right:` 顶层。`_load_section` 优先取 `retargeting`，否则按 `--hand right|left` 取对应段，兼容两种结构。

### 3.6 ★ PICO 数据单位、坐标系、quality flag

- 单位 **米**（不是毫米！）：sanity check 时 `wrist→middle_tip` ≈ 0.17 m；如果看到 ≈ 170 那是毫米。
- 坐标系 **绝对世界系**：所有点都是相对某个 PICO 自定义原点，不是相对手腕；阶段 ② 必须减 wrist。
- `active = 0`：丢帧，**不能**喂给重定向器，否则上一帧 qpos 一直被新随机解覆盖造成抖动。`teleop_*.py` 已加判断：`active != 1` 时保持上一帧目标。

### 3.7 ★ 弧度 → 0~255 命令方向不要颠倒

Linker O6 真机约定：

- `cmd = 250` 张开
- `cmd =   0` 屈曲

URDF 约定：

- `qpos = 0` 张开
- `qpos = upper` 屈曲

所以映射是 `cmd = 250 * (1 - q / upper)`，**有个 "1 -"**。曾经反过来一次，导致真机一上电就直接握拳到底。`teleop_real.py` 里有 `--dry-run` 选项可以在不连 CAN 的情况下验证 cmd 输出再上机。

### 3.8 ☆ DexPilot `project_dist` / `escape_dist` / `eta1/2`

当人手两根指尖距离 `< project_dist` 时，DexPilot 会**强制把对应 robot 指尖投影到一起**（捏合"吸附"）。Linker O6 默认 `project_dist=0.03, escape_dist=0.05`，这两个值是"米"，给 PICO 数据时候直接生效。如果发现拇指-食指总是莫名捏在一起，把这两个值在 `linker_o6.yml` 里调大。

### 3.9 ☆ 低通滤波 `low_pass_alpha`

YAML 里 `low_pass_alpha: 0.2`：`alpha` 越小越光滑、越迟钝；`>= 1.0` 表示禁用滤波。如果你看到真机/仿真有"残影感"先把它调大到 0.5~0.7。

## 4. 怎么自查（按顺序排）

跑 `python3 teleop/teleop_mujoco.py --debug --debug-every 30`，按下面 5 步看输出：

1. `**active = 1`** 才有意义；否则 PICO 没抓到手。
2. **"变换前 PICO 世界系 (mm)"** 中 wrist 数值在 `±2000 mm` 范围内 → 单位是米；如果是 `±0.1` 那是 SDK 已经给了米。
3. **"尺度判定"**：成人张开时 `wrist→middle_tip ≈ 170 mm` 左右；`thumb_tip→index_tip` 在张开 ~80 mm、捏合 ~10 mm，会动就 OK。
4. **"rot 行列式 = +1.0000 (OK ✓)"**，否则 SVD 系反了（左右手设错或共线退化）。
5. `**[Retarget rad]`** 行：张开各 `mcp_pitch ≈ 0`，握拳 `mcp_pitch ≈ 1.5~1.6`，单指弯曲只对应那根变化。

外加启动时一次：

1. `**R_mano_to_base` 校准**：`det = +1.0000`，并且 `mano +z → base_link 系下` 接近某个 ±单轴方向（说明 robot 的"手指轴"被识别出来了）。

任一项不对就照表里的"坑 #3.x"对症修复。

## 5. 文件引用速查


| 文件                                                   | 用途                                                               |
| ---------------------------------------------------- | ---------------------------------------------------------------- |
| `teleop/pico_hand.py`                                | PICO SDK 读取、26→21 映射、MANO 对齐、debug 打印                            |
| `teleop/retargeter.py`                               | 包装 `dex-retargeting`、YAML 清洗、自动 R 校准                             |
| `teleop/teleop_mujoco.py`                            | MuJoCo viewer 主循环                                                |
| `teleop/teleop_real.py`                              | CAN 真机主循环 + rad→0-255                                            |
| `load_in_mujoco.py::load_hand`                       | URDF→MJCF，注入 `<equality>` + 可选 actuator                          |
| `hand_control_example.py::kinematic_control`         | 主控关节写 qpos + 手动求解 mimic                                          |
| `linker_o6/linker_o6/linker_o6.yml`                  | DexPilot 配置：URDF 路径、关节名、scaling、滤波                               |
| `linker_o6/linker_o6/right/linkerhand_o6_right.urdf` | URDF：6 主动关节 + 5 mimic                                            |
| `o6_right_hand/controller.py`                        | CAN 真机 SDK（`O6RightHand.move(cmd)`）                              |
| `dex-retargeting/src/dex_retargeting/optimizer.py`   | DexPilot 优化器实现（`generate_link_indices`、`get_objective_function`） |


