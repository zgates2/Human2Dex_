# PICO 数据采集说明

本文档对应 `teleop/collect_data.py` 的 PKL v4 格式。新版数据按样本 bundle 保存：一个文件夹中包含 PKL 和 `images/` 图片子文件夹。数据保存 wrist 基准的六维位姿 `[x,y,z,rx,ry,rz]`，同时保留 MANO 21 点关键点、O6 命令和 Wuji Hand 弧度命令；RGB 图像以 jpg 写在 `images/`，PKL 中保存相对路径。

## 1. 采集命令

进入环境后运行：

```bash
conda activate pico
python3 teleop/collect_data.py --task-name pick_cube
```

交互控制：

```text
s：开始录制；录制中再按 s 会暂停并保存当前段
q：退出程序
Ctrl+C：退出程序；如果正在录制，会先保存当前段
```

`s` 键带防误触间隔，默认 0.8 秒，可用 `--key-debounce` 调整。暂停保存后程序不会退出，会回到待机状态；再次按 `s` 会继续录制下一段。

常用参数：

```bash
python3 teleop/collect_data.py --task-name pick_cube
python3 teleop/collect_data.py --task-name pick_cube --config teleop/collect_data.yaml
python3 teleop/collect_data.py --mvs-list
```

采集参数写在 `teleop/collect_data.yaml` 中：

```yaml
out:
hz: 60.0
cache_hz: 90.0
hand: right
retargeting_yaml:
wuji_retargeting_yaml: wuji_retargeting/config/adaptive_analytical_pico.yaml
duration:
key_debounce: 0.8
keep_invalid: false
mvs_serial:
dry_run: false
```

例如配置 `out: data/demo.pkl`，并使用 `--task-name pick_cube`，会实际保存为：

```text
data/pick_cube/demo/
  demo.pkl
  images/
    demo_rgb_000000.jpg
    demo_rgb_000001.jpg
```

如果同一次程序运行中录制多段，每一段都是一个独立 bundle 数据集目录：

```text
data/pick_cube/demo/
  demo.pkl
  images/
    demo_rgb_000000.jpg

data/pick_cube/demo_ep0002/
  demo.pkl
  images/
    demo_rgb_000000.jpg

data/pick_cube/demo_ep0003/
  demo.pkl
  images/
    demo_rgb_000000.jpg
```

YAML 字段含义：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `out` | `null` | 输出样本名或 PKL 路径；空值使用 `data/demo_<timestamp>.pkl`，任务目录会插在 episode bundle 外层 |
| `hz` | `60` | 最终写入 PKL 的 message 频率 |
| `cache_hz` | `90` | 后台 PICO 缓存刷新频率，实际使用 `max(cache_hz, hz)` |
| `duration` | `null` | 每段自动录制秒数；到时自动暂停保存并等待下一次 `s` |
| `key_debounce` | `0.8` | `s` 键开始/暂停的防误触间隔秒数 |
| `hand` | `right` | 采集左手或右手 |
| `retargeting_yaml` | `null` | 覆盖 retargeting 用的 linker_o6.yml 路径 |
| `wuji_retargeting_yaml` | `wuji_retargeting/config/adaptive_analytical_pico.yaml` | Wuji Hand 关键点重定向配置 |
| `keep_invalid` | `false` | 是否保留 PICO inactive 的占位帧 |
| `mvs_serial` | `null` | 指定 MVS 相机序列号；不指定则自动选择第一个相机 |
| `dry_run` | `false` | 只测试采集链路，不写文件；MVS 初始化失败时自动只跑 PICO |

> 命令行只保留 `--task-name`、`--config` 和 `--mvs-list`。MVS 默认强制开启，没有 `--no-mvs` 开关。如果需要在无相机的机器上测试 PICO 链路，请在 YAML 中设置 `dry_run: true`。

> MVS 相机参数不放在采集命令里调。当前默认对齐 OmniUMI-fisheye：`1440x1080`
> 全幅 probe 检测鱼眼黑边，自动得到 sensor crop，再输出 `480x480` RGB。

推荐设置：

```text
hz: 60
cache_hz: 90
```

如果后续接入真实 RGB 相机，建议相机也按类似方式维护 latest buffer，然后主循环按 `hz` 取最近 RGB、最近 PICO、最近机器人状态。

## 2. PKL 顶层结构

保存文件是一个 Python pickle：

```python
{
    "formatVersion": 4,
    "metadata": {...},
    "messages": [message_0, message_1, ...],
}
```

`formatVersion=4` 表示：

```text
o6_command = np.ndarray shape (6,), dtype uint8
wuji_command = np.ndarray shape (5,4), dtype float32, radians
trajectoryPose = np.ndarray shape (6,), dtype float32
rgbImage = str relative path under bundle directory, or None
```

这和旧版不同。旧版保存的是：

```text
formatVersion=1: trajectoryPose = [x, y, z, qx, qy, qz, qw]
formatVersion=2: trajectoryPose = np.ndarray shape (4,4), dtype float32
formatVersion=3: handCommand = np.ndarray shape (6,), dtype uint8
```

因此旧的回放脚本如果假设 `trajectoryPose` 是 7D 或 4x4，需要同步修改。

## 3. 每帧 message 字段

每个有效帧包含：

| 字段 | 类型/shape | 含义 |
|---|---|---|
| `timestamp` | `float` | `time.time()` 墙钟时间 |
| `mainClockMonotonicNs` | `int` | PICO SDK 时间戳 |
| `sampleClockNs` | `int` | 本机统一采样时刻，`time.monotonic_ns()` |
| `sourceReceiveNs` | `int` | 最近 PICO 帧进入缓存的本机单调时刻 |
| `sourceAgeNs` | `int` | `sampleClockNs - sourceReceiveNs` |
| `o6_command` | `np.ndarray(6,), uint8` | Linker O6 六维手部命令 |
| `wuji_command` | `np.ndarray(5,4), float32` | Wuji Hand 20 维控制弧度，按 `qpos.reshape(5, 4)` 保存 |
| `trajectoryPose` | `np.ndarray(6,), float32` | wrist 基准六维位姿 `[x,y,z,rx,ry,rz]` |
| `pts21_mano` | `np.ndarray(21,3), float32` | dex-retargeting 使用的 MANO 关键点 |
| `rgbImage` | `None` / `str` | bundle 内 `images/` 子目录中的 jpg 相对路径；未启用 MVS 时为空 |
| `rgbFrameId` | `None` / `int` | MVS 帧号 |
| `rgbCaptureNs` | `None` / `int` | RGB 帧进入 latest cache 的本机 `time.monotonic_ns()` |
| `rgbAlignResidualNs` | `None` / `int` | `sampleClockNs - rgbCaptureNs` |

如果使用 `--keep-invalid`，PICO 无效帧会保留，但：

```python
o6_command = None
wuji_command = None
trajectoryPose = None
pts21_mano = None
```

此时如果启用了 MVS，RGB 字段仍会按同一个 `sampleClockNs` 取最近帧，方便后续做多源时间对齐。

训练前通常应过滤掉这些无效帧。

## 4. o6_command / wuji_command

`o6_command` 是 6 维 `uint8`：

```python
[
    thumb_cmc_pitch,
    thumb_cmc_yaw,
    index_mcp_pitch,
    middle_mcp_pitch,
    ring_mcp_pitch,
    pinky_mcp_pitch,
]
```

当前编码约定：

```text
250 附近：接近张开
0   附近：接近最大屈曲
```

它不是笛卡尔坐标，也不属于 PICO world 或机器人 base 坐标系。它是手部执行器命令空间。

## 5. trajectoryPose 坐标系

`trajectoryPose` 是 6 维位姿：

```text
[x, y, z, rx, ry, rz]
```

含义：

```text
原点：PICO raw26x7 的 wrist 点，index=1
位置单位：米
旋转：rotvec，单位弧度；对应的 wrist/TCP 局部轴已从 PICO 原始轴重定义为目标 TCP 轴
```

PICO 原始 wrist 局部轴：

```text
old_x = 右方
old_y = 手背上方
old_z = 后方/朝手腕
```

存储后的目标局部轴：

```text
new_x = 上方
new_y = 右方
new_z = 前方
```

实现方式是只改局部姿态，不改 wrist 位置：

```python
R_world_tcp = R_world_pico @ R_pico_tcp
```

其中：

```python
R_pico_tcp = np.array([
    [0, 1,  0],
    [1, 0,  0],
    [0, 0, -1],
])
```

等价关系：

```text
new_x = old_y
new_y = old_x
new_z = -old_z
```

注意：这里没有做 `PICO world -> robot base` 的外参标定。

如果后处理需要 4x4 矩阵，可用前三维作为平移、后三维 rotvec 还原旋转矩阵。固定的全局 world/base 外参在相对位姿中会抵消，但局部 TCP 坐标轴定义会改变相对平移方向和旋转轴含义，所以必须统一。

## 6. pts21_mano

`pts21_mano` 是 `(21,3)` 的 MANO 局部关键点，来源流程：

```text
PICO raw26x7 世界点
-> PICO2MEDIAPIPE 映射到 21 点
-> 以 wrist 点为原点
-> 估计手部局部 frame
-> 转到 OPERATOR2MANO_RIGHT 坐标
```

保存它的目的：

```text
可以重新 retarget
可以调试手部关键点质量
可以训练额外的手部形状/姿态编码
避免只保留已经量化成 uint8 的 o6_command
```

## 7. 最近缓存采样

新版采集不是每次落盘都直接调用 PICO SDK / MVS SDK，而是：

```text
PICO 后台线程：按 cache_hz 读取 PICO，维护最新帧
MVS 后台线程：按采集 hz 读取 RGB，维护最新帧
主线程：按 hz 生成 message，每次取缓存中最近 PICO 和最近 RGB
```

这样做的好处：

```text
避免 PICO/MVS SDK 读取耗时阻塞主采样循环
后续多源数据能以同一个 sampleClockNs 为中心对齐
可以用 sourceAgeNs 评估每帧 PICO 状态有多新
可以用 rgbAlignResidualNs 评估每帧 RGB 与采样点的时间偏差
降低 RGB、手部状态、机器人状态之间的时间错位
```

检查建议：

```python
ages_ms = np.array([m["sourceAgeNs"] for m in messages if m["sourceAgeNs"] is not None]) / 1e6
print(ages_ms.mean(), ages_ms.max())
```

如果 `sourceAgeNs` 经常接近或超过 `1 / hz`，说明 PICO 缓存刷新太慢或 SDK 卡顿，需要提高 YAML 中的 `cache_hz` 或排查设备链路。

## 8. 训练前处理建议

训练 DexUMI/DP 前，建议从 PKL v4 生成最终训练数据时做以下处理：

1. 过滤无效帧：

```python
valid = [
    m for m in messages
    if m["o6_command"] is not None
    and m["trajectoryPose"] is not None
]
```

2. 取出 6D pose：

```python
poses = np.stack([m["trajectoryPose"] for m in valid]).astype(np.float32)
```

3. 对 DP action 使用相对位姿时，可按项目数据集约定把绝对 6D pose 转成相对 pose：

```python
pose_6d = poses
```

如果直接复用 DexUMI 的 `DexUMIDataset`，它内部已经会对 `pose` 做 `inv(T0) @ Tt`。这时你只需要保证最终 zarr 中的 `pose` 是 `[x,y,z,rx,ry,rz]` 绝对序列即可。

4. 保留 `o6_command` / `wuji_command`：

```python
o6_action = np.stack([m["o6_command"] for m in valid]).astype(np.float32)
wuji_action = np.stack([m["wuji_command"] for m in valid]).astype(np.float32)
```

如果使用原版 DexUMI 的 action 拼接方式，最终 action 语义是：

```text
relative_pose_6d + o6_action_6d
```

## 9. RGB 接入建议

当前 `collect_data.py` 默认就会接入 `teleop/mvs_cpp` 海康 MVS 后端。默认会把 RGB 图像保存到样本 bundle 的 `images/` 子目录，并把图像相对路径、帧号和对齐时间写入 PKL。实现方式是：

```text
启动时先用 MVS 全幅 probe 帧检测鱼眼黑边，自动设置 sensor crop
MVS 相机线程持续读取，维护 latest RGB buffer
message 采样时取最近 RGB
RGB jpg 编码和磁盘写入由后台 writer 线程完成
记录 rgbCaptureNs 和 rgbAlignResidualNs
不要在主循环里阻塞式等待相机帧
```

示例：

```bash
python3 teleop/collect_data.py --task-name pick_cube
python3 teleop/collect_data.py --mvs-list
```

MVS 默认强制开启；调试无相机机器时在 YAML 中设置 `dry_run: true`（MVS 初始化失败时自动只跑 PICO，不写 PKL）。

RGB 字段建议：

```python
rgbImage: str  # relative jpg path, e.g. images/demo_rgb_000123.jpg
rgbFrameId: int
rgbCaptureNs: int  # 本机 time.monotonic_ns()
rgbAlignResidualNs: sampleClockNs - rgbCaptureNs
```

默认所有 RGB 帧都会写成 jpg，PKL 内不再保存图像数组。
`metadata["rgb"]["writer"]` 会记录 jpg writer 的入队数、完成数、队列阻塞次数和写入错误数。

JPG 写盘参数在 `teleop/collect_data.yaml` 中配置：

```yaml
jpg_quality: 95
jpg_writer_workers: 2
jpg_writer_queue_size: 512
```

主采样循环只把 RGB 图像复制后放入队列，JPEG 编码和磁盘写入由后台 worker 完成。如果
`metadata["rgb"]["writer"]["blockedPuts"] > 0`，说明写盘队列曾经满过，采样线程被反向阻塞；
这时优先增大 `jpg_writer_queue_size`，然后再把 `jpg_writer_workers` 提到 `3-4`。如果
`blockedPuts == 0` 但 `missing_rgbImage` 仍然高，主要问题不是 JPG 写盘，而是 RGB/PICO
时间同步窗口、相机帧率、USB 抖动或 MVS ring buffer 命中率。

## 10. 质量检查清单

采集后建议检查：

```text
有效帧比例是否足够高
sourceAgeNs 是否稳定
trajectoryPose 是否全为有限数
pts21_mano 尺度是否合理
o6_command 是否有明显饱和到 0 或 255
rgbImage 指向的 jpg 是否存在
```

示例：

```python
import pickle
import numpy as np

with open("data/demo.pkl", "rb") as f:
    data = pickle.load(f)

msgs = data["messages"]
valid = [m for m in msgs if m["trajectoryPose"] is not None]
poses = np.stack([m["trajectoryPose"] for m in valid])

print("frames:", len(valid))
print("pose shape:", poses.shape)
print("finite:", np.isfinite(poses).all())
print("source age ms max:", max(m["sourceAgeNs"] for m in valid) / 1e6)
```

正常情况下：

```text
sourceAgeNs 应明显小于采样周期
手静止时 trajectoryPose 位置应稳定
手向前移动时 stored new_z 方向应符合预期
```

## 11. 当前已知兼容性

以下文件仍可能按旧格式理解 `trajectoryPose`：

```text
test/replay_traj_pico.py
teleop/PKL_REPLAY_SPEC.md
```

在使用新版 PKL v4 回放前，需要把旧的 “7D Palm pose” 或 “4x4 wrist pose” 读取逻辑改成读取 `(6,)` wrist pose，并把旧 `handCommand` 字段改为 `o6_command`。
