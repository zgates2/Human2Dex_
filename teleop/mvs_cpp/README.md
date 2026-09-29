# MVS C++ Backend

这个目录把海康 `MVS` 相机的底层取流改成了单一的 `nanobind` 扩展模块。
Python 不再通过 `ctypes` 调动态库，而是直接加载 `_mvs_camera`。

额外做了两件和稳定停机直接相关的事：

- stop 时显式执行 `ClearImageBuffer -> StopGrabbing -> CloseDevice -> DestroyHandle`
- 相机打开时显式把 SDK 内部图像缓存节点数压到可配置值，默认 `1`

图像热路径统一走同一套 `nanobind` 绑定：

- C++ 侧维护固定大小的 frame slot pool
- Python 通过 `_mvs_camera` 拿到 native `MVSCPPFrame`
- `MVSCPPFrame.as_array()` 直接返回 `numpy.ndarray` view
- 低层 `MVSCPPDevice.read_frame()` 可以直接消费这个 view
- 高层 `LatestMVSCache` 维护后台 latest buffer，采集线程直接取最新 RGB，不在主循环阻塞等图像

实现时对齐的是海康官方 SDK 提供的 C/C++ 示例思路：

- `GrabImage`: 主动取流
- `MultipleCamera`: 多相机枚举/连接
- `ConvertPixelType`: 像素格式转换

对应官方资料：

- https://www.v-club.com/home/article/10152
- https://www.v-club.com/home/article/10165
- https://www.v-club.com/home/article/1270

## 构建

默认假设 Linux 侧已经安装 MVS SDK，根目录在 `/opt/MVS`。如果不是这个路径，先设置：

```bash
export MVS_SDK_ROOT=/path/to/MVS
```

然后执行：

```bash
cd teleop/mvs_cpp
./build_backend.sh
```

构建会产出：

- `_mvs_camera*.so`

如果系统里没有 `nanobind`，CMake 会优先尝试从当前 Python 环境发现它；找不到时会自动拉取 `nanobind` 源码，因此第一次构建需要网络。
Linux 构建时会把 `MvCameraControl` 所在目录写入 RPATH；如果你的部署环境仍然找不到 SDK 动态库，再手动补：

```bash
export LD_LIBRARY_PATH="${MVS_SDK_ROOT}/Development/Libraries/64:${LD_LIBRARY_PATH}"
```

构建完成后，Python 会优先查找这些位置：

- `teleop/mvs_cpp/_mvs_camera*.so`
- `teleop/mvs_cpp/build/_mvs_camera*.so`

也可以显式指定：

```bash
export OMNIUMI_MVS_CPP_MODULE=/abs/path/to/_mvs_camera.cpython-*.so
```

## 运行时

`LatestMVSCache` 现在只保留 `cpp` 这一条 `nanobind` 链路，分成两个调用层次：

- 低层 `MVSCPPDevice` / `FrameHandle` 适合直接拿 native view 做零拷贝处理
- 高层 `LatestMVSCache` 适合采集线程/录制线程，内部维护 newest frame，优先保证主循环不阻塞

像 `1440x1080 -> 480x480` 这种路径，`LatestMVSCache` 会先临时打开全幅
probe 相机，抓几帧检测鱼眼像圈黑边，得到 sensor crop 后再重新打开正式采集。
如果检测失败，会退回中心裁剪，再交给 C++ 侧做 crop/resize。

连接相机时，Python 会额外从 `/sys/bus/usb/devices` 读取当前 USB 链路速度。若发现相机掉到了 `USB2 / 480M`，日志里会直接给出告警，避免把链路问题误判成采集代码问题。
