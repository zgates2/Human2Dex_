#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O6 底层 CAN 协议驱动

帧格式（标准帧，8 字节）：
  data[0]   = frame_type（功能码）
  data[1..] = payload

常用功能码：
  0x01  位置控制 / 位置回读
  0x02  力矩限制设置 / 回读
  0x05  速度设置
  0x33  温度回读
  0x35  故障码回读
  0x36  电流回读
  0x64 / 0xC2  固件版本
  0xC0  序列号
"""

import subprocess
import threading
import time
from typing import Optional

import can
import numpy as np


class LinkerHandO6Driver:
    """
    Linker Hand O6 CAN 底层驱动。

    一个实例对应一只手（can_id 区分左右手）。
    内部启动一个接收线程持续更新各帧缓存，外部通过属性或 get_* 方法读取。
    """

    # ── 初始化 ─────────────────────────────────────────────────────────

    def __init__(
        self,
        can_channel: str = "can0",
        can_id: int = 0x27,
        bitrate: int = 1_000_000,
        poll_hz: float = 100.0,
    ):
        """
        Parameters
        ----------
        can_channel : SocketCAN 接口名，Linux 下通常为 'can0'
        can_id      : 0x27=右手, 0x28=左手
        bitrate     : CAN 波特率，默认 1 Mbps
        poll_hz     : 后台轮询查询关节位置的频率（Hz），默认 100
        """
        self.can_id = can_id
        self.can_channel = can_channel
        self.bitrate = bitrate

        # 各帧数据缓存
        self._x01 = [0] * 6        # 关节位置
        self._x02 = [-1] * 6       # 力矩限制
        self._x05 = [0] * 6        # 速度
        self._x33 = [0] * 6        # 温度
        self._x35 = [0] * 6        # 故障码
        self._x36 = [-1] * 6       # 电流
        self._version: Optional[list] = None
        self._serial_number: list[int] = []

        # 矩阵压感缓存（10×4，-1 表示未收到数据）
        self.thumb_matrix  = np.full((10, 4), -1)
        self.index_matrix  = np.full((10, 4), -1)
        self.middle_matrix = np.full((10, 4), -1)
        self.ring_matrix   = np.full((10, 4), -1)
        self.little_matrix = np.full((10, 4), -1)

        # 力传感器缓存
        self.normal_force         = [-1.0] * 6
        self.tangential_force     = [-1.0] * 6
        self.tangential_force_dir = [-1.0] * 6
        self.approach_inc         = [-1.0] * 6

        # 矩阵帧分包行索引映射（首字节偏移 → 行号）
        self._matrix_row = {v * 16: v for v in range(12)}

        self._lock = threading.Lock()

        self._bring_up_can(can_channel, bitrate)
        self.bus = can.interface.Bus(
            channel=can_channel, interface="socketcan", bitrate=bitrate
        )
        print(f"[Driver] CAN 已连接: channel={can_channel}, id=0x{can_id:02X}")

        self._running = True
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx_thread.start()

        self._poll_interval = 1.0 / poll_hz if poll_hz > 0 else 0.01
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # 稍等接收线程就绪后读取压感类型
        time.sleep(0.1)
        self.touch_type = self._detect_touch_type()

    # ── CAN 接口管理 ────────────────────────────────────────────────────

    @staticmethod
    def _bring_up_can(channel: str, bitrate: int):
        """拉起 CAN 接口（等同于 ip link set can0 up type can bitrate 1000000）"""
        def _is_up() -> bool:
            status = subprocess.run(
                ["ip", "link", "show", channel],
                text=True,
                capture_output=True,
            )
            return status.returncode == 0 and "UP" in status.stdout

        if _is_up():
            print(f"[Driver] CAN {channel} already UP; skip ip link reconfigure.")
            return

        subprocess.run(
            ["sudo", "ip", "link", "set", channel, "down"],
            text=True,
            capture_output=True,
        )
        up = subprocess.run(
            ["sudo", "ip", "link", "set", channel, "up", "type", "can", "bitrate", str(bitrate)],
            text=True,
            capture_output=True,
        )
        if up.returncode != 0:
            if _is_up():
                print(
                    f"[Driver] CAN {channel} is UP after failed reconfigure; "
                    f"continue. stderr: {up.stderr.strip()}"
                )
                return
            raise RuntimeError(
                f"Failed to bring up CAN {channel} at bitrate {bitrate}: "
                f"{up.stderr.strip() or up.stdout.strip()}"
            )

    # ── CAN 收发 ────────────────────────────────────────────────────────

    def send_frame(
        self,
        frame_type: int,
        payload: Optional[list] = None,
        sleep_s: float = 0.005,
    ):
        """发送一帧 CAN 报文。"""
        payload = payload or []
        data = [frame_type & 0xFF] + [int(v) & 0xFF for v in payload]
        msg = can.Message(
            arbitration_id=self.can_id, data=data, is_extended_id=False
        )
        self.bus.send(msg)
        time.sleep(sleep_s)

    def _recv_loop(self):
        """后台接收线程，持续解析入帧并更新缓存。"""
        while self._running:
            try:
                msg = self.bus.recv(timeout=0.5)
                if msg is None:
                    continue
                if msg.arbitration_id != self.can_id or len(msg.data) == 0:
                    continue
                self._dispatch(msg.data[0], list(msg.data[1:]))
            except Exception:
                time.sleep(0.05)

    def _poll_loop(self):
        """后台轮询线程，以固定频率发查询帧刷新关节位置缓存。"""
        while self._running:
            try:
                self.bus.send(
                    can.Message(
                        arbitration_id=self.can_id,
                        data=[0x01],
                        is_extended_id=False,
                    )
                )
            except Exception:
                pass
            time.sleep(self._poll_interval)

    def _dispatch(self, ft: int, payload: list):
        """按功能码分发、存储帧数据。"""
        if not payload:
            return
        with self._lock:
            if ft == 0x01:
                self._x01 = payload
            elif ft == 0x02:
                self._x02 = payload
            elif ft == 0x05:
                self._x05 = payload
            elif ft == 0x20:
                self.normal_force = [float(v) for v in payload]
            elif ft == 0x21:
                self.tangential_force = [float(v) for v in payload]
            elif ft == 0x22:
                self.tangential_force_dir = [float(v) for v in payload]
            elif ft == 0x23:
                self.approach_inc = [float(v) for v in payload]
            elif ft == 0x33:
                self._x33 = payload
            elif ft == 0x35:
                self._x35 = payload
            elif ft == 0x36:
                self._x36 = payload
            elif ft in (0x64, 0xC2):
                self._version = payload
            elif ft == 0xC0:
                # 序列号分包，4 包拼接
                if payload and payload[0] in range(4):
                    self._serial_number += payload[1:]
            elif ft in (0xB1, 0xB2, 0xB3, 0xB4, 0xB5):
                matrix = [
                    self.thumb_matrix,
                    self.index_matrix,
                    self.middle_matrix,
                    self.ring_matrix,
                    self.little_matrix,
                ][ft - 0xB1]
                if len(payload) == 5:
                    row = self._matrix_row.get(payload[0])
                    if row is not None:
                        matrix[row] = payload[1:]

    # ── 运动控制 ────────────────────────────────────────────────────────

    def set_joint_positions(self, angles: list):
        """发送6轴位置指令，值域 0~255。sleep=0 以最大频率发送。"""
        self.send_frame(0x01, angles[:6], sleep_s=0)

    def set_speed(self, speed: list):
        """设置6轴速度，值域 0~255，连发 2 次提升稳定性。"""
        for _ in range(2):
            self.send_frame(0x05, speed[:6], sleep_s=0.003)

    def set_torque(self, torque: list):
        """设置6轴力矩限制，值域 0~255。"""
        self.send_frame(0x02, torque[:6])

    # ── 状态读取 ────────────────────────────────────────────────────────

    def get_joint_positions(self) -> list:
        """直接返回后台轮询线程维护的关节位置缓存，无阻塞。"""
        with self._lock:
            return list(self._x01)

    def get_torque(self) -> list:
        """查询并返回当前力矩限制。"""
        self.send_frame(0x02, [], sleep_s=0.005)
        with self._lock:
            return list(self._x02)

    def get_temperature(self) -> list:
        """查询并返回电机温度。"""
        self.send_frame(0x33, [], sleep_s=0.005)
        with self._lock:
            return list(self._x33)

    def get_fault(self) -> list:
        """查询并返回故障码。"""
        self.send_frame(0x35, [], sleep_s=0.005)
        with self._lock:
            return list(self._x35)

    def get_current(self) -> list:
        """查询并返回电机电流。"""
        self.send_frame(0x36, [], sleep_s=0.005)
        with self._lock:
            return list(self._x36)

    def get_version(self) -> Optional[list]:
        """查询固件版本，返回字节列表或 None。"""
        self.send_frame(0x64, [], sleep_s=0.1)
        time.sleep(0.1)
        with self._lock:
            v = self._version
        if v is None:
            self.send_frame(0xC2, [], sleep_s=0.1)
            time.sleep(0.1)
            with self._lock:
                v = self._version
        return v

    def get_serial_number(self) -> str:
        """查询序列号，返回字符串；失败返回 '-1'。"""
        with self._lock:
            self._serial_number = []
        self.send_frame(0xC0, [], sleep_s=0.05)
        time.sleep(0.1)
        try:
            with self._lock:
                sn = bytes(self._serial_number).decode("ascii")
            return sn if sn else "-1"
        except Exception:
            return "-1"

    # ── 压感 ────────────────────────────────────────────────────────────

    def get_force(self) -> list:
        """请求并返回 [法向力, 切向力, 切向方向, 临近量] 四组列表。"""
        self.send_frame(0x20, [], sleep_s=0.01)
        self.send_frame(0x21, [], sleep_s=0.01)
        self.send_frame(0x22, [], sleep_s=0.01)
        self.send_frame(0x23, [], sleep_s=0.01)
        with self._lock:
            return [
                list(self.normal_force),
                list(self.tangential_force),
                list(self.tangential_force_dir),
                list(self.approach_inc),
            ]

    def get_matrix_touch(self, touch_code: int = 0xA4) -> dict:
        """
        请求并返回五指矩阵压感数据。

        Returns
        -------
        dict: keys = thumb / index / middle / ring / little，值为 (10, 4) ndarray
        """
        for ft in (0xB1, 0xB2, 0xB3, 0xB4, 0xB5):
            self.send_frame(ft, [touch_code], sleep_s=0.01)
        with self._lock:
            return {
                "thumb":  self.thumb_matrix.copy(),
                "index":  self.index_matrix.copy(),
                "middle": self.middle_matrix.copy(),
                "ring":   self.ring_matrix.copy(),
                "little": self.little_matrix.copy(),
            }

    # ── 内部辅助 ─────────────────────────────────────────────────────────

    def _detect_touch_type(self) -> int:
        """
        根据序列号第5段判断压感类型：
          1 = 力传感器 (A)
          2 = 矩阵 B 型
          3 = 矩阵 J 型
          4 = 矩阵 F 型
         -1 = 无压感 (Z) 或未知
        """
        sn = self.get_serial_number()
        if sn != "-1":
            parts = sn.split("-")
            if len(parts) >= 5:
                code = parts[4]
                return {"A": 1, "B": 2, "J": 3, "F": 4, "Z": -1}.get(code, -1)
        # 无 SN 时尝试探测矩阵帧
        self.send_frame(0xB1, [], sleep_s=0.05)
        time.sleep(0.05)
        with self._lock:
            if len(self.thumb_matrix[self.thumb_matrix != -1]) > 0:
                return 2
        # 探测力传感器
        self.send_frame(0x20, [], sleep_s=0.03)
        time.sleep(0.05)
        with self._lock:
            return 1 if self.normal_force[0] != -1.0 else -1

    # ── 资源释放 ─────────────────────────────────────────────────────────

    def close(self):
        """停止所有后台线程并关闭 CAN 总线。"""
        self._running = False
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
        self.bus.shutdown()
        subprocess.run(["sudo", "ip", "link", "set", self.can_channel, "down"],
                       capture_output=True)
        print("[Driver] CAN 已关闭")
