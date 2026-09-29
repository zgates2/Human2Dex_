#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O6 右手高层控制 API

用法示例：
    from controller import O6RightHand

    hand = O6RightHand()
    hand.set_speed(150)
    hand.move([250, 250, 0, 0, 0, 0])  # 只伸食指
    hand.home()
    hand.close()
"""

from driver import LinkerHandO6Driver
from presets import INIT_POS, PRESET_ACTIONS


class O6RightHand:
    """
    Linker Hand 右手 O6 高层控制器。

    封装底层 CAN 驱动，提供面向业务的简洁接口。
    """

    # ── 初始化 ──────────────────────────────────────────────────────────

    def __init__(self, can_channel: str = "can0", bitrate: int = 1_000_000):
        """
        Parameters
        ----------
        can_channel : SocketCAN 接口名，默认 'can0'
        bitrate     : CAN 波特率，默认 1 Mbps
        """
        # 右手 CAN ID = 0x27，左手为 0x28
        self._drv = LinkerHandO6Driver(
            can_channel=can_channel,
            can_id=0x27,
            bitrate=bitrate,
        )

        touch_desc = {
            1:  "力传感器",
            2:  "矩阵压感 B 型",
            3:  "矩阵压感 J 型",
            4:  "矩阵压感 F 型",
            -1: "无/未知",
        }.get(self._drv.touch_type, "无/未知")
        print(f"[Controller] 压感类型: {self._drv.touch_type} ({touch_desc})")

        # 矩阵压感请求码（O6 统一为 0xA4）
        self._touch_code = 0xA4

    # ── 运动控制 ────────────────────────────────────────────────────────

    def move(self, positions: list):
        """
        发送6轴目标位置。

        Parameters
        ----------
        positions : 长度为 6 的列表，每个值 0~255
        """
        if len(positions) != 6:
            raise ValueError(f"需要6个关节值，实际收到 {len(positions)} 个")
        if any(not isinstance(v, (int, float)) or v < 0 or v > 255 for v in positions):
            raise ValueError("关节值必须为 0~255")
        self._drv.set_joint_positions([int(v) for v in positions])

    def home(self):
        """回到完全张开的初始位置。"""
        self.move(INIT_POS)

    def preset(self, name: str):
        """
        执行预设动作。

        Parameters
        ----------
        name : 预设名称，可选值见 presets.PRESET_ACTIONS
        """
        if name not in PRESET_ACTIONS:
            raise KeyError(f"未知预设 '{name}'，可选: {list(PRESET_ACTIONS)}")
        self.move(PRESET_ACTIONS[name])

    # ── 参数设置 ────────────────────────────────────────────────────────

    def set_speed(self, speed: int):
        """
        统一设置所有关节速度。

        Parameters
        ----------
        speed : 0~255
        """
        if not isinstance(speed, (int, float)) or speed < 0 or speed > 255:
            raise ValueError("speed 必须为 0~255")
        self._drv.set_speed([int(speed)] * 6)

    def set_speed_each(self, speeds: list):
        """
        分别设置各关节速度。

        Parameters
        ----------
        speeds : 长度为 6 的列表，每个值 0~255
        """
        if len(speeds) != 6:
            raise ValueError("需要6个速度值")
        self._drv.set_speed([int(v) for v in speeds])

    def set_torque(self, torque: int):
        """
        统一设置所有关节力矩上限。

        Parameters
        ----------
        torque : 0~255
        """
        if not isinstance(torque, (int, float)) or torque < 0 or torque > 255:
            raise ValueError("torque 必须为 0~255")
        self._drv.set_torque([int(torque)] * 6)

    # ── 状态读取 ────────────────────────────────────────────────────────

    def get_state(self) -> list:
        """返回当前各关节位置（6 个值，0~255）。"""
        return self._drv.get_joint_positions()

    def get_torque(self) -> list:
        """返回当前力矩限制（6 个值）。"""
        return self._drv.get_torque()

    def get_temperature(self) -> list:
        """返回各关节电机温度（6 个值）。"""
        return self._drv.get_temperature()

    def get_fault(self) -> list:
        """返回各关节故障码（6 个值，0 = 正常）。"""
        return self._drv.get_fault()

    def get_current(self) -> list:
        """返回各关节电流（6 个值）。"""
        return self._drv.get_current()

    # ── 压感 ────────────────────────────────────────────────────────────

    def get_touch(self) -> dict:
        """
        读取压感数据，自动适配硬件类型：

        touch_type=1  → {'normal':[...], 'tangential':[...], 'dir':[...], 'approach':[...]}
        touch_type=2/3/4 → {'thumb': ndarray(10,4), 'index':..., ...}
        touch_type=-1 → {}
        """
        tt = self._drv.touch_type
        if tt == 1:
            raw = self._drv.get_force()
            return {
                "normal":     raw[0],
                "tangential": raw[1],
                "dir":        raw[2],
                "approach":   raw[3],
            }
        if tt in (2, 3, 4):
            return self._drv.get_matrix_touch(self._touch_code)
        return {}

    # ── 资源释放 ────────────────────────────────────────────────────────

    def close(self):
        """释放 CAN 资源，退出前必须调用。"""
        self._drv.close()

    # ── 上下文管理器支持 ──────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
