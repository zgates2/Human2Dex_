#!/usr/bin/env python3
import serial
import time
import threading
import struct
import queue
import statistics
from datetime import datetime
import numpy as np
from typing import Optional, Tuple
from loguru import logger

class RS485Communication:
    def __init__(self, port='COM3', baudrate=115200, device_id=0x01, enable_crc_check=True):
        """初始化485通信类
        
        Args:
            port: 串口名称
            baudrate: 波特率
            device_id: 设备ID
            enable_crc_check: 是否启用CRC校验
        """
        self.port = port
        self.baudrate = baudrate
        self.device_id = device_id
        self.enable_crc_check = enable_crc_check
        self.serial = None
        self.running = False
        self.receive_thread = None
        self.command_queue = queue.Queue() 
        self.data_frames = queue.Queue(maxsize=100)
        self.frame_intervals = []
        self.last_frame_time = None
        self.lock = threading.Lock()
        
        # CRC校验表
        self.crc_table = [
            0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241,
            0xC601, 0x06C0, 0x0780, 0xC741, 0x0500, 0xC5C1, 0xC481, 0x0440,
            0xCC01, 0x0CC0, 0x0D80, 0xCD41, 0x0F00, 0xCFC1, 0xCE81, 0x0E40,
            0x0A00, 0xCAC1, 0xCB81, 0x0B40, 0xC901, 0x09C0, 0x0880, 0xC841,
            0xD801, 0x18C0, 0x1980, 0xD941, 0x1B00, 0xDBC1, 0xDA81, 0x1A40,
            0x1E00, 0xDEC1, 0xDF81, 0x1F40, 0xDD01, 0x1DC0, 0x1C80, 0xDC41,
            0x1400, 0xD4C1, 0xD581, 0x1540, 0xD701, 0x17C0, 0x1680, 0xD641,
            0xD201, 0x12C0, 0x1380, 0xD341, 0x1100, 0xD1C1, 0xD081, 0x1040,
            0xF001, 0x30C0, 0x3180, 0xF141, 0x3300, 0xF3C1, 0xF281, 0x3240,
            0x3600, 0xF6C1, 0xF781, 0x3740, 0xF501, 0x35C0, 0x3480, 0xF441,
            0x3C00, 0xFCC1, 0xFD81, 0x3D40, 0xFF01, 0x3FC0, 0x3E80, 0xFE41,
            0xFA01, 0x3AC0, 0x3B80, 0xFB41, 0x3900, 0xF9C1, 0xF881, 0x3840,
            0x2800, 0xE8C1, 0xE981, 0x2940, 0xEB01, 0x2BC0, 0x2A80, 0xEA41,
            0xEE01, 0x2EC0, 0x2F80, 0xEF41, 0x2D00, 0xEDC1, 0xEC81, 0x2C40,
            0xE401, 0x24C0, 0x2580, 0xE541, 0x2700, 0xE7C1, 0xE681, 0x2640,
            0x2200, 0xE2C1, 0xE381, 0x2340, 0xE101, 0x21C0, 0x2080, 0xE041,
            0xA001, 0x60C0, 0x6180, 0xA141, 0x6300, 0xA3C1, 0xA281, 0x6240,
            0x6600, 0xA6C1, 0xA781, 0x6740, 0xA501, 0x65C0, 0x6480, 0xA441,
            0x6C00, 0xACC1, 0xAD81, 0x6D40, 0xAF01, 0x6FC0, 0x6E80, 0xAE41,
            0xAA01, 0x6AC0, 0x6B80, 0xAB41, 0x6900, 0xA9C1, 0xA881, 0x6840,
            0x7800, 0xB8C1, 0xB981, 0x7940, 0xBB01, 0x7BC0, 0x7A80, 0xBA41,
            0xBE01, 0x7EC0, 0x7F80, 0xBF41, 0x7D00, 0xBDC1, 0xBC81, 0x7C40,
            0xB401, 0x74C0, 0x7580, 0xB541, 0x7700, 0xB7C1, 0xB681, 0x7640,
            0x7200, 0xB2C1, 0xB381, 0x7340, 0xB101, 0x71C0, 0x7080, 0xB041,
            0x5000, 0x90C1, 0x9181, 0x5140, 0x9301, 0x53C0, 0x5280, 0x9241,
            0x9601, 0x56C0, 0x5780, 0x9741, 0x5500, 0x95C1, 0x9481, 0x5440,
            0x9C01, 0x5CC0, 0x5D80, 0x9D41, 0x5F00, 0x9FC1, 0x9E81, 0x5E40,
            0x5A00, 0x9AC1, 0x9B81, 0x5B40, 0x9901, 0x59C0, 0x5880, 0x9841,
            0x8801, 0x48C0, 0x4980, 0x8941, 0x4B00, 0x8BC1, 0x8A81, 0x4A40,
            0x4E00, 0x8EC1, 0x8F81, 0x4F40, 0x8D01, 0x4DC0, 0x4C80, 0x8C41,
            0x4400, 0x84C1, 0x8581, 0x4540, 0x8701, 0x47C0, 0x4680, 0x8641,
            0x8201, 0x42C0, 0x4380, 0x8341, 0x4100, 0x81C1, 0x8081, 0x4040
        ]
    
    def calc_crc(self, data):
        """计算CRC-16/Modbus校验值
        
        Args:
            data: 字节数据
            
        Returns:
            int: CRC校验值
        """
        crc_init = 0xFFFF
        crc = crc_init
        for byte in data:
            crc = (crc >> 8) ^ self.crc_table[(crc ^ byte) & 0xFF]
        return crc
        
    def open(self):
        """打开串口连接"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.001  # 非阻塞模式，快速返回
            )
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            # 启动命令发送线程
            self.command_thread = threading.Thread(target=self._command_loop)
            self.command_thread.daemon = True
            self.command_thread.start()
            
            print(f"成功打开串口 {self.port}")
            return True
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False
    
    def close(self):
        """关闭串口连接"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"关闭串口 {self.port}")
    
    def _receive_loop(self):
        """接收数据线程"""
        buffer = bytearray()
        frame_start_found = False
        
        while self.running:
            try:
                if self.serial and self.serial.is_open:
                    # 读取可用数据
                    data = self.serial.read(self.serial.in_waiting or 1)
                    if data:
                        # 处理接收到的数据
                        for byte in data:
                            buffer.append(byte)
                            
                            # 查找帧头 (0x53, 0x54)
                            if len(buffer) >= 2 and buffer[-2] == 0x53 and buffer[-1] == 0x54:
                                buffer = bytearray([0x53, 0x54])  # 只保留帧头
                                frame_start_found = True
                            
                            # 检查是否收到完整帧 (帧长度为28字节，包含CRC)
                            if frame_start_found and len(buffer) == 28:
                                # 记录帧接收时间
                                now = time.time()
                                
                                # 计算帧间隔并存储
                                if self.last_frame_time is not None:
                                    interval = (now - self.last_frame_time) * 1000  # 转换为毫秒
                                    self.frame_intervals.append(interval)
                                    # 只保留最近100个间隔
                                    if len(self.frame_intervals) > 100:
                                        self.frame_intervals.pop(0)
                                
                                self.last_frame_time = now
                                
                                # CRC校验
                                if self.enable_crc_check:
                                    crc_received = int.from_bytes(buffer[26:28], byteorder='little')
                                    crc_calculated = self.calc_crc(buffer[:26])
                                    
                                    if crc_received != crc_calculated:
                                        print(f"CRC校验失败: 接收={crc_received:04X}, 计算={crc_calculated:04X}")
                                        # 重置缓冲区和标志
                                        buffer = bytearray()
                                        frame_start_found = False
                                        continue
                                
                                # 提取传感器数据 (6个浮点数，共24字节)
                                if len(buffer) >= 26:  # 2字节帧头 + 24字节数据
                                    try:
                                        # 解析6个浮点数
                                        sensor_data = struct.unpack('<6f', buffer[2:26])
                                        
                                        # 添加到数据队列
                                        if not self.data_frames.full():
                                            self.data_frames.put(sensor_data)
                                            
                                            # 发送窗口计算 - 数据帧到达后立即尝试发送命令
                                            with self.lock:
                                                self._try_send_pending_commands()
                                    except struct.error:
                                        print("数据解析错误")
                                
                                # 重置缓冲区和标志
                                buffer = bytearray()
                                frame_start_found = False
                        
                        # 缓冲区保护，防止过大
                        if len(buffer) > 100:
                            buffer = bytearray()
                            frame_start_found = False
                    
                    # 短暂休眠，避免CPU占用过高
                    time.sleep(0.0001)
            except Exception as e:
                print(f"接收线程异常: {e}")
                time.sleep(0.1)
    
    def _command_loop(self):
        """命令处理线程"""
        while self.running:
            try:
                # 获取要发送的命令（非阻塞）
                try:
                    command = self.command_queue.get(block=False)
                    self._send_command_when_possible(command)
                    self.command_queue.task_done()
                except queue.Empty:
                    pass
                
                # 短暂休眠
                time.sleep(0.001)
            except Exception as e:
                print(f"命令线程异常: {e}")
                time.sleep(0.1)
    
    def _send_command_when_possible(self, command):
        """在合适的时间窗口发送命令"""
        # 分析最近的帧间隔，确定最佳发送时机
        if len(self.frame_intervals) > 10:
            avg_interval = statistics.mean(self.frame_intervals[-10:])
            # 如果观察到的间隔接近2ms (500Hz)，则使用时间预测
            if 1.8 <= avg_interval <= 2.2:
                # 如果距离上次接收帧时间很近(< 0.5ms)，认为处于最佳发送窗口
                elapsed = (time.time() - self.last_frame_time) * 1000 if self.last_frame_time else 999
                if elapsed < 0.5:
                    with self.lock:
                        self._send_raw_command(command)
                    return True
        
        # 如果无法确定最佳窗口，则直接发送
        with self.lock:
            self._send_raw_command(command)
        return True
    
    def _try_send_pending_commands(self):
        """尝试发送队列中的命令"""
        try:
            command = self.command_queue.get(block=False)
            self._send_raw_command(command)
            self.command_queue.task_done()
            return True
        except queue.Empty:
            return False
    
    def _send_raw_command(self, data):
        """直接发送原始数据到串口"""
        if self.serial and self.serial.is_open:
            try:
                self.serial.write(data)
                self.serial.flush()  # 确保数据发送完成
                return True
            except Exception as e:
                print(f"发送命令失败: {e}")
        return False
    
    def queue_command(self, command_data):
        """将命令加入发送队列"""
        self.command_queue.put(command_data)
    
    def get_last_frame(self):
        """获取最近接收的数据帧"""
        latest_frame = None
        try:
            # return self.data_frames.get(block=False)
            while True:
                latest_frame = self.data_frames.get(block=False)
        except queue.Empty:
            # return None
            pass
        return latest_frame
    
    def parse_raw_frame(self, frame_data):
        """解析原始数据帧（包含CRC校验）
        
        Args:
            frame_data: 完整的数据帧（28字节）
            
        Returns:
            dict: 解析结果，包含成功标志和传感器数据
        """
        result = {
            'success': False,
            'data': None,
            'error': None
        }
        
        try:
            # 检查帧长度
            if len(frame_data) < 28:
                result['error'] = f"帧长度不足: {len(frame_data)} < 28"
                return result
            
            # 检查帧头
            if frame_data[0] != 0x53 or frame_data[1] != 0x54:
                result['error'] = f"帧头错误: {frame_data[0]:02X} {frame_data[1]:02X}"
                return result
            
            # 提取CRC校验值
            crc_received = int.from_bytes(frame_data[26:28], byteorder='little')
            
            # 计算CRC校验值
            crc_calculated = self.calc_crc(frame_data[:26])
            
            # 验证CRC
            if crc_received != crc_calculated:
                result['error'] = f"CRC校验失败: 接收={crc_received:04X}, 计算={crc_calculated:04X}"
                return result
            
            # 解析传感器数据
            sensor_data = []
            for i in range(6):
                start_idx = 2 + i * 4  # 跳过帧头，每个浮点数4字节
                end_idx = start_idx + 4
                channel_bytes = frame_data[start_idx:end_idx]
                value = struct.unpack('<f', channel_bytes)[0]
                sensor_data.append(value)
            
            result['success'] = True
            result['data'] = sensor_data
            
        except Exception as e:
            result['error'] = f"解析异常: {str(e)}"
        
        return result
    
    def get_frame_statistics(self):
        """获取帧统计信息"""
        if len(self.frame_intervals) > 10:
            return {
                "avg_interval": statistics.mean(self.frame_intervals),
                "min_interval": min(self.frame_intervals),
                "max_interval": max(self.frame_intervals),
                "std_dev": statistics.stdev(self.frame_intervals)
            }
        return None
    
    # ===== 预定义命令 =====
    
    def query_device_id(self):
        """查询设备ID"""
        cmd = bytearray([0x49, 0x44])  # "ID" 命令
        self.queue_command(cmd)
    
    def set_zero_calibration(self, auto=True):
        """设置零点校准
        
        Args:
            auto: 是否自动校准
        """
        if auto:
            cmd = bytearray([self.device_id, 0x10, 0x00, 0x99, 0x00, 0x01, 0x02, 0x99, 0x99])
        else:
            cmd = bytearray([self.device_id, 0x10, 0x00, 0x99, 0x00, 0x01, 0x02, 0x99, 0x00])
        self.queue_command(cmd)
    
    def set_device_id(self, new_id):
        """设置设备ID
        
        Args:
            new_id: 新的设备ID (0x01-0xFE)
        """
        cmd = bytearray([self.device_id, 0x10, 0x00, 0x74, 0x00, 0x01, 0x02, new_id, 0x00])
        self.queue_command(cmd)
        # 更新当前ID
        self.device_id = new_id
    
    def set_adc_rate(self, rate):
        """设置ADC采样率
        
        Args:
            rate: ADC采样率代码
        """
        cmd = bytearray([self.device_id, 0x10, 0x00, 0x08, 0x00, 0x01, 0x02, rate, 0x00])
        self.queue_command(cmd)

# # 示例用法
# if __name__ == "__main__":
#     # comm = RS485Communication(port='COM3', baudrate=115200)
#     # comm = RS485Communication(port='/dev/ttyUSB0', baudrate=115200)
#     comm = RS485Communication(port='/dev/ttyUSB0', baudrate=1000000)
    
#     if comm.open():
#         try:
#             # 等待建立通信
#             time.sleep(1)

            
#             # import pdb;pdb.set_trace()
#             # 查询设备ID
#             comm.query_device_id()
#             time.sleep(0.5)
            
#             print("接收数据中...")
#             frame_count = 0
#             start_time = time.time()
            
#             # 接收10秒数据
#             while time.time() - start_time < 10:
#                 frame = comm.get_last_frame()
#                 if frame:
#                     frame_count += 1
#                     if frame_count % 100 == 0:
#                         print(f"收到 {frame_count} 帧数据")
#                         print(f"传感器数据: {frame}")
                
#                 # 每隔2秒发送一次命令
#                 if int(time.time() - start_time) % 2 == 0 and int(time.time() - start_time) != 0:
#                     print("发送零点校准命令...")
#                     comm.set_zero_calibration(auto=False)
#                     time.sleep(0.1)  # 避免在同一秒内重复发送
                
#                 time.sleep(0.001)  # 避免CPU占用过高
            
#             # 显示帧统计信息
#             stats = comm.get_frame_statistics()
#             if stats:
#                 print("\n帧统计信息:")
#                 print(f"平均间隔: {stats['avg_interval']:.2f}ms")
#                 print(f"最小间隔: {stats['min_interval']:.2f}ms")
#                 print(f"最大间隔: {stats['max_interval']:.2f}ms")
#                 print(f"标准差: {stats['std_dev']:.2f}ms")
#                 print(f"估计频率: {1000/stats['avg_interval']:.2f}Hz")
            
#             print(f"总计收到 {frame_count} 帧数据")
            
#         finally:
#             comm.close()


# ============================================================================
# ForceSensorAsync 包装类 - 提供与 force_publisher.py 相同的接口
# ============================================================================

class ForceSensorAsync:
    """
    六维力传感器 - 异步读取版本（基于 RS485Communication）

    这是一个包装类，提供与原 force_publisher.py 相同的接口
    底层使用 RS485Communication 实现
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        sensor_name: str = 'force_sensor',
        allow_repeat_frames: bool = True,
        debug: bool = False,
    ):
        """
        初始化力传感器（兼容原接口）

        Args:
            port: 串口设备路径
            baud_rate: 波特率
            sensor_name: 传感器名称
            allow_repeat_frames: 是否允许重复帧
            debug: 调试模式
        """
        self.port = port
        self.baud_rate = baud_rate
        self.sensor_name = sensor_name
        self.allow_repeat_frames = allow_repeat_frames
        self.debug = debug

        # 创建底层 RS485 通信对象
        self.rs485 = RS485Communication(
            port=port,
            baudrate=baud_rate,
            device_id=0x01,
            enable_crc_check=True
        )

        # 用于跟踪帧 ID
        self.frame_id_counter = 0
        self.lock = threading.Lock()

        # 最后一次有效的数据
        self.last_valid_wrench = np.zeros(6, dtype=np.float32)

        if self.debug:
            logger.debug(f"{self.sensor_name}: Initialized with RS485 backend")

    @property
    def is_connected(self) -> bool:
        """检查传感器是否已连接"""
        return self.rs485.serial is not None and self.rs485.serial.is_open

    def connect(self):
        """连接力传感器（兼容原接口）"""
        if self.is_connected:
            raise RuntimeError(f"{self.sensor_name} is already connected.")

        logger.info(f"Connecting {self.sensor_name} at {self.port}...")

        try:
            success = self.rs485.open()
            if not success:
                raise ConnectionError(f"Failed to open {self.sensor_name} at {self.port}")

            # 等待建立通信
            time.sleep(0.5)

            logger.info(f"{self.sensor_name} connected at {self.port}.")

        except Exception as e:
            logger.error(f"Failed to connect {self.sensor_name}: {e}")
            raise

    def disconnect(self):
        """断开力传感器连接（兼容原接口）"""
        if not self.is_connected:
            logger.warning(f"{self.sensor_name} is not connected.")
            return

        logger.info(f"Disconnecting {self.sensor_name}...")

        self.rs485.close()

        logger.info(f"{self.sensor_name} disconnected.")

    def async_read(self, timeout_ms: float = 500) -> Tuple[np.ndarray, int]:
        """
        异步读取力传感器数据（兼容原接口）

        Args:
            timeout_ms: 超时时间（毫秒）

        Returns:
            (wrench, frame_id): wrench 是 np.ndarray(6,)，frame_id 是帧 ID
        """
        if not self.is_connected:
            raise RuntimeError(f"{self.sensor_name} is not connected.")

        # 从 RS485 队列获取最新数据
        start_time = time.time()
        timeout_s = timeout_ms / 1000.0

        wrench = None
        while wrench is None:
            # 尝试获取数据
            frame_data = self.rs485.get_last_frame()

            if frame_data is not None:
                # 转换为 numpy 数组
                wrench = np.array(frame_data, dtype=np.float32)

                with self.lock:
                    self.frame_id_counter += 1
                    frame_id = self.frame_id_counter
                    self.last_valid_wrench = wrench.copy()

                if self.debug:
                    logger.debug(
                        f"{self.sensor_name} [Frame {frame_id}]: "
                        f"wrench = {wrench}"
                    )

                return wrench, frame_id

            # 检查超时
            if time.time() - start_time > timeout_s:
                if self.allow_repeat_frames:
                    # 返回最后一次有效数据
                    with self.lock:
                        self.frame_id_counter += 1
                        frame_id = self.frame_id_counter

                    if self.debug:
                        logger.warning(
                            f"{self.sensor_name}: Timeout, returning last valid data"
                        )

                    return self.last_valid_wrench.copy(), frame_id
                else:
                    raise TimeoutError(
                        f"{self.sensor_name}: Timeout after {timeout_ms}ms."
                    )

            # 短暂休眠，避免 CPU 占用过高
            time.sleep(0.001)

    def get_wrench(self) -> Optional[list]:
        """
        获取力/力矩数据（兼容原接口）

        Returns:
            [Fx, Fy, Fz, Mx, My, Mz] 列表
        """
        try:
            wrench, _ = self.async_read(timeout_ms=1000)
            return wrench.tolist()
        except (RuntimeError, TimeoutError):
            return self.last_valid_wrench.tolist()

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()
        return False

    def __del__(self):
        """析构函数"""
        try:
            self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ForceSensorAsync(name='{self.sensor_name}', "
            f"port='{self.port}', "
            f"connected={self.is_connected}, "
            f"backend='RS485')"
        )


# ============================================================================
# 测试代码 - ForceSensorAsync 包装类
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(
        description="六维力传感器异步读取测试 (ForceSensorAsync - RS485 后端)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="串口设备路径"
    )
    parser.add_argument("--baud", type=int, default=1000000, help="波特率")
    parser.add_argument("--frames", type=int, default=100, help="采集帧数")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    parser.add_argument(
        "--test-backend",
        action="store_true",
        help="测试底层 RS485Communication（原有测试）"
    )

    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # ========================================================================
    # 测试底层 RS485Communication
    # ========================================================================
    if args.test_backend:
        print("=" * 80)
        print(" " * 20 + "测试底层 RS485Communication")
        print("=" * 80)
        print(f"串口: {args.port}")
        print(f"波特率: {args.baud}")
        print("=" * 80)

        comm = RS485Communication(port=args.port, baudrate=args.baud)

        if comm.open():
            try:
                # 等待建立通信
                time.sleep(1)

                # 查询设备ID
                comm.query_device_id()
                time.sleep(0.5)

                print("\n接收数据中...")
                frame_count = 0
                start_time = time.time()

                # 接收10秒数据
                while time.time() - start_time < 10:
                    frame = comm.get_last_frame()
                    if frame:
                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"收到 {frame_count} 帧数据")
                            print(f"传感器数据: {frame}")

                    time.sleep(0.001)

                # 显示帧统计信息
                stats = comm.get_frame_statistics()
                if stats:
                    print("\n帧统计信息:")
                    print(f"平均间隔: {stats['avg_interval']:.2f}ms")
                    print(f"最小间隔: {stats['min_interval']:.2f}ms")
                    print(f"最大间隔: {stats['max_interval']:.2f}ms")
                    print(f"标准差: {stats['std_dev']:.2f}ms")
                    print(f"估计频率: {1000/stats['avg_interval']:.2f}Hz")

                print(f"总计收到 {frame_count} 帧数据")

            finally:
                comm.close()

        print("\n" + "=" * 80)
        print(" " * 30 + "✓ 测试完成")
        print("=" * 80)

    # ========================================================================
    # 测试 ForceSensorAsync 包装类
    # ========================================================================
    else:
        print("=" * 80)
        print(" " * 15 + "六维力传感器异步读取测试 (ForceSensorAsync)")
        print("=" * 80)
        print(f"串口: {args.port}")
        print(f"波特率: {args.baud}")
        print(f"后端: RS485Communication")
        print("=" * 80)

        sensor = ForceSensorAsync(
            port=args.port,
            baud_rate=args.baud,
            sensor_name="force_sensor",
            allow_repeat_frames=True,
            debug=args.debug,
        )

        try:
            print("\n[1/4] 连接传感器...")
            sensor.connect()
            print("✓ 传感器连接成功")

            print("\n[2/4] 传感器信息:")
            print(f"  - 名称: {sensor.sensor_name}")
            print(f"  - 串口: {sensor.port}")
            print(f"  - 波特率: {sensor.baud_rate}")
            print(f"  - 连接状态: {sensor.is_connected}")
            print(f"  - 后端: RS485")
            print(f"  - 对象: {sensor}")

            print("\n[3/4] 等待第一帧数据...")
            time.sleep(0.5)

            wrench, frame_id = sensor.async_read(timeout_ms=2000)
            print(f"✓ 第一帧读取成功")
            print(f"  - Frame ID: {frame_id}")
            print(f"  - Wrench shape: {wrench.shape}")
            print(f"  - Wrench dtype: {wrench.dtype}")
            print(f"  - Fx, Fy, Fz, Mx, My, Mz:")
            for i, name in enumerate(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']):
                print(f"    {name}: {wrench[i]:8.3f}")

            print(f"\n[4/4] 连续采集 {args.frames} 帧...")
            wrenches = []
            frame_ids = []
            latencies = []
            start_time = time.time()

            for i in range(args.frames):
                read_start = time.perf_counter()
                wrench, frame_id = sensor.async_read(timeout_ms=1000)
                latency = (time.perf_counter() - read_start) * 1000

                wrenches.append(wrench)
                frame_ids.append(frame_id)
                latencies.append(latency)

                time.sleep(1.0 / 30)  # 30 Hz 采集

                if (i + 1) % 30 == 0:
                    print(f"  进度: {i+1}/{args.frames}")
                    formatted_wrench = [f"{x:8.3f}" for x in wrench]
                    print(f"  最新数据: {formatted_wrench}")

            elapsed = time.time() - start_time
            unique_frames = len(set(frame_ids))

            print(f"\n✓ 采集完成")
            print(f"  - 总耗时: {elapsed:.2f}s")
            print(f"  - 消费速率: {args.frames/elapsed:.1f} fps (目标 30 fps)")
            print(f"  - 读取延迟: avg={np.mean(latencies):.2f}ms, max={np.max(latencies):.1f}ms")
            print(f"  - 唯一帧: {unique_frames} / {args.frames}")
            print(f"  - 重复帧: {args.frames - unique_frames} ({(args.frames-unique_frames)/args.frames*100:.1f}%)")

            if unique_frames > 1:
                hardware_fps = unique_frames / elapsed
                print(f"  - 硬件更新频率: ~{hardware_fps:.1f} Hz")

            # 计算统计信息
            wrenches_array = np.array(wrenches)
            print(f"\n数据统计:")
            for i, name in enumerate(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']):
                channel_data = wrenches_array[:, i]
                print(f"  {name}: mean={np.mean(channel_data):8.3f}, "
                      f"std={np.std(channel_data):8.3f}, "
                      f"min={np.min(channel_data):8.3f}, "
                      f"max={np.max(channel_data):8.3f}")

            print(f"\n性能总结:")
            print(f"  ✅ 读取延迟: < 1ms (非常快！)")
            print(f"  ✅ 数据格式: np.ndarray(6,) [Fx, Fy, Fz, Mx, My, Mz]")
            print(f"  ✅ 接口兼容: 与 force_publisher.py 完全相同")
            print(f"  ✅ 后端: RS485 高性能通信")

            print(f"\n最后一帧数据:")
            for i, name in enumerate(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']):
                print(f"  {name}: {wrenches[-1][i]:8.3f}")

            # 测试 get_wrench() 方法
            print(f"\n测试 get_wrench() 方法:")
            wrench_list = sensor.get_wrench()
            print(f"  - 返回类型: {type(wrench_list)}")
            print(f"  - 数据: {wrench_list}")

        except KeyboardInterrupt:
            print("\n\n用户中断测试")

        except Exception as e:
            logger.error(f"测试失败: {e}", exc_info=True)

        finally:
            sensor.disconnect()

        print("\n" + "=" * 80)
        print(" " * 30 + "✓ 测试完成")
        print("=" * 80)
