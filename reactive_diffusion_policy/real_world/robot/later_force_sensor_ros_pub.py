#!/usr/bin/env python3
"""
Wrench sensor driver (non-ROS version)
Usage:
    sensor = WrenchSensor(port='/dev/ttyUSB3')
    sensor.start()
    wrench = sensor.get_wrench()  # Returns [fx, fy, fz, tx, ty, tz] or None
    sensor.stop()
"""

import serial
import time
import threading
import struct
import queue
import statistics
import logging


class RS485Communication:
    def __init__(self, port='/dev/ttyUSB3', baudrate=1000000, device_id=0x01, enable_crc_check=True, logger=None):
        """Initialize RS485 communication

        Args:
            port: Serial port name
            baudrate: Baud rate
            device_id: Device ID
            enable_crc_check: Enable CRC check
            logger: Logger instance
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

        self.logger = logger if logger else self._get_default_logger()

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

    def _get_default_logger(self):
        """Create default logger"""
        logger = logging.getLogger("WrenchSensor")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def calc_crc(self, data):
        """Calculate CRC-16/Modbus checksum"""
        crc = 0xFFFF
        for byte in data:
            crc = (crc >> 8) ^ self.crc_table[(crc ^ byte) & 0xFF]
        return crc

    def open(self):
        """Open serial connection"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.001
            )
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()

            self.command_thread = threading.Thread(target=self._command_loop)
            self.command_thread.daemon = True
            self.command_thread.start()

            self.logger.info(f"Opened serial port {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to open serial port: {e}")
            return False

    def close(self):
        """Close serial connection"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        if hasattr(self, 'command_thread') and self.command_thread:
            self.command_thread.join(timeout=1.0)
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.logger.info(f"Closed serial port {self.port}")

    def _receive_loop(self):
        """Receive data thread"""
        buffer = bytearray()
        frame_start_found = False

        while self.running:
            try:
                if self.serial and self.serial.is_open:
                    data = self.serial.read(self.serial.in_waiting or 1)
                    if data:
                        for byte in data:
                            buffer.append(byte)

                            if len(buffer) >= 2 and buffer[-2] == 0x53 and buffer[-1] == 0x54:
                                buffer = bytearray([0x53, 0x54])
                                frame_start_found = True

                            if frame_start_found and len(buffer) == 28:
                                now = time.time()

                                if self.last_frame_time is not None:
                                    interval = (now - self.last_frame_time) * 1000
                                    self.frame_intervals.append(interval)
                                    if len(self.frame_intervals) > 100:
                                        self.frame_intervals.pop(0)

                                self.last_frame_time = now

                                if self.enable_crc_check:
                                    crc_received = int.from_bytes(buffer[26:28], byteorder='little')
                                    crc_calculated = self.calc_crc(buffer[:26])

                                    if crc_received != crc_calculated:
                                        self.logger.warning(f"CRC check failed: received={crc_received:04X}, calculated={crc_calculated:04X}")
                                        buffer = bytearray()
                                        frame_start_found = False
                                        continue

                                if len(buffer) >= 26:
                                    try:
                                        sensor_data = struct.unpack('<6f', buffer[2:26])

                                        if not self.data_frames.full():
                                            self.data_frames.put(sensor_data)

                                            with self.lock:
                                                self._try_send_pending_commands()
                                    except struct.error:
                                        self.logger.warning("Data parsing error")

                                buffer = bytearray()
                                frame_start_found = False

                        if len(buffer) > 100:
                            buffer = bytearray()
                            frame_start_found = False

                    time.sleep(0.0001)
            except Exception as e:
                if self.running:
                    self.logger.error(f"Receive thread exception: {e}")
                    time.sleep(0.1)

    def _command_loop(self):
        """Command processing thread"""
        while self.running:
            try:
                command = self.command_queue.get(block=True, timeout=0.1)
                self._send_command_when_possible(command)
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Command thread exception: {e}")
                    time.sleep(0.1)

    def _send_command_when_possible(self, command):
        """Send command at appropriate time window"""
        if len(self.frame_intervals) > 10:
            avg_interval = statistics.mean(self.frame_intervals[-10:])
            if 1.8 <= avg_interval <= 2.2:
                elapsed = (time.time() - self.last_frame_time) * 1000 if self.last_frame_time else 999
                if elapsed < 0.5:
                    with self.lock:
                        self._send_raw_command(command)
                    return True

        with self.lock:
            self._send_raw_command(command)
        return True

    def _try_send_pending_commands(self):
        """Try to send commands in queue"""
        try:
            command = self.command_queue.get(block=False)
            self._send_raw_command(command)
            self.command_queue.task_done()
            return True
        except queue.Empty:
            return False

    def _send_raw_command(self, data):
        """Send raw data to serial port"""
        if self.serial and self.serial.is_open:
            try:
                self.serial.write(data)
                self.serial.flush()
                return True
            except Exception as e:
                self.logger.warning(f"Failed to send command: {e}")
        return False

    def queue_command(self, command_data):
        """Add command to send queue"""
        self.command_queue.put(command_data)

    def get_last_frame(self):
        """Get latest received data frame"""
        try:
            return self.data_frames.get(block=False)
        except queue.Empty:
            return None


class WrenchSensor:
    """Wrench sensor interface (non-ROS)"""

    def __init__(self, port='/dev/ttyUSB3', baudrate=1000000, device_id=0x01, enable_crc_check=True):
        """Initialize wrench sensor

        Args:
            port: Serial port name
            baudrate: Baud rate (default 1000000)
            device_id: Device ID (default 0x01)
            enable_crc_check: Enable CRC check (default True)
        """
        self.comm = RS485Communication(
            port=port,
            baudrate=baudrate,
            device_id=device_id,
            enable_crc_check=enable_crc_check
        )
        self._last_wrench = None
        self._lock = threading.Lock()

    def start(self):
        """Start sensor communication"""
        if not self.comm.open():
            raise RuntimeError(f"Failed to open serial port {self.comm.port}")
        return True

    def stop(self):
        """Stop sensor communication"""
        self.comm.close()

    def get_wrench(self):
        """Get latest wrench data

        Returns:
            list: [fx, fy, fz, tx, ty, tz] or None if no data available
        """
        # Drain queue and keep latest
        frame = self.comm.get_last_frame()
        last_frame = None
        while frame is not None:
            last_frame = frame
            frame = self.comm.get_last_frame()

        if last_frame and len(last_frame) == 6:
            with self._lock:
                self._last_wrench = list(last_frame)
            return list(last_frame)

        # Return cached value if no new data
        with self._lock:
            return self._last_wrench

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


if __name__ == '__main__':
    # Test code
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=1000000, help='Baud rate')
    args = parser.parse_args()

    print(f"Starting wrench sensor on {args.port}...")

    with WrenchSensor(port=args.port, baudrate=args.baudrate) as sensor:
        print("Sensor started. Press Ctrl+C to stop.")
        try:
            while True:
                wrench = sensor.get_wrench()
                if wrench:
                    print(f"Force: [{wrench[0]:7.2f}, {wrench[1]:7.2f}, {wrench[2]:7.2f}] "
                          f"Torque: [{wrench[3]:7.2f}, {wrench[4]:7.2f}, {wrench[5]:7.2f}]")
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping...")
