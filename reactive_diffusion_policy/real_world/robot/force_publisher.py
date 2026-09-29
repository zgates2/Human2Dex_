import time
import struct
import serial
import logging
from typing import Optional, List
from loguru import logger

class ForceSensorReader:
    """
    一个独立的类，用于从串口读取和解析六维力传感器的数据。
    """
    def __init__(self, port: str, baud_rate: int = 115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.data_buffer = b''
        self.FRAME_HEADER = b'\x01\x04\x18'
        self.FRAME_LENGTH = 29

        self.last_valid_wrench: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.success_count = 1
        self.failure_count = 1

        try:
            self.ser = serial.Serial(
                self.port, self.baud_rate, timeout=0.1,
                bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE
            )
            if self.ser.is_open:
                logger.info(f"Serial port {self.port} opened successfully.")
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            raise

    def _crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _receive_frame(self) -> Optional[bytes]:
        try:
            bytes_to_read = self.ser.in_waiting
            if bytes_to_read > 0:
                self.data_buffer += self.ser.read(bytes_to_read)
        except serial.SerialException as e:
            logger.error(f"Error reading from serial port: {e}")
            return None

        header_index = self.data_buffer.find(self.FRAME_HEADER)
        if header_index == -1:
            self.data_buffer = self.data_buffer[-len(self.FRAME_HEADER):]
            return None

        if header_index > 0:
            self.data_buffer = self.data_buffer[header_index:]

        if len(self.data_buffer) >= self.FRAME_LENGTH:
            frame = self.data_buffer[:self.FRAME_LENGTH]
            self.data_buffer = self.data_buffer[self.FRAME_LENGTH:]

            frame_data_without_crc = frame[:-2]
            crc_received = int.from_bytes(frame[-2:], byteorder='little')
            crc_calculated = self._crc16(frame_data_without_crc)

            if crc_received == crc_calculated:
                return frame_data_without_crc
            else:
                # logger.warning(f"CRC check failed: received {crc_received:04X}, calculated {crc_calculated:04X}")
                return None
        return None

    def _parse_data(self, frame_data: bytes) -> Optional[List[float]]:
        try:
            values = []
            for i in range(3, 27, 4):
                value = struct.unpack('>f', frame_data[i:i+4])[0]
                values.append(value)
            return values
        except struct.error as e:
            logger.error(f"Error parsing data: {e}")
            return None

    def get_wrench(self) -> Optional[List[float]]:
        """
        请求并获取一次力/力矩数据。
        """
        if not self.ser or not self.ser.is_open:
            logger.error("Serial port is not open.")
            return None
        try:
            self.ser.write(bytes([0x01, 0x04, 0x00, 0x00, 0x00, 0x0C, 0xF0, 0x0F]))
            time.sleep(0.006)
            raw_frame = self._receive_frame()
            if raw_frame:
                parsed_data = self._parse_data(raw_frame)
                if parsed_data is not None:
                    self.last_valid_wrench = parsed_data
                    self.success_count += 1
                    return parsed_data
        except serial.SerialException as e:
            logger.info(f"Failed to write to serial port: {e}")

        self.failure_count += 1
        logger.warning("Failed to get new wrench data, returning last known value.")
        return self.last_valid_wrench

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info(f"Failure rate is {self.failure_count / self.success_count  * 100 }%")
            logger.info("Serial port closed.")


def main():
    SERIAL_PORT = '/dev/ttyACM0' 
    
    try:
        sensor = ForceSensorReader(port=SERIAL_PORT)
    except Exception as e:
        logger.error(f"Could not start force sensor test: {e}")
        return

    try:
        logger.info("Starting force sensor test. Press Ctrl+C to stop.")
        while True:
            wrench = sensor.get_wrench()
            if wrench:
                formatted_wrench = [f"{x:8.3f}" for x in wrench]
                print(f"Fx, Fy, Fz, Mx, My, Mz: {formatted_wrench}")
            else:
                print("No data received.")
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping force sensor test.")
    finally:
        sensor.close()


if __name__ == "__main__":
    # 要测试力传感器，请取消下面的注释
    main()