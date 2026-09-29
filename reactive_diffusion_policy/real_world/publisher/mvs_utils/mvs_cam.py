from cProfile import label
import copy
from dataclasses import field, dataclass
from typing import Callable, Optional, Any, Dict
import multiprocessing as mp
import traceback

import sys
import os
import time
from ctypes import *

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

# from deploy.umi.real_world.camera.cam import (CamController,
#                                               CamControllerConfig)
from mvs_utils.cam import (CamController, CamControllerConfig)

try:
    sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
    from MvCameraControl_class import *

except ImportError:
    print("Failed to import MvCameraControl_class. Ensure the MVS SDK is installed and the path is correct.")


@dataclass
class MVSCamControllerConfig(CamControllerConfig):
    serial: str = ""

    width: int = 480
    height: int = 480

    transformed_width: int = 480
    transformed_height: int = 480

    # crop_func: Callable = lambda img: img[61:1057, 224:1235, :]
    crop_func: Callable = lambda img: img

    def validate(self):
        super().validate()
        assert isinstance(self.serial, str) and len(self.serial) > 0, f"Invalid camera serial: {self.serial}"

def get_all_mvs_dev_serial():
    """
    获取所有MVS设备的序列号。
    """

    deviceList = MV_CC_DEVICE_INFO_LIST()
    layerType = (MV_GIGE_DEVICE
                 | MV_USB_DEVICE
                 | MV_GENTL_CAMERALINK_DEVICE
                 | MV_GENTL_CXP_DEVICE
                 | MV_GENTL_XOF_DEVICE)

    ret = MvCamera.MV_CC_EnumDevices(layerType, deviceList)
    if ret != 0:
        print("Enum devices fail! ret[0x%x]" % ret)
        return []

    serial_numbers = []

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        strSerialNumber = ""

        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber += chr(per)
            serial_numbers.append(strSerialNumber)

    return serial_numbers

class MVSCamController(CamController):
    config: MVSCamControllerConfig

    def __init__(self, config: MVSCamControllerConfig):
        super().__init__(config)

    ################## cls methods ##################
    def initialize_sdk(self):
        """
        初始化SDK。
        """
        MvCamera.MV_CC_Initialize()
        sdk_version = MvCamera.MV_CC_GetSDKVersion()
        print("SDK Version: [0x%x]" % sdk_version)

    def enumerate_devices(self):
        """
        枚举可用设备，并将结果存储到 self.deviceList。
        """
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        layerType = (MV_GIGE_DEVICE
                     | MV_USB_DEVICE
                     | MV_GENTL_CAMERALINK_DEVICE
                     | MV_GENTL_CXP_DEVICE
                     | MV_GENTL_XOF_DEVICE)

        ret = MvCamera.MV_CC_EnumDevices(layerType, self.deviceList)
        if ret != 0:
            print("Enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if self.deviceList.nDeviceNum == 0:
            print("No device found!")
            sys.exit()

        print("Find %d devices!" % self.deviceList.nDeviceNum)

        self.DevSerialNumbers = []

        for i in range(0, self.deviceList.nDeviceNum):
            strSerialNumber = ""

            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            # 根据不同接口类型，打印相关信息
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))

            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print("\nCML device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print("\nXoF device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print("\nCXP device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            self.DevSerialNumbers.append(strSerialNumber)

    def connect_device(self, index):
        """
        根据索引连接指定设备，并完成打开设备、设置包大小、关闭触发模式等操作。
        """
        # 创建相机实例
        self.cam = MvCamera()

        # 获取设备信息
        self.stDeviceList = cast(self.deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

        # 创建句柄
        ret = self.cam.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # 若为GigE接口，则设置最佳包大小
        if self.stDeviceList.nTLayerType == MV_GIGE_DEVICE or self.stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # 设置触发模式为off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        self.nPayloadSize = stParam.nCurValue
        self.data_buf = (c_ubyte * self.nPayloadSize)()

    ################## abstract methods ##################
    def _process_commands(self):
        super()._process_commands()

    def _initialize(self):
        self.initialize_sdk()
        self.enumerate_devices()

        self.connect_device(self.DevSerialNumbers.index(self.config.serial))
        
        # ret = self.cam.MV_CC_SetIntValue("OffsetX", 0)
        # ret = self.cam.MV_CC_SetIntValue("OffsetY", 0)
        # # 再设置宽高（必须是 4 的倍数）
        # ret = self.cam.MV_CC_SetIntValue("Width", 640)
        # ret = self.cam.MV_CC_SetIntValue("Height", 480)
        # 最后设置居中偏移
        # ret = self.cam.MV_CC_SetIntValue("OffsetX", 0)
        # ret = self.cam.MV_CC_SetIntValue("OffsetY", 180)  # (1440-1080)/2 = 180
        # print(f"[ROI] Set to 1080x1080, ret={ret}")

        ret = self.cam.MV_CC_StartGrabbing()
        assert ret == 0, f"Start grabbing fail! ret[0x{ret:x}]"
        
        # self.cam.MV_CC_SetIntValue("Width", 640)   # 宽度
        # self.cam.MV_CC_SetIntValue("Height", 480)  # 高度
        # self.cam.MV_CC_SetIntValue("OffsetX", 0)    # X 偏移
        # self.cam.MV_CC_SetIntValue("OffsetY", 180)

        # ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateControlEnable", 1)
        # ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 30.0)
        # ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 60.0)

        # MV_EXPOSURE_AUTO_MODE_OFF, MV_EXPOSURE_AUTO_MODE_CONTINUOUS
        # ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeLowerLimit", 15)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 32000)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 20000)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 15000)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 10000)
        
        # ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        ret = self.cam.MV_CC_SetIntValue("ExposureTime", 15000)

        # MV_GAIN_MODE_OFF, MV_GAIN_MODE_CONTINUOUS
        # ret = self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
        # ret = self.cam.MV_CC_SetFloatValue("AutoGainLowerLimit", 0.0)
        # ret = self.cam.MV_CC_SetFloatValue("AutoGainUpperLimit", 16.9807)
        ret = self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_CONTINUOUS)

        # MV_BALANCEWHITE_AUTO_OFF, MV_BALANCEWHITE_AUTO_CONTINUOUS
        ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)

        ret = self.cam.MV_CC_SetBoolValue("BlackLevelEnable", 1)
        ret = self.cam.MV_CC_SetIntValue("BlackLevel", 240)

        ret = self.cam.MV_CC_SetIntValue("Brightness", 40)
        # ret = self.cam.MV_CC_SetIntValue("Brightness", 100)
        # ret = self.cam.MV_CC_SetIntValue("Brightness", 80)

        self.stOutFrame = MV_FRAME_OUT()
        memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))

        super()._initialize()

    def _update(self):
        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1000)
        if ret != 0:
            print(f"Get image buffer fail! ret[0x{ret:x}]")
            return

        self.last_timestamp = time.time()

        width = self.stOutFrame.stFrameInfo.nWidth
        height = self.stOutFrame.stFrameInfo.nHeight
        src_pixel_type = self.stOutFrame.stFrameInfo.enPixelType
        frame_len = self.stOutFrame.stFrameInfo.nFrameLen
        nRGBSize = width * height * 3

        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = width
        stConvertParam.nHeight = height
        stConvertParam.pSrcData = self.stOutFrame.pBufAddr
        stConvertParam.nSrcDataLen = frame_len
        stConvertParam.enSrcPixelType = src_pixel_type
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nDstBufferSize = nRGBSize

        ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
        if ret == 0:
            frame_data_ptr = cast(stConvertParam.pDstBuffer, POINTER(c_ubyte))
            frame_data = np.ctypeslib.as_array(frame_data_ptr, shape=(nRGBSize,))
            self.last_img = frame_data.reshape(height, width, 3)
        else:
            # SDK 转换失败，根据像素类型位域判断格式，用 OpenCV 手动转换
            # GigE Vision 像素类型: bits[23:16] = 每像素有效位数, bits[15:0] = 像素格式 ID
            if not hasattr(self, '_pixel_convert_warned'):
                print(f"[MVS] SDK pixel convert failed (ret=0x{ret:x}, "
                      f"src_type=0x{src_pixel_type:x}), using OpenCV fallback")
                self._pixel_convert_warned = True

            src_ptr = cast(self.stOutFrame.pBufAddr, POINTER(c_ubyte))
            bpp_bits = (src_pixel_type >> 16) & 0xFF
            pixel_id = src_pixel_type & 0xFFFF

            if bpp_bits <= 8:
                # 每像素 1 字节: Mono8 / BayerXX8
                raw_size = width * height
                src_data = np.ctypeslib.as_array(src_ptr, shape=(raw_size,)).copy()
                raw = src_data.reshape(height, width)
                if pixel_id == 0x0001:
                    self.last_img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                elif pixel_id == 0x0008:
                    self.last_img = cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)
                elif pixel_id == 0x0009:
                    self.last_img = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
                elif pixel_id == 0x000A:
                    self.last_img = cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
                elif pixel_id == 0x000B:
                    self.last_img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
                else:
                    self.last_img = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
            elif bpp_bits <= 16:
                # 每像素 2 字节: YUV422 等
                raw_size = width * height * 2
                src_data = np.ctypeslib.as_array(src_ptr, shape=(raw_size,)).copy()
                yuv = src_data.reshape(height, width, 2)
                self.last_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
            elif bpp_bits <= 24:
                # 每像素 3 字节: RGB8 / BGR8
                raw_size = width * height * 3
                src_data = np.ctypeslib.as_array(src_ptr, shape=(raw_size,)).copy()
                self.last_img = src_data.reshape(height, width, 3)
            else:
                if not hasattr(self, '_unsupported_format_warned'):
                    print(f"[MVS] Unsupported pixel format: "
                          f"bpp={bpp_bits}, pixel_id=0x{pixel_id:x}")
                    self._unsupported_format_warned = True
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                return

        self.last_img = self.config.crop_func(self.last_img)
        self.last_img = cv2.resize(self.last_img, (self.config.width, self.config.height), interpolation=cv2.INTER_LINEAR)
        self.last_img = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2RGB)

        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)

        super()._update()

    def _close(self):
        # TODO
        """Release SDK resources to avoid handle leaks across runs.

        This gets called after the process loop exits (BaseController.run -> finally).
        If we don't stop grabbing/close/destroy here, the driver may keep an
        exclusive handle, so the next run after a reboot-less restart can fail
        with MV_E_NODATA or open-device errors.  We also finalize the SDK to
        clear global state for the process.
        """
        try:
            if hasattr(self, "cam") and self.cam is not None:
                try:
                    self.cam.MV_CC_StopGrabbing()
                except Exception:
                    pass
                try:
                    self.cam.MV_CC_CloseDevice()
                except Exception:
                    pass
                try:
                    self.cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
        finally:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
        super()._close()

    def reset(self):
        super().reset()
