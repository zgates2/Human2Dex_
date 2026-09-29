import struct
import time
import cv2
import numpy as np
from typing import Dict
from SynexensPythonSDK import *

g_mapStreamType: Dict[int, SYStreamTypeEnum] = {}
g_mapSavePCL: Dict[int, bool] = {}
g_mapSaveDepthOrRaw: Dict[int, bool] = {}


def CreateOpencvWindow(nDeviceID: int, streamType: SYStreamTypeEnum, bDestoryOld: bool = False) -> None:
    name: str = ""
    if streamType == SYStreamTypeEnum.SYSTREAMTYPE_NULL:
        pass
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RAW:
        if bDestoryOld:
            name = "RGBD_depth_{}".format(nDeviceID)
            cv2.destroyWindow(name)
            name = "RGBD_RGB_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "raw_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTH:
        if bDestoryOld:
            name = "raw_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGB:
        if bDestoryOld:
            name = "depth_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR:
        if bDestoryOld:
            name = "RGB_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "ir_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHRGB:
        if bDestoryOld:
            name = "ir_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIRRGB:
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "ir_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
        if bDestoryOld:
            name = "depth_{}".format(nDeviceID)
            cv2.destroyWindow(name)
            name = "ir_{}".format(nDeviceID)
            cv2.destroyWindow(name)
            name = "RGB_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "RGBD_depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "RGBD_RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

bGetDepthIntric:bool = True
bGetRGBIntric:bool = True
def ProcessFrameData(nDeviceID, pFrameData: POINTER(SYFrameData)):
    global bGetDepthIntric
    global bGetRGBIntric
    if g_mapStreamType.get(nDeviceID) is not None:
        if streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
            objFrameData = pFrameData.contents
            mapIndex: Dict[SYFrameTypeEnum, int] = {}
            mapPos: Dict[SYFrameTypeEnum, int] = {}
            nPos: int = 0
            for nFrameIndex in range(objFrameData.m_nFrameCount):
                mapIndex[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nFrameIndex
                mapPos[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nPos
                nPos += objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameHeight * objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(c_short)
            itDepthIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_DEPTH)
            itRGBIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_RGB)
            nRGBDWidth = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth
            nRGBDHeight = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight
            pRGBDDepth = (c_ushort * (nRGBDWidth * nRGBDHeight))()
            pRGBDRGB = (c_ubyte * (nRGBDWidth * nRGBDHeight * 3))()
            if itDepthIndex is not None and itRGBIndex is not None:
                data_pointer_depth = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_DEPTH], POINTER(c_ushort))
                data_pointer_rgb = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_RGB], POINTER(c_ubyte))
                errorCodeGetRGBD = GetRGBD(nDeviceID,
                                           objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameWidth,
                                           objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameHeight, data_pointer_depth,
                                           objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth,
                                           objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight, data_pointer_rgb,
                                           nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB)
                if errorCodeGetRGBD == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    # Depth
                    nCount = nRGBDHeight * nRGBDWidth

                    ArrayType = c_ubyte * (nCount * 3)
                    pColor = ArrayType()

                    rgbd_depth_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)

                    gray16 = np.zeros((nRGBDHeight, nRGBDWidth), dtype=np.uint16)
                    memmove(gray16.ctypes.data, pRGBDDepth, nCount * 2)

                    if GetDepthColor(nDeviceID, nCount, pRGBDDepth, pColor) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                        memmove(rgbd_depth_bgr.ctypes.data, pColor, nRGBDHeight * nRGBDWidth * 3)
                        rgbd_depth_rgb = cv2.cvtColor(rgbd_depth_bgr, cv2.COLOR_BGR2RGB)
                        cv2.imshow("RGBD_depth_{}".format(nDeviceID), rgbd_depth_rgb)
                    else:
                        tmp = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX)
                        gray8 = cv2.convertScaleAbs(tmp)
                        rgbimg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
                        cv2.imshow("RGBD_depth_{}".format(nDeviceID), rgbimg)
                    del pColor

                    # RGB
                    rgbd_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)
                    memmove(rgbd_bgr.ctypes.data, pRGBDRGB, nRGBDHeight * nRGBDWidth * 3)
                    rgbd_rgb = cv2.cvtColor(rgbd_bgr, cv2.COLOR_BGR2RGB)
                    cv2.imshow("RGBD_RGB_{}".format(nDeviceID), rgbd_rgb)
                else:
                    PrintErrorCode("GetRGBD", errorCodeGetRGBD)

                del pRGBDDepth
                del pRGBDRGB
        else:
            objFrameData = pFrameData.contents
            nPos: int = 0
            for i in range(objFrameData.m_nFrameCount):
                nFrameHeight = objFrameData.m_pFrameInfo[i].m_nFrameHeight
                nFrameWidth = objFrameData.m_pFrameInfo[i].m_nFrameWidth
                nCount = nFrameHeight * nFrameWidth
                if objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RAW:
                    pRaw = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pRaw, ptr_new, sizeof(c_ushort) * nCount)
                    gray16 = np.frombuffer(pRaw, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)
                    gray8 = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    cv2.imshow("raw_{}".format(nDeviceID), gray8)
                    # 保存Raw图
                    if g_mapSaveDepthOrRaw.get(nDeviceID) is not None and g_mapSaveDepthOrRaw[nDeviceID]:
                        pre = f"{nDeviceID}_{objFrameData.m_pFrameInfo[i].m_nFrameWidth}x{objFrameData.m_pFrameInfo[i].m_nFrameHeight}-{int(time.time())}"
                        depth_name = f"{pre}.raw"

                        with open(depth_name, "wb") as fp:
                            fp.write(objFrameData.m_pData[nPos:].tobytes())

                        g_mapSaveDepthOrRaw[nDeviceID] = False
                    nPos += nCount * sizeof(c_short)
                elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_DEPTH:
                    pDepth = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pDepth, ptr_new, sizeof(c_ushort) * nCount)
                    ArrayType = c_ubyte * (nCount * 3)
                    pColor = ArrayType()
                    depthimg = np.zeros((nFrameHeight, nFrameWidth, 3), dtype=np.uint8)
                    if GetDepthColor(c_uint(nDeviceID), nCount, pDepth, pColor) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                        memmove(depthimg.ctypes.data, pColor,objFrameData.m_pFrameInfo[i].m_nFrameHeight * objFrameData.m_pFrameInfo[i].m_nFrameWidth * 3)
                        img = cv2.cvtColor(depthimg, cv2.COLOR_BGR2RGB)
                        cv2.imshow("depth_{}".format(nDeviceID), img)
                        itSavePCL = g_mapSavePCL.get(nDeviceID)
                        if itSavePCL is not None:
                            if itSavePCL:
                                LP_SYPointCloudData = POINTER(SYPointCloudData)
                                data_array = (SYPointCloudData * nCount)()
                                pPCLData = cast(data_array, LP_SYPointCloudData)
                                if GetDepthPointCloud(nDeviceID, nFrameWidth, nFrameHeight, pDepth,pPCLData) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    filename = str(nDeviceID) + "PointCloudData" + str(int(time.time() * 1000000)) + ".pcd"
                                    strFileName = os.path.abspath(os.path.dirname(__file__))
                                    # 打开文件并写入.pcd文件头信息
                                    with open(os.path.join(strFileName, filename), "w") as fp:
                                        fp.write("# .PCD v0.7 - Point Cloud Data file format\n")
                                        fp.write("VERSION 0.7\n")
                                        fp.write("FIELDS x y z rgb\n")
                                        fp.write("SIZE 4 4 4 4\n")
                                        fp.write("TYPE F F F U\n")
                                        fp.write("COUNT 1 1 1 1\n")
                                        fp.write(f"WIDTH  {nCount}\n")
                                        fp.write("HEIGHT 1\n")
                                        fp.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                                        fp.write(f"POINTS {nCount}\n")
                                        fp.write("DATA ascii\n")
                                        # 将颜色数据转换为整数并写入点云数据
                                        for n in range(nCount):
                                            cTempType = c_ubyte * 4
                                            cTemp = cTempType()
                                            cTemp[1] = c_ubyte(pColor[n * 3])
                                            cTemp[2] = c_ubyte(pColor[n * 3 + 1])
                                            cTemp[3] = c_ubyte(pColor[n * 3 + 2])
                                            nTemp = struct.unpack('I', cTemp)[0]
                                            fp.write(f"{pPCLData[n].m_fltX} {pPCLData[n].m_fltY} {pPCLData[n].m_fltZ} {nTemp}\n")
                                del pPCLData
                                g_mapSavePCL[nDeviceID] = False
                    else:
                        gray16 = np.zeros((nFrameHeight, nFrameWidth), dtype=np.uint16)
                        memmove(gray16.ctypes.data, pDepth, nCount * 2)
                        tmp = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX)
                        gray8 = cv2.convertScaleAbs(tmp)
                        rgbimg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
                        cv2.imshow("depth_{}".format(nDeviceID), rgbimg)
                    del pColor
                    
                    # 获取内参示例代码
                    # if bGetDepthIntric:
                    #     resolution = SYResolutionEnum()
                    #     GetFrameResolution(c_uint(nDeviceID),SYFrameTypeEnum.SYFRAMETYPE_DEPTH,resolution)

                    #     intrinsics = SYIntrinsics()
                    #     GetIntric(c_uint(nDeviceID),resolution,False,intrinsics)

                    #     print(resolution.value)
                    #     print(intrinsics)
                    #     bGetDepthIntric = False


                    # 更新数据的偏移量
                    nPos += nCount * sizeof(c_short)
                elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_IR:
                    pIR = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pIR, ptr_new, sizeof(c_ushort) * nCount)
                    gray16 = np.frombuffer(pIR, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)

                    # gray8 = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    gray8 = np.zeros((nFrameHeight, nFrameWidth), dtype=np.uint8)
                    # for i in range(0, nFrameHeight * nFrameWidth):
                    #     nTemp = gray16[i // nFrameWidth, i % nFrameWidth] // 2
                    #     if nTemp > 255:
                    #         nTemp = 255
                    #     gray8[i // nFrameWidth, i % nFrameWidth] = nTemp
                    cv2.convertScaleAbs(gray16, gray8, 0.5, 0)

                    cv2.imshow("ir_{}".format(nDeviceID), gray8)
                    nPos += nCount * sizeof(c_short)

                elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RGB:
                    data_pointer = cast(objFrameData.m_pData + nPos, POINTER(c_ubyte))
                    data_array = np.ctypeslib.as_array(data_pointer, shape=(nFrameHeight * nFrameWidth * 2,))
                    img_yuyv_cvt = data_array.reshape(nFrameHeight, nFrameWidth, 2)
                    yuyv_array = np.frombuffer(img_yuyv_cvt, dtype=np.uint8)
                    yuyv_image = yuyv_array.reshape((nFrameHeight, nFrameWidth, 2))
                    rgb_image = cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2BGR_YUYV)
                    cv2.imshow("RGB_{}".format(nDeviceID), rgb_image)

                    # 获取内参示例代码
                    # if bGetRGBIntric:
                    #     resolution = SYResolutionEnum()
                    #     GetFrameResolution(c_uint(nDeviceID),SYFrameTypeEnum.SYFRAMETYPE_RGB,resolution)

                    #     intrinsics = SYIntrinsics()
                    #     GetIntric(c_uint(nDeviceID),resolution,True,intrinsics)

                    #     print(resolution.value)
                    #     print(intrinsics)
                    #     bGetRGBIntric = False

                    nPos += nCount * 3 // 2


if __name__ == "__main__":
    print("SDKVersion:" + GetSDKVersion())
    # 初始化所有设备类型（含 CS20 系列）
    errorCodeInitSDK: SYErrorCodeEnum = InitSDK()
    # derviceList = [
    #     SYDeviceTypeEnum.SYDEVICETYPE_CS40PRO,
    #     SYDeviceTypeEnum.SYDEVICETYPE_CS30_DUAL,
    # ]
    # errorCodeInitSDK: SYErrorCodeEnum = InitSDKSelectDevices(derviceList)
    if errorCodeInitSDK != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        # 如果初始化成功
        PrintErrorCode("InitSDK", errorCodeInitSDK)

    nDeviceCount = c_int32()
    errorCodeFindDevice = FindDevice(byref(nDeviceCount), None)
    if errorCodeFindDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nDeviceCount.value > 0:
        if nDeviceCount.value <= 0:
            errorCodeRet = SYErrorCodeEnum.ERRORCODE_DEVICELISTEMPTY
        else:
            pDeviceInfo = (SYDeviceInfo * nDeviceCount.value)()
            errorCodeFindDevice = FindDevice(nDeviceCount, pDeviceInfo)
            if errorCodeFindDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                pOpen = np.zeros(nDeviceCount.value, dtype=bool)
                pIntegralTimeMin = [c_int(0)] * nDeviceCount.value
                pIntegralTimeMax = [c_int(0)] * nDeviceCount.value
                pIntegralTime = [c_int(0)] * nDeviceCount.value
                for i in range(nDeviceCount.value):
                    g_mapSavePCL[pDeviceInfo[i].m_nDeviceID] = False
                    print("DeviceID ={} Type={}".format(pDeviceInfo[i].m_nDeviceID, pDeviceInfo[i].m_deviceType.value))
                    errorCodeOpenDevice: SYErrorCodeEnum = OpenDevice(pDeviceInfo[i])
                    if errorCodeOpenDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                        strDeviceSN: str = GetDeviceSN(pDeviceInfo[i].m_nDeviceID)
                        print("DeviceSN:" + strDeviceSN)
                        strDeviceHWVersion: str = GetDeviceHWVersion(pDeviceInfo[i].m_nDeviceID)
                        print("DeviceHWVersion:" + strDeviceHWVersion)
                        supportTypeList = list()
                        errorCode = QueryDeviceSupportFrameType(pDeviceInfo[i].m_nDeviceID, supportTypeList)
                        if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                            print("Device Support Frame Type:")
                            for j in range(len(supportTypeList)):
                                print("DeviceID ={} FrameType={}".format(pDeviceInfo[i].m_nDeviceID, supportTypeList[j]))
                                supportResolutionList = list()
                                errorCodeQueryResolution = QueryDeviceSupportResolution(pDeviceInfo[i].m_nDeviceID, supportTypeList[j], supportResolutionList)
                                if errorCodeQueryResolution == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    print("DeviceID ={} FrameType={} Resolution:".format(pDeviceInfo[i].m_nDeviceID, supportTypeList[j]))
                                    for k in range(len(supportResolutionList)):
                                        print("DeviceID ={} FrameType={} Resolution:{}".format(pDeviceInfo[i].m_nDeviceID, supportTypeList[j], supportResolutionList[k]))
                                else:
                                    errorCodeRet = SYErrorCodeEnum.SYERRORCODE_FAILED
                                    PrintErrorCode("QueryDeviceSupportResolution", errorCodeRet)

                        deviceType = pDeviceInfo[i].m_deviceType.value
                        if deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS30_DUAL or deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS30_SINGLE:
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, SYResolutionEnum.SYRESOLUTION_640_480)
                            if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID,SYFrameTypeEnum.SYFRAMETYPE_RGB,SYResolutionEnum.SYRESOLUTION_1920_1080)
                                if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR
                                    errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                    if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                        g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                        pOpen[i] = True
                                        CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                        elif deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS20_P:
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID,SYFrameTypeEnum.SYFRAMETYPE_DEPTH,SYResolutionEnum.SYRESOLUTION_320_240)
                            if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR
                                errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                    pOpen[i] = True
                                    CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                        elif deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS20_SINGLE:
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, SYResolutionEnum.SYRESOLUTION_320_240)
                            if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR
                                errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                    pOpen[i] = True
                                    CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                                else:
                                    PrintErrorCode("StartStreaming", errorCode)
                            else:
                                PrintErrorCode("SetFrameResolution Depth", errorCode)
                        elif deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS20_DUAL:
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, SYResolutionEnum.SYRESOLUTION_640_480)
                            if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR
                                errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                    pOpen[i] = True
                                    CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                                else:
                                    PrintErrorCode("StartStreaming", errorCode)
                            else:
                                PrintErrorCode("SetFrameResolution Depth", errorCode)
                        if deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS40:
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID,SYFrameTypeEnum.SYFRAMETYPE_DEPTH,SYResolutionEnum.SYRESOLUTION_640_480)
                            if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR
                                errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                    pOpen[i] = True
                                    CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                                else:
                                    PrintErrorCode("StartStreaming", errorCode)
                            else:
                                PrintErrorCode("SetFrameResolution Depth", errorCode)
                        elif deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS40PRO:
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID,SYFrameTypeEnum.SYFRAMETYPE_DEPTH,SYResolutionEnum.SYRESOLUTION_640_480)
                            if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID,SYFrameTypeEnum.SYFRAMETYPE_RGB,SYResolutionEnum.SYRESOLUTION_1600_1200)
                                streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
                                errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                if errorCode == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                    pOpen[i] = True
                                    CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                                else:
                                    PrintErrorCode("StartStreaming", errorCode)
                            else:
                                PrintErrorCode("SetFrameResolution Depth", errorCode)

                for nDeviceIndex in range(nDeviceCount.value):
                    fFilterParams = (c_float * 10)()
                    num = c_int(1)
                    errorFilterParam = SetFilterParam(pDeviceInfo[nDeviceIndex].m_nDeviceID,SYFilterTypeEnum.SYFILTERTYPE_AMPLITITUD, num, fFilterParams)

                    if errorFilterParam == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                        print("SetFilterParam SUCCESS")
                    else:
                        PrintErrorCode("SetFilterParam", errorFilterParam)

                    bFilter = c_bool(False)
                    errorCodeFilter = SetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter)

                    if errorCodeFilter == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                        print(f"SetFilter success, bFilter = {bFilter}")
                    else:
                        PrintErrorCode("SetFilter", errorCodeFilter)

                while True:
                    for nDeviceIndex in range(0, nDeviceCount.value):
                        if pOpen[nDeviceIndex]:
                            pFrameData = POINTER(SYFrameData)()
                            errorCodeLastFrame: SYErrorCodeEnum = GetLastFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, byref(pFrameData))
                            if errorCodeLastFrame == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                ProcessFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pFrameData)
                            else:
                                pass
                                # PrintErrorCode("GetLastFrameData", errorCodeLastFrame)
                    bBreak = False
                    res: int = cv2.waitKey(30)
                    if res == 27:  # ESC
                        bBreak = True
                    elif res == 85:  # Windows U:85 Ubuntu U:117 Increase the integration time.
                        for nDeviceIndex in range(nDeviceCount.value):
                            nMin = c_int(0)
                            nMax = c_int(0)
                            errorCodeIntegralTime = GetIntegralTimeRange(pDeviceInfo[nDeviceIndex].m_nDeviceID, SYResolutionEnum.SYRESOLUTION_640_480, nMin, nMax)
                            if errorCodeIntegralTime == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                pIntegralTimeMin[nDeviceIndex] = nMin
                                pIntegralTimeMax[nDeviceIndex] = nMax
                                print("Current device integral time range:{} - {}".format(pIntegralTimeMin[nDeviceIndex].value, pIntegralTimeMax[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTimeRange", errorCodeIntegralTime)
                            nIntegralTime = c_int(0)
                            errorCodeIntegralTime = GetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, nIntegralTime)
                            pIntegralTime[nDeviceIndex] = nIntegralTime
                            if errorCodeIntegralTime == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("Current device integral time:{}".format(pIntegralTime[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTime", errorCodeIntegralTime)
                            if pIntegralTime[nDeviceIndex].value <= pIntegralTimeMin[nDeviceIndex].value:
                                print("IntegralTime already be Min")
                            else:
                                pIntegralTime[nDeviceIndex].value += 100
                                if pIntegralTime[nDeviceIndex].value >= pIntegralTimeMax[nDeviceIndex].value:
                                    pIntegralTime[nDeviceIndex] = pIntegralTimeMax[nDeviceIndex]
                                errorCodeIntegralTime = SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex])
                                if errorCodeIntegralTime == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    print("SetIntegralTime:{}".format(pIntegralTime[nDeviceIndex].value))
                                else:
                                    PrintErrorCode("SetIntegralTime", errorCodeIntegralTime)
                    elif res == 68:  # Windows D:68 Ubuntu D:100 Reduce the integration time
                        for nDeviceIndex in range(nDeviceCount.value):
                            nMin = c_int(0)
                            nMax = c_int(0)
                            errorCodeIntegralTime = GetIntegralTimeRange(pDeviceInfo[nDeviceIndex].m_nDeviceID,SYResolutionEnum.SYRESOLUTION_640_480, nMin, nMax)
                            if errorCodeIntegralTime == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                pIntegralTimeMin[nDeviceIndex] = nMin
                                pIntegralTimeMax[nDeviceIndex] = nMax
                                print("Current device integral time range:{} - {}".format(pIntegralTimeMin[nDeviceIndex].value, pIntegralTimeMax[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTimeRange", errorCodeIntegralTime)
                            nIntegralTime = c_int(0)
                            errorCodeIntegralTime = GetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID,nIntegralTime)
                            pIntegralTime[nDeviceIndex] = nIntegralTime
                            if errorCodeIntegralTime == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("Current device integral time:{}".format(pIntegralTime[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTime", errorCodeIntegralTime)
                            if pIntegralTime[nDeviceIndex].value <= pIntegralTimeMin[nDeviceIndex].value:
                                print("IntegralTime already be Min")
                            else:
                                pIntegralTime[nDeviceIndex].value -= 100
                                if pIntegralTime[nDeviceIndex].value <= pIntegralTimeMin[nDeviceIndex].value:
                                    pIntegralTime[nDeviceIndex] = pIntegralTimeMin[nDeviceIndex]
                                errorCodeIntegralTime = SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex])
                                if errorCodeIntegralTime == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                    print("SetIntegralTime:{}".format(pIntegralTime[nDeviceIndex].value))
                                else:
                                    PrintErrorCode("SetIntegralTime", errorCodeIntegralTime)
                    elif res == 70:  # Windows F:70 Ubuntu F:102 filter
                        for nDeviceIndex in range(nDeviceCount.value):
                            bFilter = c_bool(False)
                            errorCodeFilter = GetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter)
                            if errorCodeFilter == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("GetFilter Success, bFilter = {}".format(bFilter.value))
                            else:
                                PrintErrorCode("GetFilter", errorCodeFilter)
                            bFilter = c_bool(not bFilter.value)
                            errorCodeFilter = SetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter)
                            if errorCodeFilter == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("SetFilter Success, bFilter = {}".format(bFilter.value))
                            else:
                                PrintErrorCode("SetFilter", errorCodeFilter)
                    elif res == 77:  # Windows M:77 Ubuntu M:109 Mirror
                        for nDeviceIndex in range(nDeviceCount.value):
                            bMirror = c_bool(False)
                            errorCodeMirror = GetMirror(pDeviceInfo[nDeviceIndex].m_nDeviceID, bMirror)
                            if errorCodeMirror == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("GetMirror Success, bMirror = {}".format(bMirror.value))
                            else:
                                PrintErrorCode("GetMirror", errorCodeMirror)
                            bMirror = c_bool(not bMirror.value)
                            errorCodeMirror = SetMirror(pDeviceInfo[nDeviceIndex].m_nDeviceID, bMirror)
                            if errorCodeMirror == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("SetMirror Success, bMirror = {}".format(bMirror.value))
                            else:
                                PrintErrorCode("SetMirror", errorCodeMirror)
                    elif res == 73:  # Windows I:73 Ubuntu I:105 Flip
                        for nDeviceIndex in range(nDeviceCount.value):
                            bFlip = c_bool(False)
                            errorCodeFlip = GetFlip(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFlip)
                            if errorCodeFlip == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("GetFlip Success, bFlip = {}".format(bFlip.value))
                            else:
                                PrintErrorCode("GetFlip", errorCodeFlip)
                            bFlip = c_bool(not bFlip.value)
                            errorCodeFlip = SetFlip(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFlip)
                            if errorCodeFlip == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                                print("SetFlip Success, bFlip = {}".format(bFlip.value))
                            else:
                                PrintErrorCode("SetFlip", errorCodeFlip)
                    elif res == 80:  # Windows P:80 Ubuntu P:112 save pointcloud data
                        for nDeviceIndex in range(nDeviceCount.value):
                            itSavePCL = g_mapSavePCL.get(pDeviceInfo[nDeviceIndex].m_nDeviceID)
                            if itSavePCL is not None:
                                g_mapSavePCL[pDeviceInfo[nDeviceIndex].m_nDeviceID] = True
                    elif res == 86:  # Windows V:86 Ubuntu V:118 save raw
                        for nDeviceIndex in range(nDeviceCount.value):
                            itSaveRAW = g_mapSaveDepthOrRaw.get(pDeviceInfo[nDeviceIndex].m_nDeviceID)
                            if itSaveRAW is not None:
                                g_mapSaveDepthOrRaw[pDeviceInfo[nDeviceIndex].m_nDeviceID] = True
                    if bBreak:
                        break

                for i in range(nDeviceCount.value):
                    if pOpen[i]:
                        errorCode = StopStreaming(pDeviceInfo[i].m_nDeviceID)

                del pOpen
                del pIntegralTime
                del pIntegralTimeMin
                del pIntegralTimeMax
            else:
                errorCodeRet = SYErrorCodeEnum.SYERRORCODE_FAILED
                PrintErrorCode("FindDevice", errorCodeRet)
    else:
        errorCodeRet = SYErrorCodeEnum.SYERRORCODE_FAILED
        PrintErrorCode("FindDevice 0 code", errorCodeRet)