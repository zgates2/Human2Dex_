import os
import sys
import platform

# Linux: 未设置库路径时，设置 so 目录并重新执行进程，使链接器能找到 libSonixCamera.so
if platform.system() == 'Linux' and os.environ.get('SYNEXENS_LDPATH_DONE') != '1':
    _sdk_dir = os.path.dirname(os.path.abspath(__file__))
    _so = os.path.join(_sdk_dir, 'so')
    _env = os.environ.copy()
    _env['LD_LIBRARY_PATH'] = _so + ':' + _sdk_dir + ':' + _env.get('LD_LIBRARY_PATH', '')
    _env['SYNEXENS_LDPATH_DONE'] = '1'
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)

from typing import Any
from SYPythonDataDefine import *

absPath = os.path.abspath(os.path.dirname(__file__))
print(absPath)

if platform.system() == 'Windows':
    dllLibrary = absPath + '/dll/SynexensSDK.dll'
elif platform.system() == 'Linux':
    # dllLibrary = absPath + '/so/libSynexensSDK.so'
    machine = platform.machine().lower()
    bits, _ = platform.architecture()
    if machine in ('x86_64', 'amd64'):
        dllLibrary = os.path.join(absPath, 'so', 'libSynexensSDK.so')
    elif machine.startswith('arm') or machine.startswith('aarch'):
        # ARM boards
        if bits == '64bit':
            dllLibrary = os.path.join(
                absPath, 'so', 'arm64', 'libSynexensSDK.so')
        else:
            dllLibrary = os.path.join(
                absPath, 'so', 'arm32', 'libSynexensSDK.so')

    else:
        raise RuntimeError(
            f"Unsupported Linux architecture: {machine} ({bits})")
else:
    raise RuntimeError(f"Unsupported platform: {platform.system}")

# Linux: 仅预加载 libSonixCamera.so，避免主库加载时找不到（不预加载 libcsreconstruction2.0，因其依赖 TBB 等由主库按需加载）
if platform.system() == 'Linux':
    so_dir = os.path.dirname(dllLibrary)
    sonix = os.path.join(so_dir, 'libSonixCamera.so')
    if os.path.isfile(sonix):
        cdll.LoadLibrary(sonix)

dllLib = cdll.LoadLibrary(dllLibrary)
if dllLib == 0:
    print(" Could not open file DLL")


def PrintErrorCode(func_name, errorCode):
    print(f"Error occurred in {func_name}: {errorCode}")


def GetSDKVersion() -> str:
    nSDKVersionLength = c_int(0)
    errorCodeGetSDKVersion: SYErrorCodeEnum = dllLib.GetSDKVersion(
        byref(nSDKVersionLength), None)
    if errorCodeGetSDKVersion == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        if nSDKVersionLength.value > 0:
            pSDKVersion = (c_char * nSDKVersionLength.value)()
            errorCodeGetSDKVersion: SYErrorCodeEnum = dllLib.GetSDKVersion(
                byref(nSDKVersionLength), pSDKVersion)
            if errorCodeGetSDKVersion == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                return pSDKVersion.value.decode('utf-8')
            else:
                PrintErrorCode("GetSDKVersion2", errorCodeGetSDKVersion)
                return str()

# @ return errorCode


def InitSDK() -> SYErrorCodeEnum:
    return dllLib.InitSDK()

# Select initialization device
# @ param [in] pDeviceTyeps Device type array pointer
# @ param [in] nLen array length
# @ return errorCode


def InitSDKSelectDevices(deviceTypeList: list) -> SYErrorCodeEnum:
    nLen = len(deviceTypeList)
    DeviceArray = SYDeviceTypeEnum * nLen
    device_array = DeviceArray(*deviceTypeList)
    result = dllLib.InitSDKSelectDevices(device_array, nLen)

    return result

# @ return errorCode


def UnInitSDK() -> SYErrorCodeEnum:
    return dllLib.UnInitSDK()

# @ param [in/out] nCount: device count
# @ param [in/out] pDeviceInfo: Device information, externally allocated memory, pDevice only retrieves nCount when passed into nulliptr)
# @ return errorCode


def FindDevice(nCount: c_uint32, pDeviceInfo: POINTER(SYDeviceInfo)) -> SYErrorCodeEnum:
    dllLib.FindDevice.argtypes = [POINTER(c_int32), POINTER(SYDeviceInfo)]
    return dllLib.FindDevice(nCount, pDeviceInfo)

# @ param [in] deviceInfo
# @ return errorCode


def OpenDevice(pDevice: SYDeviceInfo) -> SYErrorCodeEnum:
    return dllLib.OpenDevice(byref(pDevice))

# @ param [in] nDeviceID
# @ return errorCode


def CloseDevice(nDeviceID: c_uint32) -> SYErrorCodeEnum:
    return dllLib.CloseDevice(nDeviceID)

# Query the frame types supported by the device.
# @ param [in] nDeviceID
# @ param [in/out] supportTypeList:List of supported frame types
# @ return errorCode


def QueryDeviceSupportFrameType(nDeviceID: c_uint32, supportTypeList: list) -> SYErrorCodeEnum:
    nSupportTypeCount = c_int(0)
    errorCodeQueryFrameType = dllLib.QueryDeviceSupportFrameType(
        nDeviceID, byref(nSupportTypeCount), None)
    if errorCodeQueryFrameType == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nSupportTypeCount.value > 0:
        pSupportType = (SYSupportTypeEnum * nSupportTypeCount.value)()
        errorCodeQueryFrameType = dllLib.QueryDeviceSupportFrameType(
            nDeviceID, byref(nSupportTypeCount), pSupportType)
        if errorCodeQueryFrameType == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nSupportTypeCount.value > 0:
            for i in range(nSupportTypeCount.value):
                supportTypeList.append(pSupportType[i].value)
            return SYErrorCodeEnum.SYERRORCODE_SUCCESS
        else:
            return SYErrorCodeEnum.SYERRORCODE_UNKOWNFRAMETYPE
    else:
        return SYErrorCodeEnum.SYERRORCODE_UNKOWNFRAMETYPE

# Query the frame resolutions supported by the device.
# @ param [in] nDeviceID
# @ param [in/out] sdkSupportType:frame type
# @ param [in/out] resolutionList:resolution list
# @ return errorCode


def QueryDeviceSupportResolution(nDeviceID: c_uint32, sdkSupportType: SYSupportTypeEnum, resolutionList: list) -> SYErrorCodeEnum:
    nResolutionCount = c_int(0)
    errorCodeQueryResolution = dllLib.QueryDeviceSupportResolution(
        nDeviceID, sdkSupportType, byref(nResolutionCount), None)
    if errorCodeQueryResolution == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nResolutionCount.value > 0:
        pResolution = (SYResolutionEnum * nResolutionCount.value)()
        errorCodeQueryResolution = dllLib.QueryDeviceSupportResolution(
            nDeviceID, sdkSupportType, byref(nResolutionCount), pResolution)
        if errorCodeQueryResolution == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nResolutionCount.value > 0:
            for i in range(nResolutionCount.value):
                resolutionList.append(pResolution[i].value)
            return SYErrorCodeEnum.SYERRORCODE_SUCCESS
        else:
            return SYErrorCodeEnum.SYERRORCODE_UNKOWNRESOLUTION
    return SYErrorCodeEnum.SYERRORCODE_UNKOWNRESOLUTION

# Get the current stream type.
# @ param [in] nDeviceID
# @ return streamType


def GetCurrentStreamType(nDeviceID: c_uint32) -> SYStreamTypeEnum:
    return dllLib.GetCurrentStreamType(nDeviceID)

# @ param [in] nDeviceID
# @ param [in] streamType
# @ return errorCode


def StartStreaming(nDeviceID: c_uint32, streamType: SYStreamTypeEnum) -> SYErrorCodeEnum:
    return dllLib.StartStreaming(nDeviceID, streamType)

# @ param [in] nDeviceID
# @ return errorCode


def StopStreaming(nDeviceID: c_uint32) -> SYErrorCodeEnum:
    return dllLib.StopStreaming(nDeviceID)

# @ param [in] nDeviceID
# @ param [in] streamType
# @ return errorCode


def ChangeStreaming(nDeviceID: c_uint32, streamType: SYStreamTypeEnum) -> SYErrorCodeEnum:
    return dllLib.ChangeStreaming(nDeviceID, streamType)


# Set resolution (if the data stream has already been started, the internal process will follow the steps of closing the stream ->setting resolution ->reopening the stream)
# @ param [in] nDeviceID
# @ param [in] frameType
# @ param [in] resolution
# @ return errorCode
def SetFrameResolution(nDeviceID: c_uint32, frameType: SYFrameTypeEnum, resolution: SYResolutionEnum) -> SYErrorCodeEnum:
    return dllLib.SetFrameResolution(nDeviceID, frameType, resolution)

# Get the device frame resolution.
# @ param [in] nDeviceID
# @ param [in] frameType
# @ param [out] resolution
# @ return errorCode


def GetFrameResolution(nDeviceID: c_uint32, frameType: SYFrameTypeEnum, resolution: SYResolutionEnum) -> SYErrorCodeEnum:
    return dllLib.GetFrameResolution(nDeviceID, frameType, byref(resolution))

# Get the filter enabled status.
# @ param [in] nDeviceID
# @ param [out] bFilter
# @ return errorCode


def GetFilter(nDeviceID: c_uint32, bFilter: c_bool) -> SYErrorCodeEnum:
    return dllLib.GetFilter(nDeviceID, byref(bFilter))

# open or close filter switch
# @ param [in] nDeviceID
# @ param [in] bFilter
# @ return errorCode


def SetFilter(nDeviceID: c_uint32, bFilter: c_bool) -> SYErrorCodeEnum:
    return dllLib.SetFilter(nDeviceID, bFilter)

# Get the hardware filter mode enabled status.
# @ param [in] nDeviceID
# @ param [out] bHardWareFilterMode:Hardware filter mode enabled status: true – hardware filter mode enabled; false – hardware filter mode disabled.


def GetHardWareFilterMode(nDeviceID: c_uint32, bHardWareFilterMode: c_bool) -> SYErrorCodeEnum:
    return dllLib.GetHardwareFilterMode(nDeviceID, byref(bHardWareFilterMode))

# Enable/disable hardware filter mode.
# @ param [in] nDeviceID
# @ param [in] bHardWareFilterMode:Hardware filter mode enabled status: true – hardware filter mode enabled; false – hardware filter mode disabled.


def SetHardWareFilterMode(nDeviceID: c_int32, bHardWareFilterMode: c_bool) -> SYErrorCodeEnum:
    return dllLib.SetHardwareFilterMode(nDeviceID, bHardWareFilterMode)

# Is hardware filter mode available
# @ param [in] nDeviceID
# @ param [out] bHaveHardwareFilterMode:Hardware filter mode availability flag，true-available ，false-disbable


def HaveHardwareFilter(nDeviceID: c_int32, bHaveHardwareFilterMode: c_bool) -> SYErrorCodeEnum:
    return dllLib.HaveHardwareFilterMode(nDeviceID, byref(bHaveHardwareFilterMode))

# Get the list of filters.
# @ param [in] nDeviceID
# @ param [in/out] nCount
# @ param [in/out] pFilterType
# @ return errorCode


def GetFilterList(nDeviceID: c_uint32, nCount: c_int, pFilterType: POINTER(SYFilterTypeEnum)) -> SYErrorCodeEnum:
    return dllLib.GetFilterList(nDeviceID, byref(nCount), pFilterType)

# Set the default filter.
# @ param [in] nDeviceID
# @ return errorCode


def SetDefaultFilter(nDeviceID: c_uint32) -> SYErrorCodeEnum:
    return dllLib.SetDefaultFilter(nDeviceID)

# @ param [in] nDeviceID
# @ param [in] filterType
# @ return errorCode


def AddFilter(nDeviceID: c_uint32, filterType: SYFilterTypeEnum) -> SYErrorCodeEnum:
    return dllLib.AddFilter(nDeviceID, filterType)

# @ param [in] nDeviceID
# @ param [in] nIndex
# @ return errorCode


def DeleteFilter(nDeviceID: c_uint32, nIndex: c_int) -> SYErrorCodeEnum:
    return dllLib.DeleteFilter(nDeviceID, nIndex)

# @ param [in] nDeviceID
# @ return errorCode


def ClearFilter(nDeviceID: c_uint32) -> SYErrorCodeEnum:
    return dllLib.ClearFilter(nDeviceID)

# @ param [in] nDeviceID
# @ param [in] filterType
# @ param [in] nParamCount
# @ param [in] pFilterParam
# @ return errorCode


def SetFilterParam(nDeviceID: c_uint32, filterType: SYFilterTypeEnum, nParamCount: c_int, pFilterParam: POINTER(c_float)) -> SYErrorCodeEnum:
    return dllLib.SetFilterParam(nDeviceID, filterType, nParamCount, pFilterParam)

# @ param [in] nDeviceID
# @ param [in] filterType
# @ param [in/out] nParamCount
# @ param [in/out] pFilterParam
# @ return errorCode


def GetFilterParam(nDeviceID: c_uint32, filterType: SYFilterTypeEnum, nParamCount: c_int, pFilterParam: POINTER(c_float)) -> SYErrorCodeEnum:
    return dllLib.GetFilterParam(nDeviceID, filterType, byref(nParamCount), pFilterParam)

# @ param [in] nDeviceID
# @ param [out] bMirror
# @ return errorCode


def GetMirror(nDeviceID: c_uint32, bMirror: c_bool) -> SYErrorCodeEnum:
    return dllLib.GetMirror(nDeviceID, byref(bMirror))

# @ param [in] nDeviceID
# @ param [in] bMirror
# @ return errorCode


def SetMirror(nDeviceID: c_uint32, bMirror: c_bool) -> SYErrorCodeEnum:
    return dllLib.SetMirror(nDeviceID, bMirror)

# @ param [in] nDeviceID
# @ param [out] bFlip
# @ return errorCode


def GetFlip(nDeviceID: c_uint32, bFilp: c_bool) -> SYErrorCodeEnum:
    return dllLib.GetFlip(nDeviceID, byref(bFilp))

# @ param [in] nDeviceID
# @ param [in] bFlip
# @ return errorCode


def SetFlip(nDeviceID: c_uint32, bFilp: c_bool) -> SYErrorCodeEnum:
    return dllLib.SetFlip(nDeviceID, bFilp)

# @ param [in] nDeviceID
# @ param [out] nIntegralTime
# @ return errorCode


def GetIntegralTime(nDeviceID: c_uint32, nIntegralTime: c_int) -> SYErrorCodeEnum:
    return dllLib.GetIntegralTime(nDeviceID, byref(nIntegralTime))

# @ param [in] nDeviceID
# @ param [in] nIntegralTime
# @ return errorCode


def SetIntegralTime(nDeviceID: c_uint32, nIntegralTime: c_int) -> SYErrorCodeEnum:
    return dllLib.SetIntegralTime(nDeviceID, nIntegralTime)

# @ param [in] nDeviceID
# @ param [in] depthResolution
# @ param [out] nMin
# @ param [out] nMax
# @ return errorCode


def GetIntegralTimeRange(nDeviceID: c_uint32, depthResolution: SYResolutionEnum, nMin: c_int, nMax: c_int) -> SYErrorCodeEnum:
    return dllLib.GetIntegralTimeRange(nDeviceID, depthResolution, byref(nMin), byref(nMax))


# Obtain the ranging range
# param [in] nDeviceID
# param [out] nMin
# param [out] nMax
# return errorCode
def GetDistanceMeasureRange(nDeviceID: c_uint32, nMin: c_int, nMax: c_int) -> SYErrorCodeEnum:
    return dllLib.GetDistanceMeasureRange(nDeviceID, byref(nMin), byref(nMax))


# Obtain the user's distance measurement range
# param [in] nDeviceID
# param [out] nMin
# param [out] nMax
# return errorCode
def GetDistanceUserRange(nDeviceID: c_uint32, nMin: c_int, nMax: c_int) -> SYErrorCodeEnum:
    return dllLib.GetDistanceUserRange(nDeviceID, byref(nMin), byref(nMax))


# Set user ranging range
# param [in] nDeviceID
# param [in] nMin
# param [in] nMax
# return errorCode
def SetDistanceUserRange(nDeviceID: c_uint32, nMin: c_int, nMax: c_int) -> SYErrorCodeEnum:
    return dllLib.SetDistanceUserRange(nDeviceID, nMin, nMax)

# Read device SN number
# param [in] nDeviceID
# return errorCode


def GetDeviceSN(nDeviceID: c_uint32):
    nStringLength = c_int(0)
    errorCodeGetSN: SYErrorCodeEnum = dllLib.GetDeviceSN(
        nDeviceID, byref(nStringLength), None)
    if errorCodeGetSN == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        if nStringLength.value > 0:
            pSerialNumber = (c_char * nStringLength.value)()
            errorCodeGetSN: SYErrorCodeEnum = dllLib.GetDeviceSN(
                nDeviceID, byref(nStringLength), pSerialNumber)
            if errorCodeGetSN == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                return pSerialNumber.value.decode('utf-8')
            else:
                PrintErrorCode("GetDeviceSN", errorCodeGetSN)
                return None
        else:
            errorCodeGetSN = SYErrorCodeEnum.SYERRORCODE_STRINGLENGTHOUTRANGE
            PrintErrorCode("GetDeviceSN", errorCodeGetSN)
            return None

# Write device SN number
# param [in] nDeviceID
# param [in] pstrSN
# return errorCode


def SetDeviceSN(nDeviceID: c_uint32, deviceSN: str) -> SYErrorCodeEnum:
    return dllLib.SetDeviceSN(nDeviceID, deviceSN.encode('utf-8'))

# Read the firmware version number of the device
# param [in] nDeviceID
# return Device firmware version


def GetDeviceHWVersion(nDeviceID: c_uint32):
    nStringLength = c_int(0)
    errorCodeGetHWVersion = dllLib.GetDeviceHWVersion(
        nDeviceID, byref(nStringLength), None)
    if errorCodeGetHWVersion == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        if nStringLength.value > 0:
            pStringFWVersion = (c_char * nStringLength.value)()
            errorCodeGetHWVersion = dllLib.GetDeviceHWVersion(
                nDeviceID, byref(nStringLength), pStringFWVersion)
            if errorCodeGetHWVersion == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                return pStringFWVersion.value.decode('utf-8')
            else:
                errorCodeRet = SYErrorCodeEnum.SYERRORCODE_UNKOWNSN
                PrintErrorCode("GetDeviceHWVersion", errorCodeRet)
                return None
        else:
            return None
    else:
        return None

# Obtain pseudo color corresponding to depth
# param [in] nDeviceID
# param [in] nCount Data volume (pDepth requires nCount * 2 bytes for memory space, pColor requires nCount * 3 bytes)
# param [in] pDepth
# param [in/out] pColor
# return errorCode


def GetDepthColor(nDeviceID: c_uint32, nCount: c_int, pDepth: POINTER(c_ushort), pColor: POINTER(c_ubyte)):
    return dllLib.GetDepthColor(nDeviceID, nCount, pDepth, pColor)


# Obtain depth corresponding point cloud data
# param [in] nDeviceID
# param [in] nWidth
# param [in] nHeight
# param [in] pDepth
# param [in/out] pPointCloud Depth corresponds to point cloud data, with external memory allocation
# param [in] bUndistort
# return errorCode
def GetDepthPointCloud(nDeviceID: int, nWidth: int, nHeight: int, pDepth: POINTER(c_ushort), pPointCloud: POINTER(SYPointCloudData), bUndistort: bool = False) -> SYErrorCodeEnum:
    return dllLib.GetDepthPointCloud(nDeviceID, c_int(nWidth), c_int(nHeight), pDepth, pPointCloud, c_bool(bUndistort))

# param [in] nDeviceID
# param [in] nSourceDepthWidth
# param [in] nSourceDepthHeight
# param [in] pSourceDepth
# param [in] nSourceRGBWidth
# param [in] nSourceRGBHeight
# param [in] pSourceRGB
# param [in] nTargetWidth
# param [in] nTargetHeight
# param [in/out] pTargetDepth The depth data in RGBD is allocated memory externally and has the same length as the source RGB data
# param [in/out] pTargetRGB The RGB data in RGBD is allocated memory externally and has the same length as the source RGB data
# return errorCode


def GetRGBD(nDeviceID: c_uint32, nSourceDepthWidth: int, nSourceDepthHeight: int, pSourceDepth: POINTER(c_ushort),
            nSourceRGBWidth: int, nSourceRGBHeight: int, pSourceRGB: POINTER(c_ubyte), nTargetWidth: int,
            nTargetHeight: int, pTargetDepth: POINTER(c_ushort),
            pTargetRGB: POINTER(c_ubyte)) -> SYErrorCodeEnum:
    return dllLib.GetRGBD(nDeviceID, c_int(nSourceDepthWidth), c_int(nSourceDepthHeight), pSourceDepth, c_int(nSourceRGBWidth), c_int(nSourceRGBHeight), pSourceRGB, c_int(nTargetWidth), c_int(nTargetHeight), pTargetDepth, pTargetRGB)


# Get the latest frame of data
# param [in] nDeviceID
# param [in/out] pFrameData
# return errorCode
def GetLastFrameData(nDeviceID: c_uint32, pFrameData: POINTER(SYFrameData)) -> SYErrorCodeEnum:
    return dllLib.GetLastFrameData(nDeviceID, pFrameData)


# Remove distortion
# param [in] nDeviceID
# param [in] pSource  Distorted data pointer to be removed
# param [in] nWidth
# param [in] nHeight
# param [in] bDepth
# param [out] pTarget  Distortion removal result data pointer, externally allocated memory, data length consistent with the length of the data pointer to be distorted
# @return errorCode
def Undistort(nDeviceID: c_uint32, pSource: POINTER(c_ushort), nWidth: c_int, nHeight: c_int, bDepth: c_bool, pTarget: POINTER(c_ushort)) -> SYErrorCodeEnum:
    return dllLib.Undistort(nDeviceID, pSource, nWidth, nHeight, bDepth, pTarget)


# Get camera parameters
# @ param [in] nDeviceID
# @ param [in] resolution
# @ param [in] bRGB  true-RGB false-TOF
# @ param [in/out] intrinsics
# @ return errorCode
def GetIntric(nDeviceID: c_uint32, resolution: SYResolutionEnum, bRGB: c_bool, intrinsics: SYIntrinsics):
    return dllLib.GetIntric(nDeviceID, resolution, bRGB, byref(intrinsics))


# Get the status of the drag filter enabled
# @param[in] nDeviceID
# @param[out] bFilter
# @return errorCode
def GetTrailFilter(nDeviceID: c_uint32, bFilter: c_bool) -> SYErrorCodeEnum:
    return dllLib.GetTrailFilter(nDeviceID, byref(bFilter))


# Set the drag filter on status
# @param[in] nDeviceID
# @param[in] bFilter
# @return errorCode
def SetTrailFilter(nDeviceID: c_uint32, bFilter: c_bool) -> SYErrorCodeEnum:
    return dllLib.SetTrailFilter(nDeviceID, bFilter)

# Modify device IP
# @ param [in] nDeviceID
# @ param [in] nIP
# @ return errorCode


def ChangeDeviceIP(nDeviceID: c_uint32, nIP: c_uint32) -> SYErrorCodeEnum:
    return dllLib.ChangeDeviceIP(nDeviceID, nIP)

# Filter depth data using IR
# @ param [in] nDeviceID
# @ param [in/out] pDepth
# @ param [in] pIR
# @ param [in] nWidth
# @ param [in] nHeight
# @ param [in] nThreshold  Filter parameter array  Maximum 10 data points
# @ return errorCode


def IRFilter(nDeviceID: c_uint32, pDepth: POINTER(c_ushort), pIR: POINTER(c_ushort), nWiddth: c_int, nHeight: c_int, nThreshold: c_int) -> SYErrorCodeEnum:
    return dllLib.IRFilter(nDeviceID, pDepth, pIR, nWiddth, nHeight, nThreshold)

# SetFrameRate
# @param[in] nDeviceID
# @param[in/out] frameRate
# @return errorCode


def SetFrameRate(nDeviceID: c_uint32, nFramerate: c_uint32) -> SYErrorCodeEnum:
    return dllLib.SetFrameRate(nDeviceID, nFramerate)

# GetFrameRate
# @param[in] nDeviceID
# @param[in/out] frameRate
# @return errorCode


def GetFrameRate(nDeviceID: c_uint32, nFramerate: c_uint32) -> SYErrorCodeEnum:
    return dllLib.GetFrameRate(nDeviceID, byref(nFramerate))

# GetTemperature
# @param[in] nDeviceID
# @param[in/out] temperature
# @return errorCode


def GetTemperature(nDeviceID: c_uint32, fltTemperature: c_float) -> SYErrorCodeEnum:
    return dllLib.GetTemperature(nDeviceID, byref(fltTemperature))

# Check if the frame rate is adjustable.
# @param[in] nDeviceID
# @param[in/out] Frame Rate Adjustability Indicator
# @return errorCode


def IsFrameRateEnabled(nDeviceID: c_uint32, bEnabled: c_bool) -> SYErrorCodeEnum:
    return dllLib.ISFrameRateEnabled(nDeviceID, byref(bEnabled))

# Parse the configuration file
# @param[in] Length of the path string
# @param[in] Full path of the configuration file
# @param[in/out] Configuration data
# @return errorCode


def ParseConfiguration(nStringLength: c_int, fileName: str, configuration: SYConfiguration) -> SYErrorCodeEnum:
    return dllLib.ParseConfiguration(nStringLength, fileName.encode('utf-8'), byref(configuration))

# Open the stream using the configuration.
# @param[in] nDeviceID
# @param[in] Configuration data
# @return errorCode


def StartStreamingWithConfiguration(nDeviceID: c_uint32, configuration: SYConfiguration) -> SYErrorCodeEnum:
    return dllLib.StartStreamingWithConfiguration(nDeviceID, byref(configuration))

# Export the configuration file
# @param[in] nDeviceID
# @param[in] Length of the path string
# @param[in] Full path of the configuration file
# @return errorCode


def ExportConfiguration(nDeviceID: c_uint32, nStringLength: c_int, fileName: str) -> SYErrorCodeEnum:
    return dllLib.ExportConfiguration(nDeviceID, nStringLength, fileName.encode('utf-8'))

# Set trigger mode
# @param[in] nDeviceID
# @param [in] SYTriggerMode trigger mode
# @return errorCode


def SetTriggerMode(nDeviceID: c_uint32, triggerMode: SYTriggerModeEnum) -> SYErrorCodeEnum:
    return dllLib.SetTriggerMode(nDeviceID, triggerMode)

# Get trigger mode
# @param[in] nDeviceID
# @param [in/out] SYTriggerMode trigger mode
# @return errorCode


def GetTriggerMode(nDeviceID: c_uint32, triggerMode: SYTriggerModeEnum) -> SYErrorCodeEnum:
    return dllLib.GetTriggerMode(nDeviceID, byref(triggerMode))

# Send software trigger signal
# @param[in] nDeviceID
# @return errorCode


def SendTriggerSignal(nDeviceID: c_uint32) -> SYErrorCodeEnum:
    return dllLib.SendTriggerSignal(nDeviceID)

# Set synchronization mode
# @param [in] syncMode synchronization mode
# @param [in] nNTPServerIP NTP server IP (small end mode: such as 192.168.1.110, data is 0x6E01A8C0)
# @return errorCode


def SetSyncMode(syncMode: SYSyncModeEnum, nNTPServerIP: c_uint32) -> SYErrorCodeEnum:
    return dllLib.SetSyncMode(syncMode, nNTPServerIP)

# Get synchronization mode
# @param [in] syncMode synchronization mode
# @return errorCode


def GetSyncMode(syncMode: SYSyncModeEnum) -> SYErrorCodeEnum:
    return dllLib.GetSyncMode(byref(syncMode))
