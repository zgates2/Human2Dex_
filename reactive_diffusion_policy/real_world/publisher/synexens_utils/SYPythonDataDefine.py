from ctypes import *

class SYErrorCodeEnum(c_int):
    SYERRORCODE_SUCCESS = 0
    SYERRORCODE_FAILED = 1
    SYERRORCODE_DEVICENOTEXIST = 2
    SYERRORCODE_DEVICENOTOPENED = 3
    SYERRORCODE_UNKOWNRESOLUTION = 4
    SYERRORCODE_DEVICEHANDLEEMPTY = 5
    SYERRORCODE_SETOUTPUTFORMATFAILED = 6
    SYERRORCODE_GETSTREAMCTRLFAILED = 7
    SYERRORCODE_STARTSTREAMINGFAILED = 8
    SYERRORCODE_COMMUNICATEOBJECTEMPTY = 9
    SYERRORCODE_UNKOWNSN = 10
    SYERRORCODE_STRINGLENGTHOUTRANGE = 11
    SYERRORCODE_UNKOWNFRAMETYPE = 12
    SYERRORCODE_UNKOWNDEVICETYPE = 13
    SYERRORCODE_DEVICEOBJECTEMPTY = 14
    SYERRORCODE_OBSERVEREMPTY = 15
    SYERRORCODE_OBSERVERNOTFOUND = 16
    SYERRORCODE_COUNTOUTRANGE = 17
    SYERRORCODE_UVCINITFAILED = 18
    SYERRORCODE_UVCFINDDEVICEFAILED = 19
    SYERRORCODE_NOFRAME = 20
    SYERRORCODE_GETAPPFOLDERPATHFAILED = 21
    SYERRORCODE_NOSTREAMING = 22
    SYERRORCODE_RECONSTRUCTIONEMPTY = 23
    SYERRORCODE_STREAMINGEXIST = 24
    SYERRORCODE_UNKOWNSTREAMTYPE = 25
    SYERRORCODE_DATABUFFEREMPTY = 26
    SYERRORCODE_REPEATED = 27
    SYERRORCODE_FIRMWARE_VERSION_TOO_LOW = 28
    SYERRORCODE_PBDOWNLOADING = 29
    SYERRORCODE_VALUEOUTOFRANGE = 30
    SYERRORCODE_FRAMEDATAOBSERVEREXIST = 31

class SYDeviceTypeEnum(c_int):
    SYDEVICETYPE_NULL = 0
    SYDEVICETYPE_CS30_DUAL = 1
    SYDEVICETYPE_CS30_SINGLE = 2
    SYDEVICETYPE_CS20_DUAL = 3
    SYDEVICETYPE_CS20_SINGLE = 4
    SYDEVICETYPE_CS20_P = 5
    SYDEVICETYPE_CS40 = 6
    SYDEVICETYPE_CS40PRO = 7

class SYStreamTypeEnum(c_int):
    SYSTREAMTYPE_NULL = 0
    SYSTREAMTYPE_RAW = 1
    SYSTREAMTYPE_DEPTH = 2
    SYSTREAMTYPE_RGB = 3
    SYSTREAMTYPE_DEPTHIR = 4
    SYSTREAMTYPE_DEPTHRGB = 5
    SYSTREAMTYPE_DEPTHIRRGB = 6
    SYSTREAMTYPE_RGBD = 7
    SYSTREAMTYPE_RAWRGB = 8
    SYSTREAMTYPE_RGBDAI = 9

class SYResolutionEnum(c_int):
    SYRESOLUTION_NULL = 0
    SYRESOLUTION_320_240 = 1
    SYRESOLUTION_640_480 = 2
    SYRESOLUTION_960_540 = 3
    SYRESOLUTION_1920_1080 = 4
    SYRESOLUTION_1600_1200 = 5
    SYRESOLUTION_800_600 = 6

class SYFrameTypeEnum(c_int):
    SYFRAMETYPE_NULL = 0
    SYFRAMETYPE_RAW = 1
    SYFRAMETYPE_DEPTH = 2
    SYFRAMETYPE_IR = 3
    SYFRAMETYPE_RGB = 4
    SYFRAMETYPE_AI = 5
    SYFRAMETYPE_TIMESTAMP = 6

class SYSupportTypeEnum(c_int):
    SYSUPPORTTYPE_NULL = 0
    SYSUPPORTTYPE_DEPTH = 1
    SYSUPPORTTYPE_RGB = 2
    SYSUPPORTTYPE_RGBD = 3
    SYSUPPORTTYPE_RGBDAI = 4

class SYEventTypeEnum(c_int):
    SYEVENTTYPE_NULL = 0
    SYEVENTTYPE_DEVICECONNECT = 1
    SYEVENTTYPE_DEVICEDISCONNECT = 2

class SYFilterTypeEnum(c_int):
    SYFILTERTYPE_NULL = 0
    SYFILTERTYPE_MEDIAN = 1
    SYFILTERTYPE_AMPLITITUD = 2
    SYFILTERTYPE_EDGE = 3
    SYFILTERTYPE_SPECKLE = 4
    SYFILTERTYPE_OKADA = 5
    SYFILTERTYPE_EDGE_MAD = 6
    SYFILTERTYPE_GAUSS = 7
    SYFILTERTYPE_EXTRA = 8
    SYFILTERTYPE_EXTRA2 = 9
    SYFILTERTYPE_EDGE_SOBEL = 10
    SYFILTERTYPE_IR = 11

class SYDeviceInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_nDeviceID', c_uint),
        ('m_deviceType', SYDeviceTypeEnum),
        ('m_nUsbBus', c_uint),
        ('m_nUsbPorts', c_uint * 7),
        ('m_nUsbPortsNumber', c_uint),
        ('m_nUsbDeviceAddress', c_uint),
        ('m_nIPAddress', c_uint),
    ]

class SYEventInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('SYEventType', SYEventTypeEnum),
        ('m_pFileData',c_void_p),
        ('m_nLength', c_int)
    ]


class SYFrameInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_frameType', c_int),
        ('m_nFrameHeight', c_int),
        ('m_nFrameWidth', c_int)
    ]

class SYFrameData(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_nFrameCount', c_int),
        ('m_pFrameInfo', POINTER(SYFrameInfo)),
        ('m_pData', c_void_p),
        ('m_nBuffferLength', c_int),
        ('m_nTimestamp', c_longlong),
    ]

class SYPointCloudData(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_fltX', c_float),
        ('m_fltY', c_float),
        ('m_fltZ', c_float)
    ]

class SYIntrinsics(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_fltFOV', c_float * 2),
        ('m_fltCoeffs', c_float * 5),
        ('m_fltFocalDistanceX', c_float),
        ('m_fltFocalDistanceY', c_float),
        ('m_fltCenterPointX', c_float),
        ('m_fltCenterPointY', c_float),
        ('m_nWidth', c_int),
        ('m_nHeight', c_int),
        ('m_fltTOF2RGBRotate', c_float * 9),
        ('m_fltTOF2RGBTranslation', c_float * 3)
    ]
    def __str__(self):
        return (f"FOV: {list(self.m_fltFOV)}\n"
                f"Coeffs: {list(self.m_fltCoeffs)}\n"
                f"Focal: ({self.m_fltFocalDistanceX}, {self.m_fltFocalDistanceY})\n"
                f"Center: ({self.m_fltCenterPointX}, {self.m_fltCenterPointY})\n"
                f"Image: {self.m_nWidth}×{self.m_nHeight}\n"
                f"TOF→RGB R: {list(self.m_fltTOF2RGBRotate)}\n"
                f"TOF→RGB T: {list(self.m_fltTOF2RGBTranslation)}")

class SYAIModelTypeEnum(c_int):
    SYAIMODELTYPE_NULL = 0
    SYAIMODELTYPE_DETECT =1
    SYAIMODELTYPE_FACE = 2
    SYAIMODELTYPE_POSE = 3

class SYAIModelInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_modelType', SYAIModelTypeEnum),
        ('m_fltNMS', c_float),
        ('m_fltBox', c_float),
        ('m_pFileData',c_void_p),
        ('m_nFileLength', c_int)
    ]

class BOX_RECT(Structure):
    _pack_ = 1
    _fields_ = [
        ('left', c_int),
        ('right', c_int),
        ('top', c_int),
        ('bottom', c_int)
    ]

class detect_result_t(Structure):
    _pack_ = 1
    _fields_ = [
        ('id', c_int),
        ('name', c_char * 16),
        ('box', BOX_RECT),
        ('prop', c_float),
    ]

class detect_result_group_t(Structure):
    _pack_ = 1
    _fields_ = [
        ('id', c_int),
        ('count', c_int),
        ('result', detect_result_t * 64)
    ]

class SYFilterParam(Structure):
    _pack_ = 1
    _fields_ = [
        ("m_FilterType", c_int),
        ("m_paramValue", c_float * 10),
        ("m_nParamCount", c_int)
    ]

class SYConfiguration(Structure):
    _pack_ = 1
    _fields_ = [
        ("m_streamType", SYStreamTypeEnum),
        ("m_resolutionDepth", SYResolutionEnum),
        ("m_resolutionRGB", SYResolutionEnum),
        ("m_nFrameRate", c_int),
        ("m_nIntegralTime", c_int),
        ("m_bMirror", c_bool),
        ("m_bFlip", c_bool),
        ("m_bFilter", c_bool),
        ("m_filterParam", SYFilterParam * 10),
        ("m_nFilterCount", c_int)
    ]

class SYTriggerModeEnum(c_int):
    SYTRIGGERMODE_NULL = 0
    SYTRIGGERMODE_HARDWARE = 1
    SYTRIGGERMODE_SOFTWARE = 2

class SYSyncModeEnum(c_int):
    SYSYNCMODE_NULL = 0
    SYSYNCMODE_NTP = 1