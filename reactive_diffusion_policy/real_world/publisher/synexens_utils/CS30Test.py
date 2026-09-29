import cv2
import numpy as np
from ctypes import *
from SynexensPythonSDK import *

# ─── 全局状态 ──────────────────────────────────────────────────────────────────
# g_depth_array: SDK 输出的深度图，未经本程序再做处理；单位 mm，暂停时悬停显示的就是该像素此值
g_paused      = False
g_depth_array = None   # 最新深度帧 (H×W, uint16, 单位 mm)
g_depth_disp  = None   # 最新深度显示图 (H×W×3, uint8, 伪彩色)
g_mouse_x     = 0
g_mouse_y     = 0
g_depth_win   = ""     # 深度窗口名称


def on_mouse(event, x, y, flags, param):
    global g_mouse_x, g_mouse_y
    g_mouse_x, g_mouse_y = x, y


def build_depth_overlay(disp_img: np.ndarray, depth_arr: np.ndarray,
                        mx: int, my: int) -> np.ndarray:
    """在伪彩色深度图上叠加鼠标悬停位置的像素坐标、深度值和暂停标记。"""
    h, w = depth_arr.shape
    mx = max(0, min(mx, w - 1))
    my = max(0, min(my, h - 1))
    depth_val = int(depth_arr[my, mx])

    overlay = disp_img.copy()
    cv2.drawMarker(overlay, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(overlay, f"({mx}, {my})  depth = {depth_val} mm",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(overlay, "[ PAUSED ]",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
    return overlay


def process_frame(nDeviceID: int, pFrameData) -> None:
    """解析一帧数据，分别处理深度、IR、RGB 子帧并刷新显示。"""
    global g_depth_array, g_depth_disp

    obj  = pFrameData.contents
    nPos = 0

    for i in range(obj.m_nFrameCount):
        h      = obj.m_pFrameInfo[i].m_nFrameHeight
        w      = obj.m_pFrameInfo[i].m_nFrameWidth
        nCount = h * w
        ftype  = obj.m_pFrameInfo[i].m_frameType

        # ── 深度帧 ────────────────────────────────────────────────────────────
        if ftype == SYFrameTypeEnum.SYFRAMETYPE_DEPTH:
            pDepth = (c_ushort * nCount)()
            ptr    = cast(c_void_p(obj.m_pData + nPos), POINTER(c_ushort))
            memmove(pDepth, ptr, sizeof(c_ushort) * nCount)

            depth_np      = np.frombuffer(pDepth, dtype=np.uint16).reshape(h, w).copy()
            g_depth_array = depth_np

            pColor = (c_ubyte * (nCount * 3))()
            disp   = np.zeros((h, w, 3), dtype=np.uint8)
            if GetDepthColor(c_uint(nDeviceID), nCount, pDepth, pColor) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                memmove(disp.ctypes.data, pColor, h * w * 3)
                disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            else:
                tmp   = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
                gray8 = cv2.convertScaleAbs(tmp)
                disp  = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)

            g_depth_disp = disp
            cv2.imshow(f"Depth_{nDeviceID}", disp)
            nPos += nCount * sizeof(c_short)

        # ── IR 帧 ─────────────────────────────────────────────────────────────
        elif ftype == SYFrameTypeEnum.SYFRAMETYPE_IR:
            pIR  = (c_ushort * nCount)()
            ptr  = cast(c_void_p(obj.m_pData + nPos), POINTER(c_ushort))
            memmove(pIR, ptr, sizeof(c_ushort) * nCount)

            gray16 = np.frombuffer(pIR, dtype=np.uint16).reshape(h, w)
            gray8  = np.zeros((h, w), dtype=np.uint8)
            cv2.convertScaleAbs(gray16, gray8, 0.5, 0)
            cv2.imshow(f"IR_{nDeviceID}", gray8)
            nPos += nCount * sizeof(c_short)

        # ── RGB 帧 (YUYV 格式) ────────────────────────────────────────────────
        elif ftype == SYFrameTypeEnum.SYFRAMETYPE_RGB:
            data_ptr = cast(obj.m_pData + nPos, POINTER(c_ubyte))
            data_arr = np.ctypeslib.as_array(data_ptr, shape=(h * w * 2,))
            yuyv     = np.frombuffer(data_arr, dtype=np.uint8).reshape(h, w, 2)
            bgr      = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            cv2.imshow(f"RGB_{nDeviceID}", bgr)
            nPos += nCount * 3 // 2

        else:
            nPos += h * w * sizeof(c_short)


def main() -> None:
    global g_paused, g_depth_win

    print(f"SDK Version: {GetSDKVersion()}")

    # ── 初始化 SDK ────────────────────────────────────────────────────────────
    if InitSDK() != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        print("[ERROR] InitSDK failed")
        return

    # ── 查找设备 ──────────────────────────────────────────────────────────────
    nDeviceCount = c_int32()
    if (FindDevice(byref(nDeviceCount), None) != SYErrorCodeEnum.SYERRORCODE_SUCCESS
            or nDeviceCount.value == 0):
        print("[ERROR] No device found. Please check USB connection.")
        print("        Hint: Try running with sudo, or configure udev rules.")
        UnInitSDK()
        return

    pDeviceInfo = (SYDeviceInfo * nDeviceCount.value)()
    if FindDevice(nDeviceCount, pDeviceInfo) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        print("[ERROR] FindDevice (2nd call) failed")
        UnInitSDK()
        return

    # ── 打开设备并启动流 ───────────────────────────────────────────────────────
    opened = []
    for i in range(nDeviceCount.value):
        dev = pDeviceInfo[i]
        print(f"  Found  DeviceID={dev.m_nDeviceID}  Type={dev.m_deviceType.value}")

        if OpenDevice(dev) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] OpenDevice {dev.m_nDeviceID} failed, skip")
            continue

        sn = GetDeviceSN(dev.m_nDeviceID)
        hw = GetDeviceHWVersion(dev.m_nDeviceID)
        print(f"         SN={sn}  HW={hw}")

        # 深度分辨率 640×480
        ec = SetFrameResolution(dev.m_nDeviceID,
                                SYFrameTypeEnum.SYFRAMETYPE_DEPTH,
                                SYResolutionEnum.SYRESOLUTION_640_480)
        if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] SetFrameResolution (Depth 640×480) failed ec={ec}")
            continue

        # RGB 分辨率 1920×1080
        ec = SetFrameResolution(dev.m_nDeviceID,
                                SYFrameTypeEnum.SYFRAMETYPE_RGB,
                                SYResolutionEnum.SYRESOLUTION_1920_1080)
        if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] SetFrameResolution (RGB 1920×1080) failed ec={ec}, skip RGB")

        # 尝试 Depth+IR+RGB 三路，失败则退回 Depth+IR
        stream_type = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIRRGB
        ec = StartStreaming(dev.m_nDeviceID, stream_type)
        if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [INFO] DEPTHIRRGB failed (ec={ec}), fallback to DEPTHIR")
            stream_type = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR
            ec = StartStreaming(dev.m_nDeviceID, stream_type)
            if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                print(f"  [WARN] StartStreaming failed for DeviceID={dev.m_nDeviceID}")
                continue

        # 创建 OpenCV 窗口
        depth_win = f"Depth_{dev.m_nDeviceID}"
        ir_win    = f"IR_{dev.m_nDeviceID}"
        cv2.namedWindow(depth_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(ir_win,    cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(depth_win, on_mouse)
        g_depth_win = depth_win

        if stream_type == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIRRGB:
            rgb_win = f"RGB_{dev.m_nDeviceID}"
            cv2.namedWindow(rgb_win, cv2.WINDOW_NORMAL)
            print(f"  [OK]   DeviceID={dev.m_nDeviceID}  Depth 640×480 + IR + RGB 1920×1080")
        else:
            print(f"  [OK]   DeviceID={dev.m_nDeviceID}  Depth 640×480 + IR  (no RGB)")

        opened.append(dev)

    if not opened:
        print("[ERROR] No device opened successfully.")
        UnInitSDK()
        return

    # 曝光（积分时间）用 640×480 对应的范围；步长 100 μs
    INTEGRAL_STEP = 10
    DEPTH_RES_FOR_INTEGRAL = SYResolutionEnum.SYRESOLUTION_640_480

    print()
    print("╔═════════════════════════════════════════╗")
    print("║  SPACE  :  暂停 / 恢复                  ║")
    print("║  U      :  曝光时间 +100 μs             ║")
    print("║  D      :  曝光时间 -100 μs             ║")
    print("║  ESC    :  退出                          ║")
    print("║  暂停后移动鼠标到深度窗口                ║")
    print("║  可查看每个像素的深度值（mm）             ║")
    print("╚═════════════════════════════════════════╝")
    print()

    # ── 主循环 ────────────────────────────────────────────────────────────────
    while True:
        if not g_paused:
            for dev in opened:
                pFrameData = POINTER(SYFrameData)()
                ec = GetLastFrameData(dev.m_nDeviceID, byref(pFrameData))
                if ec == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    process_frame(dev.m_nDeviceID, pFrameData)
        else:
            # 暂停：冻结画面，实时叠加鼠标位置的深度值
            if g_depth_disp is not None and g_depth_array is not None:
                overlay = build_depth_overlay(g_depth_disp, g_depth_array,
                                              g_mouse_x, g_mouse_y)
                cv2.imshow(g_depth_win, overlay)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:    # ESC — 退出
            break
        elif key == 32:  # SPACE — 暂停 / 恢复
            g_paused = not g_paused
            if g_paused:
                print("  >> PAUSED — 移动鼠标到深度窗口查看像素深度 (mm)")
            else:
                print("  >> RESUMED")
        elif key == 85 or key == 117:   # U — 增加曝光（积分时间）
            for dev in opened:
                nMin, nMax = c_int(0), c_int(0)
                ec = GetIntegralTimeRange(dev.m_nDeviceID, DEPTH_RES_FOR_INTEGRAL, nMin, nMax)
                if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    continue
                nCur = c_int(0)
                ec = GetIntegralTime(dev.m_nDeviceID, nCur)
                if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    continue
                new_val = min(nCur.value + INTEGRAL_STEP, nMax.value)
                if SetIntegralTime(dev.m_nDeviceID, new_val) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    print(f"  DeviceID={dev.m_nDeviceID} 曝光(积分时间): {new_val} μs")
        elif key == 68 or key == 100:   # D — 减少曝光
            for dev in opened:
                nMin, nMax = c_int(0), c_int(0)
                ec = GetIntegralTimeRange(dev.m_nDeviceID, DEPTH_RES_FOR_INTEGRAL, nMin, nMax)
                if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    continue
                nCur = c_int(0)
                ec = GetIntegralTime(dev.m_nDeviceID, nCur)
                if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    continue
                new_val = max(nCur.value - INTEGRAL_STEP, nMin.value)
                if SetIntegralTime(dev.m_nDeviceID, new_val) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    print(f"  DeviceID={dev.m_nDeviceID} 曝光(积分时间): {new_val} μs")

    # ── 清理 ──────────────────────────────────────────────────────────────────
    for dev in opened:
        StopStreaming(dev.m_nDeviceID)
    cv2.destroyAllWindows()
    UnInitSDK()
    print("Done.")


if __name__ == "__main__":
    main()
