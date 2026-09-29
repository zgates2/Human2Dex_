import cv2
import numpy as np
from ctypes import *
from SynexensPythonSDK import *

# ─── 全局状态 ──────────────────────────────────────────────────────────────────
g_paused       = False
g_depth_array  = None   # 最新深度帧 (H×W, uint16, 单位 mm)
g_depth_disp   = None   # 最新深度显示图 (H×W×3, uint8, 伪彩色)
g_mouse_x      = 0
g_mouse_y      = 0
g_depth_win    = ""     # 深度窗口名称（用于注册鼠标回调）


def on_mouse(event, x, y, flags, param):
    global g_mouse_x, g_mouse_y
    g_mouse_x, g_mouse_y = x, y


def build_depth_overlay(disp_img: np.ndarray, depth_arr: np.ndarray, mx: int, my: int) -> np.ndarray:
    """在伪彩色深度图上叠加鼠标位置、深度数值和暂停标记。"""
    h, w = depth_arr.shape
    mx = max(0, min(mx, w - 1))
    my = max(0, min(my, h - 1))
    depth_val = int(depth_arr[my, mx])

    overlay = disp_img.copy()
    cv2.drawMarker(overlay, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(overlay, f"({mx}, {my})  =  {depth_val} mm",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(overlay, "[ PAUSED ]",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
    return overlay


def process_frame(nDeviceID: int, pFrameData) -> None:
    """解析一帧数据并更新全局显示缓冲。"""
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

            # 优先用 SDK 伪彩色；失败则用 normalize 灰度转彩色
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

        else:
            # 未知帧类型，跳过
            nPos += obj.m_pFrameInfo[i].m_nFrameHeight * obj.m_pFrameInfo[i].m_nFrameWidth * sizeof(c_short)


def main() -> None:
    global g_paused, g_depth_win

    print(f"SDK Version: {GetSDKVersion()}")

    # ── 初始化 SDK（支持全部设备型号）────────────────────────────────────────
    if InitSDK() != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        print("[ERROR] InitSDK failed")
        return

    # ── 查找设备 ──────────────────────────────────────────────────────────────
    nDeviceCount = c_int32()
    if (FindDevice(byref(nDeviceCount), None) != SYErrorCodeEnum.SYERRORCODE_SUCCESS
            or nDeviceCount.value == 0):
        print("[ERROR] No device found. Please check USB connection.")
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


        ec = SetFrameResolution(dev.m_nDeviceID,
                                SYFrameTypeEnum.SYFRAMETYPE_DEPTH,
                                SYResolutionEnum.SYRESOLUTION_320_240)
        if ec != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] SetFrameResolution failed (ec={ec})")
            continue

        # 初始化：曝光 100 μs，测距量程 0–750 mm
        if SetIntegralTime(dev.m_nDeviceID, 100) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] SetIntegralTime(100) failed")
        if SetDistanceUserRange(dev.m_nDeviceID, 0, 750) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] SetDistanceUserRange(0, 750) failed")
        else:
            print(f"  [INFO] 曝光=100 μs, 测距量程=0~750 mm")

        if StartStreaming(dev.m_nDeviceID, SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            print(f"  [WARN] StartStreaming failed for DeviceID={dev.m_nDeviceID}")
            continue

        depth_win = f"Depth_{dev.m_nDeviceID}"
        ir_win    = f"IR_{dev.m_nDeviceID}"
        cv2.namedWindow(depth_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(ir_win,    cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(depth_win, on_mouse)
        g_depth_win = depth_win

        opened.append(dev)
        print(f"  [OK]   DeviceID={dev.m_nDeviceID} streaming started  (320×240, Depth+IR)")

    if not opened:
        print("[ERROR] No device opened successfully.")
        UnInitSDK()
        return

    # 曝光（积分时间）步长；CS20 深度分辨率 320×240
    INTEGRAL_STEP          = 20
    DEPTH_RES_FOR_INTEGRAL = SYResolutionEnum.SYRESOLUTION_320_240

    print()
    print("╔══════════════════════════════════════╗")
    print("║  SPACE  :  暂停 / 恢复               ║")
    print("║  U      :  曝光时间 +20 μs          ║")
    print("║  D      :  曝光时间 -20 μs          ║")
    print("║  ESC    :  退出                       ║")
    print("║  暂停后移动鼠标到深度窗口             ║")
    print("║  可查看每个像素的深度值（mm）          ║")
    print("╚══════════════════════════════════════╝")
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
            # 暂停：冻结画面，实时叠加鼠标悬停位置的深度值
            if g_depth_disp is not None and g_depth_array is not None:
                overlay = build_depth_overlay(g_depth_disp, g_depth_array,
                                              g_mouse_x, g_mouse_y)
                cv2.imshow(g_depth_win, overlay)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:    # ESC
            break
        elif key == 32:  # SPACE
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
