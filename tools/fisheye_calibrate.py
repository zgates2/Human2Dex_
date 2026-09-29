#!/usr/bin/env python3
"""
鱼眼相机标定工具

用法:
    python fisheye_calibrate.py --input_dir <图像目录> --pattern_size 7x9 --square_size 25 [--undistort]

示例:
    python fisheye_calibrate.py --input_dir ../data_calibration/fisheye_images/ --pattern_size 7x9 --square_size 25 --undistort

    rtk conda run -n l515_mvs310 python /home/zjc/Desktop/human2dex/tools/fisheye_calibrate.py \
    --input_dir /opt/MVS/bin/Temp/Data/ \
    --output_dir /home/zjc/Desktop/human2dex/converted_data \
    --pattern_size 7x9 \
    --square_size 25 \
    --undistort

输出:
    默认将标定参数和校正图像保存到项目根目录下的 data_calibration/
"""

import argparse
import os
import glob
import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    """创建文件夹"""
    if not os.path.exists(path):
        os.makedirs(path)


def _save_yaml(data, file_path):
    """将标定数据保存为 YAML"""
    serializable = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(file_path, 'w') as f:
        yaml.dump(serializable, f, default_flow_style=False)


def _per_view_rms(objpoints, imgpoints, rvecs, tvecs, K, D):
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        residual = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return np.asarray(errors, dtype=np.float64)


# ---------------------------------------------------------------------------
# 角点检测 & 姿态多样性筛选
# ---------------------------------------------------------------------------

def _compute_pose_descriptor(corners):
    """棋盘格姿态描述子: (center_x, center_y, size_x, size_y)"""
    tl = corners[0][0]
    br = corners[-1][0]
    center = np.mean(corners, axis=0)[0]
    return np.array([center[0], center[1], br[0] - tl[0], br[1] - tl[1]])


def _select_diverse(valid_images, all_corners, min_center_dist=50,
                    min_size_ratio=0.15, max_images=60):
    """
    贪心筛选姿态多样化的图像子集，去除连续帧中棋盘格位置几乎相同的冗余图像。
    """
    descriptors = [_compute_pose_descriptor(c) for c in all_corners]

    selected = [0]
    for i in range(1, len(descriptors)):
        di = descriptors[i]
        keep = True
        for j in selected:
            dj = descriptors[j]
            dist = np.sqrt((di[0] - dj[0]) ** 2 + (di[1] - dj[1]) ** 2)
            size_i = (di[2] + di[3]) / 2.0
            size_j = (dj[2] + dj[3]) / 2.0
            size_change = abs(size_i - size_j) / max(abs(size_j), 1.0)
            if dist < min_center_dist and size_change < min_size_ratio:
                keep = False
                break
        if keep:
            selected.append(i)
        if len(selected) >= max_images:
            break

    if len(selected) < len(valid_images):
        print(f"  多样性筛选: {len(valid_images)} -> {len(selected)} 张")
    return [valid_images[i] for i in selected], [all_corners[i] for i in selected]


def find_corners(images_path, pattern_size, square_size, diverse=True):
    """
    在图像目录中检测棋盘格角点，可选姿态多样性筛选。

    Returns: (objpoints, imgpoints, img_shape) 或 (None, None, None)
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
    if not images:
        images = sorted(glob.glob(os.path.join(images_path, '*.png')))
    if not images:
        images = sorted(glob.glob(os.path.join(images_path, '*.bmp')))
    if not images:
        print(f"错误: '{images_path}' 中未找到图像文件 (.jpg/.png/.bmp)")
        return None, None, None

    img_shape = None
    valid_imgs, raw_objpoints, raw_corners = [], [], []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            valid_imgs.append(fname)
            raw_objpoints.append(objp)
            raw_corners.append(refined)

    if not raw_objpoints:
        print("错误: 没有图像成功检测到角点。")
        return None, None, None

    total = len(raw_objpoints)
    if diverse and total > 5:
        _, selected_corners = _select_diverse(valid_imgs, raw_corners)
        objpoints = [objp] * len(selected_corners)
        imgpoints = selected_corners
    else:
        objpoints = raw_objpoints
        imgpoints = raw_corners

    print(f"  角点检测: {total} 张成功, 采用 {len(imgpoints)} 张")
    return objpoints, imgpoints, img_shape


# ---------------------------------------------------------------------------
# 鱼眼标定主流程
# ---------------------------------------------------------------------------

def _default_output_dir():
    """项目根目录下的 data_calibration 绝对路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'data_calibration')


def calibrate_fisheye(image_dir, pattern_size, square_size,
                      output_dir=None, undistort=False):
    """
    三步鱼眼标定：
      阶段0 - pinhole 预标定，获取稳定的 K 初值
      阶段1 - fisheye 标定（锁 k2/k3/k4）
      阶段2 - 解锁 k2 精炼
    """
    if output_dir is None:
        output_dir = _default_output_dir()
    print("=" * 50)
    print("鱼眼相机标定")
    print("=" * 50)

    objpoints, imgpoints, img_shape = find_corners(image_dir, pattern_size, square_size)
    if objpoints is None:
        return None

    N = len(objpoints)
    w, h = img_shape
    objpoints_f = [p.reshape(-1, 1, 3) for p in objpoints]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # ---- 阶段0: pinhole 预标定 ----
    print(f"\n[阶段0] pinhole 预标定 ({N} 张)...")
    try:
        ret_p, K_p, _, _, _ = cv2.calibrateCamera(
            objpoints_f, imgpoints, img_shape, None, None)
        K_init = K_p.astype(np.float64)
        print(f"  RMS={ret_p:.2f}, fx={K_init[0,0]:.1f}, fy={K_init[1,1]:.1f}")
    except cv2.error:
        diag = np.sqrt(w * w + h * h)
        K_init = np.array([[diag / 2, 0, w / 2],
                           [0, diag / 2, h / 2],
                           [0, 0, 1]], dtype=np.float64)
        print(f"  pinhole 失败，用估算 K: fx={K_init[0,0]:.0f}")

    K = K_init.copy()
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]

    # ---- 阶段1: fisheye (锁 k2/k3/k4) ----
    print(f"\n[阶段1] fisheye 标定 (锁 k2/k3/k4)...")
    flags1 = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
              | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
              | cv2.fisheye.CALIB_FIX_SKEW
              | cv2.fisheye.CALIB_FIX_K2
              | cv2.fisheye.CALIB_FIX_K3
              | cv2.fisheye.CALIB_FIX_K4)

    try:
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_f, imgpoints, img_shape, K, D, rvecs, tvecs, flags1, criteria)
        print(f"  RMS={ret:.2f}")
    except cv2.error:
        print(f"  失败，回退到自估 K...")
        K = np.zeros((3, 3))
        flags_fb = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                    | cv2.fisheye.CALIB_FIX_SKEW
                    | cv2.fisheye.CALIB_FIX_K2
                    | cv2.fisheye.CALIB_FIX_K3
                    | cv2.fisheye.CALIB_FIX_K4)
        try:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints_f, imgpoints, img_shape, K, D, rvecs, tvecs, flags_fb, criteria)
            print(f"  RMS={ret:.2f}")
        except cv2.error as e:
            print(f"  彻底失败: {e}")
            return None

    # ---- 阶段2: 解锁 k2 精炼 ----
    if ret > 3:
        print(f"\n[阶段2] 解锁 k2 精炼 (RMS={ret:.1f})...")
        flags2 = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                  | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                  | cv2.fisheye.CALIB_FIX_SKEW
                  | cv2.fisheye.CALIB_FIX_K3
                  | cv2.fisheye.CALIB_FIX_K4)
        try:
            ret2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
                objpoints_f, imgpoints, img_shape,
                K.copy(), D.copy(), rvecs, tvecs, flags2, criteria)
            if ret2 < ret * 0.9:
                print(f"  改善: RMS={ret2:.2f}")
                ret, K, D, rvecs, tvecs = ret2, K2, D2, rvecs2, tvecs2
            else:
                print(f"  无显著改善 (RMS={ret2:.2f})")
        except cv2.error:
            print(f"  失败，保留阶段1结果")

    # ---- 阶段3: 自动剔除求解退化帧并稳健重拟合 ----
    view_rms = _per_view_rms(objpoints_f, imgpoints, rvecs, tvecs, K, D)
    median_view_rms = float(np.median(view_rms))
    outlier_threshold = max(3.0, median_view_rms * 4.0)
    keep_indices = np.flatnonzero(view_rms <= outlier_threshold).tolist()
    drop_indices = np.flatnonzero(view_rms > outlier_threshold).tolist()
    if drop_indices and len(keep_indices) >= 15:
        print(f"\n[阶段3] 剔除 {len(drop_indices)} 张求解退化帧: {drop_indices}")
        robust_objpoints = [objpoints_f[i] for i in keep_indices]
        robust_imgpoints = [imgpoints[i] for i in keep_indices]
        robust_flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                        | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                        | cv2.fisheye.CALIB_FIX_SKEW)
        try:
            ret3, K3, D3, rvecs3, tvecs3 = cv2.fisheye.calibrate(
                robust_objpoints, robust_imgpoints, img_shape,
                K.copy(), D.copy(), None, None, robust_flags, criteria)
            if ret3 < ret:
                print(f"  稳健重拟合: RMS={ret3:.2f}, 使用 {len(keep_indices)} 张")
                ret, K, D, rvecs, tvecs = ret3, K3, D3, rvecs3, tvecs3
            else:
                print(f"  未改善 (RMS={ret3:.2f})，保留上一阶段结果")
        except cv2.error as e:
            print(f"  稳健重拟合失败，保留上一阶段结果: {e}")

    # ---- 结果 ----
    print(f"\n{'=' * 50}")
    print(f"RMS 重投影误差: {ret:.2f} px")
    fov_est = 2 * np.arctan(w / (2 * max(K[0, 0], 1))) * 180 / np.pi
    print(f"估计 FOV: {fov_est:.0f}°")
    print(f"\n内参 K:\n{K}")
    print(f"畸变 D: {D.ravel()}")

    # ---- 保存 ----
    params_dir = os.path.join(output_dir, "params")
    _ensure_dir(params_dir)
    _save_yaml({
        'camera_matrix': K,
        'dist_coeffs': D,
        'image_width': w,
        'image_height': h,
    }, os.path.join(params_dir, 'fisheye_calib.yaml'))
    print(f"\n参数已保存至: {params_dir}/fisheye_calib.yaml")

    # ---- 校正 ----
    if undistort:
        print(f"\n校正图像...")
        und_dir = os.path.join(output_dir, "undistorted")
        _ensure_dir(und_dir)
        images = (glob.glob(os.path.join(image_dir, '*.jpg')) +
                  glob.glob(os.path.join(image_dir, '*.png')) +
                  glob.glob(os.path.join(image_dir, '*.bmp')))
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            und_img = cv2.fisheye.undistortImage(img, K, D, Knew=K)
            out = os.path.join(und_dir, os.path.basename(fname))
            cv2.imwrite(out, und_img)

    print("完成。")
    return K, D, img_shape


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="鱼眼相机标定")
    parser.add_argument('--input_dir', required=True, help="标定图像目录")
    parser.add_argument('--pattern_size', required=True, help="棋盘格内角点数, 如 7x9")
    parser.add_argument('--square_size', type=float, required=True, help="方格尺寸 (mm)")
    parser.add_argument('--output_dir', default=None, help="输出目录 (默认: 项目根/data_calibration)")
    parser.add_argument('--undistort', action='store_true', help="校正所有输入图像")
    args = parser.parse_args()

    try:
        w, h = map(int, args.pattern_size.split('x'))
        pattern_size = (w, h)
    except ValueError:
        print("错误: --pattern_size 格式应为 WxH, 如 7x9")
        return

    calibrate_fisheye(
        image_dir=args.input_dir,
        pattern_size=pattern_size,
        square_size=args.square_size,
        output_dir=args.output_dir,
        undistort=args.undistort,
    )


if __name__ == '__main__':
    main()
