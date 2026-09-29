"""UMI QR camera timing check for the MVS fisheye camera.

This is intentionally a thin MVS port of scripts/calibrate_uvc_camera_latency.py:
show a timestamp QR code, read camera frames, decode QR timestamps, and compare
them with camera timestamps. For normal MVS inference alignment we still use the
camera hardware capture timestamp, so cameras.obs_latency should normally stay
0.0.
"""
import csv
import os
import sys
import time
from collections import deque
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import click
import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.mvs_camera import (  # noqa: E402
    MvsCamera,
    create_camera_configs_from_serials,
    is_mvs_cpp_available,
    list_mvs_serials,
)


def _parse_resolution(value):
    if value is None or value == "":
        return None
    parts = [int(x.strip()) for x in value.replace("x", ",").split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"resolution must be WIDTHxHEIGHT or WIDTH,HEIGHT, got {value!r}")
    return tuple(parts)


def _make_qr_image(timestamp, qr_size):
    # Prefer the exact UMI dependency/path. The OpenCV fallback only exists
    # because the deployment env may not have qrcode installed.
    text = f"{timestamp:.6f}"
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)
        pil_img = qr.make_image()
        img = np.array(pil_img).astype(np.uint8) * 255
    except ImportError:
        if not hasattr(cv2, "QRCodeEncoder_create"):
            raise RuntimeError(
                "OpenCV has no QRCodeEncoder_create and Python package 'qrcode' is missing."
            )
        encoder = cv2.QRCodeEncoder_create()
        img = encoder.encode(text)
        if img.ndim == 3:
            img = img[..., 0]
        border = max(4, int(round(min(img.shape[:2]) * 0.20)))
        img = cv2.copyMakeBorder(
            img,
            border,
            border,
            border,
            border,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )
    img = np.repeat(img[:, :, None], 3, axis=-1)
    return cv2.resize(img, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)


def _load_camera_config(robot_config, camera_idx, serial_override, fps, test_resolution, exposure_us):
    with open(os.path.expanduser(robot_config), "r") as f:
        cfg = yaml.safe_load(f)
    cameras_cfg = cfg.get("cameras") or {}
    serials = list(cameras_cfg.get("serials") or [])
    if serial_override:
        serial = serial_override
    elif serials:
        serial = serials[camera_idx]
    else:
        detected = list_mvs_serials()
        if not detected:
            raise RuntimeError("no MVS camera detected and no serials in robot config")
        serial = detected[0]
        print(f"[info] auto-picked serial={serial} from {detected}")

    capture_fps = int(fps)
    resolution = _parse_resolution(test_resolution)
    if resolution is None:
        resolution = tuple(cameras_cfg.get("resolution", [224, 224]))

    exposure = float(exposure_us)
    gain_db = cameras_cfg.get("gain_db", None)
    if gain_db is not None and not isinstance(gain_db, (list, tuple)):
        gain_db = float(gain_db)
    open_config = None
    if bool(cameras_cfg.get("use_collection_camera_config", True)):
        input_res = tuple(cameras_cfg.get(
            "input_res",
            cameras_cfg.get("input_resolution", (1440, 1080)),
        ))
        config_serials = serials if serials else [serial]
        camera_configs = list(create_camera_configs_from_serials(
            serials=config_serials,
            fps=capture_fps,
            output_res=resolution,
            input_res=input_res,
            exposure_time_us=exposure,
            gain_auto=cameras_cfg.get("gain_auto", "continuous"),
            gain_db=gain_db,
            prefer_sensor_roi=bool(cameras_cfg.get("prefer_sensor_roi", False)),
        ))
        for cfg_i in camera_configs:
            if "rotate_180" in cameras_cfg:
                cfg_i.rotate_180 = bool(cameras_cfg["rotate_180"])
            if "balance_white_auto" in cameras_cfg:
                cfg_i.balance_white_auto = cameras_cfg["balance_white_auto"]
            if "black_level_enable" in cameras_cfg:
                cfg_i.black_level_enable = bool(cameras_cfg["black_level_enable"])
            if "black_level" in cameras_cfg:
                cfg_i.black_level = int(cameras_cfg["black_level"])
            if "brightness" in cameras_cfg:
                cfg_i.brightness = int(cameras_cfg["brightness"])
        cfg_idx = config_serials.index(serial) if serial in config_serials else 0
        open_config = camera_configs[cfg_idx]
        resolution = (open_config.width, open_config.height)

    selected_gain_db = gain_db
    if open_config is not None:
        selected_gain_db = open_config.gain_db
    elif isinstance(gain_db, (list, tuple)):
        cfg_idx = serials.index(serial) if serial in serials else 0
        selected_gain_db = None if gain_db[cfg_idx] is None else float(gain_db[cfg_idx])

    return cameras_cfg, serial, capture_fps, resolution, exposure, selected_gain_db, open_config


def _summarize_offsets(label, offsets_s):
    offsets_ms = np.asarray(offsets_s, dtype=np.float64) * 1000.0
    print(f"{label:<34}: AVG={offsets_s.mean():.6f}s STD={offsets_s.std():.6f}s "
          f"MEDIAN={np.median(offsets_s):.6f}s")
    print(f"{'':<34}  mean={offsets_ms.mean():.3f}ms std={offsets_ms.std():.3f}ms "
          f"p95={np.percentile(offsets_ms, 95):.3f}ms")


@click.command()
@click.option("-rc", "--robot-config", default=str(ROOT_DIR / "example/eval_robots_config.yaml"))
@click.option("-ci", "--camera_idx", "camera_idx", type=int, default=0)
@click.option("--serial", default="", help="Override MVS serial.")
@click.option("-qs", "--qr_size", "qr_size", type=int, default=720)
@click.option("-f", "--fps", type=int, default=60)
@click.option("-n", "--n_frames", "n_frames", type=int, default=120)
@click.option("--test_resolution", "--test-resolution", default="720x720")
@click.option("--exposure_us", "--exposure-us", type=float, default=8000.0)
@click.option("--debug-dir", default="data_local/mvs_fisheye_qr_debug")
@click.option("--output-csv", default="data_local/mvs_fisheye_qr_latency.csv")
def main(
        robot_config,
        camera_idx,
        serial,
        qr_size,
        fps,
        n_frames,
        test_resolution,
        exposure_us,
        debug_dir,
        output_csv):
    if not is_mvs_cpp_available():
        raise RuntimeError("MVS C++ backend is not built or importable")

    cameras_cfg, serial, capture_fps, resolution, exposure, gain_db, open_config = _load_camera_config(
        robot_config=robot_config,
        camera_idx=camera_idx,
        serial_override=serial,
        fps=fps,
        test_resolution=test_resolution,
        exposure_us=exposure_us,
    )
    print(f"[info] opening serial={serial} {resolution} @ {capture_fps}Hz, "
          f"exposure {exposure:.0f}us")
    print("[info] point the fisheye camera at the QR window; press c to compute stats, q to quit")

    get_max_k = int(n_frames)
    detector = cv2.QRCodeDetector()
    qr_latency_deque = deque(maxlen=get_max_k)
    qr_capture_deque = deque(maxlen=get_max_k)
    qr_recv_deque = deque(maxlen=get_max_k)
    data = None
    last_cam_img = None
    last_qr_img = None

    with SharedMemoryManager() as shm_manager:
        camera = MvsCamera(
            shm_manager=shm_manager,
            mvs_serial=serial,
            resolution=resolution,
            open_config=open_config,
            capture_fps=capture_fps,
            put_downsample=False,
            get_max_k=get_max_k,
            receive_latency=0.0,
            exposure_time_us=exposure,
            gain_auto=cameras_cfg.get("gain_auto", "continuous"),
            gain_db=gain_db,
            balance_white_auto=cameras_cfg.get("balance_white_auto", "continuous"),
            rotate_180=bool(cameras_cfg.get("rotate_180", True)),
            auto_crop_black_border=bool(cameras_cfg.get("auto_crop_black_border", True)),
            auto_crop_probe_frames=int(cameras_cfg.get("auto_crop_probe_frames", 4)),
            auto_crop_min_probe_frames=int(cameras_cfg.get("auto_crop_min_probe_frames", 2)),
            auto_crop_warmup_frames=int(cameras_cfg.get("auto_crop_warmup_frames", 2)),
            auto_crop_black_threshold=int(cameras_cfg.get("auto_crop_black_threshold", 12)),
            auto_crop_margin_px=int(cameras_cfg.get("auto_crop_margin_px", 0)),
            auto_crop_min_size_ratio=float(cameras_cfg.get("auto_crop_min_size_ratio", 0.80)),
            auto_crop_reopen_delay_ms=int(cameras_cfg.get("auto_crop_reopen_delay_ms", 200)),
            auto_crop_detect_max_dim=int(cameras_cfg.get("auto_crop_detect_max_dim", 720)),
            launch_timeout=float(cameras_cfg.get("launch_timeout", 10.0)),
            verbose=False,
        )
        camera.start(wait=True)
        try:
            cv2.setNumThreads(1)
            while True:
                t_start = time.time()
                data = camera.get(out=data)
                cam_img = data["color"]
                last_cam_img = cam_img.copy()
                cam_vis = cam_img.copy()

                code, corners, _ = detector.detectAndDecodeCurved(cam_img)
                color = (255, 0, 0)
                if len(code) > 0:
                    color = (0, 255, 0)
                    ts_qr = float(code)
                    qr_capture_deque.append(data["camera_capture_timestamp"] - ts_qr)
                    qr_recv_deque.append(data["camera_receive_timestamp"] - ts_qr)
                else:
                    qr_capture_deque.append(float("nan"))
                    qr_recv_deque.append(float("nan"))
                if corners is not None:
                    cv2.fillPoly(cam_vis, corners.astype(np.int32), color)

                t_sample = time.time()
                qr_img = _make_qr_image(t_sample, qr_size)
                last_qr_img = qr_img.copy()
                cv2.imshow("Timestamp QRCode", qr_img)
                t_show = time.time()
                qr_latency_deque.append(t_show - t_sample)
                cv2.imshow("Camera", cam_vis[..., ::-1])
                keycode = cv2.pollKey()

                display_overhead = np.mean(qr_latency_deque)
                cap_arr = np.asarray(qr_capture_deque, dtype=np.float64)
                recv_arr = np.asarray(qr_recv_deque, dtype=np.float64)
                cap_avg = np.nanmean(cap_arr) - display_overhead
                recv_avg = np.nanmean(recv_arr) - display_overhead
                det_rate = 1.0 - np.mean(np.isnan(recv_arr))
                print("Running at {:.1f} FPS. Capture Latency: {:.3f}. Recv Latency: {:.3f}. Detection Rate: {:.2f}".format(
                    1.0 / max(time.time() - t_start, 1e-6),
                    cap_avg,
                    recv_avg,
                    det_rate,
                ))

                if keycode == ord("c"):
                    break
                if keycode == ord("q"):
                    return

            data = camera.get(k=get_max_k)
        finally:
            camera.stop(wait=True)
            cv2.destroyAllWindows()

    qr_capture_map = {}
    qr_recv_map = {}
    cap_ts = np.asarray(data["camera_capture_timestamp"], dtype=np.float64)
    recv_ts = np.asarray(data["camera_receive_timestamp"], dtype=np.float64)
    frame_ids = np.asarray(data.get("camera_frame_id", np.arange(len(cap_ts))), dtype=np.int64)
    for i in range(len(recv_ts)):
        img = data["color"][i]
        code, _, _ = detector.detectAndDecodeCurved(img)
        if len(code) > 0:
            ts_qr = float(code)
            if ts_qr not in qr_recv_map:
                qr_capture_map[ts_qr] = cap_ts[i]
                qr_recv_map[ts_qr] = recv_ts[i]

    avg_qr_latency = float(np.mean(qr_latency_deque)) if len(qr_latency_deque) else 0.0
    rows = []
    for ts_qr in sorted(qr_recv_map):
        rows.append({
            "qr_timestamp": ts_qr,
            "camera_capture_timestamp": float(qr_capture_map[ts_qr]),
            "camera_receive_timestamp": float(qr_recv_map[ts_qr]),
            "display_overhead_s": avg_qr_latency,
            "qr_to_capture_s": float(qr_capture_map[ts_qr] - ts_qr - avg_qr_latency),
            "qr_to_receive_s": float(qr_recv_map[ts_qr] - ts_qr - avg_qr_latency),
            "receive_minus_capture_s": float(qr_recv_map[ts_qr] - qr_capture_map[ts_qr]),
        })

    print()
    print(f"frames analyzed                   : {len(cap_ts)}")
    print(f"decoded unique QR frames          : {len(rows)}")
    if len(cap_ts) > 1:
        expected_ms = 1000.0 / float(capture_fps)
        cap_intervals_ms = np.diff(cap_ts) * 1000.0
        frame_deltas = np.diff(frame_ids)
        print(
            f"capture timestamp interval        : "
            f"mean={cap_intervals_ms.mean():.3f}ms "
            f"std={cap_intervals_ms.std():.3f}ms "
            f"min={cap_intervals_ms.min():.3f}ms "
            f"max={cap_intervals_ms.max():.3f}ms "
            f"(expected {expected_ms:.3f}ms)"
        )
        print(
            f"frame id delta                    : "
            f"min={frame_deltas.min()} max={frame_deltas.max()} "
            f"dropped_gaps={int(np.sum(frame_deltas > 1))}"
        )

    if debug_dir:
        debug_path = ROOT_DIR / debug_dir
        debug_path.mkdir(parents=True, exist_ok=True)
        if last_cam_img is not None:
            cv2.imwrite(str(debug_path / "last_camera_rgb.png"), last_cam_img[..., ::-1])
        if last_qr_img is not None:
            cv2.imwrite(str(debug_path / "last_qr.png"), last_qr_img)

    if len(rows) == 0:
        print("[FAIL] no QR detections in captured frames")
        if debug_dir:
            print(f"saved debug images                : {ROOT_DIR / debug_dir}")
        return

    qr_to_capture = np.array([r["qr_to_capture_s"] for r in rows], dtype=np.float64)
    qr_to_receive = np.array([r["qr_to_receive_s"] for r in rows], dtype=np.float64)
    receive_minus_capture = np.array([r["receive_minus_capture_s"] for r in rows], dtype=np.float64)
    print(f"{'display overhead estimate':<34}: {avg_qr_latency * 1000:.3f}ms")
    _summarize_offsets("QR display -> MVS capture ts", qr_to_capture)
    _summarize_offsets("QR display -> MVS receive ts", qr_to_receive)
    _summarize_offsets("receive ts - capture ts", receive_minus_capture)
    print()
    print("recommended cameras.obs_latency : 0.000000 s for MVS hardware timestamp path")
    print("  QR measures display-to-image visual delay; eval uses MVS capture timestamps.")

    if output_csv:
        out_path = ROOT_DIR / output_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved matches                    : {out_path}")


if __name__ == "__main__":
    main()
