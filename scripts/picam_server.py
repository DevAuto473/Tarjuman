#!/usr/bin/env python3
"""
picam_server.py — Picamera2 frame server for Python-version-mismatched envs
=============================================================================
Run this with the SYSTEM Python (3.13) which has picamera2 access, while the
main venv (Python 3.12 + mediapipe) reads frames from it via subprocess pipe.

Usage (called automatically by SmartCamera when source="picam_subprocess"):
    python3 scripts/picam_server.py [width] [height] [fps]

Output: raw BGR24 frames written to stdout, each exactly width*height*3 bytes.
No header, no framing — the reader knows the dimensions from its own args.
"""

import sys
import os
import struct

def main():
    width  = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 480
    fps    = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    try:
        from picamera2 import Picamera2
    except ImportError:
        sys.stderr.write("[picam_server] ERROR: picamera2 not found. "
                         "Run with system python3, not the venv.\n")
        sys.exit(1)

    picam = Picamera2()

    # ── اختر أكبر وضع حساس متاح (نفس ما يفعله rpicam-hello) ──────────────────
    # rpicam-hello يستخدم الوضع الذي يغطّي أكبر مساحة من الحساس = أوسع زاوية.
    # بدون هذا، Picamera2 يختار وضعاً مقصوصاً تلقائياً فيبدو الإطار مقرَّباً.
    sensor_cfg = {}
    try:
        modes = list(picam.sensor_modes or [])
        if modes:
            # الوضع الأكبر مساحةً هو الأوسع زاويةً
            best = max(modes, key=lambda m: m["size"][0] * m["size"][1])
            sensor_cfg = {
                "output_size": best["size"],
                "bit_depth":   best.get("bit_depth", 10),
            }
            sys.stderr.write(
                f"[picam_server] Sensor mode: {best['size'][0]}x{best['size'][1]} "
                f"@ {best.get('bit_depth', '?')}bit  (widest FOV — same as rpicam-hello)\n"
            )
    except Exception as e:
        sys.stderr.write(f"[picam_server] sensor_modes unavailable: {e} — using default\n")

    config = picam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": float(fps)},
        **({"sensor": sensor_cfg} if sensor_cfg else {}),
    )
    picam.configure(config)
    picam.start()

    # ── ScalerCrop = الحساس كاملاً = لا زوم رقمي ──────────────────────────────
    try:
        props  = picam.camera_properties
        full_w = props.get("PixelArraySize", (0, 0))[0]
        full_h = props.get("PixelArraySize", (0, 0))[1]
        if full_w and full_h:
            picam.set_controls({"ScalerCrop": (0, 0, full_w, full_h)})
            sys.stderr.write(
                f"[picam_server] ScalerCrop → full sensor {full_w}x{full_h} "
                f"(wide angle, no digital zoom)\n"
            )
    except Exception as e:
        sys.stderr.write(f"[picam_server] ScalerCrop unavailable: {e}\n")

    sys.stderr.write(f"[picam_server] streaming {width}x{height}@{fps} BGR24 -> stdout\n")
    sys.stderr.flush()

    out = sys.stdout.buffer

    try:
        while True:
            frame = picam.capture_array("main")   # (H, W, 3) RGB uint8
            out.write(frame.tobytes())
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            picam.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
