"""
bench_pipeline.py - where does each millisecond actually go?
============================================================
    npm run bench

Why this exists
---------------
The frame rate did not move after the capture pipeline was rewritten, and there
are only two possible reasons: the changes are not taking effect, or they were
never the bottleneck. Those need completely different fixes, and guessing which
one it is wastes an evening. This measures it on YOUR machine.

It reports three things:

  1. What the camera actually GRANTED. Asking a webcam for MJPEG at 60 fps is a
     request, not a command - drivers silently downgrade. If the sensor is still
     handing over 30 fps in YUY2, no amount of threading will help.

  2. Where the per-frame time goes, stage by stage. MediaPipe is normally ~95%
     of it, in which case optimising anything else is wasted effort.

  3. Blocking read versus the background grabber, measured back to back on the
     same camera. The grabber only pays off when frame DECODE is expensive
     enough to be worth overlapping with inference; on a slow sensor it can do
     nothing at all, and it is better to know that than to assume.
"""

# -- Import bootstrap ---------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import statistics
import time

import cv2
import mediapipe as mp
import numpy as np

from tarjuman_core.camera_manager import SmartCamera, choose_camera_interactive
from tarjuman_core.feature_extractor import (
    PoseTracker, extract_frame_features, prepare_frame, split_hands,
)
from tarjuman_core.gesture_segmenter import GestureSegmenter

WARMUP = 15
FRAMES = 120


def ms(values):
    return statistics.mean(values) * 1000 if values else 0.0


def describe_camera(cam):
    print("\n" + "=" * 68)
    print("   WHAT THE CAMERA ACTUALLY GRANTED")
    print("=" * 68)
    inner = getattr(cam, "_cam", None)
    if inner is None or not hasattr(inner, "get"):
        print(f"   backend   : {getattr(cam, '_backend', '?')} (no OpenCV handle)")
        return
    fourcc = int(inner.get(cv2.CAP_PROP_FOURCC))
    tag = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)) if fourcc else "?"
    w = int(inner.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(inner.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = inner.get(cv2.CAP_PROP_FPS)
    print(f"   backend   : {getattr(cam, '_backend', '?')}")
    print(f"   asked for : {SmartCamera.WIDTH}x{SmartCamera.HEIGHT} @ "
          f"{SmartCamera.FPS} fps, MJPG")
    print(f"   granted   : {w}x{h} @ {fps:.0f} fps, format '{tag}'")
    if tag.upper() not in ("MJPG", "MJPEG"):
        print("   [!] NOT MJPEG. Most webcams cap raw formats at 30 fps or less,")
        print("       which puts a hard ceiling on everything downstream.")
    if fps and fps <= 31:
        print("   [!] Sensor is delivering <=30 fps. The background grabber")
        print("       cannot invent frames, so it will show no gain here.")


def measure_sensor_rate(cam, seconds=2.0):
    """Raw delivery rate, with no processing at all in the loop."""
    cam.stop_grabber()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        ok, _ = cam.read()
        if ok:
            n += 1
    return n / seconds


def profile(cam, hands, pose, seg):
    t = {k: [] for k in ("read", "prep", "cvt", "hands", "pose",
                         "feat", "seg", "total")}
    for i in range(WARMUP + FRAMES):
        t0 = time.perf_counter()
        ok, frame = cam.read()
        t1 = time.perf_counter()
        if not ok or frame is None:
            continue
        frame = prepare_frame(frame)
        t2 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t3 = time.perf_counter()
        res = hands.process(rgb)
        t4 = time.perf_counter()
        anchors = pose.update(rgb)
        t5 = time.perf_counter()
        left, right = split_hands(res)
        feats = extract_frame_features(res, anchors)
        t6 = time.perf_counter()
        seg.update(feats, (left is not None) or (right is not None),
                   now=time.monotonic())
        t7 = time.perf_counter()

        if i < WARMUP:
            continue
        t["read"].append(t1 - t0)
        t["prep"].append(t2 - t1)
        t["cvt"].append(t3 - t2)
        t["hands"].append(t4 - t3)
        t["pose"].append(t5 - t4)
        t["feat"].append(t6 - t5)
        t["seg"].append(t7 - t6)
        t["total"].append(t7 - t0)
    return t



def sweep_modes(cam):
    """
    Ask the camera for every plausible mode and time what it actually delivers.

    A datasheet figure is not a measurement, and "supports 60 fps" is usually
    true only at some resolution/format pair the driver will not tell you about.
    Since lowering the resolution costs MediaPipe nothing - it resizes to a
    fixed tensor regardless - a lower mode is worth taking IF it buys frames.
    This is the only way to find out which, if any, does.
    """
    inner = getattr(cam, "_cam", None)
    if inner is None or not hasattr(inner, "set"):
        return None

    print("\n" + "=" * 68)
    print("   MODE SWEEP - what this camera can really deliver")
    print("=" * 68)
    print(f"   {'resolution':>12s} {'format':>8s} {'asked':>6s} {'MEASURED':>9s}")
    print("   " + "-" * 42)

    # Remember what we started with. Without this the sweep leaves the camera
    # in whatever mode it tested LAST, and every measurement after it silently
    # describes a different resolution than the one actually configured - which
    # is precisely what happened: a 320x240 profile presented as 640x480.
    orig = (int(inner.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(inner.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(inner.get(cv2.CAP_PROP_FOURCC)))

    sizes = [(640, 480), (800, 600), (1280, 720), (424, 240), (320, 240)]
    results = []
    for (w, h) in sizes:
        for fmt in ("MJPG", "YUY2"):
            inner.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            inner.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            inner.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
            inner.set(cv2.CAP_PROP_FPS, 60)
            got_w = int(inner.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(inner.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tag = cam._fourcc_tag(inner)
            rate = cam._measure_rate(inner, seconds=0.6)
            results.append((got_w, got_h, tag, rate))
            flag = "  <-- best" if rate == max(r[3] for r in results) else ""
            print(f"   {got_w:>5d}x{got_h:<6d} {tag:>8s} {fmt:>6s} "
                  f"{rate:>8.1f}{flag}")

    # Put the camera back before anything else measures it.
    inner.set(cv2.CAP_PROP_FRAME_WIDTH, orig[0])
    inner.set(cv2.CAP_PROP_FRAME_HEIGHT, orig[1])
    inner.set(cv2.CAP_PROP_FOURCC, orig[2])
    cam._measure_rate(inner, seconds=0.15)      # let it settle
    print(f"   restored: {int(inner.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(inner.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    best = max(results, key=lambda r: r[3])
    print(f"\n   Best mode: {best[0]}x{best[1]} {best[2]} at {best[3]:.1f} fps")
    if best[3] <= max(r[3] for r in results if r[0] >= 640) * 1.05:
        print("   No mode beats the current one - this sensor is rate-capped,")
        print("   so keep the HIGHER resolution: it costs MediaPipe nothing")
        print("   and gives it more detail to find small hands with.")
    return best


def main() -> int:
    print("=" * 68)
    print("   TARJUMAN PIPELINE BENCHMARK")
    print("=" * 68)

    cam = SmartCamera(source=choose_camera_interactive())
    cam.start()
    if not cam.is_running:
        print("[FAIL] Camera did not open.")
        return 1

    describe_camera(cam)

    best_mode = sweep_modes(cam)

    print("\n   Measuring raw sensor rate (no processing)...")
    raw = measure_sensor_rate(cam)
    print(f"   sensor delivers   : {raw:5.1f} fps with an empty loop")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = PoseTracker()
    seg = GestureSegmenter()

    print("\n" + "=" * 68)
    print(f"   PER-FRAME BREAKDOWN  ({FRAMES} frames, blocking read)")
    print("=" * 68)
    cam.stop_grabber()
    blocking = profile(cam, hands, pose, seg)
    total_b = ms(blocking["total"])
    labels = {"read": "camera read", "prep": "prepare_frame (flip)",
              "cvt": "BGR->RGB", "hands": "MediaPipe Hands",
              "pose": "MediaPipe Pose (throttled)", "feat": "feature extraction",
              "seg": "segmenter"}
    for k, label in labels.items():
        v = ms(blocking[k])
        share = v / total_b * 100 if total_b else 0
        bar = "#" * int(round(share / 4))
        print(f"   {label:<28s} {v:7.2f} ms  {share:5.1f}%  {bar}")
    print(f"   {'-' * 60}")
    print(f"   {'TOTAL':<28s} {total_b:7.2f} ms  ->  {1000/total_b:5.1f} fps")

    print("\n" + "=" * 68)
    print(f"   SAME AGAIN, WITH THE BACKGROUND GRABBER")
    print("=" * 68)
    cam.start_grabber()
    time.sleep(0.3)
    grab = profile(cam, hands, pose, seg)
    total_g = ms(grab["total"])
    print(f"   camera read                  {ms(grab['read']):7.2f} ms "
          f"(was {ms(blocking['read']):.2f})")
    print(f"   {'TOTAL':<28s} {total_g:7.2f} ms  ->  {1000/total_g:5.1f} fps")
    cam.stop_grabber()

    print("\n" + "=" * 68)
    print("   VERDICT")
    print("=" * 68)
    infer = ms(blocking["hands"]) + ms(blocking["pose"])
    print(f"   MediaPipe alone costs {infer:.1f} ms -> a hard ceiling of "
          f"{1000/max(infer,1e-6):.1f} fps")
    gain = total_b / total_g if total_g else 1.0
    print(f"   Grabber gain: {gain:.2f}x  "
          f"({1000/total_b:.1f} -> {1000/total_g:.1f} fps)")
    print()
    print(f"   Sensor ceiling: {raw:.1f} fps measured with an EMPTY loop.")
    print()
    if raw < 1000 / max(infer, 1e-6) * 0.9:
        print("   THE CAMERA IS THE BOTTLENECK, not MediaPipe. An empty loop")
        print(f"   only reaches {raw:.0f} fps, so no amount of processing work")
        print("   can go faster. Check the negotiated format line above: if it")
        print("   is not MJPG, the driver is bandwidth-capping the sensor.")
    elif infer / total_b > 0.8:
        print("   MediaPipe dominates. Capture-side work is finished; the only")
        print("   remaining levers are running Hands in its own thread, or")
        print("   accepting this rate. Lowering the RESOLUTION will not help -")
        print("   MediaPipe resizes internally to a fixed tensor either way.")
    elif ms(blocking["read"]) / total_b > 0.25:
        print("   Camera read is still a large share. Check the format line")
        print("   above: if it is not MJPG, the driver refused the request.")
    else:
        print("   Time is spread out; no single stage dominates.")

    if best_mode and best_mode[3] > raw * 1.15:
        print()
        print(f"   A faster mode EXISTS: {best_mode[0]}x{best_mode[1]} "
              f"{best_mode[2]} reached {best_mode[3]:.0f} fps vs {raw:.0f} now.")
        print("   Set it in .env and the whole pipeline follows:")
        print(f"     CAMERA_WIDTH={best_mode[0]}")
        print(f"     CAMERA_HEIGHT={best_mode[1]}")

    cam.release()
    return 0


if __name__ == "__main__":
    try:
        _sys.exit(main())
    except KeyboardInterrupt:
        print("\n[stop] Interrupted.")
        _sys.exit(130)
