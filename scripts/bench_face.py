"""
bench_face.py - what does the face cost?
=========================================
    npm run bench:face

Why this is separate from bench_pipeline.py
-------------------------------------------
That one measures the capture pipeline as it stands and passes judgement on it.
This one answers a single question: how much does `FaceLandmarker` cost on top of
what already runs, and at what throttle is it affordable? Mixing the two questions
corrupts the first one's verdict.

What it measures
----------------
  1. The baseline: hands (every frame) and body (once every five, as `PoseTracker`
     does) - that is, what you pay today.
  2. The face call alone: `detect_for_video` time, measured rather than guessed.
  3. The amortised cost at 1, 2, 3 and 5 - because the face does not need the
     hands' frame rate. Expression changes slowly and is smoothed between calls.

WARNING: sit in front of the camera while this runs. With no face detected the
detector runs but the landmark model after it does not, so the number comes out
far below the truth. This script measures the detection rate and refuses to give
a verdict when it is low.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import statistics
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from tarjuman_core.camera_manager import SmartCamera, choose_camera_interactive
from tarjuman_core.feature_extractor import (
    POSE_EVERY_N_FRAMES, PoseTracker, prepare_frame,
)

WARMUP = 15
FRAMES = 120

# The face model is not part of the mediapipe package; download once and keep.
#
# Two URLs in order: the documented one (`latest`) first, then the pinned release
# (`1`). `latest` may one day move to a model mediapipe 0.10.14 cannot read, so
# the second stays as a safety net.
MODEL_URLS = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task",
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def ms(values):
    return statistics.mean(values) * 1000 if values else 0.0


def p95(values):
    """The peak matters as much as the mean: one slow call a second reads as a stutter."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] * 1000


def ensure_model() -> Path:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   Downloading the face model (once) -> {MODEL_PATH}")
    last = None
    for url in MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"   Done ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
            return MODEL_PATH
        except Exception as exc:
            last = exc
            print(f"   Failed from {url.rsplit('/', 3)[1]}: {type(exc).__name__}")
    print(f"\n[FAIL] Could not download: {type(last).__name__}: {last}")
    print("       Download it by hand and put it here:")
    print(f"       {MODEL_PATH}")
    for url in MODEL_URLS:
        print(f"       {url}")
    raise SystemExit(1)


def make_face_landmarker(path: Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            # This is the point: 52 values with ARKit names, the same names the
            # model's morph targets carry - so no translation table between them.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
    )


def run(cam, hands, pose, face, every_n):
    """
    One pass. `every_n=0` means no face at all - that is the baseline.

    Face time is collected in its own list, so any throttle can be computed later
    without re-measuring: amortised = sum of calls / number of frames.
    """
    t_frame, t_face = [], []
    faces_seen = 0
    face_calls = 0
    counter = 0
    ts = 0
    for i in range(WARMUP + FRAMES):
        t0 = time.perf_counter()
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        frame = prepare_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hands.process(rgb)
        pose.update(rgb)

        if every_n and counter % every_n == 0:
            rgb_c = rgb.copy()
            rgb_c.flags.writeable = True
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)
            ts += 33
            f0 = time.perf_counter()
            res = face.detect_for_video(img, ts)
            f1 = time.perf_counter()
            if i >= WARMUP:
                t_face.append(f1 - f0)
                face_calls += 1
                if res.face_blendshapes:
                    faces_seen += 1
        counter += 1
        t1 = time.perf_counter()
        if i >= WARMUP:
            t_frame.append(t1 - t0)
    return {"frame": t_frame, "face": t_face,
            "calls": face_calls, "seen": faces_seen}


def main() -> int:
    print("=" * 68)
    print("   FACE COST - FaceLandmarker on top of the current pipeline")
    print("=" * 68)

    model = ensure_model()

    cam = SmartCamera(source=choose_camera_interactive())
    cam.start()
    if not cam.is_running:
        print("[FAIL] Camera did not open.")
        return 1
    cam.stop_grabber()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = PoseTracker()
    face = make_face_landmarker(model)

    print("\n   Sit in front of the camera. Measuring now...")

    print(f"\n   [1/2] Baseline: hands every frame + body every {POSE_EVERY_N_FRAMES}")
    base = run(cam, hands, pose, face, every_n=0)
    base_ms = ms(base["frame"])
    print(f"         {base_ms:6.1f} ms/frame  ->  {1000/max(base_ms,1e-6):5.1f} fps")

    print("\n   [2/2] With the face on every frame - to time a single call")
    full = run(cam, hands, pose, face, every_n=1)
    face_ms, face_p95 = ms(full["face"]), p95(full["face"])
    seen = full["seen"] / max(1, full["calls"]) * 100

    print(f"         face call      : {face_ms:6.1f} ms  (95th pct = {face_p95:.1f})")
    print(f"         face detected in: {seen:5.1f}% of calls")

    print("\n" + "=" * 68)
    print("   Amortised cost and the resulting frame rate")
    print("=" * 68)
    print(f"   {'every N frames':<18s}{'amortised':>11s}{'ms/frame':>11s}{'fps':>8s}{'loss':>9s}")
    print("   " + "-" * 57)
    rows = []
    base_fps = 1000 / max(base_ms, 1e-6)
    for n in (1, 2, 3, 5, 0):
        amort = face_ms / n if n else 0.0
        total = base_ms + amort
        fps = 1000 / max(total, 1e-6)
        loss = (1 - fps / base_fps) * 100
        label = "no face" if n == 0 else f"1 in {n}"
        rows.append((n, fps, loss))
        print(f"   {label:<18s}{amort:10.1f} {total:10.1f} {fps:7.1f} {loss:8.1f}%")

    print("\n" + "=" * 68)
    print("   VERDICT")
    print("=" * 68)
    if seen < 60:
        print("   [!] No face was detected in most calls, so these numbers are")
        print("       well below the truth. Re-run sitting in front of the")
        print("       camera in decent light.")
        cam.release()
        return 2

    # The face is driven at 10 Hz or better; below that expression looks choppy.
    ok_n = None
    for n, fps, loss in rows:
        if n and loss <= 12 and fps / n >= 8:
            ok_n = (n, fps, loss)
            break
    if ok_n:
        n, fps, loss = ok_n
        print(f"   Run the face once every {n} frame(s).")
        print(f"   Loss {loss:.0f}% ({base_fps:.0f} -> {fps:.0f} fps), "
              f"and the face updates {fps/n:.0f} times/s.")
        print("   Put it in .env:  FACE_EVERY_N_FRAMES=%d" % n)
    else:
        print("   No throttle gives both an acceptable loss and a usable face rate.")
        print("   Options: turn the face off on this hardware (FACE_ENABLED=0),")
        print("   accept a bigger loss, or settle for the mouth from BlazePose")
        print("   with no eyebrows.")

    print(f"\n   For comparison: the body already costs {POSE_EVERY_N_FRAMES} frames' throttle.")
    cam.release()
    return 0


if __name__ == "__main__":
    try:
        _sys.exit(main())
    except KeyboardInterrupt:
        print("\n   Stopped.")
        _sys.exit(130)
