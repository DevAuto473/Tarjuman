"""
list_cameras.py — find and identify every camera Windows can see
=================================================================
OpenCV exposes cameras as bare numbers, with no names. When a phone is attached
through DroidCam / iVCam / Camo it just becomes "another index", so picking the
wrong one silently records the laptop webcam instead of the phone — and nothing
in the data says so afterwards.

This tool reports the DEVICE NAMES where the platform allows it, probes every
index, and can show a live preview so the right one is chosen by eye.

    python list_cameras.py             # names + probe
    python list_cameras.py --preview   # also show each camera live
    npm run cameras
"""

import argparse
import os
import platform
import sys

import cv2

MAX_INDEX = 10
IS_WINDOWS = platform.system() == "Windows"

# Probing an index that does not exist makes OpenCV print a wall of backend
# warnings. That noise buries the actual answer, which is the whole point of
# this tool, so the logger is silenced for the duration.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")


# ─────────────────────────────────────────────────────────────────────────────
#  Device names (Windows / DirectShow)
# ─────────────────────────────────────────────────────────────────────────────

def device_names() -> list[str]:
    """
    Return DirectShow device names, in the same order OpenCV indexes them.

    Names are what make this tool actually useful — "DroidCam Source 3" is
    unambiguous, "index 2" is a guess. Requires pygrabber; if it is missing we
    degrade to resolution-based hints rather than failing.
    """
    if not IS_WINDOWS:
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except ImportError:
        return []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Probing
# ─────────────────────────────────────────────────────────────────────────────

def open_capture(index: int):
    """
    Open an index, trying each Windows backend, and report which one worked.

    Reporting the backend matters: Windows exposes cameras through both
    DirectShow and Media Foundation, and a device may answer on only one of
    them. A diagnostic that quietly succeeds via MSMF while the recorder forces
    DSHOW would send the user chasing a camera that "exists" but never opens.

    Returns (capture | None, backend_name).
    """
    attempts = ([("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("default", None)]
                if IS_WINDOWS else [("default", None)])

    for name, api in attempts:
        cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name
        cap.release()
    return None, None


def probe(max_index: int = MAX_INDEX) -> list[dict]:
    """Test each index and report the ones that actually deliver a frame."""
    results = []
    for idx in range(max_index):
        cap = None
        try:
            cap, backend = open_capture(idx)
            if cap is None:
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                results.append({"index": idx, "opened": True, "frame": False})
                continue
            h, w = frame.shape[:2]
            results.append({
                "index": idx, "opened": True, "frame": True, "backend": backend,
                "w": w, "h": h, "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            })
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return results


# Clients that pipe a PHONE camera into Windows.
PHONE_HINTS = ("droidcam", "ivcam", "camo", "epoccam", "iriun", "e2esoft")

# Not a camera at all — OBS re-broadcasts whatever scene is on its canvas.
# Selecting it records OBS's output, not the phone, which is a subtle and very
# confusing way to end up with the wrong footage.
RELAY_HINTS = ("obs virtual", "streamlabs", "vtube", "manycam", "xsplit")

# Built-in laptop webcams.
BUILTIN_HINTS = ("integrated", "built-in", "truevision", "true vision",
                 "hd camera", "hd webcam", "facetime")


def classify(name: str, w: int) -> str:
    """Best guess at what a device is, name first and resolution as fallback."""
    low = (name or "").lower()
    if any(h in low for h in RELAY_HINTS):
        return "  <-- NOT a camera (relays another app's output)"
    if any(h in low for h in PHONE_HINTS):
        return "  <-- PHONE (virtual camera)"
    if any(h in low for h in BUILTIN_HINTS):
        return "  <-- built-in webcam"
    if w and w >= 1280:
        return "  <-- probably the phone (high resolution)"
    if w and w <= 640:
        return "  <-- probably the built-in webcam"
    return ""


def preview(index: int, label: str) -> str | None:
    """Live window for one index. Returns 'stop' if the user pressed q."""
    cap, _backend = open_capture(index)
    if cap is None:
        print(f"     index {index}: could not open for preview")
        return None

    print(f"     index {index} ({label}) — any key = next, q = stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            cv2.putText(frame, f"index {index}  {label}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 90), 2, cv2.LINE_AA)
            cv2.imshow("Camera preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                return "stop" if key == ord("q") else None
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Client detection
# ─────────────────────────────────────────────────────────────────────────────

def installed_clients() -> list[str]:
    """Look for the PC clients that create a virtual camera."""
    if not IS_WINDOWS:
        return []
    candidates = {
        "DroidCam": [r"C:\Program Files (x86)\DroidCam", r"C:\Program Files\DroidCam"],
        "iVCam":    [r"C:\Program Files (x86)\e2eSoft\iVCam", r"C:\Program Files\e2eSoft\iVCam"],
        "Camo":     [r"C:\Program Files\Reincubate\Camo", r"C:\Program Files (x86)\Reincubate\Camo"],
        "EpocCam":  [r"C:\Program Files (x86)\Elgato\EpocCam", r"C:\Program Files\Elgato\EpocCam"],
    }
    return [name for name, paths in candidates.items()
            if any(os.path.isdir(p) for p in paths)]


# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="List and identify camera devices.")
    ap.add_argument("--preview", action="store_true", help="show each camera live")
    ap.add_argument("--max", type=int, default=MAX_INDEX, help="highest index to probe")
    args = ap.parse_args()

    print("=" * 66)
    print("  CAMERA DIAGNOSTIC")
    print("=" * 66)

    # ── 1. PC clients ───────────────────────────────────────────────────────
    clients = installed_clients()
    print("\n1) Virtual-camera clients installed:")
    if clients:
        for c in clients:
            print(f"     [OK] {c}")
    elif IS_WINDOWS:
        print("     [!] None found (DroidCam / iVCam / Camo / EpocCam).")
        print("         A phone over USB needs one of these RUNNING —")
        print("         the phone app alone is not enough.")
    else:
        print("     (skipped: not Windows)")

    # ── 2. Device names ─────────────────────────────────────────────────────
    names = device_names()
    print("\n2) Device names reported by Windows:")
    if names:
        for i, n in enumerate(names):
            print(f"     {i}: {n}")
    elif IS_WINDOWS:
        print("     (names unavailable — install pygrabber for them:)")
        print("       venv\\Scripts\\pip install pygrabber")
    else:
        print("     (not available on this platform)")

    # ── 3. Probe ────────────────────────────────────────────────────────────
    print("\n3) Probing indices (a few seconds)...\n")
    results = probe(args.max)
    working = [r for r in results if r.get("frame")]

    if not results:
        print("     [!] No camera devices responded at all.")
    for r in results:
        idx = r["index"]
        name = names[idx] if idx < len(names) else ""
        if r.get("frame"):
            tag = classify(name, r["w"])
            label = f" — {name}" if name else ""
            print(f"     index {idx}: {r['w']}x{r['h']} @ {r['fps']:.0f}fps"
                  f"  [{r.get('backend', '?')}]{label}{tag}")
        else:
            print(f"     index {idx}: opened but sent no frames "
                  f"(in use by another app?)")

    # ── 4. Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    if not working:
        print("  NO USABLE CAMERA FOUND")
        print("=" * 66)
        print("  Checklist for a phone over USB:")
        print("   1. Install the PC client (DroidCam / iVCam / Camo) — not just")
        print("      the phone app. The client is what creates the camera.")
        print("   2. iPhone also needs Apple 'Apple Devices' or iTunes installed")
        print("      for the USB drivers.")
        print("   3. Open the PC client, choose USB mode, press CONNECT and wait")
        print("      until video appears IN THE CLIENT WINDOW first.")
        print("   4. Unlock the phone and accept 'Trust this computer'.")
        print("   5. Close Zoom / Teams / Windows Camera — they lock the device.")
        return 1

    def name_of(r):
        return names[r["index"]].lower() if r["index"] < len(names) else ""

    # Never recommend a relay device — it would silently record another app's
    # canvas instead of the camera.
    candidates = [r for r in working
                  if not any(h in name_of(r) for h in RELAY_HINTS)] or working

    phone = next((r for r in candidates
                  if any(h in name_of(r) for h in PHONE_HINTS)), None)
    if phone is None:
        phone = max(candidates, key=lambda r: r["w"])

    print(f"  RECOMMENDED:  index {phone['index']}  ({phone['w']}x{phone['h']})")
    print("=" * 66)
    print("  Put this in .env:")
    print(f"     CAMERA_SOURCE=index:{phone['index']}")
    print("  then run:  npm run collect")
    print("\n  Not sure it is the right one? Confirm visually:")
    print("     npm run cameras -- --preview")

    if args.preview:
        print("\n  Previewing...")
        for r in working:
            idx = r["index"]
            label = names[idx] if idx < len(names) else f"{r['w']}x{r['h']}"
            if preview(idx, label) == "stop":
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
