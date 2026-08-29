"""
camera_manager.py — Smart Camera Abstraction Layer for Tarjuman
================================================================
Provides a unified SmartCamera interface that automatically detects
the runtime environment and uses the most appropriate camera backend:

  • Raspberry Pi (ARM/Linux)  ->  Picamera2 (Camera Module 3, 12MP)
  • Laptop / Dev Machine      ->  OpenCV VideoCapture (cv2)

The .read() method mirrors OpenCV's exact (ret, frame) tuple contract,
making it a drop-in replacement anywhere cv2.VideoCapture is used.
"""

import os
import platform
import sys
import time
import cv2
import numpy as np

try:                                   # optional: lets .env supply the IP/port
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -----------------------------------------------------------------------------
#  Network camera (DroidCam / IP Webcam)
# -----------------------------------------------------------------------------

DROIDCAM_IP   = os.getenv("DROIDCAM_IP",   "192.168.8.177")
DROIDCAM_PORT = os.getenv("DROIDCAM_PORT", "4747")


def droidcam_url(host: str | None = None) -> str:
    """
    Build the DroidCam MJPEG endpoint.

    DROIDCAM_URL wins if set, so an unusual app or port layout can be handled
    from .env without touching code. DroidCam serves its stream at /video.
    """
    explicit = os.getenv("DROIDCAM_URL")
    if explicit and host is None:
        return explicit
    return f"http://{host or DROIDCAM_IP}:{DROIDCAM_PORT}/video"


# -----------------------------------------------------------------------------
#  USB tunnel (adb) — the reliable alternative to Wi-Fi
# -----------------------------------------------------------------------------
#
# Wi-Fi MJPEG stutters because the stream competes with everything else on the
# network and has no flow control: when bandwidth dips, frames are simply lost.
# For dataset recording that is worse than it sounds — dropped frames distort
# the very timing and trajectory the model learns from.
#
# `adb forward` tunnels the phone's port over the USB cable, so the exact same
# HTTP stream arrives via localhost with no radio involved.

def adb_forward(port: str | int = None, adb_path: str = None) -> tuple[bool, str]:
    """
    Forward the phone's DroidCam port to localhost over USB.

    ANDROID ONLY. adb is part of the Android SDK and cannot talk to an iPhone
    at all — iOS exposes no equivalent debugging bridge. For an iPhone use the
    virtual-webcam route instead (DroidCam / iVCam / Camo Windows client),
    which appears to OpenCV as an ordinary camera index.

    ADB_PATH in .env can point at adb.exe directly, so there is no need to edit
    the system PATH.

    Returns (ok, message). On success the stream is reachable at
    http://127.0.0.1:<port>/video and behaves identically to the Wi-Fi URL.
    """
    import shutil
    import subprocess

    port = str(port or DROIDCAM_PORT)
    adb_path = adb_path or os.getenv("ADB_PATH", "adb")

    # Accept a folder as well as the executable itself — people usually have
    # the platform-tools directory to hand, not the exact exe path.
    if os.path.isdir(adb_path):
        candidate = os.path.join(adb_path, "adb.exe" if os.name == "nt" else "adb")
        if os.path.isfile(candidate):
            adb_path = candidate

    if not os.path.isfile(adb_path) and shutil.which(adb_path) is None:
        return False, (
            f"adb not found ({adb_path}).\n"
            "   Set ADB_PATH in .env to the platform-tools folder, e.g.\n"
            "   ADB_PATH=C:\\Users\\HP\\Desktop\\platform-tools-latest-windows\n"
            "   NOTE: adb is Android-only. iPhone users should pick the\n"
            "   virtual-webcam option instead."
        )

    try:
        devices = subprocess.run(
            [adb_path, "devices"], capture_output=True, text=True, timeout=15
        )
        lines = [l.strip() for l in devices.stdout.splitlines()[1:] if l.strip()]
        attached = [l for l in lines if l.endswith("device")]
        unauthorised = [l for l in lines if l.endswith("unauthorized")]

        if unauthorised:
            return False, ("Phone is connected but UNAUTHORIZED.\n"
                           "   Unlock the phone and accept the 'Allow USB debugging' prompt.")
        if not attached:
            return False, ("No phone detected over USB.\n"
                           "   1. Enable Developer Options (tap Build number 7x)\n"
                           "   2. Enable USB debugging\n"
                           "   3. Reconnect the cable and accept the prompt")

        subprocess.run([adb_path, "forward", f"tcp:{port}", f"tcp:{port}"],
                       capture_output=True, text=True, timeout=15, check=True)
        return True, f"USB tunnel active: localhost:{port} -> phone:{port}"

    except subprocess.TimeoutExpired:
        return False, "adb timed out — try unplugging and reconnecting the cable."
    except subprocess.CalledProcessError as exc:
        return False, f"adb forward failed: {exc.stderr.strip() or exc}"
    except Exception as exc:
        return False, f"adb error: {type(exc).__name__}: {exc}"


def _is_blank(frame, std_threshold: float = 6.0) -> bool:
    """
    True when a frame carries no picture — a flat colour rather than a scene.

    Virtual cameras fail in a peculiar way: instead of erroring, they deliver a
    perfectly valid image that happens to be solid green (a YUV buffer read as
    BGR) or solid black (source not streaming yet). `isOpened()` says yes and
    `read()` says yes, so nothing downstream notices until an entire dataset has
    been recorded from a blank screen.

    A real scene always has SPATIAL variation; a flat colour has almost none.

    The variation must be measured per channel. Solid green is B=0, G=200, R=0 —
    taken as one array its standard deviation is large, because the channels
    differ from each other, even though every pixel is identical. Measuring each
    channel separately asks the right question: does the picture change from
    pixel to pixel?
    """
    try:
        import numpy as _np
        sample = frame[::8, ::8]          # subsample: this runs per attempt
        if sample.ndim == 2:              # greyscale
            return float(_np.std(sample)) < std_threshold
        # Spatial spread within each channel; flat if the busiest one is flat.
        per_channel = [float(_np.std(sample[:, :, c])) for c in range(sample.shape[2])]
        return max(per_channel) < std_threshold
    except Exception:
        return False


def list_local_cameras(max_index: int = 6) -> list[tuple[int, int, int]]:
    """
    Probe local device indices and report which ones deliver a frame.

    Used to tell an external USB webcam apart from the built-in one — they are
    distinguished only by index, and an external camera is never index 0.
    Returns [(index, width, height), ...].
    """
    # Probing indices that do not exist makes OpenCV print a wall of backend
    # errors, which buries the list this function exists to produce.
    try:
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        previous = None

    found = []
    try:
        for idx in range(max_index):
            cap = None
            try:
                # Try each backend: a device may register with only one of them.
                if platform.system() == "Windows":
                    apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
                else:
                    apis = [None]

                for api in apis:
                    cap = (cv2.VideoCapture(idx, api) if api is not None
                           else cv2.VideoCapture(idx))
                    if cap.isOpened():
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            h, w = frame.shape[:2]
                            found.append((idx, w, h))
                            break
                    cap.release()
                    cap = None
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
    finally:
        if previous is not None:
            try:
                cv2.utils.logging.setLogLevel(previous)
            except Exception:
                pass

    return found


# -----------------------------------------------------------------------------
#  Environment Detection
# -----------------------------------------------------------------------------

def _is_raspberry_pi() -> bool:
    """
    Returns True if the current machine is a Raspberry Pi.

    Detection strategy (layered for reliability):
      1. OS must be Linux (rules out Windows / macOS immediately).
      2. CPU architecture must be ARM (armv7l, aarch64, etc.).
      3. Belt-and-suspenders: read /proc/cpuinfo and confirm
         the 'Raspberry Pi' hardware model string is present.
    """
    # Fast path: non-Linux OSes are never a Pi
    if platform.system() != "Linux":
        return False

    machine = platform.machine().lower()
    is_arm = machine.startswith("arm") or machine.startswith("aarch")
    if not is_arm:
        return False

    # Confirm via /proc/cpuinfo (always present on real Pi hardware)
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        return "raspberry pi" in cpuinfo.lower()
    except OSError:
        # Cannot read the file; trust the ARM architecture check alone
        return True


# -----------------------------------------------------------------------------
#  SmartCamera Class
# -----------------------------------------------------------------------------

class SmartCamera:
    """
    A unified camera abstraction that works identically on both:
      • Laptop / development machine  (uses OpenCV VideoCapture)
      • Raspberry Pi / production      (uses Picamera2 for Camera Module 3)

    Usage
    -----
        cam = SmartCamera()
        cam.start()

        while True:
            ret, frame = cam.read()   # identical to cv2.VideoCapture.read()
            if not ret:
                break
            # ... process frame ...

        cam.release()

    Configuration
    -------------
    WIDTH, HEIGHT : Target resolution (default 640x480).
    FPS           : Target framerate for Picamera2 (default 30).
    LENS_POSITION : Fixed focus distance for Picamera2 manual-focus mode.
                    Range 0.0 (infinity) to 10.0 (macro). 1.0 ≈ 1 m distance,
                    which is suitable for a signer seated in front of the Pi.
    """

    # -- Shared configuration constants --------------------------------------
    # Overridable from .env, because the mode that delivers the most frames is
    # a property of YOUR camera and can only be found by measuring it
    # (`npm run bench` sweeps them). MediaPipe resizes to a fixed tensor
    # internally, so a lower resolution costs it nothing - which makes trading
    # pixels for frames a genuinely free win when the sensor allows it.
    WIDTH         = int(os.getenv("CAMERA_WIDTH", "640"))
    HEIGHT        = int(os.getenv("CAMERA_HEIGHT", "480"))
    # Ask the sensor for more than we expect to consume. Delivery rate and
    # processing rate are different things: MediaPipe costs ~20 ms per frame
    # whatever the resolution, so the camera should never be the limit. A
    # sensor running at 60 also halves the age of the newest frame, which is
    # what stops the hand tracker losing the hand between detections.
    FPS           = int(os.getenv("CAMERA_FPS", "60"))
    LENS_POSITION = 1.0   # Manual-focus position (≈ 1 m; adjust as needed)

    # -- Low-light manual exposure / gain --------------------------------------
    # `_force_frame_rate()` below already SEARCHES for a fast-enough exposure
    # automatically. These two let you PIN the result instead of searching —
    # useful once a physical LED light is added and you know exactly what
    # works, or if you want the same settings every run without re-measuring.
    #
    #   CAMERA_EXPOSURE : log2(seconds) on Windows/DirectShow (-6 = 1/64 s;
    #                     more negative = shorter = faster but darker) and
    #                     backend-specific raw units on V4L2/Linux. Leave
    #                     unset (None) to keep the automatic search below.
    #   CAMERA_GAIN     : sensor gain/ISO to brighten the image without
    #                     lengthening exposure. Raise this first when a
    #                     fixed exposure looks too dark — it costs noise,
    #                     not frame rate. 0-255 on most UVC webcams.
    #
    # Both are read once at import time so `npm run bench` / `.env` changes
    # take effect on the next run without touching this file.
    _manual_exposure_raw = os.getenv("CAMERA_EXPOSURE")
    _manual_gain_raw     = os.getenv("CAMERA_GAIN")
    MANUAL_EXPOSURE = float(_manual_exposure_raw) if _manual_exposure_raw else None
    MANUAL_GAIN     = float(_manual_gain_raw) if _manual_gain_raw else None

    # Mean luma (0-255) below which a frame is considered too dark to use.
    # 45 was far too permissive - 45/255 is about 18% grey, a picture you can
    # barely see. Brightness is a HARD constraint: a fast black frame is worth
    # nothing to a landmark detector.
    BRIGHTNESS_MIN = float(os.getenv("CAMERA_BRIGHTNESS_MIN", "85"))

    # Set CAMERA_FORCE_FPS=0 to switch the whole exposure-forcing search off
    # and simply leave the camera on auto-exposure.
    FORCE_FPS = os.getenv("CAMERA_FORCE_FPS", "1").strip().lower() not in (
        "0", "false", "no", "off")

    # -- Backend identifiers --------------------------------------------------
    BACKEND_PICAMERA2 = "picamera2"
    BACKEND_OPENCV    = "opencv"
    BACKEND_NETWORK   = "network"      # DroidCam / IP Webcam / any MJPEG stream
    BACKEND_USB_DSHOW = "usb_dshow"    # External USB camera using DSHOW

    def __init__(self, device_index: int = 0, source=None):
        """
        Parameters
        ----------
        device_index : int
            Index passed to cv2.VideoCapture for a locally attached webcam.
        source : str | int | None
            • None            -> auto-detect (Picamera2 on a Pi, else webcam)
            • "laptop"        -> force the local webcam
            • "droidcam"      -> use the DroidCam URL from the environment
            • int (e.g. 1, 2) -> use USB camera with DSHOW backend at this index
            • an http(s)/rtsp URL -> use that stream directly

        Why a network option
        --------------------
        A phone camera is usually far sharper than a laptop webcam, and sharper
        frames mean cleaner MediaPipe landmarks — which is the raw material the
        whole model is built from. Recording the dataset over DroidCam is a
        genuine quality win.
        """
        self._device_index = device_index
        self._backend      = None   # Will be set during start()
        self._cam          = None   # Backend camera object
        self._last_frame   = None   # Cached latest frame (numpy ndarray)
        self._url          = None   # Stream URL when in network mode
        # True when the CSI camera was ASKED FOR by name rather than merely
        # auto-detected. It decides whether a failure may fall back to a USB
        # webcam: auto-detection may, an explicit choice may not.
        self._explicit_picamera = False

        # If source is an integer, it means the user selected DroidCam USB
        if isinstance(source, int):
            self._device_index = source
            self._requested_backend = self.BACKEND_USB_DSHOW
        else:
            source = source if source is not None else os.getenv("CAMERA_SOURCE", "auto")
            source = str(source).strip()

            if source.lower() in ("droidcam-usb", "usb-tunnel", "adb"):
                # Same MJPEG stream as Wi-Fi, but carried over the USB cable by
                # `adb forward`. Wi-Fi MJPEG stutters because it has no flow
                # control — when bandwidth dips, frames are simply dropped, and
                # dropped frames distort the exact timing the model learns from.
                ok, msg = adb_forward()
                print(f"[SmartCamera] {msg}")
                if not ok:
                    print("[SmartCamera] USB tunnel unavailable — see the steps above.")
                self._url = droidcam_url(host="127.0.0.1")
                self._requested_backend = self.BACKEND_NETWORK
            elif source.lower().startswith("index:"):
                # DroidCam Client's virtual webcam, e.g. CAMERA_SOURCE=index:1
                self._device_index = int(source.split(":", 1)[1])
                self._requested_backend = self.BACKEND_USB_DSHOW
            elif source.lower() in ("droidcam", "phone", "network"):
                self._url = droidcam_url()
                self._requested_backend = self.BACKEND_NETWORK
            elif "://" in source:
                self._url = source
                self._requested_backend = self.BACKEND_NETWORK
            elif source.lower() in ("picamera", "picamera2", "pi", "picam",
                                    "csi", "pi-camera", "rpi"):
                # The CSI-attached Camera Module, requested by name. This is
                # how the Pi records its dataset; see _start_picamera2.
                self._requested_backend = self.BACKEND_PICAMERA2
                self._explicit_picamera = True
            elif source.lower() == "laptop":
                self._requested_backend = self.BACKEND_OPENCV
            elif source.lower() == "usb_dshow":
                self._requested_backend = self.BACKEND_USB_DSHOW
            else:                                    # "auto"
                self._requested_backend = (
                    self.BACKEND_PICAMERA2 if _is_raspberry_pi() else self.BACKEND_OPENCV
                )

        if self._requested_backend == self.BACKEND_NETWORK:
            print(f"[SmartCamera] Source              : network stream")
            print(f"[SmartCamera] URL                 : {self._url}")
        else:
            env_label = ("Raspberry Pi (Production)" if _is_raspberry_pi()
                         else "Laptop/Dev (Development)")
            print(f"[SmartCamera] Environment detected : {env_label}")
            print(f"[SmartCamera] Requested backend    : {self._requested_backend}")

    # -- Public API ----------------------------------------------------------

    def start(self) -> "SmartCamera":
        """
        Initialize and start the camera backend.

        Network  : opens the MJPEG/RTSP stream (no fallback — a silent switch
                   to the laptop webcam would poison the dataset with frames
                   from the wrong camera).
        Pi       : tries Picamera2 first; falls back to OpenCV on error.
        Laptop   : uses OpenCV directly.

        Returns self to allow chaining:  cam = SmartCamera().start()
        """
        if self._requested_backend == self.BACKEND_NETWORK:
            self._start_network()
        elif self._requested_backend == self.BACKEND_PICAMERA2:
            self._start_picamera2()
        elif self._requested_backend == self.BACKEND_USB_DSHOW:
            self._start_usb_dshow()
        else:
            self._start_opencv()

        return self


    # -- Stream format negotiation --------------------------------------------

    def _fourcc_tag(self, cap) -> str:
        f = int(cap.get(cv2.CAP_PROP_FOURCC))
        if not f:
            return "?"
        return "".join(chr((f >> (8 * i)) & 0xFF) for i in range(4)).strip()

    def _measure_rate(self, cap, seconds: float = 0.7, settle: int = 8) -> float:
        """
        Frames actually delivered per second, with nothing else in the loop.

        The settle frames are not optional. A camera re-negotiating its media
        type delivers erratically for the first handful of frames, and timing
        those gave 20 fps for a sensor genuinely running at 30 - a wrong number
        that could make this function pick the worse of two formats.
        """
        for _ in range(settle):
            cap.read()
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < seconds:
            ok, _ = cap.read()
            if ok:
                n += 1
        return n / (time.perf_counter() - t0)

    def _negotiate_format(self, cap) -> dict:
        """
        Get MJPEG at the requested rate to actually STICK, and prove it did.

        Setting properties on a capture device is a negotiation, not a command,
        and the order matters in a way that is easy to get wrong: on DirectShow,
        setting the RESOLUTION renegotiates the media type and quietly reverts
        the pixel format to the driver's default. Requesting MJPG first and the
        size second therefore ends up back in YUY2 - which is exactly what
        happened here, while the log cheerfully reported "CAP_DSHOW + MJPG".

        So several orders are tried and each is VERIFIED by reading the format
        back and timing real frames. Whichever genuinely delivers the most
        frames per second wins. Believing the request instead of checking the
        result is what hid a 20 fps ceiling behind a 60 fps log line.
        """
        MJPG = cv2.VideoWriter_fourcc(*"MJPG")

        def apply(steps):
            for step in steps:
                if step == "size":
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
                elif step == "mjpg":
                    cap.set(cv2.CAP_PROP_FOURCC, MJPG)
                elif step == "fps":
                    cap.set(cv2.CAP_PROP_FPS, self.FPS)

        plans = [
            ("size -> MJPG -> fps",        ["size", "mjpg", "fps"]),
            ("MJPG -> size -> MJPG -> fps", ["mjpg", "size", "mjpg", "fps"]),
            ("MJPG -> size -> fps",        ["mjpg", "size", "fps"]),
            ("size -> fps (no MJPG)",      ["size", "fps"]),
        ]

        best = None
        for label, steps in plans:
            apply(steps)
            tag = self._fourcc_tag(cap)
            rate = self._measure_rate(cap)
            cand = {"plan": label, "steps": steps, "fourcc": tag, "rate": rate}
            if best is None or rate > best["rate"] + 1.0:
                best = cand
            # Good enough to stop early: MJPEG and close to what we asked for.
            if tag.upper().startswith("MJP") and rate >= self.FPS * 0.75:
                break

        if best["steps"] != plans[-1][1]:
            apply(best["steps"])
            for _ in range(3):
                cap.read()

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        best["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        best["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        best["fourcc"] = self._fourcc_tag(cap)
        print(f"[SmartCamera] format negotiated: {best['fourcc']} "
              f"{best['width']}x{best['height']} ~{best['rate']:.0f} fps "
              f"(via {best['plan']})")
        if not best["fourcc"].upper().startswith("MJP"):
            print("[SmartCamera] [!] driver refused MJPEG; raw formats are "
                  "bandwidth-capped, so the rate will stay low.")

        # Format is settled; now make sure auto-exposure is not halving the rate.
        self._force_frame_rate(cap, min(self.FPS, 30))
        return best


    def _frame_brightness(self, cap) -> float:
        """Mean luma of a fresh frame, 0-255. Used to refuse a too-dark trade."""
        ok, frame = cap.read()
        if not ok or frame is None:
            return 0.0
        return float(np.mean(frame))

    def _enable_auto_exposure(self, cap) -> None:
        """
        Hand exposure control back to the camera.

        This matters far more than it looks. UVC webcams keep manual exposure
        settings in the DEVICE, not the process - so a short exposure written
        by a previous run is still in force the next time the camera is
        opened, and every run after that, until something sets it back or the
        camera is physically unplugged.

        That is what turned the picture black in a brightly lit room: an
        earlier run had switched auto-exposure off and pinned a very short
        exposure. The next run then measured a healthy 30 fps, concluded
        there was nothing to fix, and returned early - leaving the dark
        setting exactly where it was. Starting from a known state removes a
        whole class of "it was fine yesterday" faults.

        Auto values are backend-specific: 0.75 on DirectShow, 3 on V4L2.
        """
        for value in (0.75, 3):
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
            except Exception:
                pass
        for _ in range(6):          # let the AE loop converge
            cap.read()

    def _disable_auto_exposure(self, cap) -> bool:
        """
        Turn off auto-exposure, regardless of which backend is behind `cap`.

        DirectShow (Windows) and V4L2 (Linux/Raspberry Pi) do not agree on
        what "manual" even means numerically: DirectShow wants 0.25, V4L2
        wants 1 (3 is its "aperture priority" auto mode), and some UVC
        drivers only accept 0. Setting an unsupported value is usually
        just ignored rather than raising, so every candidate is tried
        behind its own try/except - one backend's quirk must not be able
        to abort the others or crash the caller.
        """
        applied_any = False
        for value in (0.25, 1, 0):
            try:
                if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value):
                    applied_any = True
            except Exception:
                pass
        return applied_any

    def _apply_fixed_exposure_gain(self, cap, exposure=None, gain=None) -> dict:
        """
        Pin exposure and gain to fixed values instead of searching for them.

        Each property is set behind its own try/except: `CAP_PROP_EXPOSURE`,
        `CAP_PROP_GAIN` and `CAP_PROP_ISO_SPEED` are all backend-dependent and
        a webcam driver that rejects one should not take down the others.
        Gain is tried before ISO_SPEED because most UVC webcams on Windows
        expose GAIN; ISO_SPEED is the fallback some Linux/V4L2 drivers use
        instead. Returns whatever actually got applied, so the caller can log
        the truth instead of the intent.
        """
        applied: dict = {}
        self._disable_auto_exposure(cap)

        if exposure is not None:
            try:
                cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
                applied["exposure"] = float(exposure)
            except Exception as exc:
                print(f"[SmartCamera] [!] could not set CAP_PROP_EXPOSURE: {exc}")

        if gain is not None:
            gain_set = False
            try:
                if cap.set(cv2.CAP_PROP_GAIN, float(gain)):
                    applied["gain"] = float(gain)
                    gain_set = True
            except Exception:
                pass
            if not gain_set:
                try:
                    if cap.set(cv2.CAP_PROP_ISO_SPEED, float(gain)):
                        applied["iso_speed"] = float(gain)
                except Exception as exc:
                    print(f"[SmartCamera] [!] could not set gain/ISO "
                          f"(GAIN and ISO_SPEED both unsupported): {exc}")

        return applied

    def _brighten_without_exposure(self, cap, bright: float) -> float:
        """
        Brighten the image using every control EXCEPT exposure time.

        Exposure is the one knob that costs frame rate, because a frame can
        never be produced faster than it is exposed - 1/32 s is a hard 32 fps
        ceiling no matter what else is configured. Everything else here is
        free in time terms.

        Tried in order of how little damage they do:
          GAIN       - analogue amplification. Costs noise. Best option, but
                       plenty of laptop webcams do not expose it at all.
          BRIGHTNESS - a digital offset. Lifts shadows, flattens contrast.
          GAMMA      - non-linear lift. Brightens midtones, keeps highlights.

        Each is applied cumulatively and re-measured, because whether any of
        them does anything at all is entirely up to the driver, and a set()
        that returns True is not a promise that the picture changed.
        """
        ladders = (
            (cv2.CAP_PROP_GAIN, "gain", (64, 128, 192, 255)),
            (cv2.CAP_PROP_BRIGHTNESS, "brightness", (128, 160, 192, 224)),
            (cv2.CAP_PROP_GAMMA, "gamma", (100, 140, 180)),
        )
        for prop, label, steps in ladders:
            if bright >= self.BRIGHTNESS_MIN:
                return bright
            for value in steps:
                try:
                    if not cap.set(prop, float(value)):
                        break
                except Exception:
                    break
                new_bright = self._frame_brightness(cap)
                if new_bright > bright + 1.0:
                    bright = new_bright
                    if bright >= self.BRIGHTNESS_MIN:
                        print(f"[SmartCamera] {label}={value} lifted the image "
                              f"to {bright:.0f}/255 without touching exposure")
                        return bright
        return bright

    def _force_frame_rate(self, cap, target_rate: float) -> dict:
        """
        Stop the camera halving its own frame rate in dim light.

        UVC webcams run auto-exposure by default. In a dim room the driver
        lengthens the exposure time to brighten the picture, and since a frame
        cannot be produced faster than it is exposed, the sensor silently drops
        from 30 fps to 15 - or lower. Nothing in software reports this; the
        frames simply arrive half as often.

        That is a bad trade for sign language. A longer exposure also MOTION
        BLURS a moving hand, so the frames you do get are worse for landmark
        detection, not better. Fewer, blurrier frames is the opposite of what
        the tracker needs.

        So auto-exposure is switched off and progressively shorter exposures are
        tried, each one MEASURED for both frame rate and brightness. The
        shortest exposure that hits the target rate while keeping the image
        usable wins; if none does, the original settings are restored rather
        than leaving you with a fast, black picture.
        """
        # Always start from a known state. See _enable_auto_exposure: the
        # camera REMEMBERS a manual exposure across runs, so without this the
        # function can inherit a dark frame from a previous session, measure a
        # fine frame rate, and "correctly" decide to change nothing.
        if not self.FORCE_FPS:
            self._enable_auto_exposure(cap)
            return {"changed": False, "rate": self._measure_rate(cap, 0.3),
                    "brightness": self._frame_brightness(cap)}

        self._enable_auto_exposure(cap)
        before_rate = self._measure_rate(cap, seconds=0.5)
        before_bright = self._frame_brightness(cap)

        # Manual override: skip the search entirely and pin the values you
        # already know work (e.g. once a physical LED light is in place).
        # Set via CAMERA_EXPOSURE / CAMERA_GAIN in .env.
        if self.MANUAL_EXPOSURE is not None:
            try:
                applied = self._apply_fixed_exposure_gain(
                    cap, exposure=self.MANUAL_EXPOSURE, gain=self.MANUAL_GAIN)
                rate = self._measure_rate(cap, seconds=0.45)
                bright = self._frame_brightness(cap)
                print(f"[SmartCamera] manual exposure/gain applied {applied} "
                      f"-> {rate:.0f} fps, brightness {bright:.0f}/255 "
                      f"[was {before_rate:.0f} fps]")
                return {"changed": True, "rate": rate, "brightness": bright,
                        **applied}
            except Exception as exc:
                print(f"[SmartCamera] [!] manual exposure/gain failed ({exc}); "
                      f"falling back to automatic search.")

        if before_rate >= target_rate * 0.85:
            return {"changed": False, "rate": before_rate,
                    "brightness": before_bright}

        prev_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        prev_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        try:
            prev_gain = cap.get(cv2.CAP_PROP_GAIN)
        except Exception:
            prev_gain = None

        # 0.25 is DirectShow's "manual"; 1/0 are what other backends expect.
        for manual in (0.25, 1, 0):
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
            except Exception:
                pass
            # Exposure is log2 seconds on Windows: -5 is 1/32 s, -6 is 1/64 s.
            # Anything at or below -6 comfortably clears 30 fps.
            for exposure in (-5, -6, -7):
                try:
                    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                except Exception:
                    pass
                # A configured gain compensates for the shorter exposure
                # without costing any frame rate, unlike lengthening exposure.
                if self.MANUAL_GAIN is not None:
                    try:
                        cap.set(cv2.CAP_PROP_GAIN, self.MANUAL_GAIN)
                    except Exception:
                        pass
                rate = self._measure_rate(cap, seconds=0.45)
                bright = self._frame_brightness(cap)

                # Too dark? Raise GAIN, not exposure. Gain brightens the image
                # without lengthening the shutter, so it costs neither frame
                # rate nor motion sharpness - it costs noise, which is the
                # cheaper price of the two for landmark detection.
                if bright < self.BRIGHTNESS_MIN and self.MANUAL_GAIN is None:
                    bright = self._brighten_without_exposure(cap, bright)
                    if bright >= self.BRIGHTNESS_MIN:
                        rate = self._measure_rate(cap, seconds=0.3)

                if rate >= target_rate * 0.85 and bright >= self.BRIGHTNESS_MIN:
                    print(f"[SmartCamera] exposure fixed at {exposure} "
                          f"(auto off) -> {rate:.0f} fps, brightness "
                          f"{bright:.0f}/255  [was {before_rate:.0f} fps]")
                    return {"changed": True, "rate": rate,
                            "brightness": bright, "exposure": exposure}

        # Nothing acceptable. Hand control back to the camera's own exposure
        # loop rather than restoring prev_auto/prev_exp: those may themselves
        # be a stale manual setting left behind by an earlier run, which is
        # the exact bug that produced a black picture in a well-lit room. A
        # usable image at a lower frame rate beats a fast unusable one.
        try:
            if prev_gain is not None:
                cap.set(cv2.CAP_PROP_GAIN, prev_gain)
        except Exception:
            pass
        self._enable_auto_exposure(cap)
        rate = self._measure_rate(cap, seconds=0.3)
        print(f"[SmartCamera] [!] Sensor is running at {before_rate:.0f} fps, "
              f"not {target_rate:.0f}.")
        print("[SmartCamera]     Shorter exposures were too dark to use, so the")
        print("[SmartCamera]     camera was left as it was. MORE LIGHT on the")
        print("[SmartCamera]     signer is the real fix - it raises the frame")
        print("[SmartCamera]     rate AND removes motion blur from moving hands.")
        print("[SmartCamera]     Once a light is in place, you can also pin exact")
        print("[SmartCamera]     values via CAMERA_EXPOSURE / CAMERA_GAIN in .env")
        print("[SmartCamera]     instead of relying on this search.")
        return {"changed": False, "rate": rate, "brightness": before_bright}

    # -- Background grabber ---------------------------------------------------
    # cv2's read() BLOCKS until the next frame arrives. In a serial loop you
    # therefore pay the wait AND the inference back to back, so the two never
    # overlap - measured as the single largest avoidable cost in the pipeline.
    #
    # A grabber thread keeps only the NEWEST frame and throws away anything
    # older. Dropping stale frames is the point: a sign is judged on where the
    # hand is now, and showing the recogniser a frame from 80 ms ago is worse
    # than showing it nothing.

    def start_grabber(self) -> "SmartCamera":
        """Begin decoding frames in the background. Safe to call twice."""
        import threading
        if getattr(self, "_grab_thread", None) is not None:
            return self
        self._grab_lock = threading.Lock()
        self._grab_frame = None
        self._grab_seq = 0
        self._served_seq = None
        self._grab_stop = threading.Event()

        def loop():
            while not self._grab_stop.is_set():
                ok, frame = self._read_direct()
                if ok and frame is not None:
                    with self._grab_lock:
                        self._grab_frame = frame
                        self._grab_seq += 1
                else:
                    time.sleep(0.005)

        self._grab_thread = threading.Thread(target=loop, daemon=True,
                                             name="tarjuman-camera")
        self._grab_thread.start()
        return self

    def stop_grabber(self) -> None:
        if getattr(self, "_grab_thread", None) is None:
            return
        self._grab_stop.set()
        self._grab_thread.join(timeout=1.0)
        self._grab_thread = None

    def read_latest(self, since: int = None, wait: float = 0.5):
        """
        Newest decoded frame, skipping any that piled up while you were busy.

        Returns (ok, frame, seq). `seq` counts decoded frames, so a caller can
        tell a genuinely new frame from a repeat.

        `since` is what makes this correct rather than merely fast. Without it
        the grabber happily returns the SAME frame again whenever inference
        outruns the sensor, and a duplicate frame is not harmless here: it has
        zero displacement from its predecessor, so it drags the measured speed
        down, pollutes the motion features the model is trained on, and inflates
        the frame-rate estimate with frames that were never captured. Waiting
        for the counter to move keeps every frame handed out a real one.
        """
        if getattr(self, "_grab_thread", None) is None:
            ok, frame = self._read_direct()
            return ok, frame, -1
        deadline = time.time() + wait
        while True:
            with self._grab_lock:
                if self._grab_frame is not None and self._grab_seq != since:
                    return True, self._grab_frame, self._grab_seq
            if time.time() > deadline:
                return False, None, -1
            time.sleep(0.001)

    def _read_direct(self):
        """The original blocking read, used by the grabber thread."""
        if self._cam is None:
            return False, None
        if self._backend == self.BACKEND_PICAMERA2:
            return self._read_picamera2()
        elif self._backend == self.BACKEND_NETWORK:
            return self._read_network()
        return self._read_opencv()

    def read(self):
        """
        Capture and return the next frame.

        Returns
        -------
        (success : bool, frame : np.ndarray or None)
            Mirrors the exact return signature of cv2.VideoCapture.read().
            frame is a BGR numpy array of shape (HEIGHT, WIDTH, 3).
            Returns (False, None) if the camera is not started or on error.
        """
        # When a grabber is running, serve the newest frame from it so callers
        # that never migrated to read_latest() still stop blocking.
        if getattr(self, "_grab_thread", None) is not None:
            # Remember what we last handed out so callers that never migrated to
            # read_latest() still get a NEW frame on every call, exactly as a
            # blocking read() always did.
            last = getattr(self, "_served_seq", None)
            ok, frame, seq = self.read_latest(since=last)
            if ok:
                self._served_seq = seq
            return ok, frame
        return self._read_direct()

    def release(self) -> None:
        """
        Stop the camera and release all held resources.
        Safe to call even if start() was never called.
        """
        if self._cam is None:
            return

        try:
            if self._backend == self.BACKEND_PICAMERA2:
                self._cam.stop()
                print("[SmartCamera] Picamera2 stopped and released.")
            elif self._backend == self.BACKEND_NETWORK:
                self._cam.release()
                print("[SmartCamera] Network stream closed.")
            else:
                self._cam.release()
                print("[SmartCamera] OpenCV VideoCapture released.")
        except Exception as exc:
            print(f"[SmartCamera] Warning during release: {exc}")
        finally:
            self._cam        = None
            self._last_frame = None
            self._backend    = None

    # -- Picamera2 Backend ---------------------------------------------------

    def _pick_sensor_mode(self, picam):
        """
        Pick the sensor mode with the WIDEST field of view, not the fastest.

        This exists because of a trap that costs a whole dataset. Camera
        Module 3 advertises, among others:

            1536x864   120 fps   crop (768,432)/3072x1728   <- CROPPED
            2304x1296   56 fps   crop (0,0)/4608x2592       <- full sensor

        The fast mode is fast precisely BECAUSE it reads a smaller patch of
        the sensor - roughly two thirds of the width. That is a narrower lens
        in everything but name. Ask Picamera2 for a small main size and it
        will happily pick that cropped mode, because it is only trying to
        satisfy the pixel count you asked for.

        For sign language that is the wrong trade in both directions. Hands
        travel far off-centre, so horizontal field of view is the thing you
        cannot afford to lose - while the frame rate is never the limit
        anyway, because MediaPipe on a Pi CPU is far slower than 56 fps.

        So: prefer the largest sensor crop area, and among equally wide modes
        take the cheapest one that still clears the target rate.

        CAMERA_SENSOR_MODE=WxH in .env overrides all of this.
        """
        try:
            modes = list(getattr(picam, "sensor_modes", []) or [])
        except Exception:
            return None
        if not modes:
            return None

        forced = os.getenv("CAMERA_SENSOR_MODE", "").lower().strip()
        if forced and "x" in forced:
            try:
                fw, fh = (int(v) for v in forced.split("x", 1))
                for m in modes:
                    if tuple(m.get("size", ())) == (fw, fh):
                        print(f"[SmartCamera] sensor mode {fw}x{fh} "
                              f"(forced by CAMERA_SENSOR_MODE)")
                        return {"output_size": (fw, fh),
                                "bit_depth": m.get("bit_depth", 10)}
                print(f"[SmartCamera] [!] CAMERA_SENSOR_MODE={forced} is not a "
                      f"mode this sensor offers; choosing automatically.")
            except ValueError:
                pass

        def crop_area(m):
            crop = m.get("crop_limits") or ()
            return (crop[2] * crop[3]) if len(crop) == 4 else 0

        widest = max(crop_area(m) for m in modes)
        if widest <= 0:
            return None

        # Within 2% of the widest crop counts as "same field of view".
        full_fov = [m for m in modes if crop_area(m) >= widest * 0.98]
        target = float(min(self.FPS, 30))
        fast_enough = [m for m in full_fov if float(m.get("fps", 0)) >= target]
        pool = fast_enough or full_fov

        # Cheapest of the acceptable ones: fewer pixels off the sensor is less
        # bandwidth and less ISP work, and the ISP scales to our output anyway.
        chosen = min(pool, key=lambda m: m.get("size", (1 << 30, 1 << 30))[0])
        size = tuple(chosen.get("size", ()))
        if len(size) != 2:
            return None

        narrowest = min(crop_area(m) for m in modes)
        print(f"[SmartCamera] sensor mode {size[0]}x{size[1]} @ "
              f"{float(chosen.get('fps', 0)):.0f} fps — full sensor "
              f"(widest field of view)")
        if narrowest < widest * 0.98:
            lost = 1.0 - (narrowest / widest) ** 0.5
            print(f"[SmartCamera]   (a faster cropped mode exists but would cut "
                  f"~{lost * 100:.0f}% of the frame width — hands would leave "
                  f"the shot)")
        return {"output_size": size, "bit_depth": chosen.get("bit_depth", 10)}

    def _mount_transform(self, picam):
        """
        Undo the sensor's physical mounting rotation, if it reports one.

        libcamera exposes how the module is mounted as a Rotation property.
        Module 3 on a Pi 5 commonly reports 180. An inverted frame does not
        raise anything - it just makes MediaPipe much worse at finding hands,
        which is exactly the sort of failure that gets blamed on the model
        weeks later.

        CAMERA_ROTATE_180=0 disables this; =1 forces it on.
        """
        override = os.getenv("CAMERA_ROTATE_180", "").strip()
        try:
            from libcamera import Transform  # type: ignore[import]
        except Exception:
            if override == "1":
                print("[SmartCamera] [!] CAMERA_ROTATE_180=1 but libcamera's "
                      "Transform is unavailable; frame left as-is.")
            return None

        if override in ("0", "false", "no"):
            return None
        if override in ("1", "true", "yes"):
            print("[SmartCamera] rotating frame 180° (CAMERA_ROTATE_180=1)")
            return Transform(hflip=1, vflip=1)

        try:
            rotation = int(picam.camera_properties.get("Rotation", 0))
        except Exception:
            return None

        if rotation == 180:
            print("[SmartCamera] sensor reports Rotation=180 — flipping frame "
                  "upright (set CAMERA_ROTATE_180=0 if the image ends up "
                  "upside down)")
            return Transform(hflip=1, vflip=1)
        return None

    def _start_picamera2(self) -> None:
        """
        Start the CSI-attached Raspberry Pi camera (Module 3 and friends).

        Three things here are easy to get wrong and expensive to discover late:

        PIXEL FORMAT. Picamera2 inherits its format names from libcamera, and
        they describe memory layout rather than the numpy channel order you
        end up with - so they read BACKWARDS. "RGB888" is the one that hands
        OpenCV a (B, G, R) array; "BGR888" gives you (R, G, B). This file
        previously asked for BGR888 with a comment saying "native BGR, no
        cvtColor needed", which was exactly inverted: every frame would have
        had red and blue swapped. That does not throw, it just quietly makes
        skin tones blue - degrading MediaPipe's hand detection AND making Pi
        recordings inconsistent with the laptop's, which is far worse for a
        shared dataset than an outright crash.

        LATENCY. queue=False stops the camera handing back a frame it had
        already prepared before the request. That queued frame is by
        definition stale, and a sign is judged on where the hand is NOW.

        AUTOFOCUS. AfMode/LensPosition exist only on autofocus modules
        (Camera Module 3). On Module 2, the HQ camera or the Global Shutter
        camera they raise, so they are applied separately after start and
        allowed to fail - a fixed-focus camera simply does not need them.
        """
        try:
            # Deferred import: this module is imported on the Windows laptop
            # too, where picamera2 does not exist and must not be required.
            from picamera2 import Picamera2  # type: ignore[import]

            picam = Picamera2()

            config_kwargs = dict(
                main={
                    "size": (self.WIDTH, self.HEIGHT),
                    # See the note above - RGB888 is what OpenCV wants.
                    "format": "RGB888",
                },
                controls={"FrameRate": self.FPS},
                buffer_count=4,
            )

            # Choose the sensor mode explicitly - see _pick_sensor_mode for why
            # letting it default silently narrows the field of view.
            sensor = self._pick_sensor_mode(picam)
            if sensor:
                config_kwargs["sensor"] = sensor
                # Having just chosen the widest sensor crop, do not throw the
                # width away again at the output stage. The IMX708 is natively
                # 16:9, so asking for a 4:3 frame makes the ISP crop the SIDES -
                # precisely where the hands are. Worth a warning, not a silent
                # override, because the rest of the pipeline was built at 4:3.
                sw, sh = sensor["output_size"]
                want = self.WIDTH / max(1, self.HEIGHT)
                native = sw / max(1, sh)
                if abs(want - native) > 0.08:
                    sug_h = int(round(self.WIDTH / native / 2) * 2)
                    print(f"[SmartCamera] [!] output {self.WIDTH}x{self.HEIGHT} "
                          f"({want:.2f}) does not match the sensor's "
                          f"{native:.2f} — the sides of the frame get cropped.")
                    print(f"[SmartCamera]     For the full width, set "
                          f"CAMERA_HEIGHT={sug_h} in .env "
                          f"({self.WIDTH}x{sug_h}).")

            # The camera reports how it is physically mounted. Module 3 on a Pi 5
            # commonly reports Rotation 180, and an upside-down frame is quietly
            # destructive here: MediaPipe is trained on upright images, so hand
            # detection degrades badly while still "working" enough to record a
            # whole dataset before anyone notices.
            transform = self._mount_transform(picam)
            if transform is not None:
                config_kwargs["transform"] = transform

            # queue=False is not supported by every Picamera2 version, and it
            # is a latency optimisation rather than a requirement, so losing
            # it must not lose the camera.
            try:
                video_config = picam.create_video_configuration(
                    queue=False, **config_kwargs)
            except TypeError:
                video_config = picam.create_video_configuration(**config_kwargs)

            picam.configure(video_config)
            picam.start()

            # Manual focus, best-effort. Continuous autofocus is actively
            # harmful here: the lens hunts during fast hand movement, which
            # both blurs the frame and costs frame rate.
            focus_note = "fixed-focus module (no AF controls)"
            try:
                picam.set_controls({"AfMode": 0,
                                    "LensPosition": self.LENS_POSITION})
                focus_note = f"LensPosition={self.LENS_POSITION} (manual focus)"
            except Exception:
                pass

            self._cam     = picam
            self._backend = self.BACKEND_PICAMERA2

            # Report what the camera actually configured, not what was asked
            # for - the same rule the USB path learned the hard way.
            try:
                actual = picam.camera_configuration()["main"]
                got_w, got_h = actual["size"]
                got_fmt = actual["format"]
            except Exception:
                got_w, got_h, got_fmt = self.WIDTH, self.HEIGHT, "RGB888"

            # "@ N fps" here is the REQUESTED rate - the sensor may clamp it to
            # what the chosen mode supports. Labelled as such rather than
            # stated as fact; `npm run bench` measures what really arrives.
            print(f"[SmartCamera] Picamera2 started — {got_w}x{got_h} "
                  f"{got_fmt}, {self.FPS} fps requested, {focus_note}")
            if (got_w, got_h) != (self.WIDTH, self.HEIGHT):
                print(f"[SmartCamera] [!] requested {self.WIDTH}x{self.HEIGHT}; "
                      f"the sensor mode nearest to it was chosen instead.")

        except ImportError:
            print("[SmartCamera] picamera2 is not installed.")
            if self._explicit_picamera:
                # Explicitly asked for the Pi camera: do NOT quietly record
                # from some other lens. Mixing cameras in one dataset shifts
                # every landmark and is very hard to notice afterwards.
                print("[SmartCamera] Install it on the Pi with:")
                print("[SmartCamera]     sudo apt install -y python3-picamera2")
                print("[SmartCamera] (apt, not pip - it is built against the")
                print("[SmartCamera]  system libcamera stack.)")
                print("[SmartCamera] If you are on the laptop, pick option 1-4.")
                self._cam = None
                self._backend = None
            else:
                print("[SmartCamera] Falling back to OpenCV VideoCapture.")
                self._start_opencv()

        except Exception as exc:
            print(f"[SmartCamera] Picamera2 failed to initialise ({exc}).")
            if self._explicit_picamera:
                print("[SmartCamera] Checklist:")
                print("   1. Ribbon cable seated in the CSI port, contacts")
                print("      facing the right way (they are easy to reverse).")
                print("   2. `libcamera-hello --list-cameras` sees the module.")
                print("   3. Nothing else is holding the camera open.")
                self._cam = None
                self._backend = None
            else:
                print("[SmartCamera] Falling back to OpenCV VideoCapture.")
                self._start_opencv()

    def _read_picamera2(self):
        """
        Capture one frame from Picamera2 and return it as a BGR numpy array.
        """
        try:
            # capture_array("main") returns shape (H, W, 3), dtype uint8.
            # The stream is configured as "RGB888", which - per libcamera's
            # inverted naming - is the format that yields (B, G, R) channel
            # order. So this really is BGR, matching every other backend and
            # the rest of the pipeline. No conversion needed or wanted.
            frame = self._cam.capture_array("main")
            if frame is None or frame.size == 0:
                return False, None
            self._last_frame = frame
            return True, frame
        except Exception as exc:
            print(f"[SmartCamera] Picamera2 read error: {exc}")
            return False, None

    # -- Network Backend (DroidCam / IP Webcam) ------------------------------

    def _start_network(self) -> None:
        """
        Open an MJPEG/RTSP stream.

        Two details matter for a phone camera over Wi-Fi:

        • BUFFERSIZE = 1 — OpenCV otherwise queues frames, and every queued
          frame is added latency. For gesture recording that means the saved
          landmarks lag behind what the signer actually did.

        • No silent fallback — if the stream cannot be reached we fail loudly.
          Quietly dropping back to the laptop webcam would mix two cameras in
          one dataset, which is very hard to notice and ruins the data.
        """
        try:
            cap = cv2.VideoCapture(self._url)

            if not cap.isOpened():
                raise RuntimeError(f"could not connect to {self._url}")

            # Keep only the newest frame — minimise latency
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass                       # not supported by every backend

            # Ask the source for our working resolution. Many MJPEG endpoints
            # ignore this, which is why _read_network() also downscales.
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
            except Exception:
                pass

            # Verify we actually receive a frame before declaring success:
            # an MJPEG endpoint can "open" and then never deliver anything.
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"connected to {self._url} but received no frames")

            h, w = frame.shape[:2]
            self._cam     = cap
            self._backend = self.BACKEND_NETWORK
            print(f"[SmartCamera] Network stream connected — {w}x{h}")

        except Exception as exc:
            print(f"[SmartCamera] Network camera FAILED: {exc}")
            print( "[SmartCamera] Checklist:")
            print( "  1. DroidCam app is open on the phone (not just installed)")
            print( "  2. Phone and PC are on the SAME Wi-Fi network")
            print(f"  3. The IP/port shown in the app match: {self._url}")
            print( "  4. Open that URL in a browser — you should see video")
            print( "  5. Windows Firewall is not blocking python.exe")
            self._cam     = None
            self._backend = None

    def _read_network(self):
        """
        Read one frame from the network stream, normalised to WIDTH×HEIGHT.

        Why normalise at all
        --------------------
        The reason is CONSISTENCY, not speed. The webcam and Picamera2 paths
        already pin themselves to WIDTH×HEIGHT, so without this a phone stream
        would deliver a different frame geometry — different aspect ratio and
        different effective field of view. Landmarks are normalised 0-1 against
        the frame, so the same hand in the same place yields different numbers
        at 16:9 than at 4:3. Recording on the phone and running inference on the
        webcam would then quietly feed the model a shifted coordinate space.

        Note on cost: downscaling is not free (~2 ms at 720p, ~4.5 ms at 1080p),
        and MediaPipe crops the hand region from whatever it is given, so a
        higher-resolution source can actually yield a slightly sharper hand
        crop. If profiling on the Pi shows this resize is a bottleneck, raise
        NETWORK_MAX_WIDTH/HEIGHT — but change it for the collector AND the
        server together, or training and inference will disagree.
        """
        try:
            ret, frame = self._cam.read()
            if not ret or frame is None:
                return False, None

            h, w = frame.shape[:2]
            if w > self.WIDTH or h > self.HEIGHT:
                # INTER_AREA is the correct filter for shrinking: it averages
                # the discarded pixels instead of point-sampling, which keeps
                # finger edges clean rather than aliased.
                frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT),
                                   interpolation=cv2.INTER_AREA)

            self._last_frame = frame
            return True, frame
        except Exception as exc:
            print(f"[SmartCamera] Network read error: {exc}")
            return False, None

    # -- OpenCV Backend ------------------------------------------------------

    def _start_opencv(self) -> None:
        """
        Open an OpenCV VideoCapture and pin it to 640x480 so the resolution
        matches the production stream (useful for consistent model inference).
        """
        try:
            cap = cv2.VideoCapture(self._device_index)

            if not cap.isOpened():
                raise RuntimeError(
                    f"cv2.VideoCapture({self._device_index}) failed to open. "
                    "Check that a webcam is connected and not in use."
                )

            # MJPG BEFORE resolution. Most UVC webcams default to raw YUY2,
            # whose bandwidth caps 640x480 at 30 fps and higher modes lower
            # still; the same sensor will happily deliver 60 fps as MJPG. This
            # is the one setting that raises the DELIVERY rate, which is what
            # keeps MediaPipe from losing the hand between frames.
            self._negotiate_format(cap)

            # Confirm what the driver actually provided (may differ on some webcams)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            self._cam     = cap
            self._backend = self.BACKEND_OPENCV
            print(
                f"[SmartCamera] OpenCV VideoCapture started — "
                f"requested {self.WIDTH}x{self.HEIGHT}@{self.FPS}, "
                f"driver returned {actual_w}x{actual_h}"
                f"@{actual_fps:.0f}" if actual_fps else
                f"requested {self.WIDTH}x{self.HEIGHT}, got {actual_w}x{actual_h}"
            )

        except Exception as exc:
            print(f"[SmartCamera] OpenCV VideoCapture failed: {exc}")
            # Both backends failed; _cam stays None.
            # read() will return (False, None) gracefully.
            self._cam     = None
            self._backend = None

    def _read_opencv(self):
        """
        Read one frame from the OpenCV VideoCapture.
        """
        try:
            ret, frame = self._cam.read()
            if ret:
                self._last_frame = frame
            return ret, (frame if ret else None)
        except Exception as exc:
            print(f"[SmartCamera] OpenCV read error: {exc}")
            return False, None

    def _start_usb_dshow(self) -> None:
        """
        Open the camera at `self._device_index`, trying each Windows backend.

        Why not just force DSHOW
        ------------------------
        Windows exposes cameras through two different APIs — DirectShow and
        Media Foundation — and a virtual camera may register with only one of
        them. Worse, the SAME index can mean different devices under each API.
        Forcing DSHOW therefore fails outright on devices that only speak MSMF,
        with an error that looks like "the camera is missing" when it is simply
        being asked in the wrong language.

        Trying each backend in turn, and verifying a real frame arrives before
        declaring success, makes this work without the user having to know any
        of the above.
        """
        if platform.system() == "Windows":
            # (backend, force MJPG?) — MJPG first because virtual cameras almost
            # always offer it and OpenCV decodes it reliably. Raw YUV modes are
            # where the "solid green picture" comes from: the source hands over
            # NV12/YUY2 and OpenCV hands it on as if it were BGR.
            attempts = [
                ("CAP_DSHOW + MJPG", cv2.CAP_DSHOW, True),
                ("CAP_DSHOW",        cv2.CAP_DSHOW, False),
                ("CAP_MSMF + MJPG",  cv2.CAP_MSMF,  True),
                ("CAP_MSMF",         cv2.CAP_MSMF,  False),
                ("default",          None,          False),
            ]
        else:
            attempts = [("default", None, False)]

        errors = []
        for name, api, force_mjpg in attempts:
            cap = None
            try:
                cap = (cv2.VideoCapture(self._device_index, api) if api is not None
                       else cv2.VideoCapture(self._device_index))

                if not cap.isOpened():
                    errors.append(f"{name}: device did not open")
                    cap.release()
                    continue

                # Ask the driver to hand us BGR rather than raw YUV planes.
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                if force_mjpg:
                    self._negotiate_format(cap)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, self.FPS)

                # "Opened" is not the same as "working": a virtual camera whose
                # source app is idle opens happily and then never sends a frame.
                ok, frame = None, None
                for _ in range(5):          # first frames are often junk
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break

                if not ok or frame is None:
                    errors.append(f"{name}: opened but sent no frames")
                    cap.release()
                    continue

                if _is_blank(frame):
                    # The classic solid-green / solid-black frame. It looks like
                    # success to isOpened() and read(), so without this check we
                    # would happily record a whole dataset of flat colour.
                    errors.append(f"{name}: delivered a blank frame "
                                  f"(format mismatch or source not streaming)")
                    cap.release()
                    continue

                h, w = frame.shape[:2]
                self._cam = cap
                self._backend = self.BACKEND_USB_DSHOW
                # Report the format that was NEGOTIATED, not the one requested.
                # "started via CAP_DSHOW + MJPG" while the device was actually
                # handing over YUY2 is how a 20 fps ceiling stayed hidden for
                # two rounds of debugging. A log that flatters the request is
                # worse than no log.
                print(f"[SmartCamera] Camera index {self._device_index} started "
                      f"via {name.split(' +')[0]} — {w}x{h} "
                      f"{self._fourcc_tag(cap)}")
                return

            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                if cap is not None:
                    cap.release()

        print(f"[SmartCamera] Could not open camera index {self._device_index}.")
        for e in errors:
            print(f"    - {e}")
        print("  Checklist:")
        print("   1. Is the DroidCam Windows client OPEN and Connected?")
        print("      The virtual camera only exists while it is running.")
        print("   2. Close anything else using the camera: OBS, Zoom, Teams,")
        print("      Windows Camera, browser tabs with camera access.")
        print("   3. Re-run `npm run cameras` — indices shift when devices")
        print("      are plugged, unplugged, or clients restart.")
        self._cam = None
        self._backend = None

    # -- Context manager support ----------------------------------------------

    def __enter__(self) -> "SmartCamera":
        """Enables:  with SmartCamera() as cam: ..."""
        return self.start()

    def __exit__(self, *_) -> None:
        self.release()

    # -- Dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SmartCamera(backend={self._backend!r}, "
            f"resolution={self.WIDTH}x{self.HEIGHT}, "
            f"fps={self.FPS})"
        )

    # -- Read-only convenience properties -------------------------------------

    @property
    def backend(self):
        """The active backend identifier string, or None if not started."""
        return self._backend

    @property
    def is_running(self) -> bool:
        """True if the camera has been successfully started."""
        return self._cam is not None

    @property
    def last_frame(self):
        """The most recently captured frame (BGR numpy array), or None."""
        return self._last_frame


# -----------------------------------------------------------------------------
#  Quick smoke-test  (run:  python camera_manager.py)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  SmartCamera — live preview  (press 'q' to quit)")
    print("=" * 60)

    cam = SmartCamera()
    cam.start()

    if not cam.is_running:
        print("[SmartCamera] Could not open any camera. Exiting.")
        sys.exit(1)

    print(f"[SmartCamera] Active: {cam}")

    while True:
        ret, frame = cam.read()

        if not ret or frame is None:
            print("[SmartCamera] Empty frame — skipping.")
            continue

        # Overlay backend info on the preview window
        label = f"Backend: {cam.backend}  |  {cam.WIDTH}x{cam.HEIGHT}"
        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)

        cv2.imshow("SmartCamera Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[SmartCamera] Preview closed.")


# -----------------------------------------------------------------------------
#  Device identification
# -----------------------------------------------------------------------------

# Words that appear in the NAME of a camera soldered into a laptop lid. Anything
# without one of these had to be plugged in.
_INTEGRATED_HINTS = ("integrated", "built-in", "builtin", "internal",
                     "facetime", "user facing", "front camera", "laptop",
                     "true vision", "hd user-facing", "ir camera")

# Vendors name lid cameras whatever they like ("HP True Vision FHD Camera"),
# so no keyword list will ever be complete. When nothing matches by name, the
# lowest index is the built-in one - which is the rule that held before names
# were available at all.


def camera_device_names() -> tuple:
    """
    Real device names in enumeration order, on Windows.

    Returns (names, exact). OpenCV identifies cameras by bare index, which tells
    you nothing: index 1 could be a USB webcam, a phone acting as one, or a
    virtual camera from OBS. Windows knows the product names, so it is asked
    rather than guessed at.

    Two sources, and the difference matters:

      pygrabber  - enumerates DirectShow devices in exactly the order OpenCV
                   assigns indices, so name and index line up by construction.
                   `exact` is True.
      PowerShell - lists PnP camera devices. Usually the same order, but that
                   is a convention rather than a guarantee, so `exact` is False
                   and the caller keeps showing the index and resolution for
                   the user to check against.
    """
    if sys.platform != "win32":
        return [], False

    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        if names:
            return list(names), True
    except Exception:
        pass

    query = ("Get-CimInstance Win32_PnPEntity | "
             "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
             "Select-Object -ExpandProperty Name")
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True, timeout=20,
        )
        names = [ln.strip() for ln in
                 out.stdout.decode("utf-8", "replace").splitlines() if ln.strip()]
        return names, False
    except Exception:
        return [], False


def describe_cameras() -> list:
    """
    Every working camera, with a name and a kind.

    Each entry: {"index", "width", "height", "name", "kind"} where kind is
    "integrated" or "usb".

    Entries also carry "exact": True when the name provably belongs to that
    index (pygrabber enumerates in OpenCV's own order), False when it is a
    best-effort match. Callers show the index and resolution either way - a
    label you can sanity-check beats a confident label you cannot.
    """
    found = list_local_cameras()
    names, exact = camera_device_names()

    out = []
    for pos, (idx, w, h) in enumerate(found):
        name = names[pos] if pos < len(names) else None
        if name:
            kind = ("integrated"
                    if any(h_ in name.lower() for h_ in _INTEGRATED_HINTS)
                    else "usb")
        else:
            # No names available: index 0 is the built-in camera on a laptop.
            kind = "integrated" if idx == 0 else "usb"
        out.append({"index": idx, "width": w, "height": h,
                    "name": name, "kind": kind, "exact": bool(name) and exact})

    if out and not any(c["kind"] == "integrated" for c in out):
        lowest = min(out, key=lambda c: c["index"])
        lowest["kind"] = "integrated"
    return out


# -----------------------------------------------------------------------------
#  Interactive chooser
# -----------------------------------------------------------------------------
# Shared by data_collector.py and websocket_server.py. Recording and inference
# MUST use the same camera: the model learns the lens it was trained on, and a
# different field of view shifts every landmark. One prompt, one behaviour.

def choose_camera_interactive(*, allow_prompt: bool = True):
    """
    Ask which camera to record with.

    A phone camera over DroidCam is usually much sharper than a laptop webcam,
    and sharper frames give cleaner MediaPipe landmarks — the raw material the
    whole model is built from. Worth the extra prompt.

    Returns a source string (or int for USB index) understood by SmartCamera.

    When there is nothing to read an answer from, it falls back to
    CAMERA_SOURCE instead of raising: an unguarded input() at startup dies with
    EOFError, and inside a task runner that reads as a crash with no cause.

    Why this does NOT test isatty()
    -------------------------------
    Under `npm run dev:all` the process is a child of `concurrently`, so stdin
    is a PIPE, not a terminal — even though `--handle-input` is forwarding your
    keystrokes down it perfectly well. Testing isatty() therefore refused to ask
    in exactly the case the question was most needed. What actually matters is
    whether stdin can be READ, so that is what is tested, and EOF is caught.
    """
    preset = os.getenv("CAMERA_SOURCE")
    if preset and preset.lower() != "auto":
        print(f"\nCamera: {preset}  (from CAMERA_SOURCE)")
        return preset

    if os.getenv("TARJUMAN_NO_PROMPT"):
        print("\nCamera: auto  (TARJUMAN_NO_PROMPT set)")
        return "auto"

    if not allow_prompt or not _stdin_readable():
        print("\nCamera: auto  (no input available — set CAMERA_SOURCE in .env)")
        return "auto"

    on_pi = _is_raspberry_pi()
    pi_cam = picamera2_available()

    print("\n" + "=" * 60)
    print("  Select camera")
    print("=" * 60)
    print("  1. Integrated camera             (built into the laptop)")
    print("  2. External USB webcam           (plugged in)")
    print("  3. DroidCam / phone")
    print("  4. Other stream URL")
    if pi_cam:
        print("  5. Raspberry Pi Camera           (Module 3 / CSI, Picamera2)")
    elif on_pi:
        print("  5. Raspberry Pi Camera           [picamera2 NOT installed]")
    print()

    # On the Pi the CSI module is the point of the exercise, so make it the
    # default there and leave the USB webcam as the default on the laptop.
    default = "5" if pi_cam else "2"

    try:
        return _camera_menu(default=default, pi_cam=pi_cam, on_pi=on_pi)
    except (EOFError, KeyboardInterrupt):
        # stdin closed under us, or the user pressed Ctrl-C at the prompt.
        print("\nCamera: auto  (no answer given)")
        return "auto"


def picamera2_available() -> bool:
    """
    True when the CSI camera stack can actually be used.

    Deliberately checks that picamera2 IMPORTS rather than that the machine
    looks like a Pi. A Pi with the library missing and a laptop without it
    fail in the same way, and both need to be steered to a different option
    rather than into a traceback.

    importlib.util.find_spec avoids paying the (slow) picamera2 import just
    to answer a menu question.
    """
    if _is_raspberry_pi() is False and sys.platform == "win32":
        return False          # fast path: never available on Windows
    try:
        import importlib.util
        return importlib.util.find_spec("picamera2") is not None
    except Exception:
        return False


def _stdin_readable() -> bool:
    """True when stdin is something we could actually read an answer from."""
    stream = sys.stdin
    if stream is None or getattr(stream, "closed", False):
        return False
    try:
        stream.fileno()
    except (OSError, ValueError, AttributeError):
        return False
    return True


def _camera_menu(default: str = "2", pi_cam: bool = False,
                 on_pi: bool = False):
    """
    The question loop itself. Raises EOFError if stdin ends.

    `pi_cam`/`on_pi` are passed in rather than re-probed so the menu shown and
    the choices accepted can never disagree - offering an option that is then
    rejected is a small thing that feels broken.
    """
    top = "5" if (pi_cam or on_pi) else "4"
    while True:
        choice = input(f"Choice [1-{top}, default {default}]: ").strip() or default

        if choice in ("1", "2"):
            want = "integrated" if choice == "1" else "usb"
            title = ("Integrated camera" if want == "integrated"
                     else "External USB webcam")
            print(f"\n  [{title}]")
            print("  Scanning camera devices...")
            cams = describe_cameras()

            if not cams:
                print("   [!] No camera devices responded.")
                print("       - Is the webcam plugged in and its light on?")
                print("       - Close Zoom / Teams / Windows Camera / OBS.")
                print("       - Try a different USB port.")
                continue

            for c in cams:
                mark = " <-" if c["kind"] == want else "   "
                label = c["name"] or ("built-in (index 0)" if c["index"] == 0
                                      else "external")
                print(f"    {mark} index {c['index']}: {c['width']}x{c['height']}"
                      f"  {label}  [{c['kind']}]")

            matching = [c for c in cams if c["kind"] == want]

            # One obvious answer: take it. Making someone type an index they
            # cannot verify is how you end up recording on the wrong lens and
            # only noticing after thirty samples.
            if len(matching) == 1:
                pick = matching[0]
                print(f"\n   Using index {pick['index']}"
                      f"{' - ' + pick['name'] if pick['name'] else ''}")
                return f"index:{pick['index']}"

            if not matching:
                print(f"\n   [!] No {want} camera was detected.")
                if want == "usb":
                    print("       Check the cable and the USB port, then re-run.")
                continue

            print("\n   More than one matched. Not sure which is which?")
            print("     npm run cameras -- --preview")
            default = str(matching[0]["index"])
            raw = input(f"\n   Camera index [default {default}]: ").strip() or default
            if raw.isdigit():
                return f"index:{raw}"
            print("   [!] Enter one of the numbers listed above.")
            continue

        if choice == "3":
            print("\n  [DroidCam Setup]")
            print("  Wi-Fi MJPEG has no flow control: when bandwidth dips it")
            print("  silently DROPS frames, and lost frames distort the exact")
            print("  timing the model learns from. Prefer USB for recording.")
            print()
            print("  1. USB - virtual webcam    <- iPhone AND Android")
            print("  2. USB - adb tunnel        <- ANDROID ONLY (no PC client)")
            print("  3. Wi-Fi                   (requires IP, may stutter)")
            print()
            print("  iPhone: adb cannot talk to iOS at all — use option 1.")
            dc_mode = input("  Choose mode [1-3, default 1]: ").strip() or "1"

            if dc_mode == "3":
                print("\n  Before continuing:")
                print("   - DroidCam app is OPEN on the phone")
                print("   - Phone and PC are on the same Wi-Fi")
                print(f"   - The app shows {DROIDCAM_IP}:{DROIDCAM_PORT}")
                print("   (change via DROIDCAM_IP / DROIDCAM_PORT in .env)")
                return "droidcam"

            if dc_mode == "2":
                print("\n  [adb tunnel - ANDROID ONLY]")
                print("   This will NOT work with an iPhone.")
                print("   1. Platform Tools folder set in .env as ADB_PATH")
                print("   2. Developer Options ON (tap Build number 7 times)")
                print("   3. USB debugging ON")
                print("   4. DroidCam app OPEN on the phone")
                print("   5. Cable connected, 'Allow USB debugging' accepted")
                return "droidcam-usb"

            # Default: virtual webcam — works for iPhone and Android alike
            print("\n  [USB - virtual webcam]")
            print("   1. DroidCam (or iVCam / Camo) PC client is OPEN")
            print("   2. Mode set to USB, 'Connect' pressed")
            print("   3. Phone connected by cable, app open on the phone")
            print("   iPhone also needs Apple's iTunes / Apple Devices installed")
            print("   (it provides the USB drivers Windows needs).")
            print("\n  Scanning local camera devices...")
            cams = list_local_cameras()
            if cams:
                for idx, w, h in cams:
                    hint = "   <- likely the phone (higher resolution)" if w >= 1280 else ""
                    print(f"     index {idx}:  {w}x{h}{hint}")
            else:
                print("     [!] No devices responded. Is the PC client connected?")
            cam_idx = input("\n   Camera index [default 1]: ").strip() or "1"
            try:
                return int(cam_idx)
            except ValueError:
                return "usb_dshow"

        if choice == "4":
            url = input("  Stream URL: ").strip()
            if "://" in url:
                return url
            print("  [!] Must be a full URL, e.g. http://192.168.8.177:4747/video")
            continue

        if choice == "5":
            if pi_cam:
                print("\n  [Raspberry Pi Camera — CSI / Picamera2]")
                print(f"   Capturing at {SmartCamera.WIDTH}x{SmartCamera.HEIGHT}"
                      f" @ {SmartCamera.FPS} fps (CAMERA_WIDTH / CAMERA_HEIGHT")
                print("   / CAMERA_FPS in .env change this).")
                print("   Focus is locked to avoid the lens hunting mid-sign.")
                return "picamera"

            if on_pi:
                print("\n   [!] picamera2 is not installed on this Pi.")
                print("       sudo apt install -y python3-picamera2")
                print("       Use apt, NOT pip: it is built against the system")
                print("       libcamera stack and the pip build will not work.")
            else:
                print("\n   [!] This is not a Raspberry Pi — there is no CSI")
                print("       camera here. Use 1-4 on the laptop.")
            continue

        print(f"  [!] Enter a number from 1 to {top}.")
