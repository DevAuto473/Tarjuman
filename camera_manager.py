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
    WIDTH         = 640
    HEIGHT        = 480
    FPS           = 30
    LENS_POSITION = 1.0   # Manual-focus position (≈ 1 m; adjust as needed)

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
        if self._cam is None:
            return False, None

        if self._backend == self.BACKEND_PICAMERA2:
            return self._read_picamera2()
        elif self._backend == self.BACKEND_NETWORK:
            return self._read_network()
        else:
            # OpenCV handles both BACKEND_OPENCV and BACKEND_USB_DSHOW
            return self._read_opencv()

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

    def _start_picamera2(self) -> None:
        """
        Configure and start Picamera2 with:
          • Fixed 640x480 video stream at 30 FPS
          • Manual (fixed) lens position to prevent autofocus hunting,
            which causes motion blur and FPS drops during fast
            sign-language gestures.

        Falls back to OpenCV automatically on any import or runtime error.
        """
        try:
            # Import is deferred so this file can be imported on non-Pi systems
            from picamera2 import Picamera2  # type: ignore[import]

            picam = Picamera2()

            # -- Build video configuration ---------------------------------
            # main stream -> BGR888 so OpenCV receives frames with no colour conversion
            video_config = picam.create_video_configuration(
                main={
                    "size":   (self.WIDTH, self.HEIGHT),
                    "format": "BGR888",   # Native BGR -> no cvtColor needed
                },
                controls={
                    # Lock framerate: both min and max to FPS (prevents dipping)
                    "FrameRate": self.FPS,

                    # -- Manual focus — critical for gesture recognition ---
                    # AfMode 0 = Manual; prevents the lens from continuously
                    # hunting for focus, which causes blur during fast hand
                    # movements and can drop the effective FPS significantly.
                    "AfMode":       0,                  # 0=Manual, 2=Continuous
                    "LensPosition": self.LENS_POSITION, # Fixed focal distance
                },
                # Optimise internal queue depth for low-latency live capture
                buffer_count=4,
            )

            picam.configure(video_config)
            picam.start()

            self._cam     = picam
            self._backend = self.BACKEND_PICAMERA2
            print(
                f"[SmartCamera] Picamera2 started — "
                f"{self.WIDTH}x{self.HEIGHT} @ {self.FPS} FPS, "
                f"LensPosition={self.LENS_POSITION} (manual focus)"
            )

        except ImportError:
            print(
                "[SmartCamera] Picamera2 not found (ImportError). "
                "Falling back to OpenCV VideoCapture."
            )
            self._start_opencv()

        except Exception as exc:
            print(
                f"[SmartCamera] Picamera2 failed to initialise ({exc}). "
                f"Falling back to OpenCV VideoCapture."
            )
            self._start_opencv()

    def _read_picamera2(self):
        """
        Capture one frame from Picamera2 and return it as a BGR numpy array.
        """
        try:
            # capture_array("main") returns a numpy array in the configured
            # pixel format: BGR888 -> shape (H, W, 3), dtype uint8
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

            # Pin resolution to match the Pi production stream
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

            # Confirm what the driver actually provided (may differ on some webcams)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._cam     = cap
            self._backend = self.BACKEND_OPENCV
            print(
                f"[SmartCamera] OpenCV VideoCapture started — "
                f"requested {self.WIDTH}x{self.HEIGHT}, "
                f"driver returned {actual_w}x{actual_h}"
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

                if force_mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                # Ask the driver to hand us BGR rather than raw YUV planes.
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

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
                print(f"[SmartCamera] Camera index {self._device_index} started "
                      f"via {name} — {w}x{h}")
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
                     "facetime", "user facing", "front camera", "laptop")


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

    print("\n" + "=" * 60)
    print("  Select camera")
    print("=" * 60)
    print("  1. Integrated camera             (built into the laptop)")
    print("  2. External USB webcam           (plugged in)")
    print("  3. DroidCam / phone")
    print("  4. Other stream URL")
    print()

    try:
        return _camera_menu()
    except (EOFError, KeyboardInterrupt):
        # stdin closed under us, or the user pressed Ctrl-C at the prompt.
        print("\nCamera: auto  (no answer given)")
        return "auto"


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


def _camera_menu():
    """The question loop itself. Raises EOFError if stdin ends."""
    while True:
        choice = input("Choice [1-4, default 2]: ").strip() or "2"

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

        print("  [!] Enter 1, 2, 3 or 4.")
