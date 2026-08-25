"""
websocket_server.py — Tarjuman Backend (Zero-Bandwidth Headless Camera)
========================================================================
Architecture overview
---------------------
  • The React frontend NO LONGER streams video frames.
  • The frontend sends two simple JSON control commands:
        {"type": "start_camera"}   -> launches the background camera loop
        {"type": "stop_camera"}    -> cancels the loop and releases the camera
  • The backend owns the camera (SmartCamera), runs MediaPipe Hands
    locally, and emits a lightweight JSON heartbeat to the client:
        {"type": "tracking_status", "body_visible": bool, "hands_visible": bool}
  • cam.read() is a blocking call — it is always dispatched through
    loop.run_in_executor() to avoid freezing the asyncio event loop.

Message protocol (text / JSON)
-------------------------------
  Client  -> Server  |  {"type": "start_camera"}
  Client  -> Server  |  {"type": "stop_camera"}
  Client  -> Server  |  {"type": "speak", "value": "..."}
  Client  -> Server  |  {"type": "user_question", "data": "...", "history": [...]}
  Server  -> Client  |  {"type": "tracking_status", "body_visible": bool, "hands_visible": bool}
  Server  -> Client  |  {"type": "letter",    "value": "أ"}
  Server  -> Client  |  {"type": "pose_data", "distance": {...}}
  Server  -> Client  |  {"type": "ai_response", "data": {...}}
  Server  -> Client  |  <binary 0x01 header> + MP3 audio bytes  (TTS)
  Server  -> Client  |  {"type": "stt_result", "purpose": str, "value": str}

Binary audio protocol (unchanged)
----------------------------------
  0x01  + <bytes>  -> audio to translate via STT
  0x02  + <bytes>  -> audio for conversational chat via STT
"""

# -- Import bootstrap ---------------------------------------------------------
# Puts src/ on the path so `tarjuman_core` resolves when this file is run
# directly (`python websocket_server.py`). Running through `npm run ...` sets PYTHONPATH
# instead, and `pip install -e .` makes both unnecessary - this is the belt to
# those braces, so a plain `python` invocation never fails with ImportError.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.basename(_os.path.dirname(_os.path.abspath(__file__))) == "scripts"
    else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "src"))

import asyncio
import collections
import json
import os
import re
import sys
import time
import traceback
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp
import cv2
import edge_tts
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import websockets
from dotenv import load_dotenv
from openai import AsyncOpenAI

from tarjuman_core.paths import data, root
from tarjuman_core.camera_manager import (
    SmartCamera, choose_camera_interactive, describe_cameras, droidcam_url,
)
from tarjuman_core.dtw_matcher import SignReferenceLibrary
from tarjuman_core.pose_to_bones import frame_to_bone_dirs
from tarjuman_core.gesture_segmenter import GestureSegmenter
from tarjuman_core.feature_extractor import (
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    VALS_PER_FRAME,
    PoseTracker,
    estimate_distance,
    extract_frame_features,
    prepare_frame,
    split_hands,
)


# -----------------------------------------------------------------------------
#  Global configuration
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore")

# -- Secrets: loaded from .env, never hardcoded -------------------------------
# Create a .env file next to this script (see .env.example) containing:
#     OPENROUTER_API_KEY=sk-or-v1-...
#     GROQ_API_KEY=gsk_...
# .env is git-ignored and must NEVER be committed.
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")

_missing_keys = [
    name for name, value in (
        ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
        ("GROQ_API_KEY",       GROQ_API_KEY),
    ) if not value
]
if _missing_keys:
    print("[FAIL]  Missing required environment variable(s): " + ", ".join(_missing_keys))
    print("    -> Create a .env file next to websocket_server.py containing:")
    for name in _missing_keys:
        print(f"        {name}=<your key>")
    print("    -> See .env.example for the expected format.")
    sys.exit(1)

# -- Network binding ----------------------------------------------------------
# Bound to loopback ONLY. The Tauri frontend runs on the same machine and
# connects via ws://localhost:8765, so there is no reason to expose this
# server to the local network. Binding to 0.0.0.0 would let any device on the
# network control the camera and consume paid API credits without any auth.
BIND_HOST = "127.0.0.1"
BIND_PORT = 8765

# -- Camera source ------------------------------------------------------------
#   auto      -> Picamera2 on a Pi, otherwise the local webcam
#   laptop    -> force the local webcam
#   droidcam  -> phone over DroidCam (uses DROIDCAM_IP / DROIDCAM_PORT)
#   <url>     -> any MJPEG / RTSP stream
# Set CAMERA_SOURCE in .env. Whatever the DATASET was recorded with is usually
# what inference should use too — a model trained on sharp phone frames sees a
# different world through a soft laptop webcam.
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "auto")

# The ACTIVE source, which the user can change from the app at runtime.
# `.env` only supplies the starting value: a camera that turns out to be the
# wrong one should not require editing a file and restarting the server, which
# is exactly the situation this variable exists to avoid.
_active_camera_source = CAMERA_SOURCE

CONFIDENCE_THRESHOLD = 0.65

# -- Recognition mode ---------------------------------------------------------
#   "segmented" (default) — event-driven: detect when a sign starts and ends,
#                           classify ONCE. Latency ≈ the sign's own duration.
#   "sliding"             — legacy: keep a rolling 30-frame window and re-run
#                           inference every frame. Kept for A/B comparison.
RECOGNITION_MODE = "segmented"

# -- Camera preview -----------------------------------------------------------
# The original "zero-bandwidth" design withheld video from the UI to save
# bandwidth — but frontend and backend run on the SAME machine over loopback,
# so there was no bandwidth to save. The cost of that choice was real though:
# a Deaf user could not see themselves and had no way to tell whether they were
# framed correctly.
#
# Measured cost of restoring it: 0.30 ms/frame to encode 320x240 JPEG q=50
# (~1 % of MediaPipe's per-frame budget) and ~220 KB/s over localhost.
PREVIEW_ENABLED   = True
PREVIEW_WIDTH     = 320
PREVIEW_HEIGHT    = 240
PREVIEW_QUALITY   = 50      # JPEG quality 0-100
PREVIEW_FPS       = 12      # throttle; recognition still runs at full rate
PREVIEW_SKELETON  = True    # draw detected hand landmarks onto the preview

# Live-mirror stream rate. Recognition still sees every frame; only the avatar
# update is throttled, since the eye cannot use more than this.
MIRROR_FPS = 20

# Binary frame headers, server -> client
BIN_AUDIO   = 0x01          # MP3 speech (existing)
BIN_PREVIEW = 0x02          # JPEG camera preview (new)

# Used by the legacy sliding mode only: how many consecutive identical
# predictions confirm a letter. Segmented mode needs no such counter because
# it classifies once per detected gesture instead of continuously.
FRAME_STABILITY_LIMIT = 2

# Landmark geometry is defined ONCE in feature_extractor.py and imported above
# (SEQUENCE_LENGTH = 30, VALS_PER_FRAME = 126, TOTAL_FEATURES = 3 780).
# Never redeclare these here — divergence between collector and server is
# exactly the silent train/inference mismatch this module exists to prevent.

# Arabic label map (English folder names -> Arabic glyphs)
LABEL_TO_ARABIC = {
    "Alef": "أ", "Al": "ال", "Ain": "ع", "Beh": "ب", "Teh": "ت",
    "Theh": "ث", "Jeem": "ج", "Hah": "ح", "Khah": "خ", "Dal": "د",
    "Thal": "ذ", "Reh": "ر", "Zain": "ز", "Seen": "س", "Sheen": "ش",
    "Sad": "ص", "Dad": "ض", "Tah": "ط", "Zah": "ظ", "Ghain": "غ",
    "Feh": "ف", "Qaf": "ق", "Kaf": "ك", "Lam": "ل", "Laa": "لا",
    "Meem": "م", "Noon": "ن", "Heh": "هـ", "Waw": "و", "Yeh": "ي",
    "Teh_Marbuta": "ة", "Alslam_3lykm": "السلام عليكم",
    "ma_esmk": "ما اِسمك؟", "kyf_7alk": "كيفَ حالُكَ؟", "salam": "السلام عليكم",

 }


# -----------------------------------------------------------------------------
#  ONNX model + label map loading
# -----------------------------------------------------------------------------

ONNX_MODEL_PATH = root("sign_model.onnx")
LABELS_JSON     = data("labels.json")

# Graceful loading: if the model files have not been trained yet, the server
# still starts — camera, TTS, and STT will work; only sign recognition will
# be disabled until the model files are generated via train_model.py.
onnx_session = None       # onnxruntime.InferenceSession
onnx_input_name = None    # str — name of the (N, TOTAL_FEATURES) float32 input
onnx_prob_name  = None    # str — name of the (N, n_classes) probabilities output
label_map: dict | None = None       # {"0": "Alef", "1": "Beh", ...}


def _disable_recognition(*reason_lines: str) -> None:
    """
    Hard-disable sign recognition and explain why, loudly.

    Session and label map are cleared together so no half-loaded state can
    ever reach process_frame_sync(). Camera / TTS / STT are unaffected —
    the server stays usable, just without sign recognition.
    """
    global onnx_session, onnx_input_name, onnx_prob_name, label_map
    onnx_session = None
    onnx_input_name = None
    onnx_prob_name = None
    label_map = None
    print("\n" + "!" * 70)
    for line in reason_lines:
        print(line)
    print("   -> Sign recognition is DISABLED. Camera / TTS / STT still work.")
    print("!" * 70 + "\n")


def _validate_onnx_session(sess, labels: dict) -> tuple[str, str]:
    """
    Verify the loaded ONNX graph is actually the dynamic-gesture model this
    server expects, and not a stale/incompatible one.

    Why this exists
    ---------------
    `train_model.py` (9 000 features: 30 frames × 300 landmarks) and the older
    `learn_model.py` (63 features: one static hand) historically wrote to the
    SAME model filename. Loading the wrong one used to fail *silently* — the
    server started fine, then threw on every single frame inside
    process_frame_sync() while the user just saw recognition mysteriously stop.
    Failing loudly here, once, at startup is far better than failing quietly
    30 times a second.

    Returns
    -------
    (input_name, probabilities_output_name)

    Raises
    ------
    ValueError with a human-readable message if the graph is unusable.
    """
    inputs  = sess.get_inputs()
    outputs = sess.get_outputs()

    # -- 1. Single input tensor ----------------------------------------------
    if len(inputs) != 1:
        raise ValueError(
            f"Expected exactly 1 input tensor, found {len(inputs)}: "
            f"{[i.name for i in inputs]}"
        )

    inp = inputs[0]

    # -- 2. Input shape — the critical check ---------------------------------
    # Shape is [batch, features]; batch is dynamic (None/'N'), features fixed.
    if len(inp.shape) != 2:
        raise ValueError(
            f"Expected a 2-D input [batch, features], got shape {inp.shape}."
        )

    n_features = inp.shape[1]
    if not isinstance(n_features, int):
        raise ValueError(
            f"Input feature dimension is dynamic ({n_features!r}) — cannot "
            f"verify the model's input shape, so it is unsafe to use."
        )
    if n_features != TOTAL_FEATURES:
        raise ValueError(
            f"Incompatible model detected. Expected {TOTAL_FEATURES} features "
            f"({SEQUENCE_LENGTH} frames × {VALS_PER_FRAME} landmarks), "
            f"got {n_features}.\n"
            f"   -> A {n_features}-feature model is almost certainly built from "
            f"a different feature layout (e.g. the old static-hand pipeline).\n"
            f"   -> Re-run train_model.py to regenerate {ONNX_MODEL_PATH}."
        )

    # -- 3. Probabilities output must exist ----------------------------------
    # With zipmap=False, skl2onnx emits [0]='label', [1]='probabilities'.
    # Resolve by name where possible, fall back to index 1.
    if len(outputs) < 2:
        raise ValueError(
            f"Expected 2 outputs (label + probabilities), found "
            f"{len(outputs)}: {[o.name for o in outputs]}.\n"
            f"   -> Was the model exported without options={{'clf': "
            f"{{'zipmap': False}}}}?"
        )

    prob_out = next(
        (o for o in outputs if "prob" in o.name.lower()), outputs[1]
    )

    # -- 4. Class count <-> label-map consistency ------------------------------
    n_classes = prob_out.shape[1] if len(prob_out.shape) == 2 else None
    if isinstance(n_classes, int) and n_classes != len(labels):
        raise ValueError(
            f"Label map is out of sync with the model: the model outputs "
            f"{n_classes} class probabilities but {LABELS_JSON} defines "
            f"{len(labels)}.\n"
            f"   -> {ONNX_MODEL_PATH} and {LABELS_JSON} must come from the SAME "
            f"train_model.py run."
        )

    # -- 5. Label map must cover every class index 0..n-1 --------------------
    # Inference maps argmax(probabilities) directly to a label_map key, which
    # is only valid because LabelEncoder produces sorted integer classes 0..n-1.
    expected_classes = n_classes if isinstance(n_classes, int) else len(labels)
    missing = [str(i) for i in range(expected_classes) if str(i) not in labels]
    if missing:
        raise ValueError(
            f"{LABELS_JSON} is missing entries for class index/indices: "
            f"{', '.join(missing)}."
        )

    return inp.name, prob_out.name


try:
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        candidate_labels = json.load(f)

    candidate_session = ort.InferenceSession(
        ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]
    )

    # Validate BEFORE publishing to the globals used by the inference path
    _in_name, _prob_name = _validate_onnx_session(candidate_session, candidate_labels)

    onnx_session    = candidate_session
    onnx_input_name = _in_name
    onnx_prob_name  = _prob_name
    label_map       = candidate_labels

    print(
        f"[OK]  ONNX model loaded and validated: {ONNX_MODEL_PATH}\n"
        f"    |-- input  : '{onnx_input_name}' "
        f"[batch, {TOTAL_FEATURES}] ({SEQUENCE_LENGTH} frames × {VALS_PER_FRAME})\n"
        f"    |-- output : '{onnx_prob_name}'\n"
        f"    `-- classes: {len(label_map)}"
    )

except FileNotFoundError as exc:
    _disable_recognition(
        f"[!]  Model file not found: {exc.filename}",
        "   -> Run train_model.py first to generate it.",
    )
except ValueError as exc:
    # Raised by _validate_onnx_session — the incompatible-model case
    _disable_recognition(f"[FAIL]  ERROR: {exc}")
except Exception as exc:
    _disable_recognition(
        f"[FAIL]  ERROR: Failed to load {ONNX_MODEL_PATH}: "
        f"{type(exc).__name__}: {exc}"
    )

# -- DTW reference library ("تعلم مع ترجمان") ---------------------------------
# Built from the same CSV the classifier trains on, so practice mode works
# without recording a separate reference set. Adding a new word to the
# dictionary needs only one new recording — no retraining.
REFERENCE_CSV = data("dynamic_gestures_v4.csv")
dtw_library = SignReferenceLibrary.from_csv(REFERENCE_CSV)


# Fail here rather than inside a client session. Hands() is constructed
# per-connection below, so a dependency mismatch would otherwise surface as a
# handler exception on the first client instead of at server startup.
from tarjuman_core.runtime_check import check_mediapipe_stack
check_mediapipe_stack()

# MediaPipe module reference (instances created per-session to avoid races).
# Hands ONLY — Holistic additionally ran BlazePose (33 pts) and Face Mesh
# (468 pts) on every frame, and the face landmarks were never used at all.
mp_hands = mp.solutions.hands

# Drawing helpers, used only for the optional preview overlay
mp_drawing = mp.solutions.drawing_utils
_preview_dot_spec  = mp_drawing.DrawingSpec(color=(80, 220, 120), thickness=1, circle_radius=2)
_preview_line_spec = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)


# -----------------------------------------------------------------------------
#  AI / TTS / STT clients
# -----------------------------------------------------------------------------

or_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# -- Assistant model ----------------------------------------------------------
# Overridable from .env so the model can be swapped without touching code:
#     AI_MODEL=perplexity/sonar-pro
#
# perplexity/* models have live web search built in. "sonar" is the small,
# fast, cheap tier; "sonar-pro" runs deeper multi-step searches and returns
# roughly double the citations, which shows up directly in answer quality.
AI_MODEL    = os.getenv("AI_MODEL", "perplexity/sonar")
AI_TIMEZONE = os.getenv("AI_TIMEZONE", "Asia/Riyadh")


def _now_in_local_tz() -> datetime:
    """Current time in the configured timezone, falling back to system local."""
    try:
        return datetime.now(ZoneInfo(AI_TIMEZONE))
    except Exception:
        return datetime.now().astimezone()


_ARABIC_DAYS = {
    0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
    4: "الجمعة", 5: "السبت", 6: "الأحد",
}


def _arabic_clock(now: datetime) -> str:
    """Format the time the way an Arabic speaker says it, e.g. '١١:٤٤ مساءً'."""
    hour12 = now.strftime("%I").lstrip("0") or "12"
    suffix = "صباحاً" if now.strftime("%p") == "AM" else "مساءً"
    return f"{hour12}:{now.strftime('%M')} {suffix}"


def build_system_prompt() -> str:
    """
    Build the system prompt fresh for every request.

    Why it is rebuilt each time
    ---------------------------
    A language model has NO access to the current date or time. Asking one
    "كم الساعة؟" with a static prompt cannot work — not because the model is
    weak, but because nothing ever told it. Injecting the clock here is what
    makes time and date questions answerable at all.

    Why it is written this way
    --------------------------
    Tuned for `perplexity/sonar`, a small fast model. Small models follow
    CONCRETE EXAMPLES far more reliably than abstract rules, and they degrade
    when handed a long list of constraints. So this prompt is short, front-loads
    the facts, and demonstrates the desired answer shape instead of describing
    it. The previous version did the opposite — many strict rules, no examples —
    which is the worst possible shape for a model this size.
    """
    now = _now_in_local_tz()
    day = _ARABIC_DAYS[now.weekday()]
    clock = _arabic_clock(now)
    date_str = now.strftime("%Y-%m-%d")

    return f"""أنت "ترجمان"، مساعد عربي ذكي.

الوقت الآن: {clock} — يوم {day} الموافق {date_str} بتوقيت {AI_TIMEZONE}.
هذه معلومة مؤكدة لديك. لا تقل أبداً إنك لا تعرف الوقت أو التاريخ.

أجب دائماً بالعربية الفصحى، بنص مباشر بلا Markdown وبلا روابط وبلا أرقام مراجع مثل [1].
أجب بجملة إلى ثلاث جمل. ابدأ بالإجابة نفسها فوراً.
لأي سؤال عن الطقس أو الأخبار أو الأسعار أو أي معلومة متغيّرة: ابحث في الإنترنت الآن واذكر الرقم أو الخبر الفعلي.

أمثلة على الأسلوب المطلوب:

س: كم الساعة؟
ج: الساعة الآن {clock} بتوقيت {AI_TIMEZONE}.

س: ما تاريخ اليوم؟
ج: اليوم {day}، الموافق {date_str}.

س: كيف حالك؟
ج: بخيرٍ والحمد لله، كيف أستطيع مساعدتك؟

س: ما طقس الرياض اليوم؟
ج: درجة الحرارة في الرياض الآن ٣٨ مئوية، والجو صحو مع رياح خفيفة."""


# -----------------------------------------------------------------------------
#  Utility: safe JSON send
# -----------------------------------------------------------------------------

async def send_to_client(ws, msg_dict: dict) -> None:
    """
    Serialise msg_dict to JSON and send it.

    A closed connection is normal (the client navigated away mid-send) and is
    ignored quietly. Anything else — a payload that will not serialise, for
    instance — is a real bug and is reported, because swallowing it would make
    messages vanish with no trace at all.
    """
    try:
        await ws.send(json.dumps(msg_dict, ensure_ascii=False))
    except websockets.exceptions.ConnectionClosed:
        pass      # expected during disconnect
    except Exception as exc:
        print(f"[!]  send failed for {msg_dict.get('type', '?')!r}: "
              f"{type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------------
#  Client session state
# -----------------------------------------------------------------------------

class ClientSession:
    """
    Holds all mutable state for one connected WebSocket client.

    Each session owns its own Hands instance to guarantee thread-safety
    (MediaPipe native objects are NOT safe to share across threads).
    """

    def __init__(self, ws):
        self.ws = ws

        # -- Sign-recognition streak tracking --------------------------------
        self.current_streak        = 0
        self.last_predicted_letter = None
        self.confirmed_letter      = None

        # -- Practice mode ("تعلم مع ترجمان") --------------------------------
        # When set, a completed gesture is graded against the DTW reference for
        # this label instead of being classified and appended as translated
        # text. None means normal translation mode.
        self.practice_target: str | None = None

        # -- Live mirror ------------------------------------------------------
        # When on, every frame's bone directions are streamed to the avatar so
        # the robot copies the user in real time. Throttled separately from
        # recognition: the model wants every frame, the eye does not.
        self.mirror_live = False
        self.mirror_next_at = 0.0

        # -- Gesture segmentation (segmented mode) ---------------------------
        # Detects the real start/end of a sign so inference runs once per
        # gesture instead of once per frame.
        self.segmenter = GestureSegmenter()
        self.was_capturing = False       # for capture-state change events

        # -- Sliding window (legacy mode only) -------------------------------
        # Holds the last 30 extracted feature frames (each VALS_PER_FRAME
        # floats). When full, the deque is flattened into a TOTAL_FEATURES
        # vector for ONNX inference.
        self.frames_window = collections.deque(maxlen=SEQUENCE_LENGTH)

        # -- Camera-loop task handle ------------------------------------------
        # Guarded by the single asyncio thread — no locking needed.
        self.camera_task = None   # asyncio.Task | None

        # -- Per-session body anchors -----------------------------------------
        # Location is a defining parameter of a sign, so hands are expressed
        # relative to the body rather than to the picture. Pose is throttled
        # internally; one tracker per session because MediaPipe graph objects
        # are not safe to share across threads.
        self.pose_tracker = PoseTracker()

        # -- Per-session MediaPipe Hands --------------------------------------
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,        # fastest; lightest on CPU
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset_streak(self) -> None:
        """Reset per-letter streak counters, the window and the segmenter."""
        self.current_streak        = 0
        self.last_predicted_letter = None
        self.confirmed_letter      = None
        self.frames_window.clear()
        self.segmenter.reset()
        self.was_capturing = False

    def close(self) -> None:
        """Release native MediaPipe resources. Call on disconnect."""
        try:
            self.hands.close()
        except Exception as exc:
            print(f"[!]  Error closing MediaPipe Hands: {type(exc).__name__}: {exc}")
        try:
            self.pose_tracker.close()
        except Exception as exc:
            print(f"[!]  Error closing PoseTracker: {type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------------
#  Feature extraction helpers
# -----------------------------------------------------------------------------

# Feature extraction lives in feature_extractor.py — imported at the top of
# this file so the collector, the server and the migration script can never
# drift apart. Do not re-implement it here.


# -----------------------------------------------------------------------------
#  Frame processing (sync — runs in executor thread)
# -----------------------------------------------------------------------------

# De-duplication state for per-frame error reporting (see except block below).
# process_frame_sync runs ~30×/second, so raw printing would flood the console.
_frame_error_state = {"signature": None, "count": 0}
_FRAME_ERROR_REPEAT_EVERY = 100


def _encode_preview(frame: np.ndarray, results) -> bytes | None:
    """
    Downscale the (already mirrored) frame to a small JPEG for the UI.

    Optionally draws the detected hand skeleton so the user can see exactly
    what the tracker sees — far more informative than a bare video feed when
    diagnosing "why isn't it recognising me?".

    Returns the JPEG bytes, or None if encoding failed.
    """
    try:
        small = cv2.resize(
            frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_AREA
        )

        if PREVIEW_SKELETON and results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    small, landmarks, mp_hands.HAND_CONNECTIONS,
                    _preview_dot_spec, _preview_line_spec,
                )

        ok, buf = cv2.imencode(
            ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_QUALITY]
        )
        return buf.tobytes() if ok else None

    except Exception as exc:
        print(f"[!]  Preview encode failed: {type(exc).__name__}: {exc}")
        return None


def _classify(flattened: np.ndarray) -> tuple[str, float]:
    """
    Run one ONNX inference on a (1, TOTAL_FEATURES) float32 vector.

    Returns
    -------
    (label, confidence)
        label      : raw English label, or "Unknown" below CONFIDENCE_THRESHOLD
        confidence : top probability, always reported so the caller can show
                     the user *how* uncertain a rejection was
    """
    probas = onnx_session.run(
        [onnx_prob_name], {onnx_input_name: flattened}
    )[0][0]

    confidence = float(np.max(probas))
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence

    # LabelEncoder produces sorted integer classes 0..n-1, so the probabilities
    # column index IS the class index (validated at startup).
    predicted_idx = int(np.argmax(probas))
    return label_map.get(str(predicted_idx), str(predicted_idx)), confidence


def process_frame_sync(frame: np.ndarray,
                       loop: asyncio.AbstractEventLoop,
                       session: ClientSession,
                       want_preview: bool = False) -> dict:
    """
    Run MediaPipe Hands on a single BGR frame (synchronous / blocking).

    For every frame:
      1. Mirror the frame (prepare_frame) so orientation matches recording.
      2. Extract exactly VALS_PER_FRAME (126) hybrid values:
         [left hand 63] + [right hand 63], each = raw wrist (3) + 20 landmarks
         relative to the wrist (60). A missing hand contributes 63 zeros.
      3. Append the vector to the sliding window deque.
      4. When the deque reaches SEQUENCE_LENGTH frames, flatten ->
         (1, TOTAL_FEATURES) float32 and run ONNX inference.

    Dispatched via loop.run_in_executor() so heavy CPU work does not stall
    the asyncio event loop.

    Returns
    -------
    dict with keys:
        body_visible  : bool  — deprecated; mirrors hands_visible (see below)
        hands_visible : bool
        messages      : list[dict]  — zero or more JSON payloads to send
    """
    result_payload = {
        "body_visible":  False,
        "hands_visible": False,
        "messages":      [],
        "preview":       None,     # JPEG bytes, or None when not due this frame
    }

    try:
        # Mirror BEFORE processing — identical to data_collector.py.
        # Skipping this would swap MediaPipe's Left/Right handedness labels
        # relative to the recorded training data.
        frame = prepare_frame(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = session.hands.process(rgb)
        # Same RGB frame for both models, so they agree on the world.
        anchors = session.pose_tracker.update(rgb)
        rgb.flags.writeable = True

        # -- 0. Preview (throttled) -------------------------------------------
        # Built here, inside the executor thread, so JPEG encoding never runs
        # on the event loop. Recognition below is unaffected by the throttle.
        if PREVIEW_ENABLED and want_preview:
            result_payload["preview"] = _encode_preview(frame, results)

        # -- 1. Hand visibility -----------------------------------------------
        left_hand, right_hand = split_hands(results)
        has_any_hand = (left_hand is not None) or (right_hand is not None)

        if has_any_hand:
            result_payload["hands_visible"] = True

        # `body_visible` reports something real again: whether Pose located the
        # shoulders. When it is False the hands are being measured against the
        # picture instead of the body, so location-based signs (forehead vs.
        # chin) cannot be told apart — worth surfacing to the user.
        result_payload["body_visible"] = bool(anchors.valid)

        # -- 2. Distance feedback (now hand-based, Pose is gone) -------------
        distance = estimate_distance(results)
        if distance is not None:
            result_payload["messages"].append({
                "type":     "pose_data",     # type name kept for compatibility
                "distance": distance,
            })

        # -- 3. Hybrid feature extraction (126 values per frame) -------------
        # Single source of truth: feature_extractor.extract_frame_features
        frame_features = extract_frame_features(results, anchors)

        # -- Live mirror: stream bone directions to the avatar --------------
        if session.mirror_live and has_any_hand:
            now_t = time.monotonic()
            if now_t >= session.mirror_next_at:
                session.mirror_next_at = now_t + (1.0 / MIRROR_FPS)
                try:
                    result_payload["messages"].append({
                        "type": "live_pose",
                        "pose": frame_to_bone_dirs(frame_features),
                        "body": bool(anchors.valid),
                    })
                except Exception as exc:
                    print(f"[!]  live_pose failed: {type(exc).__name__}: {exc}")

        recognition_ready = onnx_session is not None and label_map is not None

        if RECOGNITION_MODE == "segmented":
            # -- 4a. Event-driven: classify ONCE per detected gesture -------
            # Pass a real timestamp so gesture DURATION is measured rather than
            # inferred — several signs differ only by how fast they are made.
            captured = session.segmenter.update(
                frame_features, has_any_hand, now=time.monotonic()
            )
            sequence = captured["sequence"] if captured else None

            # Tell the client when a sign is being captured, so the UI can show
            # "recording" feedback instead of leaving the user guessing.
            if session.segmenter.is_capturing != session.was_capturing:
                session.was_capturing = session.segmenter.is_capturing
                result_payload["messages"].append({
                    "type":      "capture_state",
                    "capturing": session.was_capturing,
                })

            # -- Practice mode: grade the attempt, don't translate it -------
            # DTW compares raw pose trajectories, so it uses the resampled
            # sequence only — the global block is a classifier input.
            if sequence is not None and session.practice_target is not None:
                verdict = dtw_library.score(session.practice_target, sequence)
                if verdict is None:
                    result_payload["messages"].append({
                        "type":    "practice_result",
                        "error":   "no_reference",
                        "target":  session.practice_target,
                        "message": "لا يوجد مرجع مسجَّل لهذه الإشارة.",
                    })
                else:
                    result_payload["messages"].append({
                        "type": "practice_result", **verdict,
                    })
                sequence = None      # consumed; skip the translation path

            if sequence is not None and recognition_ready:
                # Per-frame block + the global block that survives resampling
                flattened = np.concatenate([
                    sequence.reshape(-1),
                    np.asarray(captured["globals"], dtype=np.float32),
                ]).reshape(1, TOTAL_FEATURES).astype(np.float32)
                predicted_raw, confidence = _classify(flattened)

                if predicted_raw != "Unknown":
                    result_payload["messages"].append({
                        "type":       "letter",
                        "value":      LABEL_TO_ARABIC.get(predicted_raw, predicted_raw),
                        "confidence": round(confidence, 3),
                    })
                else:
                    # A sign WAS performed but did not clear the threshold.
                    # Silence here is ambiguous for a Deaf user — they cannot
                    # tell "camera didn't see me" from "sign not recognised".
                    # Say so explicitly instead.
                    result_payload["messages"].append({
                        "type":       "unrecognized",
                        "confidence": round(confidence, 3),
                    })

        else:
            # -- 4b. Legacy sliding window: re-classify every frame ---------
            session.frames_window.append(frame_features)

            if recognition_ready and len(session.frames_window) == SEQUENCE_LENGTH:
                flattened = np.array(
                    session.frames_window, dtype=np.float32
                ).flatten().reshape(1, TOTAL_FEATURES)

                predicted_raw, _confidence = _classify(flattened)
                arabic_letter = LABEL_TO_ARABIC.get(predicted_raw, predicted_raw)

                # Stability counter — only meaningful in this continuous mode
                if predicted_raw == session.last_predicted_letter:
                    session.current_streak += 1
                else:
                    session.current_streak        = 1
                    session.last_predicted_letter = predicted_raw

                if (session.current_streak >= FRAME_STABILITY_LIMIT
                        and predicted_raw != "Unknown"
                        and predicted_raw != session.confirmed_letter):
                    session.confirmed_letter = predicted_raw
                    result_payload["messages"].append(
                        {"type": "letter", "value": arabic_letter}
                    )

        # If neither hand is visible, reset the streak, window and segmenter
        if not has_any_hand:
            session.reset_streak()

    except Exception as e:
        # Never fail silently — but never spam the console either.
        # This runs up to 30×/second, so an unhandled error would otherwise
        # produce thousands of identical lines and bury the real cause.
        # Strategy: print the FIRST occurrence in full (with traceback), then
        # collapse repeats of the same error into a periodic counter.
        signature = f"{type(e).__name__}: {e}"

        if signature != _frame_error_state["signature"]:
            _frame_error_state["signature"] = signature
            _frame_error_state["count"] = 1
            print(f"\n[FAIL]  Error processing frame: {signature}")
            traceback.print_exc()
            print(
                "    -> This will be reported every "
                f"{_FRAME_ERROR_REPEAT_EVERY} occurrences while it persists.\n"
            )
        else:
            _frame_error_state["count"] += 1
            if _frame_error_state["count"] % _FRAME_ERROR_REPEAT_EVERY == 0:
                print(
                    f"[FAIL]  Error processing frame (still failing, "
                    f"×{_frame_error_state['count']}): {signature}"
                )

    return result_payload


# -----------------------------------------------------------------------------
#  Async camera loop — the core background task
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#  Single-camera arbitration
# -----------------------------------------------------------------------------
#
# There is exactly ONE physical camera. Previously every client session built
# its own SmartCamera, so a second client connecting and pressing "start"
# would fight the first for the device: on Linux the second open typically
# fails, on Windows it can yield black frames — and neither client was told
# anything. This makes the constraint explicit and refuses politely instead.
_camera_owner = None      # ClientSession currently holding the camera


async def camera_loop(ws, session: ClientSession) -> None:
    """
    Async background task: open the camera, read frames in a tight loop,
    process each frame off-thread, and emit tracking + recognition events.

    Lifecycle
    ---------
    • Started via asyncio.create_task() when the client sends start_camera.
    • Terminated (asyncio.CancelledError) when:
        – client sends stop_camera
        – client disconnects (register() finally block cancels the task)
    • Camera is ALWAYS released in the finally block regardless of cause.

    Non-blocking guarantee
    ----------------------
    cam.read() is a synchronous blocking call (OpenCV or Picamera2).
    It is dispatched through loop.run_in_executor(None, cam.read) so the
    asyncio event loop stays free to handle other coroutines between frames.
    """
    global _camera_owner

    # Refuse rather than fight over the single physical device
    if _camera_owner is not None and _camera_owner is not session:
        print("[CameraLoop] Refused — camera already in use by another client.")
        await send_to_client(ws, {
            "type":    "error",
            "code":    "camera_busy",
            "message": "الكاميرا مستخدَمة من جلسة أخرى. أغلِق الجلسة الأخرى ثم أعد المحاولة.",
        })
        return

    _camera_owner = session

    cam  = SmartCamera(source=_active_camera_source)
    loop = asyncio.get_event_loop()

    cam.start()
    # Decode frames in the background. read() then returns the NEWEST frame
    # instead of blocking until the sensor produces one, so capture and
    # MediaPipe inference overlap rather than running end to end.
    cam.start_grabber()

    if not cam.is_running:
        _camera_owner = None
        await send_to_client(ws, {
            "type":    "error",
            "code":    "camera_failed",
            "message": "تعذَّر فتح الكاميرا في الخادم.",
        })
        return

    print(f"[CameraLoop] Started — backend: {cam.backend}")

    preview_interval = 1.0 / PREVIEW_FPS if PREVIEW_FPS > 0 else 0.0
    next_preview_at  = 0.0

    try:
        while True:
            # Offload the blocking cam.read() call to a thread pool worker
            ret, frame = await loop.run_in_executor(None, cam.read)

            if not ret or frame is None:
                # Transient read failure: yield control and retry
                await asyncio.sleep(0)
                continue

            # Preview is throttled independently of recognition: the model
            # still sees every frame, the UI only needs ~12 fps.
            now = loop.time()
            want_preview = PREVIEW_ENABLED and now >= next_preview_at
            if want_preview:
                next_preview_at = now + preview_interval

            # Offload heavy MediaPipe + model inference to a thread pool worker
            result = await loop.run_in_executor(
                None, process_frame_sync, frame, loop, session, want_preview
            )

            # Emit lightweight tracking heartbeat (every frame, very cheap)
            await send_to_client(ws, {
                "type":          "tracking_status",
                "body_visible":  result["body_visible"],
                "hands_visible": result["hands_visible"],
            })

            # Ship the preview frame, if one was produced
            if result["preview"]:
                try:
                    await ws.send(bytes([BIN_PREVIEW]) + result["preview"])
                except websockets.exceptions.ConnectionClosed:
                    pass      # client vanished; the loop will notice shortly
                except Exception as exc:
                    print(f"[!]  preview send failed: {type(exc).__name__}: {exc}")

            # Emit any recognition events produced this frame (letter, pose_data)
            for msg in result["messages"]:
                await send_to_client(ws, msg)

            # Yield control to the event loop between frames
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        # Normal cancellation path — must re-raise so asyncio marks task done
        print("[CameraLoop] Cancelled gracefully.")
        raise

    finally:
        # Guaranteed cleanup regardless of exit reason
        cam.stop_grabber()
        cam.release()
        if _camera_owner is session:
            _camera_owner = None      # free the device for the next client
        print("[CameraLoop] Camera released.")


# -----------------------------------------------------------------------------
#  Camera task lifecycle helpers
# -----------------------------------------------------------------------------

async def _stop_camera_task(session: ClientSession) -> None:
    """
    Cancel the running camera task and await its completion.
    Safe to call even when no task is running (idempotent).
    """
    task = session.camera_task
    if task is None or task.done():
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass      # expected: this is how the loop is asked to stop
    except Exception as exc:
        print(f"[!]  camera loop ended with an error: "
              f"{type(exc).__name__}: {exc}")
    finally:
        session.camera_task = None


# -----------------------------------------------------------------------------
#  AI helpers (unchanged from original)
# -----------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    """Strip markdown formatting from AI reply text (for clean TTS input)."""
    text = re.sub(r'\*{1,3}|_{1,3}', '', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'`{1,3}', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'  +', ' ', text).strip()
    return text


async def ask_turjuman_ai(user_question: str, history: list) -> str:
    """
    Ask the OpenRouter model and return the assistant's reply as plain text.

    No JSON parsing, no regex recovery ladder — the model is asked for plain
    Arabic prose, so the "parsing" is just markdown cleanup for the TTS engine.
    """
    try:
        valid_roles  = {"user", "assistant"}
        safe_history = [
            {"role": t["role"], "content": str(t["content"])}
            for t in history
            if isinstance(t, dict)
               and t.get("role") in valid_roles
               and isinstance(t.get("content"), str)
               and t["content"].strip()
        ]
        messages = [
            # Rebuilt per request so the clock is always current
            {"role": "system", "content": build_system_prompt()},
            *safe_history,
            {"role": "user",   "content": user_question},
        ]
        completion = await or_client.chat.completions.create(
            model=AI_MODEL,
            # Small models drift and hallucinate more as temperature rises.
            # 0.2 keeps `sonar` anchored to the facts it was given / found.
            temperature=0.2,
            messages=messages,
        )

        raw = completion.choices[0].message.content or ""
        raw = re.sub(r'\[\d+\]', '', raw)      # strip citation markers [1], [2]
        reply = _clean_markdown(raw)

        return reply or "لا يوجد رد."

    except Exception as exc:
        print(f"[!]  AI request failed: {type(exc).__name__}: {exc}")
        return "عُذراً، حَدَثَ خطَأْ في الاتِّصالِ."


# -----------------------------------------------------------------------------
#  TTS helper
# -----------------------------------------------------------------------------

async def generate_and_send_speech(text: str, ws) -> None:
    """
    Synthesise Arabic speech via edge-tts and stream the MP3 bytes to the
    client as a single binary message prefixed with the 0x01 header byte.

    bytearray.extend() is O(N) — avoids the O(N²) cost of b'' += chunk.
    """
    try:
        communicate = edge_tts.Communicate(text, "ar-SA-HamedNeural")
        audio_data  = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        if audio_data:
            await ws.send(bytes([BIN_AUDIO]) + bytes(audio_data))
        else:
            print("[!]  TTS produced no audio for the requested text.")
    except websockets.exceptions.ConnectionClosed:
        pass      # client disconnected while speech was being synthesised
    except Exception as exc:
        # TTS failing silently made the app look "mute" for no visible reason.
        print(f"[!]  TTS failed: {type(exc).__name__}: {exc}")
        await send_to_client(ws, {
            "type":    "error",
            "code":    "tts_failed",
            "message": "تعذَّر توليد الصوت.",
        })


# -----------------------------------------------------------------------------
#  STT helper — runs in executor thread
# -----------------------------------------------------------------------------

GROQ_STT_URL     = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL   = "whisper-large-v3"
GROQ_STT_TIMEOUT = 15      # seconds, total


async def run_stt(audio_bytes: bytes, purpose: str, ws) -> None:
    """
    Transcribe audio via Groq Whisper — fully async.

    Previously this used the synchronous `requests` library inside a thread
    pool. That works, but it burns a worker thread for the entire round trip
    and competes with the frame-processing executor for the SAME thread pool —
    on a Raspberry Pi that directly steals capacity from sign recognition.
    aiohttp keeps the whole call on the event loop instead.
    """
    transcript = ""
    try:
        form = aiohttp.FormData()
        form.add_field(
            "file", audio_bytes,
            filename="audio.webm", content_type="audio/webm",
        )
        form.add_field("model", GROQ_STT_MODEL)
        form.add_field("language", "ar")

        timeout = aiohttp.ClientTimeout(total=GROQ_STT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                data=form,
            ) as response:
                if response.status == 200:
                    payload = await response.json()
                    transcript = (payload.get("text") or "").strip()
                else:
                    body = (await response.text())[:200]
                    print(f"[!]  Groq STT returned {response.status}: {body}")

    except asyncio.TimeoutError:
        print(f"[!]  Groq STT timed out after {GROQ_STT_TIMEOUT}s")
    except Exception as exc:
        print(f"[!]  Groq STT failed: {type(exc).__name__}: {exc}")

    await send_to_client(
        ws, {"type": "stt_result", "purpose": purpose, "value": transcript}
    )


# -----------------------------------------------------------------------------
#  WebSocket connection handler
# -----------------------------------------------------------------------------

async def register(websocket) -> None:
    """
    Handle one WebSocket client for its entire lifetime.

    Control flow
    ------------
    • start_camera  -> launch camera_loop as an asyncio Task (one at a time)
    • stop_camera   -> cancel + await the Task; camera released in finally
    • speak         -> fire-and-forget TTS task
    • user_question -> fire-and-forget AI query task
    • Binary 0x01   -> STT for translation (executor thread)
    • Binary 0x02   -> STT for chat (executor thread)
    • Disconnect    -> finally block cancels camera task + closes Hands
    """
    session = ClientSession(websocket)
    loop    = asyncio.get_event_loop()

    try:
        async for raw_msg in websocket:

            # -- Binary protocol (audio only) ---------------------------------
            if isinstance(raw_msg, bytes):
                header  = raw_msg[0]
                payload = raw_msg[1:]
                if header == 0x01:
                    asyncio.create_task(run_stt(payload, "translate", websocket))
                elif header == 0x02:
                    asyncio.create_task(run_stt(payload, "chat", websocket))
                continue

            # -- JSON text protocol -------------------------------------------
            try:
                msg      = json.loads(raw_msg)
                msg_type = msg.get("type", "")
            except (json.JSONDecodeError, Exception):
                continue

            # -- Camera control commands --------------------------------------
            if msg_type == "start_camera":
                # Guard: only one camera loop per session at a time
                if session.camera_task and not session.camera_task.done():
                    print("[register] Camera already running — ignoring duplicate start.")
                    continue
                print("[register] Launching camera loop...")
                session.camera_task = asyncio.create_task(
                    camera_loop(websocket, session)
                )

            elif msg_type == "list_cameras":
                # Probing device indices takes a second or two and blocks, so
                # it runs off the event loop — otherwise the whole app freezes
                # while it scans, including the preview stream.
                await send_to_client(websocket, {"type": "camera_scan_started"})
                loop_ = asyncio.get_event_loop()
                try:
                    found = await loop_.run_in_executor(None, describe_cameras)
                except Exception as exc:
                    print(f"[!]  camera scan failed: {type(exc).__name__}: {exc}")
                    found = []

                options = []
                for cam in found:
                    # Windows supplies the real product name; where it does not,
                    # fall back to the index heuristic rather than inventing one.
                    integrated = cam["kind"] == "integrated"
                    options.append({
                        "source": str(cam["index"]),
                        "label":  ("الكاميرا المدمجة" if integrated
                                   else "كاميرا USB خارجية"),
                        "detail": (f"{cam['name']} — {cam['width']}×{cam['height']}"
                                   if cam.get("name")
                                   else f"index {cam['index']} — "
                                        f"{cam['width']}×{cam['height']}"),
                        "kind":   cam["kind"],
                    })
                options.append({
                    "source": droidcam_url(),
                    "label":  "الهاتف (DroidCam)",
                    "detail": droidcam_url(),
                    "kind":   "network",
                })
                options.append({
                    "source": "auto",
                    "label":  "اختيار تلقائي",
                    "detail": "يجرِّب المتاح بالترتيب",
                    "kind":   "auto",
                })

                await send_to_client(websocket, {
                    "type":    "camera_list",
                    "cameras": options,
                    "active":  str(_active_camera_source),
                })

            elif msg_type == "set_camera":
                # Switching source means reopening the device. If the camera is
                # live it is stopped and restarted, so the user sees the new
                # feed immediately instead of having to toggle it themselves.
                new_source = str(msg.get("source", "")).strip()
                if not new_source:
                    continue

                was_running = bool(session.camera_task
                                   and not session.camera_task.done())
                if was_running:
                    await _stop_camera_task(session)

                globals()["_active_camera_source"] = new_source
                print(f"[register] camera source -> {new_source}")

                await send_to_client(websocket, {
                    "type":   "camera_source_changed",
                    "source": new_source,
                })

                if was_running:
                    session.camera_task = asyncio.create_task(
                        camera_loop(websocket, session)
                    )

            elif msg_type == "stop_camera":
                print("[register] stop_camera received — shutting down loop.")
                await _stop_camera_task(session)

            # -- Other commands -----------------------------------------------
            # -- Practice mode control ----------------------------------------
            elif msg_type == "start_practice":
                target = str(msg.get("target", "")).strip()
                if target in dtw_library.references:
                    session.practice_target = target
                    session.reset_streak()      # clear any half-captured gesture
                    print(f"[register] Practice mode ON — target: {target}")
                    await send_to_client(websocket, {
                        "type":   "practice_state",
                        "active": True,
                        "target": target,
                        "label":  LABEL_TO_ARABIC.get(target, target),
                    })
                else:
                    await send_to_client(websocket, {
                        "type":    "error",
                        "code":    "unknown_sign",
                        "message": f"لا توجد إشارة مرجعية باسم «{target}».",
                    })

            elif msg_type == "stop_practice":
                session.practice_target = None
                session.reset_streak()
                await send_to_client(websocket, {
                    "type": "practice_state", "active": False, "target": None,
                })

            elif msg_type == "list_signs":
                await send_to_client(websocket, {
                    "type": "sign_list",
                    "signs": [
                        {"id": lbl, "label": LABEL_TO_ARABIC.get(lbl, lbl)}
                        for lbl in dtw_library.labels
                    ],
                })

            elif msg_type == "start_mirror":
                session.mirror_live = True
                session.mirror_next_at = 0.0
                print("[register] Live mirror ON")
                await send_to_client(websocket, {"type": "mirror_state", "active": True})

            elif msg_type == "stop_mirror":
                session.mirror_live = False
                print("[register] Live mirror OFF")
                await send_to_client(websocket, {"type": "mirror_state", "active": False})

            elif msg_type == "speak":
                text = msg.get("value", "").strip()
                if text:
                    asyncio.create_task(generate_and_send_speech(text, websocket))

            elif msg_type == "user_question":
                question = msg.get("data", "").strip()
                history  = msg.get("history", [])
                if not isinstance(history, list):
                    history = []
                if question:
                    async def _handle_ai(q: str, h: list, ws):
                        # `data` is now a plain string (was {"reply": "..."}).
                        # The frontend's extractReply() accepts both.
                        reply = await ask_turjuman_ai(q, h)
                        await send_to_client(ws, {"type": "ai_response", "data": reply})
                    asyncio.create_task(_handle_ai(question, history, websocket))

    finally:
        # Always runs on disconnect or unhandled exception
        print("[register] Client disconnected — cleaning up.")
        await _stop_camera_task(session)   # cancel camera loop + release camera
        session.close()                     # release MediaPipe Hands


# -----------------------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------------------

async def main() -> None:
    # Ask BEFORE the server starts listening. The desktop app waits on port
    # 8765, so it simply waits through the prompt instead of opening against a
    # camera the user has not chosen yet.
    #
    # Skipped automatically when CAMERA_SOURCE is set in .env, or when there is
    # no terminal attached — see choose_camera_interactive().
    global _active_camera_source
    _active_camera_source = choose_camera_interactive()

    print("=" * 60)
    print("  Tarjuman WebSocket Server — Headless Camera Mode")
    print(f"  ws://{BIND_HOST}:{BIND_PORT}")
    print("  Access : LOCAL ONLY (loopback — not reachable from the network)")
    print(f"  Camera : {_active_camera_source}  (changeable from the app)")
    print("  Video  : DISABLED (zero-bandwidth architecture)")
    print("=" * 60)

    server = await websockets.serve(
        register,
        BIND_HOST, BIND_PORT,
        reuse_address=True,
        ping_interval=30,
        ping_timeout=10,
    )
    print("RDDY[OK]")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
