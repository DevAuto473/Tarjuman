"""
data_collector.py — Dynamic Gesture Sequence Recorder for Tarjuman
===================================================================
Captures sliding-window gesture sequences and saves them to
`dynamic_gestures.csv` for training the hybrid DTW/Random Forest model.

Recording architecture
----------------------
  • One sequence  = ضغطةُ `r` تُسلّح، ثمّ يقتطع `GestureSegmenter` ما بين
                    السكون والسكون، ويُعاد تشكيله إلى SEQUENCE_LENGTH (30).
                    الإيقاعُ بيدك، والقصُّ بيد المقطِّع نفسه الذي يستعمله
                    الخادم وقت التعرّف — فيرى التدريبُ والاستدلال الإشارةَ
                    بالعين ذاتها.
  • One CSV row   = label  +  flattened features of all 30 frames
  • Feature set per frame (hands only — see feature_extractor.py):
        - Left hand  : 63 values (raw wrist x,y,z + 20 landmarks relative)
        - Right hand : 63 values (same layout)
        = 126 values / frame
        × 30 frames  = 3 780 values per row  (+1 label column)

  The geometry above is NOT defined here — it is imported from
  feature_extractor.py so the collector and the live server can never drift.

Controls
--------
  r   ->  عيّنةٌ واحدة (وضغطةٌ ثانية تُلغي ما لم يبدأ بعد)
  p   ->  إيقاف مؤقّت
  c   ->  إعادة المعايرة
  q   ->  إنهاء الكلمة الحالية
"""

# -- Import bootstrap ---------------------------------------------------------
# Puts src/ on the path so `tarjuman_core` resolves when this file is run
# directly (`python data_collector.py`). Running through `npm run ...` sets PYTHONPATH
# instead, and `pip install -e .` makes both unnecessary - this is the belt to
# those braces, so a plain `python` invocation never fails with ImportError.
import argparse
import os as _os
import platform as _platform
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.basename(_os.path.dirname(_os.path.abspath(__file__))) == "scripts"
    else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "src"))

import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# Import our smart camera abstraction (works on laptop and Raspberry Pi)
from tarjuman_core.paths import dataset_csv
from tarjuman_core.camera_manager import (
    SmartCamera, choose_camera_interactive, _stdin_readable,
)

# Feature geometry + extraction — single source of truth shared with the server
from tarjuman_core.feature_extractor import (
    GLOBAL_FEATURE_NAMES,
    compute_global_features,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    VALS_PER_FRAME,
    PoseTracker,
    extract_frame_features,
    prepare_frame,
    split_hands,
)
from tarjuman_core.gesture_segmenter import GestureSegmenter, resample_sequence
from tarjuman_core.take_quality import (
    analyse as analyse_take,
    spread,
    suggest_samples,
    take_distance,
)


# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------

OUTPUT_CSV = dataset_csv()   # اسمها في tarjuman_core/paths.py

# Samples per label before a class is considered "done".
# 30+ per class is the realistic floor for a 100-class problem; below that the
# model memorises rather than generalises.
TARGET_PER_LABEL = 50


# -----------------------------------------------------------------------------
#  MediaPipe Hands setup
# -----------------------------------------------------------------------------
# Hands ONLY. Holistic additionally ran BlazePose (33 pts) and Face Mesh
# (468 pts) every frame — the face landmarks were never used at all.

# Hands is constructed at import time here, so the compatibility check has to
# come first - otherwise a protobuf mismatch surfaces as a traceback from
# inside MediaPipe before any of this file's own output appears.
from tarjuman_core.runtime_check import check_mediapipe_stack
check_mediapipe_stack()

mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Body anchors. Location is a defining parameter of a sign (forehead vs. chin),
# so the hands are recorded relative to the body rather than to the picture.
# Pose is throttled internally — see PoseTracker.
#
# Created lazily: building it loads a MediaPipe model, and doing that at import
# time means merely importing this module for a helper function pays the cost —
# and fails outright if the model cannot be loaded.
_pose_tracker = None


def get_pose_tracker() -> PoseTracker:
    global _pose_tracker
    if _pose_tracker is None:
        _pose_tracker = PoseTracker()
    return _pose_tracker


# -----------------------------------------------------------------------------
#  Landmark extraction
# -----------------------------------------------------------------------------
# Deliberately NOT implemented here — see feature_extractor.py. Duplicating
# the layout between collector and server is exactly how a silent
# train/inference mismatch gets introduced.

def extract_frame_landmarks(results, anchors=None) -> np.ndarray:
    """Thin wrapper: shared extractor -> float32 array of (VALS_PER_FRAME,)."""
    return np.array(extract_frame_features(results, anchors), dtype=np.float32)


# -----------------------------------------------------------------------------
#  CSV helper
# -----------------------------------------------------------------------------

# -- كيف تُقتطَع الإشارة -------------------------------------------------------
# كان المسجِّل يقتطع بنافذةٍ زمنية ثابتة: تضغط `r`، فيسجّل ثانيتين، وما وقع
# خارجهما ضاع. والخادمُ وقت التعرّف لا يفعل ذلك أبداً — هو يستعمل
# `GestureSegmenter` الذي يرقب السكون والحركة فيعرف أين تبدأ الإشارة وأين
# تنتهي.
#
# فكان النموذج يُدرَّب على مقاطع مقصوصة بمِقصٍّ، ويُستدَلّ عليه بمقاطع مقصوصة
# بمِقصٍّ آخر. هذا هو التفاوت الصامت بين التدريب والاستدلال، وأشدّ ما فيه أنّه
# لا يظهر في أيّ اختبار: دقّة التدريب ممتازة، والأداء الحقيقي رديء.
#
# الآن يستعمل الاثنان المقطِّع نفسه. وسقط معه كلُّ ما كان يخدم النافذة:
# RECORD_SECONDS و MIN_CAPTURE_FRAMES و MAX_CAPTURE_FRAMES و CALIBRATION_SECONDS
# — لم تعد هناك نافذةٌ يُضبَط طولها.

# -- التسليح والارتداد ---------------------------------------------------------
# `r` تُسلّح المسجِّل، ثمّ المقطِّع هو من يقرّر أين تبدأ الإشارة وأين تنتهي.
# لكنّ المقطِّع يحتاج حركةً ليُدرك البداية، وبعض الإشارات صغيرة السَّعة بطبعها
# — 'sorry' مثلاً دائرةٌ على الصدر مسارها 0.68 — فقد لا يراها تبدأ أصلاً.
#
# فلو اكتفينا بالمقطِّع وحده، لوقف المستخدم يضغط `r` ولا يُستجاب له، ولا يعرف
# لماذا. ARM_GRACE هي المهلة التي بعدها نكفّ عن انتظاره ونسجّل بالنافذة
# المُعايَرة، ونُعلِم أنّ هذه العيّنة جاءت بالارتداد لا بالمقطِّع.
ARM_GRACE = float(_os.getenv("TARJUMAN_ARM_GRACE", "2.5"))

# ونافذةُ الارتداد حين لا معايرة بعد (أثناء المعايرة نفسها).
FALLBACK_SECONDS = 3.0
MIN_FALLBACK_FRAMES = 8
MAX_FALLBACK_FRAMES = 300

# -- المعايرة ------------------------------------------------------------------
# قبل خمسين عيّنة، تُؤدّى الكلمة ثلاثاً. ولماذا ثلاثاً لا واحدة: تسجيلةٌ واحدة
# تقديرٌ هشّ — إن جاءت أسرع أو أبطأ من عادتك بُني عليها كلُّ ما بعدها. الوسيط
# بين ثلاث يُهمل الشاذّة منها.
CALIBRATION_TAKES = 3

# -- الحكم على الشاذّ ----------------------------------------------------------
# كلّ عيّنة تُقاس بـDTW إلى مرجع الكلمة؛ فإن بعُدت أكثر من الحدّ رُدَّت. والحدّ
# لا يُفرَض رقماً، بل يُشتقّ من تباعد التسجيلات نفسها: مَن كان أداؤه ثابتاً ضاق
# حدُّه، ومَن كان متنوّعاً اتّسع.
#
# العددان مقيسان على 388 تسجيلة حقيقية من dynamic_gestures_v4: بمعامل 2.40
# يُقبَل ما نسبته 90% من تسجيلات الكلمة الواحدة، ويظلّ الحدّ دون أقرب كلمةٍ
# أخرى في أغلب الحالات. والأرضيةُ 1.80 تحمي من معايرةٍ جاءت متقاربةً بالصدفة
# فأنتجت حدّاً خانقاً — وهي الحالة التي كانت تردّ رُبع التسجيلات في 'please'.
OUTLIER_FLOOR = 1.80
OUTLIER_FACTOR = 2.40

# ومع ذلك تبقى ثلاثُ تسجيلاتٍ أساساً ضيّقاً. فبعد كلّ دفعةٍ مقبولة يُعاد اشتقاق
# المرجع والحدّ من العيّنات المقبولة نفسها — عشرٌ خيرٌ من ثلاث. قِيس أثر ذلك:
# أسوأ نسبة ردٍّ في 'hello' هبطت من 17% إلى 7%.
OUTLIER_REVISE = True

# وإن ارتفعت نسبة الردّ فوق هذا، فالخلل ليس في تسجيلةٍ بعينها. إمّا أنّ
# المعايرة لم تُمثّل أداءك، وإمّا أنّ أداءك نفسه يتأرجح. البرنامج لا يستطيع
# التمييز بينهما، فلا يُخفي الأمر ولا يُصلحه صامتاً — يقولهما معاً ويترك
# القرار لك (c لإعادة المعايرة).
OUTLIER_ALERT = 0.35

# -- الإشباع -------------------------------------------------------------------
# متى تكفي العيّنات؟ لا جواب من عددٍ مجرّد. الجوابُ أنّ العيّنات الجديدة لم تعد
# تضيف تنوّعاً: إن ثبت تباعدُ آخر SATURATION_WINDOW عيّنة بين دفعةٍ والتي قبلها،
# فأنت تُعيد تسجيل ما عندك. هذا إخبارٌ لا إيقاف — القرار لك.
SATURATION_WINDOW = 20
SATURATION_CHANGE = 0.08

# -- الراحة بين الدفعات --------------------------------------------------------
# خمسون عيّنة متتابعة بلا توقّف تُنتج خمسين عيّنة متعبة. الدفعة تُقطَع بوقفةٍ
# قصيرة تُعيد فيها يديك ووضعيتك، وتُعرَض عدّاً تنازلياً في نافذة الكاميرا.
BATCH_SIZE = int(_os.getenv("TARJUMAN_BATCH", "10"))
REST_SECONDS = float(_os.getenv("TARJUMAN_REST", "3"))


def _ensure_csv_header(filepath: str) -> None:
    """
    Create the CSV with a header row if it does not exist.

    Layout:  label, f0_v0 … f29_v125, g_duration_s … g_openness_change
    The trailing global columns are what let duration and tempo survive the
    fixed-length resampling — without them, signs that differ only by speed
    become identical rows.
    """
    if os.path.exists(filepath):
        return   # Append mode: header already written

    header = ["label"]
    for frame_idx in range(SEQUENCE_LENGTH):
        for val_idx in range(VALS_PER_FRAME):
            header.append(f"f{frame_idx}_v{val_idx}")
    header += [f"g_{name}" for name in GLOBAL_FEATURE_NAMES]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    print(f"[Collector] Created CSV: {filepath}  ({len(header)} columns)")


def save_prepared(label: str, sequence, globals_, duration: float,
                  filepath: str) -> None:
    """
    تكتب إيماءةً جاهزة صفّاً في الـCSV.

    «جاهزة» تعني أنّ `GestureSegmenter` أعادها مُعادَ التشكيل إلى
    SEQUENCE_LENGTH، وحسب مُجمَلاتها (المدّة والإيقاع والاتّجاه والانفتاح) على
    الالتقاط الخام قبل إعادة التشكيل — لأنّ إعادة التشكيل هي بالضبط ما يُتلفها.

    فلا يُعاد هنا تشكيلٌ ولا حساب: إعادة تشكيل المُعاد تشكيله تُفسد، وحسابُ
    المُجمَلات بعد التشكيل يُعطي مدّةً واحدة لكلّ الكلمات. الدالّة تكتب فقط.
    """
    frames = np.asarray(sequence, dtype=np.float32)
    assert frames.shape == (SEQUENCE_LENGTH, VALS_PER_FRAME), (
        f"sequence is {frames.shape}, expected "
        f"{(SEQUENCE_LENGTH, VALS_PER_FRAME)} — was it resampled already?"
    )

    row = [label] + frames.reshape(-1).tolist() + list(globals_)

    assert len(row) == TOTAL_FEATURES + 1, (
        f"row has {len(row)} values, expected {TOTAL_FEATURES + 1}"
    )

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# -----------------------------------------------------------------------------
#  Visual overlay helpers
# -----------------------------------------------------------------------------

# Colour palette
_GREEN  = (50,  220,  50)
_RED    = (30,   30, 220)
_WHITE  = (240, 240, 240)
_SHADOW = (20,   20,  20)
_AMBER  = (30,  180, 230)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.65
_THICKNESS  = 2


def _put_text_with_shadow(frame, text: str, pos: tuple, color: tuple,
                           scale=_FONT_SCALE, thickness=_THICKNESS) -> None:
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), _FONT, scale, _SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),          _FONT, scale, color,   thickness,     cv2.LINE_AA)


def draw_landmarks_on_frame(frame: np.ndarray, results, pose_results=None) -> None:
    """Draw MediaPipe hand and face skeletons on the frame (in-place)."""
    hand_spec = mp_drawing.DrawingSpec(color=(105, 40, 193), thickness=2, circle_radius=2)
    left, right = split_hands(results)
    for landmarks in (left, right):
        if landmarks is not None:
            mp_drawing.draw_landmarks(
                frame, landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_spec, hand_spec,
            )
            
    if pose_results and getattr(pose_results, 'pose_landmarks', None):
        face_spec = mp_drawing.DrawingSpec(color=(151, 237, 61), thickness=1, circle_radius=1)
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            face_spec, face_spec
        )


# -----------------------------------------------------------------------------
#  Zoom via mouse scroll
# -----------------------------------------------------------------------------
# زوم بالعجلة — البُعد البؤري يتبدّل بالسكرول دون الحاجة إلى نقرة.
# _zoom_level = 1.0  → الإطار الكامل (لا زوم).
# _zoom_level = 2.0  → يُظهِر نصف العرض ونصف الارتفاع مُكبَّراً × 2.
# الحدّ العلوي 5.0 يبقي الصورة قابلة للقراءة قبل أن تصبح بيكسلات.
_zoom_level   = 1.0
_ZOOM_MIN     = 1.0
_ZOOM_MAX     = 5.0
_ZOOM_STEP    = 0.15    # حجم الخطوة في كلّ نقرة عجلة


def _on_mouse(event, x, y, flags, param) -> None:
    """معالج الماوس: السكرول للأعلى يكبِّر، للأسفل يصغِّر."""
    global _zoom_level
    if event == cv2.EVENT_MOUSEWHEEL:
        # FLAGS موجب = العجلة للأعلى = تكبير
        if flags > 0:
            _zoom_level = min(_ZOOM_MAX, _zoom_level + _ZOOM_STEP)
        else:
            _zoom_level = max(_ZOOM_MIN, _zoom_level - _ZOOM_STEP)
    # بعض إصدارات OpenCV ترفع MOUSEHWHEEL بدل MOUSEWHEEL
    elif event == cv2.EVENT_MOUSEHWHEEL:
        pass   # أفقي — لا نفعل شيئاً


def _apply_zoom(frame: np.ndarray) -> np.ndarray:
    """
    يقصّ المنطقة المركزية بنسبة _zoom_level ثمّ يُعيد تحجيمها إلى حجم الإطار الأصلي.
    النتيجة مطابقة لتضييق البُعد البؤري: لا تشويه في النسب، لا حركة في الإطار.
    """
    if _zoom_level <= 1.0:
        return frame
    h, w = frame.shape[:2]
    # أبعاد المنطقة المقصوصة
    ch = int(h / _zoom_level)
    cw = int(w / _zoom_level)
    # مركّزة على منتصف الإطار
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    cropped = frame[y1:y1 + ch, x1:x1 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _draw_zoom_indicator(frame: np.ndarray) -> None:
    """شريطٌ صغير أسفل اليمين يُظهر مستوى الزوم الحالي."""
    if _zoom_level <= 1.01:
        return   # لا حاجة للمؤشّر عند الزوم الطبيعي
    h, w = frame.shape[:2]
    label = f"x{_zoom_level:.1f}"
    scale, thick = 0.55, 1
    (tw, th), _ = cv2.getTextSize(label, _FONT, scale, thick)
    pad = 6
    x0, y0 = w - tw - pad * 2 - 4, h - th - pad * 2 - 4
    # خلفية شبه شفّافة
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 4, h - 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    _put_text_with_shadow(frame, label, (x0 + pad, h - 4 - pad), _AMBER, scale=scale, thickness=thick)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def _choose_camera():
    """Kept as a thin alias; the implementation lives in camera_manager."""
    return choose_camera_interactive()


def _existing_counts(filepath: str) -> dict:
    """How many samples each label already has, so you can see what's missing."""
    counts = {}
    if not os.path.exists(filepath):
        return counts
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    except OSError as exc:
        print(f"Could not read existing samples: {exc}")
    return counts


def _choose_label() -> str:
    """Interactive picker over vocabulary.py, with per-label progress."""
    try:
        from tarjuman_core.vocabulary import as_dicts
        vocab = as_dicts()
    except Exception as exc:
        print(f"[!]  vocabulary.py unavailable ({exc}) — falling back to free text.")
        label = input("\nEnter the gesture label: ").strip()
        if not label:
            print("[FAIL] Label cannot be empty. Exiting.")
            sys.exit(1)
        return label

    counts = _existing_counts(OUTPUT_CSV)
    total = len(vocab)
    done = sum(1 for e in vocab if counts.get(e["id"], 0) >= TARGET_PER_LABEL)

    print(f"\nVocabulary: {total} terms | complete: {done} | "
          f"target {TARGET_PER_LABEL} samples each")
    print("Arabic words are in learn.csv — terminals cannot render them correctly.\n")

    # Ids only: Windows consoles do not shape or bidi-order Arabic, so printing
    # the words here produces reversed, unreadable text.
    for i, e in enumerate(vocab, 1):
        n = counts.get(e["id"], 0)
        mark = "[x]" if n >= TARGET_PER_LABEL else ("[~]" if n else "[ ]")
        print(f"  {mark} {i:3d}. {e['id']:<17s} {n:>2d}/{TARGET_PER_LABEL}", end="")
        if i % 3 == 0:
            print()
    print("\n")

    while True:
        raw = input("Pick a number (or type the id): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= total:
            return vocab[int(raw) - 1]["id"]
        for e in vocab:
            if raw == e["id"] or raw == e["arabic"]:
                return e["id"]
        print("   [!] Not found - try again.")


# -----------------------------------------------------------------------------
#  Where am I running, and what should the camera do?
# -----------------------------------------------------------------------------
#
# The same recorder runs on two very different machines, and the camera wants
# different things from each:
#
#   laptop — a USB/UVC webcam, opened through OpenCV.
#   pi     — the CSI camera, opened through libcamera/Picamera2.
#
# Both ask for 60, and the reason is the same on each: a high frame rate is
# what a SIGN needs. It is not about having more samples — every take is
# resampled to SEQUENCE_LENGTH anyway. It is that 60 fps means each frame is
# exposed for at most 1/60 s, and a hand moving through a longer exposure
# smears. A blurred hand is a hand MediaPipe places badly, and every feature
# downstream inherits that error. The frame grabber keeps only the newest
# frame, so a faster sensor also halves how stale the frame handed to the
# tracker is, even when inference itself runs slower than the sensor.
#
# The cost of a short exposure is light, and light is a room problem, not a
# settings problem. Both paths now measure the picture and say so rather than
# recording a dark one silently: the USB path in `_force_frame_rate`, the Pi
# path in `_tune_picamera_ae`. If either reports a dark frame, the honest
# answers are a lamp in FRONT of the signer, `--fps 30` to double the exposure
# time, or `CAMERA_EV` to bias the metering brighter.
#
# The resolution is deliberately the same in both: MediaPipe normalises x by
# width and y by height SEPARATELY, so recording at one aspect ratio and
# running at another quietly distorts every landmark, and with it every feature
# derived from the body frame.

DEVICE_PROFILES = {
    "laptop": {"source": "auto",      "width": 640, "height": 480, "fps": 60},
    "pi":     {"source": "picamera2", "width": 640, "height": 480, "fps": 60},
}


def _parse_size(text: str) -> tuple:
    """`640x480` -> (640, 480). Accepts x or X as the separator."""
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"--size wants WIDTHxHEIGHT, e.g. 640x480 (got {text!r})")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="data_collector.py",
        description="Record sign-language samples for Tarjuman.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python data_collector.py                          ask which machine, then record
  python data_collector.py --device pi              Pi profile, no questions
  python data_collector.py --device pi --fps 20     Pi profile but slower (dim room)
  python data_collector.py --size 1280x720          a different sensor mode
  python data_collector.py --device laptop --camera droidcam

anything given on the command line wins over .env, and .env wins over the
device profile. The recorder prints where every value came from before it
opens the camera.""")

    ap.add_argument("--device", choices=sorted(DEVICE_PROFILES),
                    help="which machine this is; omit to be asked")
    ap.add_argument("--camera", metavar="SRC",
                    help="camera source: auto, picamera2, laptop, droidcam, "
                         "a USB index, or an http(s)/rtsp URL")
    ap.add_argument("--fps", type=int, metavar="N",
                    help="frame rate to request from the sensor")
    ap.add_argument("--width", type=int, metavar="N", help="frame width")
    ap.add_argument("--height", type=int, metavar="N", help="frame height")
    ap.add_argument("--size", type=_parse_size, metavar="WxH",
                    help="frame size in one argument, e.g. 640x480")
    ap.add_argument("--ev", type=float, metavar="STOPS",
                    help="Pi only: bias auto-exposure brighter, e.g. 1.5 when "
                         "the signer is backlit (costs noise, not frame rate)")
    ap.add_argument("--samples", type=int, metavar="N",
                    help=f"samples per word (default {TARGET_PER_LABEL})")
    ap.add_argument("--yes", action="store_true",
                    help="never prompt; take the profile and the defaults")

    args = ap.parse_args(argv)
    if args.size:
        if args.width or args.height:
            ap.error("--size cannot be combined with --width/--height")
        args.width, args.height = args.size
    for name in ("fps", "width", "height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            ap.error(f"--{name} must be positive (got {value})")
    return args


def _ask_device() -> str:
    """
    Which machine is this? Auto-detected, but always confirmed.

    Guessing silently is the failure that costs a whole session: a Pi profile
    applied on a laptop, or worse a laptop profile on the Pi, where 60 fps
    darkens the picture with nothing to warn you.
    """
    detected = "pi" if _looks_like_pi() else "laptop"

    if not _stdin_readable():
        print(f"\n  Device: {detected}  (detected; no input available)")
        return detected

    print("\n" + "=" * 60)
    print("  Which machine is this?")
    print("=" * 60)
    for i, (key, prof) in enumerate(sorted(DEVICE_PROFILES.items()), 1):
        mark = "  <- detected" if key == detected else ""
        print(f"  {i}. {key:<8} {prof['width']}x{prof['height']} @ "
              f"{prof['fps']} fps, camera={prof['source']}{mark}")
    print()

    try:
        raw = input(f"Choice [1-{len(DEVICE_PROFILES)}, "
                    f"Enter = {detected}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return detected

    if not raw:
        return detected
    keys = sorted(DEVICE_PROFILES)
    if raw.isdigit() and 1 <= int(raw) <= len(keys):
        return keys[int(raw) - 1]
    if raw.lower() in DEVICE_PROFILES:
        return raw.lower()
    print(f"  [!] '{raw}' is not one of them - using {detected}.")
    return detected


def _looks_like_pi() -> bool:
    """Same test camera_manager uses, without importing its private helper."""
    if _sys.platform != "linux":
        return False
    if not _platform.machine().lower().startswith(("arm", "aarch")):
        return False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def _resolve_camera_settings(args) -> dict:
    """
    Settle source / width / height / fps, and say where each value came from.

    Order: command line > .env > device profile. The environment outranks the
    profile because someone who wrote CAMERA_FPS in .env meant it; the command
    line outranks both because it is this run, deliberately.
    """
    profile = DEVICE_PROFILES[args.device]
    settled = {}

    for key, flag, env_name in (
        ("source", args.camera, "CAMERA_SOURCE"),
        ("width",  args.width,  "CAMERA_WIDTH"),
        ("height", args.height, "CAMERA_HEIGHT"),
        ("fps",    args.fps,    "CAMERA_FPS"),
    ):
        env_value = _os.getenv(env_name)
        # CAMERA_SOURCE=auto is the file's "no opinion" value, not a choice.
        if key == "source" and env_value and env_value.lower() == "auto":
            env_value = None

        if flag is not None:
            settled[key] = (flag, "command line")
        elif env_value:
            settled[key] = (env_value, f".env ({env_name})")
        else:
            settled[key] = (profile[key], f"{args.device} profile")

    return settled


def _print_camera_plan(device: str, settled: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  Camera plan — {device}")
    print("=" * 60)
    for key in ("source", "width", "height", "fps"):
        value, origin = settled[key]
        print(f"  {key:<8}: {str(value):<14}  {origin}")
    fps = int(settled["fps"][0])
    if fps >= 60:
        print(f"\n  {fps} fps caps each exposure at 1/{fps} s — sharp hands, but")
        print("  it needs light. The camera measures the picture on startup and")
        print("  will say so if there is not enough.")
    print()


def _actual_camera_settings(cam) -> dict:
    """What the camera really produced — measured from a frame, not requested."""
    try:
        for _ in range(5):
            ok, frame = cam.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                fps = None
                probe = getattr(cam, "_cam", None)
                if probe is not None and hasattr(probe, "get"):
                    try:
                        fps = float(probe.get(cv2.CAP_PROP_FPS)) or None
                    except Exception:
                        fps = None
                return {"width": w, "height": h, "fps": fps}
            time.sleep(0.05)
    except Exception:
        pass
    return {}


def main(argv=None) -> None:
    global TARGET_PER_LABEL

    args = _parse_args(argv)

    # -- Terminal prompt: get gesture label before opening camera window ------
    print("=" * 60)
    print("  Tarjuman — Dynamic Gesture Sequence Recorder")
    print("=" * 60)
    print(f"  Output file     : {OUTPUT_CSV}")
    print(f"  Sequence length : {SEQUENCE_LENGTH} frames")
    print(f"  Values per row  : 1 (label) + {SEQUENCE_LENGTH * VALS_PER_FRAME} (coords)")
    print("=" * 60)

    if args.samples:
        TARGET_PER_LABEL = args.samples


    # Which machine, then what the camera should do about it.
    if args.device is None:
        if args.yes:
            args.device = "pi" if _looks_like_pi() else "laptop"
        else:
            args.device = _ask_device()
    settled = _resolve_camera_settings(args)
    _print_camera_plan(args.device, settled)

    camera_source = settled["source"][0]
    # Only ask WHICH camera when nothing has already decided. A profile or a
    # flag is an answer; asking again after it would be noise.
    if camera_source == "auto" and not args.yes:
        camera_source = _choose_camera()

    # -- Ensure CSV exists with the correct header ----------------------------
    _ensure_csv_header(OUTPUT_CSV)

    # -- Camera and models open ONCE, for the whole session -------------------
    # Opening a webcam and loading the MediaPipe graphs takes several seconds.
    # Doing that per word made recording a vocabulary of a hundred signs a
    # hundred separate startups; keeping them alive across words turns the
    # session into one continuous flow.
    pose_tracker = get_pose_tracker()
    # `args.ev` goes through the constructor for the same reason the size and
    # rate do: CAMERA_EV is read into a class attribute when camera_manager is
    # imported, which happens long before argv is parsed. Setting the
    # environment variable here would be read by nobody.
    cam = SmartCamera(source=camera_source,
                      width=int(settled["width"][0]),
                      height=int(settled["height"][0]),
                      fps=int(settled["fps"][0]),
                      exposure_value=args.ev)
    cam.start()
    # Decode frames in the background. read() then returns the NEWEST frame
    # instead of blocking until the sensor produces one, so capture and
    # MediaPipe inference overlap rather than running end to end.
    cam.start_grabber()

    if not cam.is_running:
        print("[FAIL] Camera did not open. See the checklist above. Exiting.")
        sys.exit(1)

    # What was asked for is not what was necessarily delivered: a sensor may
    # have no mode at the requested size and silently pick its own. Recording a
    # whole session at a resolution you did not choose is the kind of thing you
    # want to find out now, not while training.
    print(f"[Collector] Camera backend: {cam.backend}")
    actual = _actual_camera_settings(cam)
    if actual:
        want = (int(settled["width"][0]), int(settled["height"][0]))
        print(f"[Collector] Delivered      : {actual['width']}x{actual['height']}"
              + (f" @ {actual['fps']:.0f} fps" if actual.get("fps") else ""))
        if (actual["width"], actual["height"]) != want:
            print(f"[Collector] [!] Asked for {want[0]}x{want[1]} — the sensor "
                  "chose otherwise.")
            print("[Collector]     Record every session at ONE size: MediaPipe "
                  "normalises x and y")
            print("[Collector]     separately, so a changed aspect ratio moves "
                  "every landmark.")

    session_totals = {}
    try:
        while True:
            label = _record_one_word(cam, pose_tracker, session_totals)
            if label is None:
                break
    finally:
        hands.close()
        pose_tracker.close()
        cam.stop_grabber()
        cam.release()
        cv2.destroyAllWindows()
        _print_session_summary(session_totals)



def _draw_banner(frame, title: str, subtitle: str = "", tone=(60, 60, 200)) -> None:
    """لوحةٌ كبيرة في منتصف الإطار — للعدّ التنازلي وحالة المقطِّع."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 90), (w, h // 2 + 70), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    size = cv2.getTextSize(title, _FONT, 2.0, 4)[0]
    _put_text_with_shadow(frame, title, ((w - size[0]) // 2, h // 2 + 10),
                          tone, scale=2.0, thickness=4)
    if subtitle:
        s2 = cv2.getTextSize(subtitle, _FONT, 0.7, 2)[0]
        _put_text_with_shadow(frame, subtitle, ((w - s2[0]) // 2, h // 2 + 52),
                              _WHITE, scale=0.7, thickness=2)


# -----------------------------------------------------------------------------
#  حلقة الالتقاط — بالمقطِّع نفسه الذي يستعمله الخادم
# -----------------------------------------------------------------------------

def _segment_session(cam, pose_tracker, title, on_take, status,
                     window=None) -> str:
    """
    عيّنةٌ لكلّ ضغطة `r` — والقصُّ بيد المقطِّع.

    هنا يجتمع أمران كانا يبدوان متعارضين:

      • أنت تملك الإيقاع. لا شيء يُلتقَط ما لم تضغط `r`. تستعدّ، تتنفّس،
        تصحّح وقفتك، ثمّ تأذن. وبعد كلّ عيّنة يعود ساكناً حتّى تأذن ثانية.

      • والقصُّ يبقى للمقطِّع. `r` تُسلّح فقط — أمّا أين تبدأ الإشارة وأين
        تنتهي فيقرّره `GestureSegmenter`، وهو المقطِّع نفسه الذي يستعمله
        الخادم وقت التعرّف. فلا يعود التدريبُ يرى الإشارة بهوامشَ ميّتة
        والاستدلالُ يراها مقصوصة.

    وهذا هو الفرق عن المسجِّل القديم: `r` هناك كانت تفتح نافذةً زمنية وتحفظ
    ما وقع فيها؛ و`r` هنا تأذن للمقطِّع أن ينظر.

    الارتداد: بعض الإشارات صغيرة السَّعة (كـ'sorry'، دائرةٌ على الصدر) فقد لا
    يراها المقطِّع تبدأ أصلاً. فإن مضت `ARM_GRACE` ثانية بعد التسليح دون أن
    يلتقط شيئاً، تُسجَّل العيّنة بالنافذة المُعايَرة ويُعلَم أنّها ارتداد —
    خيرٌ من أن تقف تضغط `r` ولا يُستجاب لك.

    `on_take(result)` تُستدعى لكلّ عيّنة مكتملة وتُعيد:
        "continue" — واصِل        "rest" — واصِل بعد راحة        "stop" — انتهِ

    تُعيد سبب الخروج: "done" أو "quit" أو "recalibrate".
    """
    segmenter = GestureSegmenter()
    span = float(window) if window else FALLBACK_SECONDS
    rest_until = time.time() + REST_SECONDS      # «استعدّ» قبل أوّل عيّنة
    paused = False

    armed = False          # هل أذِنتَ بعيّنة؟
    armed_at = 0.0
    forced = None          # إطارات الارتداد، حين يعجز المقطِّع عن الرؤية
    forced_at = 0.0

    def disarm():
        nonlocal armed, forced
        armed, forced = False, None
        segmenter.reset()

    # سجِّل نافذة OpenCV وأرفق معالج الماوس — يُتيح السكرول للتكبير/التصغير
    # قبل أوّل `imshow`، وإلّا لا تُوجد النافذة فعليًا بعد.
    global _zoom_level
    _zoom_level = 1.0    # أعِد ضبط الزوم مع كلّ كلمة جديدة
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, _on_mouse)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = prepare_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            anchors = pose_tracker.update(rgb)
            rgb.flags.writeable = True
            draw_landmarks_on_frame(frame, results, pose_tracker.last_results)

            left_hand, right_hand = split_hands(results)
            hands_present = (left_hand is not None) or (right_hand is not None)
            now = time.time()
            resting = now < rest_until
            captured = None

            if resting or paused:
                # التسليح لا يبقى عبر الراحة: تعود، فتأذن من جديد.
                disarm()
            elif armed:
                features = extract_frame_features(results, anchors)

                if forced is not None:
                    # -- ارتداد: نافذةٌ صريحة، لأنّ المقطِّع لم يرَ بداية ----
                    forced.append(features)
                    if now - forced_at >= span or len(forced) >= MAX_FALLBACK_FRAMES:
                        if len(forced) >= MIN_FALLBACK_FRAMES:
                            captured = {
                                "sequence": resample_sequence(forced),
                                "globals": compute_global_features(
                                    forced, now - forced_at),
                                "duration": now - forced_at,
                                "frames": len(forced),
                                "fallback": True,
                            }
                        else:
                            print("  [SKIP] too few frames reached the recorder.")
                        disarm()
                else:
                    captured = segmenter.update(features, hands_present,
                                                now=time.monotonic())
                    if captured is not None:
                        captured["fallback"] = False
                        disarm()
                    elif (not segmenter.is_capturing
                          and now - armed_at >= ARM_GRACE):
                        # لم يرَ بدايةً خلال المهلة — نسجّل بالنافذة
                        forced, forced_at = [features], now

            if captured is not None:
                verdict = on_take(captured)
                if verdict == "stop":
                    return "done"
                if verdict == "rest":
                    rest_until = time.time() + REST_SECONDS

            # -- الحالة على الشاشة ------------------------------------------
            if paused:
                _draw_banner(frame, "||", "PAUSED - press p to resume",
                             (200, 200, 200))
            elif resting:
                _draw_banner(frame, f"{max(0.0, rest_until - now):.0f}",
                             status() + "  -  get ready", (120, 200, 120))
            elif forced is not None:
                _draw_banner(frame, "REC*",
                             f"window {now - forced_at:.1f}s / {span:.1f}s",
                             (60, 150, 220))
            elif segmenter.is_capturing:
                _draw_banner(frame, "REC",
                             f"{segmenter.captured_frames} frames", (60, 60, 220))
            elif armed:
                _draw_banner(frame, "ARMED", "sign now", (60, 200, 200))
            elif not hands_present:
                _draw_banner(frame, "...", "show me your hands", (150, 150, 150))
            else:
                _draw_banner(frame, "r", "press r for one sample", (150, 150, 150))

            _put_text_with_shadow(frame, status(), (12, 28), _WHITE, scale=0.7)
            _put_text_with_shadow(frame,
                                  "r=one sample  p=pause  c=recalibrate  q=quit",
                                  (12, frame.shape[0] - 16), _WHITE, scale=0.5)

            display_frame = _apply_zoom(frame)
            _draw_zoom_indicator(display_frame)
            cv2.imshow(title, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return "quit"
            if key == ord("c"):
                return "recalibrate"
            if key == ord("p"):
                paused = not paused
                disarm()
            elif key == ord("r"):
                if armed:
                    disarm()          # ضغطةٌ ثانية تُلغي ما لم يبدأ بعد
                elif not resting:
                    armed, armed_at, forced = True, now, None
                    segmenter.reset()
    finally:
        cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
#  المرحلة الأولى: المعايرة
# -----------------------------------------------------------------------------

def _calibrate_word(cam, pose_tracker, label: str):
    """
    تُؤدّى الكلمة CALIBRATION_TAKES مرّات، فتُقاس ويُبنى عليها الباقي.

    ولماذا أكثر من مرّة: تسجيلةٌ واحدة تقديرٌ هشّ — إن جاءت أسرع أو أبطأ من
    عادتك بُني عليها كلُّ شيء. الوسيط بين ثلاث يُهمل الشاذّة منها.

    تُعيد dict فيه:
        reference — التسجيلة الوسيطة، مرجعُ الحكم على الشاذّ لاحقاً
        tolerance — أقصى بُعدٍ مقبول عن المرجع، مشتقٌّ من تباعد المعايرات
                    نفسها لا من رقمٍ مفروض
        seconds   — المدّة الوسيطة
    أو None إن تخطّاها المستخدم.
    """
    title = f"Tarjuman Calibration - '{label}'"
    print("\n" + "=" * 62)
    print(f"  CALIBRATION - '{label}'")
    print("=" * 62)
    print(f"  Sign the word {CALIBRATION_TAKES} times, exactly as you always will.")
    print("  Press r for each take. r only arms the recorder - it still finds")
    print("  the start and end of the motion itself, exactly as the live")
    print("  recogniser does. Stand still, press r, sign, lower your hands.\n")

    takes, reports = [], []

    def on_take(result):
        report = analyse_take(list(result["sequence"]), result["duration"],
                              segmented=True)
        index = len(takes) + 1
        print(f"\n  [{index}/{CALIBRATION_TAKES}]  "
              f"{result['duration']:.2f}s  ·  path {report['path']:.2f}  ·  "
              f"R {report['hands']['right'] * 100:.0f}%  "
              f"L {report['hands']['left'] * 100:.0f}%")
        if not report["ok"]:
            for problem in report["problems"]:
                print(f"        - {problem}")
            print("        Not counted - do it again.")
            return "rest"

        takes.append(result)
        reports.append(report)
        if len(takes) >= CALIBRATION_TAKES:
            return "stop"
        return "rest"

    def status():
        return f"calibration {len(takes)}/{CALIBRATION_TAKES}"

    reason = _segment_session(cam, pose_tracker, title, on_take, status)

    if len(takes) < 2:
        print("\n  [SKIP] not enough calibration - recording without a reference.")
        return None

    durations = [t["duration"] for t in takes]
    order = sorted(range(len(takes)), key=lambda i: durations[i])
    middle = order[len(order) // 2]

    sequences = [t["sequence"] for t in takes]
    agreement = spread(sequences)
    # التسامح مشتقٌّ من تباعد المعايرات نفسها: من كان أداؤه ثابتاً ضاق
    # تسامحه، ومن كان متنوّعاً اتّسع. رقمٌ مفروضٌ سلفاً كان سيظلم أحدهما.
    tolerance = max(OUTLIER_FLOOR, agreement["mean"] * OUTLIER_FACTOR)

    print("\n  Calibration summary:")
    print(f"    median duration  : {durations[middle]:.2f}s"
          f"   (range {min(durations):.2f}-{max(durations):.2f})")
    print(f"    take-to-take spread : {agreement['mean']:.3f}")
    print(f"    outlier limit    : {tolerance:.3f}")
    print(f"    hand path        : {reports[middle]['path']:.2f}")

    return {
        "reference": sequences[middle],
        "tolerance": tolerance,
        "seconds": durations[middle],
        # نافذةُ الارتداد: مدّةُ الكلمة كما قِيست، وعليها هامشٌ فلا تُقتطع
        # محاولةٌ جاءت أبطأ قليلاً من معايرتها.
        "window": reports[middle]["window"],
        "path": reports[middle]["path"],
        "two_handed": reports[middle]["two_handed"],
        "quit": reason == "quit",
    }


# -----------------------------------------------------------------------------
#  تسجيل كلمة واحدة
# -----------------------------------------------------------------------------

def _record_one_word(cam, pose_tracker, session_totals):
    """
    Record one label end to end.

    Returns a truthy value if the user wants another word, or None to finish.
    The camera and MediaPipe graphs are owned by the caller and are NOT torn
    down here - that is the whole point of splitting this out.
    """
    label = _choose_label()

    calibration = _calibrate_word(cam, pose_tracker, label)
    if calibration is None:
        reference, tolerance = None, None
        default_target = TARGET_PER_LABEL
    else:
        reference = calibration["reference"]
        tolerance = calibration["tolerance"]
        default_target = suggest_samples(calibration, TARGET_PER_LABEL)

    answer = input(
        f"\n How many samples for '{label}'? [default: {default_target}]: ").strip()
    target_sequences = int(answer) if answer.isdigit() else default_target

    print(f"\n[OK] recording {target_sequences} samples of '{label}'")
    print(f"   batches of {BATCH_SIZE}, {REST_SECONDS:.0f}s rest between them")
    print("   Press r for each sample; the recorder cuts it at the motion.\n")

    expect_path = calibration["path"] if calibration else None
    fallback_window = calibration["window"] if calibration else None

    saved, rejected, by_window = [], 0, 0
    last_spread = None
    attempts_at_batch = 0        # كم محاولة مضت عند آخر دفعة، لقياس نسبة الردّ

    def on_take(result):
        nonlocal rejected, last_spread, reference, tolerance
        nonlocal attempts_at_batch, by_window
        sequence = result["sequence"]
        # `expect_path` هو ما يمنع ردَّ إشارةٍ صغيرةِ السَّعة بطبعها. بدونه
        # كانت عتبةٌ واحدة تردّ 'sorry' ستّين مرّة من ستّين.
        report = analyse_take(list(sequence), result["duration"],
                              segmented=True, expect_path=expect_path)

        # ١. جودة التسجيلة في ذاتها
        if not report["ok"]:
            rejected += 1
            print(f"  [DROP] {report['problems'][0]}")
            return "continue"

        # ٢. وهل تشبه ما عايرتَ عليه؟ عيّنةٌ تنحرف عن المرجع ليست مثالاً
        #    للكلمة. وبدون هذا الفحص كانت العيّنات من ٢ إلى ٥٠ لا تُفحَص
        #    إطلاقاً، فينحرف الأداء تدريجياً ولا يظهر ذلك إلّا بعد ساعة.
        if reference is not None:
            distance = take_distance(sequence, reference)
            if distance > tolerance:
                rejected += 1
                print(f"  [OUTLIER] {distance:.3f} from calibration "
                      f"(limit {tolerance:.3f}) - sign it as you calibrated.")
                return "continue"

        save_prepared(label, sequence, result["globals"],
                      result["duration"], OUTPUT_CSV)
        saved.append(sequence)
        # العيّنة التي جاءت بالارتداد قُصّت بنافذة لا بالمقطِّع — تُحسب،
        # لكن يُقال ذلك: إن كثُرت، فالمقطِّع لا يرى هذه الكلمة تبدأ.
        if result.get("fallback"):
            by_window += 1
        mark = "  [window]" if result.get("fallback") else ""
        print(f"  [OK] {len(saved)}/{target_sequences}  "
              f"({result['duration']:.2f}s){mark}")

        if len(saved) >= target_sequences:
            return "stop"

        if not (BATCH_SIZE and len(saved) % BATCH_SIZE == 0):
            return "continue"

        # ــــ نهاية دفعة ــــــــــــــــــــــــــــــــــــــــــــــــــــــ
        batch_no = len(saved) // BATCH_SIZE
        recent = saved[-SATURATION_WINDOW:]
        current = spread(recent)["mean"]
        print(f"\n  [BATCH {batch_no}]  spread of last {len(recent)} samples: "
              f"{current:.3f}")

        # ٣. هل ما زالت العيّنات تضيف جديداً؟
        if last_spread is not None and last_spread > 0:
            change = abs(current - last_spread) / last_spread
            if change < SATURATION_CHANGE:
                print(f"        [SATURATED] only {change * 100:.0f}% change from the "
                      "previous batch - new samples add no more variety.")
                print("        You can stop here (q) and move to another word.")
        last_spread = current

        # ٤. وهل كانت نسبة الردّ معقولة في هذه الدفعة؟
        #    ثلاثُ تسجيلاتٍ أساسٌ ضيّق، وقد تجيء متقاربةً بالصدفة فيخنق الحدُّ
        #    ما بعدها. فإن ارتفع الردّ، فالخلل ليس في تسجيلةٍ بعينها: إمّا أنّ
        #    المعايرة لم تُمثّلك، وإمّا أنّ أداءك نفسه يتأرجح. لا سبيل للبرنامج
        #    أن يميّز بينهما، فيقولهما معاً بدل أن يُخفي أحدهما.
        attempts = len(saved) + rejected
        span = attempts - attempts_at_batch
        batch_rejects = span - BATCH_SIZE
        attempts_at_batch = attempts
        if span and batch_rejects / span > OUTLIER_ALERT:
            print(f"        [WARN] {batch_rejects} of {span} attempts rejected in "
                  f"this batch ({batch_rejects / span * 100:.0f}%).")
            print("        Either the calibration did not represent you, or your")
            print("        performance is drifting between takes.")
            print("        Watch yourself once and decide: c to recalibrate, or go on.")

        # ٥. وبعدها: يُعاد اشتقاق المرجع والحدّ من العيّنات المقبولة نفسها.
        #    عشرٌ خيرٌ من ثلاث — وقد قِيس أثر ذلك على 388 تسجيلة حقيقية: أسوأ
        #    نسبة ردٍّ في 'hello' هبطت من 17% إلى 7%.
        if OUTLIER_REVISE and reference is not None and len(saved) >= BATCH_SIZE:
            agreement = spread(saved)
            reference = saved[agreement["medoid"]]
            revised = max(OUTLIER_FLOOR, agreement["mean"] * OUTLIER_FACTOR)
            if abs(revised - tolerance) / max(tolerance, 1e-6) > 0.05:
                print(f"        [TUNE] outlier limit {tolerance:.3f} -> "
                      f"{revised:.3f}  (from {len(saved)} samples, not 3)")
            tolerance = revised

        return "rest"

    def status():
        return (f"'{label}'  {len(saved)}/{target_sequences}"
                + (f"  ·  rejected {rejected}" if rejected else ""))

    # `c` أثناء التسجيل تعني «المعايرة كانت خطأً» — فتُعاد، ويُستأنف ما بقي
    # من العدد بمرجعٍ جديد. العيّنات المحفوظة قبلها تبقى: هي لم تُردّ، ولا
    # سبب لإتلافها.
    while True:
        reason = _segment_session(cam, pose_tracker,
                                  f"Tarjuman Recorder - '{label}'",
                                  on_take, status, window=fallback_window)
        if reason != "recalibrate":
            break
        print(f"\n  [RECALIBRATE] saved so far: {len(saved)} samples.")
        again = _calibrate_word(cam, pose_tracker, label)
        if again is None:
            print("  Calibration incomplete - continuing with the previous limit.")
            continue
        reference = again["reference"]
        tolerance = again["tolerance"]
        expect_path = again["path"]
        fallback_window = again["window"]
        if again["quit"] or len(saved) >= target_sequences:
            break

    session_totals[label] = session_totals.get(label, 0) + len(saved)

    print(f"\n Finished '{label}':")
    print(f"   saved    : {len(saved)}")
    print(f"   rejected : {rejected}")
    if by_window:
        print(f"   by window: {by_window}  (segmenter saw no onset - "
              "sign it a little wider or slower)")
    print(f"   file     : {os.path.abspath(OUTPUT_CSV)}")
    if not saved:
        print("[!]  No samples were saved for this word.")

    return _ask_next()


def _ask_next():
    """
    Offer another word, or finish.

    Returns a truthy value to continue the session, or None to stop. Recording
    a vocabulary means dozens of words in a sitting, and quitting to the shell
    after each one meant re-picking the camera and reloading MediaPipe every
    time - several seconds of waiting for no reason.
    """
    print("\n" + "=" * 60)
    print("  What next?")
    print("=" * 60)
    print("  1. Finish and close")
    print("  2. Record another word  (camera stays open)")
    print()

    while True:
        try:
            choice = input("Choice [1-2, default 2]: ").strip() or "2"
        except (EOFError, KeyboardInterrupt):
            print("\n[stop] Finishing.")
            return None
        if choice == "1":
            return None
        if choice == "2":
            print()
            return "continue"
        print("  [!] Enter 1 or 2.")


def _print_session_summary(session_totals):
    """Everything recorded across the whole sitting, not just the last word."""
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    if not session_totals:
        print("  Nothing was recorded.")
        return
    total = 0
    for word, n in session_totals.items():
        print(f"   {word:<20s} {n:>4d} sequences")
        total += n
    print("   " + "-" * 30)
    print(f"   {'TOTAL':<20s} {total:>4d} sequences")
    print("\n   Next:  npm run train")


if __name__ == "__main__":
    main()