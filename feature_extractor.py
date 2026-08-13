"""
feature_extractor.py — Single source of truth for Tarjuman's feature layout
============================================================================
Every component that touches landmark features imports from THIS file:

    • data_collector.py    — records training sequences
    • websocket_server.py  — runs live inference
    • migrate_dataset.py   — converts the old Holistic dataset
    • train_model.py       — reads TOTAL_FEATURES for the ONNX input shape

Why a shared module
-------------------
The feature layout used to be re-declared independently in the collector and
the server. Any drift between them produces a model that trains on one layout
and infers on another — a bug that is completely silent and extremely hard to
diagnose (the model just "gets worse" for no visible reason). Defining it once
here makes that class of bug structurally impossible.

Feature layout (v4 — hands anchored to the body)
------------------------------------------------
Per frame:
    [ Left hand  68 ] + [ Right hand 68 ] + [ Body 4 ]  =  140 values

Per hand (68 values):
      [0:3]    wrist in BODY coordinates (see below)
      [3:63]   landmarks 1..20 relative to the wrist, divided by palm length
               → hand SHAPE, invariant to position and distance
      [63:68]  distance from the wrist to five body anchors:
               nose, mouth, ear, shoulder, chest-centre
A hand that is not detected contributes 68 zeros.

Body block (4 values):
      shoulder width, face size, torso tilt, pose-detected flag

Why body coordinates (this is the point of v4)
----------------------------------------------
Location is one of the defining parameters of a sign — أب is made at the
forehead and أم at the chin, أخ at the forehead and أخت at the cheek. In this
project's own 100-term vocabulary, 47 terms are anchored to a body location and
7 pairs are separated by NOTHING ELSE.

The previous layout stored the wrist in FRAME coordinates. That says "the hand
is at 0.35, 0.22 of the picture" — it does not say "the hand is at the chin".
Sit higher, move the camera, or hand the system to a taller signer and the same
sign produces different numbers, because there is no anchor.

v4 re-expresses the wrist in a coordinate system built from the body itself:
      origin = midpoint between the shoulders
      unit   = shoulder width
"Hand at the chin" then yields the same numbers for any person, at any distance,
in any seating position. That is a kind of invariance no amount of extra
training data can recover from raw frame coordinates.

The five explicit anchor distances carry the discriminating signal directly,
rather than asking the model to infer it from coordinates.

Why Pose and not Holistic
-------------------------
`mp.solutions.pose` provides nose, eyes, ears, mouth corners and shoulders —
every anchor this vocabulary needs. Holistic would additionally run Face Mesh
(468 points) on every frame, none of which is used. Pose alone, at
model_complexity=0, is the cheap way to get the anchors.

Why the division matters (NORMALIZE_SCALE)
------------------------------------------
Subtracting the wrist only removes TRANSLATION. If the signer leans closer to
the camera, every relative offset grows proportionally and an identical hand
shape looks like a different one to the model. Dividing by palm length —
the wrist → middle-finger-MCP distance, which barely moves as fingers open and
close — removes SCALE too, so the same sign reads the same at 0.5 m and 1.5 m.
No information is lost: absolute distance is still available from the raw wrist
coordinates and from estimate_distance().

Sequence:   SEQUENCE_LENGTH (30) frames × 126 = 3 780 features per sample.

Mirroring (important)
---------------------
Both recording and inference MUST see the frame in the same orientation, or
MediaPipe's Left/Right handedness labels get swapped between training and
inference and the model silently learns the wrong hand. `MIRROR_FRAME` below
is the single switch that guarantees this; use `prepare_frame()` everywhere.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry constants
# ─────────────────────────────────────────────────────────────────────────────

N_HAND_LANDMARKS = 21
COORDS_PER_LM    = 3                       # x, y, z  (hands carry no visibility)

# ── Body anchors (MediaPipe Pose landmark indices) ───────────────────────────
# Only these are used. Pose reports 33 points; the rest (elbows, hips, legs)
# add nothing for a seated signer and would only dilute the feature vector.
POSE_NOSE        = 0
POSE_MOUTH_L     = 9
POSE_MOUTH_R     = 10
POSE_EAR_L       = 7
POSE_EAR_R       = 8
POSE_SHOULDER_L  = 11
POSE_SHOULDER_R  = 12

# Anchors each hand is measured against, in this exact order.
ANCHOR_NAMES = ("nose", "mouth", "ear", "shoulder", "chest")
N_ANCHORS    = len(ANCHOR_NAMES)                        # 5

HAND_SHAPE_VALS  = (N_HAND_LANDMARKS - 1) * COORDS_PER_LM   # 60
VALS_PER_HAND    = COORDS_PER_LM + HAND_SHAPE_VALS + N_ANCHORS   # 3 + 60 + 5 = 68

# Body block: shoulder width, face size, torso tilt, pose-detected flag.
BODY_BLOCK_VALS  = 4

VALS_PER_FRAME   = VALS_PER_HAND * 2 + BODY_BLOCK_VALS  # 140

SEQUENCE_LENGTH  = 30
FRAME_FEATURES   = VALS_PER_FRAME * SEQUENCE_LENGTH     # 3 780

# ── Global (whole-gesture) features ──────────────────────────────────────────
# Appended once per sample, AFTER the per-frame block.
#
# Why they exist
# --------------
# gesture_segmenter resamples every capture to SEQUENCE_LENGTH frames, and
# training augments with time-warping. Both are deliberate — they make the same
# sign recognisable at any signing speed. But they also DELETE duration, and
# some signs mean what they mean *because* of their speed: طوارئ / إسعاف /
# ساعدني فوراً are urgent versions of ordinary movements. After resampling they
# become byte-identical to their calm counterparts.
#
# These features are computed BEFORE resampling, so tempo survives. They also
# summarise direction and hand openness, which is what separates the many pairs
# that differ by only one attribute (forward vs. back, 3 fingers vs. 4).
GLOBAL_FEATURE_NAMES = (
    "duration_s",        # seconds the gesture actually took
    "mean_speed",        # average wrist travel per second
    "peak_speed",        # fastest instant — urgency shows up here
    "speed_variance",    # smooth glide vs. sharp jab
    "path_length",       # total distance travelled
    "net_dx",            # signed horizontal displacement (start → end)
    "net_dy",            # signed vertical displacement — up vs. down signs
    "range_x",           # horizontal extent covered
    "range_y",           # vertical extent covered
    "hands_used",        # 0.0 none, 0.5 one hand, 1.0 both
    "mean_openness",     # average finger extension — separates 3 vs. 4 fingers
    "openness_change",   # how much the hand shape morphed during the sign
)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)           # 12

TOTAL_FEATURES   = FRAME_FEATURES + N_GLOBAL_FEATURES   # 3 792

WRIST_IDX        = 0
MIDDLE_MCP_IDX   = 9                       # used as a stable hand-size proxy


# ─────────────────────────────────────────────────────────────────────────────
#  Frame orientation
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirror the frame so the signer sees themselves as in a mirror. This MUST be
# identical in data_collector.py and websocket_server.py — see module docstring.
MIRROR_FRAME = True


def prepare_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply the canonical orientation. Call before ANY MediaPipe processing."""
    return cv2.flip(frame_bgr, 1) if MIRROR_FRAME else frame_bgr


# ─────────────────────────────────────────────────────────────────────────────
#  Scale normalisation
# ─────────────────────────────────────────────────────────────────────────────
#
# Divide the relative (shape) block by palm length so hand shape is invariant
# to how far the signer sits from the camera. See module docstring.
# Changing this REQUIRES re-running migrate_dataset.py and train_model.py —
# a model trained with one setting is meaningless under the other.
NORMALIZE_SCALE = True

# Guard against division by ~0 when the hand is edge-on or badly detected.
_MIN_PALM_LENGTH = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
#  Hand splitting (mp.solutions.hands has no left/right attributes)
# ─────────────────────────────────────────────────────────────────────────────

def split_hands(results):
    """
    Turn a mp.solutions.hands result into an explicit (left, right) pair.

    Unlike Holistic — which exposes `.left_hand_landmarks` / `.right_hand_
    landmarks` directly — Hands returns an unordered list plus a parallel
    handedness classification. Without this split, hand order would depend on
    detection order and flip randomly between frames.

    Returns
    -------
    (left_landmarks | None, right_landmarks | None)
    """
    left = right = None

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return left, right

    for landmarks, handedness in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):
        label = handedness.classification[0].label      # "Left" | "Right"
        if label == "Left" and left is None:
            left = landmarks
        elif label == "Right" and right is None:
            right = landmarks

    return left, right


# ─────────────────────────────────────────────────────────────────────────────
#  Hybrid feature extraction:  raw wrist (location) + relative shape
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Body anchors — the reference frame that makes location meaningful
# ─────────────────────────────────────────────────────────────────────────────

# Pose is markedly slower than Hands, and a seated signer's torso barely moves
# between frames. Running it every Nth frame and reusing the last anchors keeps
# the reference accurate while cutting most of the cost — important on a Pi.
POSE_EVERY_N_FRAMES = 3

# Below this, the pose is treated as unreliable (person turned away, occluded).
MIN_POSE_VISIBILITY = 0.5


class PoseTracker:
    """
    Owns a MediaPipe Pose instance and the throttling around it.

    Pose costs meaningfully more than Hands, but a seated signer's torso barely
    moves between consecutive frames — the hands do all the work. Running Pose
    every Nth frame and reusing the previous anchors keeps the reference frame
    accurate while paying a fraction of the cost, which matters on a Pi.

    One instance per client/session: MediaPipe graph objects are not safe to
    share across threads.
    """

    def __init__(self, every_n: int = POSE_EVERY_N_FRAMES, model_complexity: int = 0):
        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,   # 0 = lite; anchors need no more
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._every_n = max(1, int(every_n))
        self._counter = 0
        self._anchors = BodyAnchors()

    def update(self, rgb_frame) -> "BodyAnchors":
        """
        Return current anchors, refreshing them only every Nth call.

        `rgb_frame` must already be RGB and mirrored — the same frame handed to
        Hands, so both models see an identical world.
        """
        if self._counter % self._every_n == 0:
            try:
                results = self._pose.process(rgb_frame)
                self._anchors = BodyAnchors.from_pose(results)
            except Exception as exc:
                print(f"[PoseTracker] pose failed: {type(exc).__name__}: {exc}")
        self._counter += 1
        return self._anchors

    @property
    def anchors(self) -> "BodyAnchors":
        return self._anchors

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass


class BodyAnchors:
    """
    A body-centred coordinate system derived from MediaPipe Pose.

        origin = midpoint of the shoulders
        unit   = shoulder width

    Expressing the hands in these terms is what turns "the hand is at 0.35 of
    the picture" into "the hand is at the chin" — the same numbers for a tall
    signer and a short one, near the camera or far from it.

    `valid` is False when no usable pose was found; callers then fall back to
    raw frame coordinates and the flag tells the model the anchors are absent,
    rather than silently feeding it a different coordinate space.
    """

    __slots__ = ("valid", "origin", "scale", "points", "face_size", "tilt")

    def __init__(self):
        self.valid = False
        self.origin = np.zeros(2, dtype=np.float32)
        self.scale = 1.0
        self.points = {}          # name -> (x, y) in body coordinates
        self.face_size = 0.0
        self.tilt = 0.0

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_pose(cls, pose_results) -> "BodyAnchors":
        a = cls()
        if pose_results is None or not getattr(pose_results, "pose_landmarks", None):
            return a

        lms = pose_results.pose_landmarks.landmark

        def pt(i):
            lm = lms[i]
            return np.array([lm.x, lm.y], dtype=np.float32), getattr(lm, "visibility", 1.0)

        sh_l, vis_l = pt(POSE_SHOULDER_L)
        sh_r, vis_r = pt(POSE_SHOULDER_R)

        # Shoulders define the whole frame of reference; without them there is
        # nothing to anchor to.
        if min(vis_l, vis_r) < MIN_POSE_VISIBILITY:
            return a

        origin = (sh_l + sh_r) / 2.0
        width = float(np.linalg.norm(sh_l - sh_r))
        if width < 1e-4:                      # degenerate (person side-on)
            return a

        a.valid = True
        a.origin = origin
        a.scale = width

        nose, _    = pt(POSE_NOSE)
        mouth_l, _ = pt(POSE_MOUTH_L)
        mouth_r, _ = pt(POSE_MOUTH_R)
        ear_l, _   = pt(POSE_EAR_L)
        ear_r, _   = pt(POSE_EAR_R)

        mouth = (mouth_l + mouth_r) / 2.0
        # "ear" is taken as whichever is better placed — for a signer facing the
        # camera both are visible; when turned, one becomes meaningless.
        ear = (ear_l + ear_r) / 2.0

        to_body = lambda p: (p - origin) / width

        a.points = {
            "nose":     to_body(nose),
            "mouth":    to_body(mouth),
            "ear":      to_body(ear),
            "shoulder": to_body(sh_r),          # dominant-side shoulder
            "chest":    to_body(origin + np.array([0.0, width * 0.35], np.float32)),
        }

        # Head size relative to the shoulders: a second scale cue, and a rough
        # proxy for how far away the signer is.
        a.face_size = float(np.linalg.norm(ear_l - ear_r) / width)
        # Shoulder-line angle: lets the model tolerate a signer leaning sideways.
        delta = sh_l - sh_r
        a.tilt = float(np.arctan2(delta[1], delta[0]))

        return a

    # ── Use ─────────────────────────────────────────────────────────────────

    def to_body_coords(self, x: float, y: float) -> tuple[float, float]:
        """Map a normalised frame point into body coordinates."""
        if not self.valid:
            return float(x), float(y)
        return (float((x - self.origin[0]) / self.scale),
                float((y - self.origin[1]) / self.scale))

    def anchor_distances(self, x: float, y: float) -> list[float]:
        """
        Distance from a point to each body anchor, in shoulder-width units.

        These are the features that separate أب from أم and أخ from أخت: the
        hand shape and the movement are identical, only the anchor it sits near
        differs. Handing the model the distance directly is far more learnable
        than expecting it to derive the same thing from coordinates.
        """
        if not self.valid:
            return [0.0] * N_ANCHORS
        p = np.array([x, y], dtype=np.float32)
        bx, by = self.to_body_coords(x, y)
        p_body = np.array([bx, by], dtype=np.float32)
        return [float(np.linalg.norm(p_body - self.points[name]))
                for name in ANCHOR_NAMES]

    def body_block(self) -> list[float]:
        """The 4 per-frame body values appended after both hands."""
        return [
            float(self.scale),      # shoulder width in frame units → distance cue
            float(self.face_size),  # head size relative to shoulders
            float(self.tilt),       # torso lean
            1.0 if self.valid else 0.0,
        ]


def hand_features_from_array(coords: np.ndarray) -> list[float]:
    """
    THE canonical hand → 63-feature transform. Everything else delegates here.

    Both the live pipeline (MediaPipe landmark objects) and the offline dataset
    migration (raw CSV numbers) funnel into this one function, so the two can
    never compute features differently.

    Parameters
    ----------
    coords : ndarray, shape (21, 3)
        Hand landmarks as x, y, z.

    Returns
    -------
    list[float] of length 63:
        [0:3]   raw wrist x, y, z (frame coordinates)
        [3:63]  landmarks 1..20 relative to the wrist,
                divided by palm length if NORMALIZE_SCALE — shape

    The caller re-expresses the wrist in body coordinates and appends the
    anchor distances; this function deliberately stays anchor-agnostic so the
    dataset migration can reuse it unchanged.
    """
    wrist = coords[WRIST_IDX]

    # a) Absolute wrist position — preserves WHERE the hand is
    features: list[float] = [float(wrist[0]), float(wrist[1]), float(wrist[2])]

    # b) Every other landmark relative to the wrist — preserves SHAPE
    relative = coords[1:] - wrist

    # c) Remove scale so the same sign reads identically near and far
    if NORMALIZE_SCALE:
        palm_vec    = coords[MIDDLE_MCP_IDX] - wrist
        palm_length = float(np.hypot(palm_vec[0], palm_vec[1]))
        if palm_length > _MIN_PALM_LENGTH:
            relative = relative / palm_length

    features.extend(relative.reshape(-1).astype(float).tolist())
    return features


def extract_hand_features(hand_landmarks, anchors: "BodyAnchors" = None) -> list[float]:
    """
    Convert one MediaPipe hand into its 68-value feature block.

        [0:3]    wrist in body coordinates (frame coordinates if no pose)
        [3:63]   wrist-relative, palm-normalised finger shape
        [63:68]  distance from the wrist to each body anchor

    Returns 68 zeros when the hand is absent, so the frame vector always has
    exactly VALS_PER_FRAME values regardless of how many hands are visible.
    """
    if hand_landmarks is None:
        return [0.0] * VALS_PER_HAND

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )
    block = hand_features_from_array(coords)      # 3 raw wrist + 60 shape

    wrist_x, wrist_y, wrist_z = block[0], block[1], block[2]

    if anchors is not None and anchors.valid:
        bx, by = anchors.to_body_coords(wrist_x, wrist_y)
        block[0], block[1] = bx, by
        # z stays as MediaPipe's relative depth — there is no reliable body-space
        # depth reference from a single camera, and pretending otherwise would
        # invent numbers.
        block[2] = wrist_z
        distances = anchors.anchor_distances(wrist_x, wrist_y)
    else:
        distances = [0.0] * N_ANCHORS

    return block + distances


def extract_frame_features(results, anchors: "BodyAnchors" = None) -> list[float]:
    """
    Build the full VALS_PER_FRAME vector for one frame.

    Layout: [left hand 68] + [right hand 68] + [body 4].

    `anchors` may be None (or invalid) — the hands are then expressed in frame
    coordinates and the body flag is 0, so the model can tell the difference
    instead of being handed two incompatible coordinate spaces as if they were
    the same.
    """
    left, right = split_hands(results)
    body = anchors if anchors is not None else BodyAnchors()
    return (extract_hand_features(left, body)
            + extract_hand_features(right, body)
            + body.body_block())


# ─────────────────────────────────────────────────────────────────────────────
#  Distance feedback — replaces the old Pose shoulder-ratio heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _hand_openness(hand_block: list) -> float:
    """
    How extended the fingers are, 0 (closed) → ~1 (fully open).

    Uses the mean distance of the wrist-relative landmarks. Because that block
    is already palm-length normalised, this is comparable across users and
    distances — which is what makes it usable to tell three fingers from four.
    """
    # Shape only: skip the 3 wrist values and stop before the anchor distances.
    relative = np.asarray(hand_block[3:3 + HAND_SHAPE_VALS], dtype=np.float32)
    if not np.any(relative):
        return 0.0
    return float(np.mean(np.abs(relative)))


def compute_global_features(frames, duration_seconds: float) -> list[float]:
    """
    Summarise a whole gesture into N_GLOBAL_FEATURES values.

    MUST be called on the RAW captured frames, before resampling — the point of
    these features is to preserve exactly what resampling throws away.

    Parameters
    ----------
    frames : list of per-frame vectors, each VALS_PER_FRAME long
    duration_seconds : real wall-clock length of the capture
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return [0.0] * N_GLOBAL_FEATURES

    n_frames = arr.shape[0]
    duration = max(float(duration_seconds), 1e-3)

    # ── Wrist trajectory (whichever hand is present, averaged) ──────────────
    positions = []
    hands_present = []
    for row in arr:
        pts = []
        for hand_idx in range(2):
            base = hand_idx * VALS_PER_HAND
            if np.any(row[base:base + VALS_PER_HAND]):
                pts.append((row[base + 0], row[base + 1]))
        hands_present.append(len(pts))
        positions.append(np.mean(pts, axis=0) if pts else None)

    known = [p for p in positions if p is not None]
    if len(known) < 2:
        return [duration] + [0.0] * (N_GLOBAL_FEATURES - 1)

    track = np.asarray(known, dtype=np.float32)

    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    path_length = float(np.sum(steps))
    speeds = steps * (n_frames / duration)          # units per second

    net = track[-1] - track[0]
    span = track.max(axis=0) - track.min(axis=0)

    # ── Hand openness over time ─────────────────────────────────────────────
    openness = []
    for row in arr:
        vals = [
            _hand_openness(row[h * VALS_PER_HAND:(h + 1) * VALS_PER_HAND])
            for h in range(2)
            if np.any(row[h * VALS_PER_HAND:(h + 1) * VALS_PER_HAND])
        ]
        if vals:
            openness.append(float(np.mean(vals)))

    mean_open = float(np.mean(openness)) if openness else 0.0
    open_change = float(np.max(openness) - np.min(openness)) if len(openness) > 1 else 0.0

    max_hands = max(hands_present) if hands_present else 0
    hands_used = 0.0 if max_hands == 0 else (0.5 if max_hands == 1 else 1.0)

    return [
        duration,
        float(np.mean(speeds)) if speeds.size else 0.0,
        float(np.max(speeds)) if speeds.size else 0.0,
        float(np.var(speeds)) if speeds.size else 0.0,
        path_length,
        float(net[0]),
        float(net[1]),
        float(span[0]),
        float(span[1]),
        hands_used,
        mean_open,
        open_change,
    ]


def estimate_distance(results) -> dict | None:
    """
    Estimate how far the signer is, using hand size instead of shoulder width
    (Pose is no longer computed).

    Uses palm length — the wrist → middle-finger-MCP distance — because unlike
    a bounding box it barely changes when fingers open or close, making it a
    much steadier distance proxy.

    Returns None when no hand is visible.

    NOTE: thresholds are first-pass estimates for a 640×480 stream and should
    be calibrated against the real Pi camera mounting distance.
    """
    left, right = split_hands(results)
    hand = right or left
    if hand is None:
        return None

    wrist  = hand.landmark[WRIST_IDX]
    middle = hand.landmark[MIDDLE_MCP_IDX]
    palm_length = float(np.hypot(middle.x - wrist.x, middle.y - wrist.y))

    if   palm_length > 0.22:  return {"label": "قريب جداً — ابعد للخلف",  "quality": "bad",  "palm": palm_length}
    elif palm_length > 0.11:  return {"label": "ممتاز — المسافة مثالية",   "quality": "good", "palm": palm_length}
    elif palm_length > 0.06:  return {"label": "مقبول — اقترب قليلاً",     "quality": "ok",   "palm": palm_length}
    else:                     return {"label": "بعيد جداً — اقترب للأمام", "quality": "bad",  "palm": palm_length}
