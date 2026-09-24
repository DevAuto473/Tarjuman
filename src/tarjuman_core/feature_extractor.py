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
               -> hand SHAPE, invariant to position and distance
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
the wrist -> middle-finger-MCP distance, which barely moves as fingers open and
close — removes SCALE too, so the same sign reads the same at 0.5 m and 1.5 m.
No information is lost: absolute distance is still available from the raw wrist
coordinates and from estimate_distance().

Sequence:   SEQUENCE_LENGTH (30) frames × 144 = 4 320 features per sample.

Mirroring (important)
---------------------
Both recording and inference MUST see the frame in the same orientation, or
MediaPipe's Left/Right handedness labels get swapped between training and
inference and the model silently learns the wrong hand. `MIRROR_FRAME` below
is the single switch that guarantees this; use `prepare_frame()` everywhere.
"""

import os as _os
import numpy as np

# OpenCV is deliberately NOT imported at module level. This module is imported
# by train_model.py and export_signs_3d.py purely for its CONSTANTS, and pulling
# OpenCV's native libraries into those processes buys nothing while adding one
# more DLL that can clash with scikit-learn's or ONNX's. It is loaded on first
# use instead, which is inside the camera path where it is genuinely needed.
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _c
        _cv2 = _c
    return _cv2


# -----------------------------------------------------------------------------
#  Geometry constants
# -----------------------------------------------------------------------------

N_HAND_LANDMARKS = 21
COORDS_PER_LM    = 3                       # x, y, z  (hands carry no visibility)

# -- Body anchors (MediaPipe Pose landmark indices) ---------------------------
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

# Body block: shoulder width, face size, torso tilt, pose-detected flag, plus 4 face features (head yaw, head pitch, mouth width, mouth pitch).
BODY_BLOCK_VALS  = 8

VALS_PER_FRAME   = VALS_PER_HAND * 2 + BODY_BLOCK_VALS  # 144

SEQUENCE_LENGTH  = 30
FRAME_FEATURES   = VALS_PER_FRAME * SEQUENCE_LENGTH     # 4 320

# -- Global (whole-gesture) features ------------------------------------------
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
    "net_dx",            # signed horizontal displacement (start -> end)
    "net_dy",            # signed vertical displacement — up vs. down signs
    "range_x",           # horizontal extent covered
    "range_y",           # vertical extent covered
    "hands_used",        # 0.0 none, 0.5 one hand, 1.0 both
    "mean_openness",     # average finger extension — separates 3 vs. 4 fingers
    "openness_change",   # how much the hand shape morphed during the sign
)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)           # 12

TOTAL_FEATURES   = FRAME_FEATURES + N_GLOBAL_FEATURES   # 4 332

WRIST_IDX        = 0
MIDDLE_MCP_IDX   = 9                       # used as a stable hand-size proxy


# -----------------------------------------------------------------------------
#  Frame orientation
# -----------------------------------------------------------------------------
#
# Mirror the frame so the signer sees themselves as in a mirror. This MUST be
# identical in data_collector.py and websocket_server.py — see module docstring.
MIRROR_FRAME = True


def prepare_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply the canonical orientation. Call before ANY MediaPipe processing."""
    return _get_cv2().flip(frame_bgr, 1) if MIRROR_FRAME else frame_bgr


# -----------------------------------------------------------------------------
#  Scale normalisation
# -----------------------------------------------------------------------------
#
# Divide the relative (shape) block by palm length so hand shape is invariant
# to how far the signer sits from the camera. See module docstring.
# Changing this REQUIRES re-running migrate_dataset.py and train_model.py —
# a model trained with one setting is meaningless under the other.
NORMALIZE_SCALE = True

# Guard against division by ~0 when the hand is edge-on or badly detected.
_MIN_PALM_LENGTH = 1e-6


# -----------------------------------------------------------------------------
#  Hand splitting (mp.solutions.hands has no left/right attributes)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
#  Hybrid feature extraction:  raw wrist (location) + relative shape
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#  Body anchors — the reference frame that makes location meaningful
# -----------------------------------------------------------------------------

# Pose is markedly slower than Hands, and a seated signer's torso barely moves
# between frames. Running it every Nth frame and reusing the last anchors keeps
# the reference accurate while cutting most of the cost — important on a Pi.
# Measured at 24 ms per Pose call, which is 8 ms amortised at 1-in-3 - a sixth
# of the whole frame budget spent locating shoulders that barely move between
# frames. 1-in-5 costs 4.8 ms and the anchors are cached in between, so nothing
# downstream can tell the difference.
POSE_EVERY_N_FRAMES = 5

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
        self.last_results = None

    def update(self, rgb_frame) -> "BodyAnchors":
        """
        Return current anchors, refreshing them only every Nth call.

        `rgb_frame` must already be RGB and mirrored — the same frame handed to
        Hands, so both models see an identical world.
        """
        if self._counter % self._every_n == 0:
            try:
                results = self._pose.process(rgb_frame)
                self.last_results = results
                h, w = rgb_frame.shape[:2]
                self._anchors = BodyAnchors.from_pose(results, aspect=h / w)
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

    __slots__ = ("valid", "origin", "scale", "points", "face_size", "tilt",
                 "head_yaw", "head_pitch", "mouth_width", "mouth_pitch",
                 "aspect")

    def __init__(self):
        self.valid = False
        self.origin = np.zeros(2, dtype=np.float32)
        self.scale = 1.0
        # height / width of the frame these landmarks came from. See from_pose.
        self.aspect = 1.0
        self.points = {}          # name -> (x, y) in body coordinates
        self.face_size = 0.0
        self.tilt = 0.0
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.mouth_width = 0.0
        self.mouth_pitch = 0.0

    # -- Construction --------------------------------------------------------

    @classmethod
    def from_pose(cls, pose_results, aspect: float = 1.0) -> "BodyAnchors":
        """
        Build the body frame from a Pose result.

        `aspect` is the frame's HEIGHT / WIDTH, and it is not optional in
        practice — passing the default of 1.0 is only correct for a square
        frame, which no camera produces.

        Why it is needed
        ----------------
        MediaPipe normalises `lm.x` by the image WIDTH and `lm.y` by its
        HEIGHT. On a 640x480 frame one unit of x spans 640 px and one unit of y
        spans 480, so the two axes are not the same length and the coordinates
        are not isotropic: the same physical distance reads 1.33x larger
        vertically than horizontally.

        Everything downstream then divides BOTH axes by `scale`, the shoulder
        width - a purely HORIZONTAL measurement. So every vertical distance in
        this pipeline was overstated by width/height. The chin was recorded a
        third higher above the shoulders than it really sits, and the avatar,
        told to put its hand where the recording said, put it somewhere else.
        Worse, the error changes with the camera: the same sign shot at 16:9
        produced different numbers than at 4:3, so the recogniser was learning
        part of the lens.

        Multiplying y by height/width here restores square units - x and y then
        both measure in image WIDTHS - and every distance, angle and anchor
        below inherits that. Callers must hand hand landmarks through the same
        correction; `extract_hand_features` does it from `self.aspect`.
        """
        a = cls()
        if pose_results is None or not getattr(pose_results, "pose_landmarks", None):
            return a

        aspect = float(aspect) if aspect and aspect > 0 else 1.0
        a.aspect = aspect
        lms = pose_results.pose_landmarks.landmark

        def pt(i):
            lm = lms[i]
            return (np.array([lm.x, lm.y * aspect], dtype=np.float32),
                    getattr(lm, "visibility", 1.0))

        sh_l, vis_l = pt(POSE_SHOULDER_L)
        sh_r, vis_r = pt(POSE_SHOULDER_R)

        # Shoulders define the whole frame of reference; without them there is
        # nothing to anchor to.
        if min(vis_l, vis_r) < MIN_POSE_VISIBILITY:
            return a                      # a.aspect is already set

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

        # Face expressions / Head orientation features
        a.head_yaw = float((nose[0] - ear[0]) / width)
        a.head_pitch = float((nose[1] - ear[1]) / width)
        a.mouth_width = float(np.linalg.norm(mouth_l - mouth_r) / width)
        a.mouth_pitch = float((mouth[1] - nose[1]) / width)

        return a

    # -- Use -----------------------------------------------------------------

    def to_body_coords(self, x: float, y: float) -> tuple[float, float]:
        """
        Map an ISOTROPIC frame point into body coordinates.

        `y` must already carry the aspect correction (y_raw * self.aspect).
        Applying it here instead would double-correct the points built in
        `from_pose`, which are already in isotropic units.
        """
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
        """The 8 per-frame body values appended after both hands."""
        return [
            float(self.scale),      # shoulder width in frame units -> distance cue
            float(self.face_size),  # head size relative to shoulders
            float(self.tilt),       # torso lean
            1.0 if self.valid else 0.0,
            float(self.head_yaw),   # horizontal head orientation
            float(self.head_pitch), # vertical head orientation
            float(self.mouth_width),# smiling / mouth shape
            float(self.mouth_pitch) # jaw drop / vertical mouth position
        ]


def hand_features_from_array(coords: np.ndarray) -> list[float]:
    """
    THE canonical hand -> 63-feature transform. Everything else delegates here.

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

    # Same aspect correction as the pose landmarks, for the same reason: MediaPipe
    # normalises y by the frame HEIGHT and x by its WIDTH. Left uncorrected, the
    # hand's own shape is stretched vertically too, so finger directions and the
    # palm normal come out of a squashed hand - which is a quieter error than the
    # misplaced wrist, and the one that survives every position fix.
    aspect = float(getattr(anchors, "aspect", 1.0) or 1.0)
    coords = np.array(
        [[lm.x, lm.y * aspect, lm.z] for lm in hand_landmarks.landmark],
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


# ═════════════════════════════════════════════════════════════════════════════
#  Cleaning a captured take — gaps and jitter
# ═════════════════════════════════════════════════════════════════════════════
#
# Both of these run on the frames a gesture was captured from, BEFORE the
# globals are measured and before the sequence is resampled. That ordering is
# the whole point: the globals (speed, tempo, path) are measured on the raw
# frames precisely because resampling destroys them — so anything that corrupts
# the raw frames corrupts the globals, and it has to be cleaned first.
#
# They live here, and are called from exactly one place (GestureSegmenter), so
# the recorder and the live server clean a take identically. Cleaning in only
# one of the two would rebuild the train/inference mismatch this pipeline was
# rewritten to remove.
#
# Both are off with TARJUMAN_CLEAN_TAKES=0, which is how any A/B of them is run.

# How many consecutive frames a hand may be missing and still be filled in.
# Measured on 391 recorded takes: 28% contained an interior dropout, and 74% of
# those gaps were 3 frames or shorter. Beyond that it stops being a tracking
# blink and starts being the hand genuinely leaving, which is not ours to
# invent.
MAX_HELD_FRAMES = int(_os.getenv("TARJUMAN_MAX_HELD_FRAMES", "3"))

# A hand must actually be tracked before its gaps are worth repairing.
#
# Without this floor the repair does the opposite of its job. Measured on the
# 391 recorded takes: filling every interior gap moved `hands_used` from
# one-handed to two-handed in 42 of them — and in NONE of those 42 were both
# hands present in even 40% of the frames. 33 had a hand below 15%: one take of
# 'hello' had a left hand in a single frame out of thirty. That is a false
# detection, and holding it across its neighbours was enough to make the take
# read as two-handed.
#
# So a hand seen this rarely is left exactly as it is. The threshold matches
# MIN_BONE_PRESENCE in export_signs_3d.py, which draws the same line for the
# same reason.
MIN_HAND_PRESENCE = float(_os.getenv("TARJUMAN_MIN_HAND_PRESENCE", "0.40"))

# One Euro filter. `min_cutoff` sets how hard a motionless hand is smoothed,
# `beta` how quickly smoothing relaxes as the hand speeds up — the property
# that makes this filter worth using over a plain average: it removes tremor
# while standing still without adding lag to a fast sign.
ONE_EURO_MIN_CUTOFF = float(_os.getenv("TARJUMAN_EURO_MIN_CUTOFF", "1.2"))
ONE_EURO_BETA = float(_os.getenv("TARJUMAN_EURO_BETA", "0.35"))
CLEAN_TAKES = _os.getenv("TARJUMAN_CLEAN_TAKES", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _hand_present(frame, hand: int) -> bool:
    """
    Is hand `hand` (0 left, 1 right) in this frame?

    An absent hand is 68 zeros — and that is not a defect to be repaired, it is
    how "this sign uses one hand" is written down. `compute_global_features`
    reads exactly this to decide `hands_used`, so every function below has to
    leave a fully absent hand untouched.
    """
    base = hand * VALS_PER_HAND
    return any(frame[base:base + VALS_PER_HAND])


def fill_tracking_gaps(frames, max_gap: int = None) -> list:
    """
    Carry the last good reading across a brief loss of tracking.

    MediaPipe drops a hand for a frame or two all the time — a finger crosses
    the palm, the hand turns edge-on, the light flickers. The pipeline wrote 68
    zeros for each of those frames, which is not a small error: the hand does
    not move to the origin and back, so every measure taken from the raw frames
    sees a violent excursion that never happened. It is the single largest
    source of the speed spikes that forced `take_quality.analyse()` to use a
    90th percentile instead of the actual maximum.

    Only INTERIOR gaps are filled, and only for a hand that is genuinely being
    tracked (`MIN_HAND_PRESENCE`) — a stretch with a real reading on both sides
    of it, belonging to a hand that is really there. Leading and trailing absences are left alone, because there is no
    reading to carry and nothing to prove the hand was ever there. A hand absent
    for the whole take therefore stays 68 zeros in every frame, and one-handed
    signs are encoded exactly as before.
    """
    if max_gap is None:
        max_gap = MAX_HELD_FRAMES
    out = [list(f) for f in frames]
    n = len(out)
    if n < 3 or max_gap <= 0:
        return out

    for hand in (0, 1):
        base = hand * VALS_PER_HAND
        end = base + VALS_PER_HAND
        seen = [_hand_present(f, hand) for f in out]
        if not any(seen):
            continue                       # never there: leave it that way

        # Barely there is not there. Repairing around a handful of frames turns
        # a false detection into a confident one — see MIN_HAND_PRESENCE.
        if (sum(seen) / n) < MIN_HAND_PRESENCE:
            continue

        first, last = seen.index(True), n - 1 - seen[::-1].index(True)
        i = first
        while i <= last:
            if seen[i]:
                i += 1
                continue
            j = i
            while j <= last and not seen[j]:
                j += 1
            # j <= last is guaranteed by construction, so both sides are real
            if (j - i) <= max_gap:
                carried = out[i - 1][base:end]
                for k in range(i, j):
                    out[k][base:end] = list(carried)
            i = j
    return out


class _OneEuro:
    """
    One Euro filter over a vector, with dt supplied per sample.

    Chosen over a fixed low-pass because its cutoff follows the speed of the
    signal: nearly still -> smooth hard, so tremor disappears; moving fast ->
    barely smooth, so a quick sign is not delayed or rounded off. A fixed
    filter has to pick one of those and lose the other.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x", "_dx")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._x = None
        self._dx = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self._x is None:
            self._x = np.asarray(x, dtype=np.float64)
            self._dx = np.zeros_like(self._x)
            return self._x
        dx = (x - self._x) / max(dt, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))) / max(dt, 1e-6))
        self._x = a * x + (1.0 - a) * self._x
        return self._x


def smooth_tracking_jitter(frames, fps: float = None) -> list:
    """
    Take the tremor out of the landmarks without blunting the sign.

    Measured on 391 takes: the fastest single step in a take is a median 1.4x
    the 90th-percentile step, but reaches 19.7x — a hand appearing to move
    twenty times its own top speed between two frames. Nothing physical does
    that; it is the tracker guessing. Left in, it inflates `peak_speed` and
    `path_length`, which the model reads as if they were the sign's own tempo.

    Each hand is filtered on its own and only across frames where it is
    actually present. The filter is reset whenever a hand disappears, so a
    reading from before a gap is never blended into one after it.
    """
    out = [list(f) for f in frames]
    n = len(out)
    if n < 3:
        return out

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / GLOBALS_REFERENCE_FPS

    for hand in (0, 1):
        base = hand * VALS_PER_HAND
        end = base + VALS_PER_HAND
        euro = _OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
        for i in range(n):
            if not _hand_present(out[i], hand):
                euro.reset()               # do not bridge across an absence
                continue
            block = np.asarray(out[i][base:end], dtype=np.float64)
            out[i][base:end] = euro(block, dt).tolist()
    return out


def clean_take(frames, fps: float = None) -> list:
    """
    Gaps first, then jitter — the order matters.

    Filling a gap after smoothing would feed the filter a zero-excursion it
    then spreads over the neighbouring frames, so the spike survives as a
    smeared version of itself. Filling first means the filter only ever sees
    plausible readings.
    """
    if not CLEAN_TAKES:
        return [list(f) for f in frames]
    return smooth_tracking_jitter(fill_tracking_gaps(frames), fps)


def extract_frame_features(results, anchors: "BodyAnchors" = None) -> list[float]:
    """
    Build the full VALS_PER_FRAME vector for one frame.

    Layout: [left hand 68] + [right hand 68] + [body 8].

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


# -----------------------------------------------------------------------------
#  Distance feedback — replaces the old Pose shoulder-ratio heuristic
# -----------------------------------------------------------------------------

def _hand_openness(hand_block: list) -> float:
    """
    How extended the fingers are, 0 (closed) -> ~1 (fully open).

    Uses the mean distance of the wrist-relative landmarks. Because that block
    is already palm-length normalised, this is comparable across users and
    distances — which is what makes it usable to tell three fingers from four.
    """
    # Shape only: skip the 3 wrist values and stop before the anchor distances.
    relative = np.asarray(hand_block[3:3 + HAND_SHAPE_VALS], dtype=np.float32)
    if not np.any(relative):
        return 0.0
    return float(np.mean(np.abs(relative)))


# The sampling density these features are DEFINED at. Speed, path length and
# openness range are all measured from frame-to-frame steps, so they depend on
# how finely the motion was sampled: the same sign recorded at 30 fps yields a
# 5-21% different peak_speed than at 20 fps. Left alone, that turns capture rate
# into a hidden input the model was never meant to see - and since the existing
# dataset was recorded at 20 fps, running inference at 30 would quietly feed the
# model out-of-distribution values.
#
# Normalising to a fixed density removes the dependence entirely, and 20 is
# chosen so previously recorded data remains exactly what it always was.
GLOBALS_REFERENCE_FPS = 20.0


def _resample_to_rate(arr, duration_seconds: float):
    """Re-grid a capture to GLOBALS_REFERENCE_FPS, whatever rate it arrived at."""
    target = max(2, int(round(duration_seconds * GLOBALS_REFERENCE_FPS)))
    n = arr.shape[0]
    if n == target:
        return arr
    idx = np.linspace(0.0, n - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo).astype(np.float32)[:, None]
    return arr[lo] * (1.0 - w) + arr[hi] * w


def compute_global_features(frames, duration_seconds: float) -> list[float]:
    """
    Summarise a whole gesture into N_GLOBAL_FEATURES values.

    MUST be called on the RAW captured frames, before the sequence is resampled
    to SEQUENCE_LENGTH — the point of these features is to preserve exactly what
    that resampling throws away.

    The raw frames ARE, however, re-gridded to a fixed sampling density first.
    That is a different operation and it is what makes these values comparable
    across capture rates: without it, a faster camera reports a faster sign.

    Parameters
    ----------
    frames : list of per-frame vectors, each VALS_PER_FRAME long
    duration_seconds : real wall-clock length of the capture
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return [0.0] * N_GLOBAL_FEATURES

    duration = max(float(duration_seconds), 1e-3)
    arr = _resample_to_rate(arr, duration)
    if arr.shape[0] < 2:
        return [duration] + [0.0] * (N_GLOBAL_FEATURES - 1)
    n_frames = arr.shape[0]

    # -- Wrist trajectory (whichever hand is present, averaged) --------------
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

    # -- Hand openness over time ---------------------------------------------
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

    # -- How many hands this sign uses ---------------------------------------
    # This was `max(hands_present)` — the largest count seen in ANY SINGLE
    # frame — which let one frame decide a feature for the whole take. One
    # spurious detection of a second hand, and a one-handed sign was recorded
    # as two-handed.
    #
    # It was not a rare edge: of the 391 takes in the previous dataset, 47 were
    # labelled two-handed and 23 of those — 49% — had a second hand present in
    # under 15% of their frames. One take of 'name' was called two-handed on
    # the strength of a left hand appearing in a single frame out of thirty.
    #
    # A hand now counts when it is actually tracked through the sign rather
    # than glimpsed once. Same threshold as the gap repair above, for the same
    # reason: below it, a detection is noise.
    n_frames = len(arr)
    active = 0
    for h in range(2):
        seen = sum(1 for row in arr
                   if np.any(row[h * VALS_PER_HAND:(h + 1) * VALS_PER_HAND]))
        if n_frames and (seen / n_frames) >= MIN_HAND_PRESENCE:
            active += 1
    hands_used = 0.0 if active == 0 else (0.5 if active == 1 else 1.0)

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

    Uses palm length — the wrist -> middle-finger-MCP distance — because unlike
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
