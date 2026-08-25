"""
gesture_segmenter.py — Event-driven gesture boundary detection for Tarjuman
============================================================================
Replaces the "classify the last 30 frames, every single frame, forever" model
with "detect when a sign actually starts and ends, then classify it once".

The latency problem this solves
-------------------------------
The old pipeline could not emit anything until a fixed 30-frame window had
filled — roughly 1.2-2 s on a Raspberry Pi — for EVERY sign, no matter how
short. It then re-ran inference on every subsequent frame, which both wasted
CPU while the hand sat still and caused flickering predictions during the
transition between two signs.

Here, latency is bounded by how long the signer actually takes, plus a short
stillness debounce. A quick sign is recognised quickly; a slow one still works.

How it works
------------
Wrist speed (per-frame displacement) drives a small state machine:

    IDLE --speed > START--▶ CAPTURING --still for N frames--▶ emit -▶ COOLDOWN
      ▲                         │                                        │
      `--------------------------┴---- hand disappears / too long --------┘

Hysteresis (START > STOP) prevents rapid toggling around the threshold.

Variable-length -> fixed-length
------------------------------
A captured gesture may be 14 frames or 60. `resample_sequence()` linearly
interpolates it to exactly SEQUENCE_LENGTH frames, so:
  • the existing (30 × 126 = 3 780) model shape keeps working unchanged, and
  • signing speed stops mattering — the same sign performed fast or slow maps
    onto the same normalised representation.
"""

import collections

import numpy as np

from tarjuman_core.feature_extractor import (
    SEQUENCE_LENGTH,
    VALS_PER_FRAME,
    VALS_PER_HAND,
    WRIST_IDX,
    compute_global_features,
)

# The rate every threshold below was originally tuned at, and the rate the
# existing dataset was recorded at. It is now a REFERENCE point rather than an
# assumption: the segmenter measures the real frame rate and rescales itself, so
# raising the capture rate no longer changes when a gesture starts or ends.
#
# Why that matters more than it sounds
# ------------------------------------
# Every threshold used to be counted in FRAMES. At 20 fps "5 still frames" is a
# quarter-second pause; at 45 fps it is a tenth, so signs would be chopped off
# mid-motion. And a per-frame speed threshold gets HARDER to trip as the rate
# rises, because each frame covers less ground - so a faster camera would have
# silently stopped detecting slow signs at all. Worse, the speed and duration
# features feed the model, so a model trained at 20 fps would quietly mispredict
# at 45. Raising fps without this change would have broken the dataset.
REFERENCE_FPS = 20.0
ASSUMED_FPS = REFERENCE_FPS      # kept: older callers import this name

# Bounds for the measured rate, so one hiccuped frame cannot warp the thresholds.
MIN_TRACKED_FPS = 5.0
MAX_TRACKED_FPS = 120.0


# -----------------------------------------------------------------------------
#  Tuning constants
# -----------------------------------------------------------------------------
# Speeds are in normalised image units per frame (a wrist crossing the whole
# frame in 1 s at 30 fps ≈ 0.033/frame). Calibrate on the real Pi camera.

# Expressed per SECOND now. 0.012 per frame at 20 fps is the same motion as
# 0.24 per second, so behaviour at 20 fps is unchanged to the last decimal.
MOTION_START_SPEED_PER_S = 0.012 * REFERENCE_FPS      # 0.24 units/second
MOTION_START_SPEED = 0.012   # per-frame equivalent at REFERENCE_FPS

# A slow, deliberate sign never exceeds MOTION_START_SPEED on any single frame,
# yet it clearly travels. Speed alone would silently ignore it entirely — so we
# ALSO trigger on total displacement away from where the hand was resting.
# DISTANCES are deliberately NOT rescaled. How far a hand has drifted is the
# same physical fact whatever rate you sample it at; only rates and durations
# depend on the clock. Scaling these too would have been a real bug.
MOTION_START_DISPLACEMENT = 0.045   # cumulative drift from the resting anchor
ANCHOR_HISTORY_S          = 12 / REFERENCE_FPS   # 0.60 s of resting history
ANCHOR_HISTORY_FRAMES     = 12      # equivalent at REFERENCE_FPS

STILL_WINDOW_S = 5 / REFERENCE_FPS   # 0.25 s of stillness ends a gesture
STILL_FRAMES_TO_END = 5      # equivalent at REFERENCE_FPS

# Stillness is judged by TOTAL travel across the last STILL_FRAMES_TO_END
# frames, not by instantaneous speed. Per-frame speed is unusable here: a very
# slow sign moves less per frame than sensor noise, so a speed threshold would
# declare it "finished" mid-gesture and truncate it.
STILL_DISPLACEMENT = 0.018   # travel below this over the window = hand at rest
MIN_GESTURE_S      = 6 / REFERENCE_FPS    # 0.30 s - shorter is a twitch
MIN_GESTURE_FRAMES = 6       # equivalent at REFERENCE_FPS

# Sensor noise occasionally spikes past the start threshold, producing short
# "gestures" that never actually went anywhere. A real sign always travels —
# in position, in finger configuration, or both. Segments whose peak deviation
# from their own first frame stays under this are discarded as noise.
MIN_GESTURE_TRAVEL = 0.040
MAX_GESTURE_S      = 90 / REFERENCE_FPS   # 4.5 s hard stop
MAX_GESTURE_FRAMES = 90      # equivalent at REFERENCE_FPS
COOLDOWN_S         = 3 / REFERENCE_FPS    # 0.15 s pause, avoids double-fire
COOLDOWN_FRAMES    = 3       # equivalent at REFERENCE_FPS

# Frames kept before motion is detected. A sign's first moments happen just
# before the speed threshold trips, so we prepend a little history.
PRE_ROLL_S = 4 / REFERENCE_FPS       # 0.20 s of onset kept before the trigger
PRE_ROLL_FRAMES = 4          # equivalent at REFERENCE_FPS


# -----------------------------------------------------------------------------
#  Sequence resampling
# -----------------------------------------------------------------------------

def resample_sequence(frames: list, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Linearly resample a variable-length gesture to exactly `target_len` frames.

    Parameters
    ----------
    frames : list of per-frame feature vectors, each VALS_PER_FRAME long
    target_len : desired number of frames

    Returns
    -------
    ndarray, shape (target_len, VALS_PER_FRAME), dtype float32
    """
    arr = np.asarray(frames, dtype=np.float32)

    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)

    src_positions = np.linspace(0.0, 1.0, num=arr.shape[0])
    dst_positions = np.linspace(0.0, 1.0, num=target_len)

    resampled = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        resampled[:, col] = np.interp(dst_positions, src_positions, arr[:, col])

    return resampled


# -----------------------------------------------------------------------------
#  Wrist speed
# -----------------------------------------------------------------------------

def _wrist_points(frame_features) -> list[tuple[float, float]]:
    """
    Pull the (x, y) wrist coordinates of whichever hands are present.

    Relies on the layout guaranteed by feature_extractor: each 63-value hand
    block begins with the RAW wrist x, y, z, and an absent hand is all zeros.
    """
    points = []
    for hand_idx in range(2):
        base = hand_idx * VALS_PER_HAND
        block = frame_features[base:base + VALS_PER_HAND]
        if any(block):
            points.append((block[WRIST_IDX], block[WRIST_IDX + 1]))
    return points


def _wrist_speed(current, previous) -> float:
    """
    Mean per-frame wrist displacement between two frames.

    Returns 0.0 when there is nothing comparable (no hands, or the set of
    visible hands changed), so an appearing/disappearing hand is never
    mistaken for motion.
    """
    if previous is None:
        return 0.0

    cur = _wrist_points(current)
    prev = _wrist_points(previous)

    if not cur or not prev or len(cur) != len(prev):
        return 0.0

    return float(np.mean([
        np.hypot(c[0] - p[0], c[1] - p[1]) for c, p in zip(cur, prev)
    ]))


def _shape_change(current, previous) -> float:
    """
    Mean absolute change across the wrist-relative (finger) block of each hand.

    Wrist movement alone is not enough to detect a sign: finger-spelling and
    many letters articulate the fingers while the wrist stays essentially
    still. Those gestures would be completely invisible to a wrist-only motion
    metric — the segmenter would never trigger and recognition would appear
    dead for exactly the signs the user makes most often.
    """
    if previous is None:
        return 0.0

    total, counted = 0.0, 0
    for hand_idx in range(2):
        base = hand_idx * VALS_PER_HAND
        cur_block = current[base + 3:base + VALS_PER_HAND]
        prv_block = previous[base + 3:base + VALS_PER_HAND]

        # Only compare when the hand is present in BOTH frames
        if any(cur_block) and any(prv_block):
            total += float(np.mean(np.abs(
                np.asarray(cur_block) - np.asarray(prv_block)
            )))
            counted += 1

    return total / counted if counted else 0.0


# The shape block is normalised by palm length, so its units are much larger
# than the 0-1 image coordinates the wrist lives in. This factor puts the two
# signals on a comparable footing before they are combined.
SHAPE_WEIGHT = 0.35


def _motion(current, previous) -> float:
    """
    Combined motion metric: whichever of "the hand moved" or "the fingers
    moved" is stronger. Covers travelling signs and in-place finger articulation
    with a single threshold.
    """
    return max(
        _wrist_speed(current, previous),
        _shape_change(current, previous) * SHAPE_WEIGHT,
    )


def _displacement(current, anchor) -> float:
    """
    How far the hand has travelled from a reference frame overall — position
    and finger configuration combined — regardless of how gently it got there.
    """
    return _motion(current, anchor)


# -----------------------------------------------------------------------------
#  The state machine
# -----------------------------------------------------------------------------

IDLE      = "idle"
CAPTURING = "capturing"
COOLDOWN  = "cooldown"


class GestureSegmenter:
    """
    Per-client gesture boundary detector.

    Feed it one frame at a time via `update()`. It returns None while a gesture
    is still in progress (or nothing is happening), and a fixed-length
    (SEQUENCE_LENGTH, VALS_PER_FRAME) array the moment a gesture completes —
    that is the only point at which the caller should run inference.
    """

    def __init__(self):
        self.state = IDLE
        self.buffer: list = []
        self.buffer_times: list = []      # wall-clock stamp per buffered frame
        self.pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)
        self.pre_roll_times = collections.deque(maxlen=PRE_ROLL_FRAMES)
        # Longer history used to locate where the hand was "resting", so slow
        # movement is detected by total travel rather than per-frame speed.
        self.anchor_history = collections.deque(maxlen=ANCHOR_HISTORY_FRAMES)
        self.prev_frame = None
        self.still_count = 0
        self.cooldown_count = 0
        self.last_speed = 0.0

        # Measured capture rate, smoothed. Starts at the reference rate so the
        # very first frames behave exactly as before rather than as an outlier.
        self.fps = REFERENCE_FPS
        self.dt = 1.0 / REFERENCE_FPS
        self._last_now = None

    # -- Frame-rate tracking -------------------------------------------------

    def _tick(self, now):
        """Update the measured frame rate and resize the frame-count windows."""
        if now is not None and self._last_now is not None:
            gap = now - self._last_now
            if 1.0 / MAX_TRACKED_FPS <= gap <= 1.0 / MIN_TRACKED_FPS:
                # Exponential smoothing: a single late frame (a GC pause, a
                # dropped USB packet) must not yank every threshold with it.
                self.fps += (1.0 / gap - self.fps) * 0.15
        self._last_now = now
        self.dt = 1.0 / max(self.fps, 1e-6)

        # deques are sized in frames, so they follow the measured rate.
        want_anchor = max(2, round(ANCHOR_HISTORY_S * self.fps))
        if self.anchor_history.maxlen != want_anchor:
            self.anchor_history = collections.deque(self.anchor_history,
                                                    maxlen=want_anchor)
        want_pre = max(1, round(PRE_ROLL_S * self.fps))
        if self.pre_roll.maxlen != want_pre:
            self.pre_roll = collections.deque(self.pre_roll, maxlen=want_pre)
            self.pre_roll_times = collections.deque(self.pre_roll_times,
                                                    maxlen=want_pre)

    # -- Public API ----------------------------------------------------------

    def reset(self) -> None:
        """Drop all in-flight state (call when hands leave or client stops)."""
        self.state = IDLE
        self.buffer = []
        self.buffer_times = []
        self._last_now = None
        self.pre_roll.clear()
        self.pre_roll_times.clear()
        self.anchor_history.clear()
        self.prev_frame = None
        self.still_count = 0
        self.cooldown_count = 0
        self.last_speed = 0.0

    def update(self, frame_features: list, hands_present: bool, now: float | None = None):
        """
        Advance the state machine by one frame.

        Parameters
        ----------
        now : optional wall-clock timestamp (seconds). Supplying it lets the
              real gesture duration be measured instead of inferred from the
              frame count, which matters for the speed-critical signs.

        Returns
        -------
        None while nothing has completed, otherwise a dict:
            {"sequence": ndarray (SEQUENCE_LENGTH, VALS_PER_FRAME),
             "globals":  list[N_GLOBAL_FEATURES],
             "duration": float seconds,
             "frames":   int raw frames captured}
        """
        # Hands gone -> abandon whatever was in flight
        if not hands_present:
            self.reset()
            return None

        self._tick(now)

        speed = _motion(frame_features, self.prev_frame)
        self.last_speed = speed
        self.prev_frame = frame_features

        if self.state == COOLDOWN:
            self.cooldown_count += 1
            if self.cooldown_count >= max(1, round(COOLDOWN_S * self.fps)):
                self.state = IDLE
                self.cooldown_count = 0
            self.pre_roll.append(frame_features)
            self.pre_roll_times.append(now)
            self.anchor_history.append(frame_features)
            return None

        if self.state == IDLE:
            self.pre_roll.append(frame_features)
            self.pre_roll_times.append(now)

            # Two independent triggers:
            #   • fast movement  -> per-frame speed
            #   • slow movement  -> total travel away from the resting anchor
            anchor = self.anchor_history[0] if self.anchor_history else None
            drifted = _displacement(frame_features, anchor) > MOTION_START_DISPLACEMENT

            # Speed is a per-frame displacement, so the threshold has to be
            # scaled by how long a frame lasts. Without this, doubling the frame
            # rate halves the distance covered per frame and the trigger simply
            # stops firing for anything but a very fast sign.
            if speed > MOTION_START_SPEED_PER_S * self.dt or drifted:
                # Start with the pre-roll so the sign's onset isn't clipped
                self.buffer = list(self.pre_roll)
                self.buffer_times = list(self.pre_roll_times)
                self.state = CAPTURING
                self.still_count = 0
                self.anchor_history.clear()
            else:
                self.anchor_history.append(frame_features)
            return None

        # -- CAPTURING -------------------------------------------------------
        self.buffer.append(frame_features)
        self.buffer_times.append(now)

        # Stillness = how far the wrist travelled across the whole window.
        # Using cumulative travel (not per-frame speed) is what lets a very
        # slow sign keep recording instead of being cut off mid-gesture.
        still_window = max(2, round(STILL_WINDOW_S * self.fps))
        at_rest = False
        if len(self.buffer) > still_window:
            window_start = self.buffer[-(still_window + 1)]
            at_rest = _displacement(frame_features, window_start) < STILL_DISPLACEMENT

        ended = at_rest or len(self.buffer) >= round(MAX_GESTURE_S * self.fps)
        if not ended:
            return None

        captured = self.buffer
        times = self.buffer_times
        self.buffer = []
        self.buffer_times = []
        self.state = COOLDOWN
        self.cooldown_count = 0
        self.still_count = 0

        if len(captured) < max(2, round(MIN_GESTURE_S * self.fps)):
            return None      # twitch, not a sign

        # Reject segments that never meaningfully moved (sensor noise spikes)
        origin = captured[0]
        peak_travel = max(_displacement(f, origin) for f in captured)
        if peak_travel < MIN_GESTURE_TRAVEL:
            return None

        # Real elapsed time when the caller supplied timestamps, otherwise
        # inferred from the frame count.
        stamps = [t for t in times if t is not None]
        duration = ((stamps[-1] - stamps[0]) if len(stamps) >= 2
                    else len(captured) / self.fps)
        duration = max(duration, 1e-3)

        # Globals are computed on the RAW capture — resampling would erase them
        globals_ = compute_global_features(captured, duration)

        return {
            "sequence": resample_sequence(captured),
            "globals":  globals_,
            "duration": duration,
            "frames":   len(captured),
        }

    # -- Introspection (useful for the frontend / debugging) -----------------

    @property
    def is_capturing(self) -> bool:
        return self.state == CAPTURING

    @property
    def captured_frames(self) -> int:
        return len(self.buffer)
