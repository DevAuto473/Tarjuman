"""
augment.py — grow the dataset without recording anything
========================================================

Why this exists
---------------
Every sample was produced by one person. The model therefore learned one
person's hands, and nothing measures how it behaves on anyone else. Recording
more signers is the real fix, but it is slow and needs people. Augmentation
buys a large part of the same robustness from the data already on disk.

What it can and cannot do
-------------------------
It CAN remove sensitivity to things that are accidents of the recording:
which hand was used, how far away the signer sat, how fast they moved, and
landmark tracking noise.

It CANNOT invent signing STYLE. How a person forms a sign is a property of
that person, and no transform conjures a second human. Augmentation narrows
the signer-independent gap; only more signers close it.

Working in feature space
------------------------
Raw landmarks were never stored, so every transform operates on the 4212-value
feature vector directly. That makes the geometry easy to get subtly wrong, so
each transform below states exactly which columns it touches and why.

    layout per frame (VALS_PER_FRAME = 140):
        [  0: 68]  left hand    [0:3] wrist in body coords
                                [3:63] 20 landmarks relative to wrist / palm
                                [63:68] distances to nose, mouth, ear,
                                        shoulder, chest
        [ 68:136]  right hand   (same 68-value layout)
        [136:140]  body         shoulder width, face size, tilt, pose flag
    then SEQUENCE_LENGTH frames, then N_GLOBAL_FEATURES global values.

Body coordinates: origin = midpoint between the shoulders, unit = shoulder
width. So the body frame is already scale- and translation-normalised; these
transforms perturb what is left.
"""

import numpy as np

from .feature_extractor import (
    ANCHOR_NAMES, GLOBAL_FEATURE_NAMES, N_ANCHORS, N_GLOBAL_FEATURES,
    SEQUENCE_LENGTH, TOTAL_FEATURES, VALS_PER_FRAME, VALS_PER_HAND,
)

FRAME_BLOCK = VALS_PER_FRAME * SEQUENCE_LENGTH
HAND_SHAPE_START = 3                      # after the 3 wrist values
HAND_SHAPE_END = 63                       # 20 landmarks x 3
ANCHOR_START = 63

_ANCHOR_IX = {n: i for i, n in enumerate(ANCHOR_NAMES)}
_G = {n: i for i, n in enumerate(GLOBAL_FEATURE_NAMES)}

# The dominant-side shoulder anchor, in body coordinates.
#
# Not a guess. Recovered from the recorded dataset two independent ways:
#
#   * solving each frame's stored wrist position and shoulder distance for the
#     anchor consistent with them -> x = -0.501, IQR -0.50..-0.50
#   * circle-circle intersection on the 315 frames where BOTH hands are
#     visible, which over-determines the anchor -> x = -0.4998, y = -0.0038,
#     IQR x -0.500..-0.499, y -0.021..+0.008
#
# Both land on exactly what the definition implies, since the origin is the
# midpoint between the shoulders and the unit is the distance between them.
#
# Error budget: reproducing the STORED distance from this idealised anchor is
# accurate to a median of 0.0063 shoulder widths (~2.5 mm) across 11,362
# frames, p90 0.030, worst 0.066. That residual is pose-tracker jitter, not a
# modelling error, and is smaller than the noise `jitter()` adds on purpose.
#
# This anchor matters because it is the ONE that is not on the body midline,
# and therefore the only one whose distance a mirror does not preserve. Note
# the consequence: mirror() is deliberately NOT an exact involution. It
# re-derives the shoulder distance from this anchor instead of carrying the
# original noisy value back, so mirroring twice returns a sample that is
# geometrically right but not bit-identical.
SHOULDER_XY = np.array([-0.5, 0.0], dtype=np.float64)


def _frames_view(sample: np.ndarray) -> np.ndarray:
    """(SEQUENCE_LENGTH, VALS_PER_FRAME) view of the per-frame block."""
    return sample[:FRAME_BLOCK].reshape(SEQUENCE_LENGTH, VALS_PER_FRAME)


def _hand_present(frames: np.ndarray, offset: int) -> np.ndarray:
    """
    Per-frame mask of whether a hand was actually detected.

    An absent hand is written as 68 zeros. Those zeros are meaningful — they
    tell the model "no hand here" — so every transform must leave them alone.
    Rotating or jittering a block of zeros would fabricate a hand that was
    never in the frame.
    """
    block = frames[:, offset:offset + HAND_SHAPE_END]
    return np.abs(block).sum(axis=1) > 1e-9


# ── Mirror ───────────────────────────────────────────────────────────────────

def mirror(sample: np.ndarray) -> np.ndarray:
    """
    Reflect the whole gesture left-to-right.

    The single most valuable transform here, for two reasons. It doubles the
    dataset, and without it a LEFT-HANDED signer is simply unrecognisable —
    every sample was recorded right-handed, so a mirrored gesture is a shape
    the model has never seen.

    Three things must happen together, and skipping any one corrupts the sample:

      1. the two hand blocks swap places (the left hand becomes the right)
      2. every x component flips sign (wrist position and finger shape)
      3. the shoulder anchor distance is RECOMPUTED

    Point 3 is the subtle one. nose, mouth, ear and chest all sit on the body
    midline, so a mirrored hand is exactly as far from them as before and their
    stored distances stay valid. The shoulder anchor does not: it sits at
    x = -0.5, and reflecting the hand across the midline genuinely changes how
    far it is from that shoulder. Keeping the old number would describe a body
    that does not exist.
    """
    out = sample.copy()
    frames = _frames_view(out)

    left = frames[:, 0:VALS_PER_HAND].copy()
    right = frames[:, VALS_PER_HAND:2 * VALS_PER_HAND].copy()

    for src, dst_off in ((right, 0), (left, VALS_PER_HAND)):
        blk = src.copy()
        blk[:, 0] *= -1.0                                     # wrist x
        blk[:, HAND_SHAPE_START:HAND_SHAPE_END:3] *= -1.0     # every landmark x
        frames[:, dst_off:dst_off + VALS_PER_HAND] = blk

    # Recompute the shoulder distance from the now-mirrored wrist position.
    si = ANCHOR_START + _ANCHOR_IX["shoulder"]
    for off in (0, VALS_PER_HAND):
        present = _hand_present(frames, off)
        if not present.any():
            continue
        p = frames[:, [off + 0, off + 1]]
        d = np.linalg.norm(p - SHOULDER_XY, axis=1)
        frames[present, off + si] = d[present]

    # Torso tilt is an angle about the vertical, so it flips with the image.
    frames[:, 2 * VALS_PER_HAND + 2] *= -1.0

    # Horizontal displacement reverses; every other global is a magnitude.
    out[FRAME_BLOCK + _G["net_dx"]] *= -1.0
    return out


# ── Scale ────────────────────────────────────────────────────────────────────

def scale(sample: np.ndarray, factor: float) -> np.ndarray:
    """
    Make the gesture larger or smaller in body units.

    Simulates a signer with different proportions, or one who simply signs
    more expansively. Hand SHAPE is left alone: it is already divided by palm
    length, so scaling it again would model a hand whose fingers grew relative
    to its own palm — an anatomy that does not occur.
    """
    out = sample.copy()
    frames = _frames_view(out)
    for off in (0, VALS_PER_HAND):
        present = _hand_present(frames, off)
        frames[present, off + 0] *= factor          # wrist x
        frames[present, off + 1] *= factor          # wrist y
        for a in range(N_ANCHORS):
            frames[present, off + ANCHOR_START + a] *= factor

    g = out[FRAME_BLOCK:]
    for k in ("mean_speed", "peak_speed", "path_length",
              "net_dx", "net_dy", "range_x", "range_y"):
        g[_G[k]] *= factor
    g[_G["speed_variance"]] *= factor * factor      # a variance of a scaled speed
    return out


# ── Rotate ───────────────────────────────────────────────────────────────────

def rotate(sample: np.ndarray, degrees: float) -> np.ndarray:
    """
    Tilt the hand shape in the image plane.

    Covers signers who hold the hand at a slightly different angle, and camera
    roll. Only the wrist-relative shape rotates; the wrist POSITION does not,
    because moving it would change which body anchor the sign is made against
    and that is the very thing separating several words in this vocabulary.
    """
    out = sample.copy()
    frames = _frames_view(out)
    t = np.radians(degrees)
    cos_t, sin_t = np.cos(t), np.sin(t)
    for off in (0, VALS_PER_HAND):
        present = _hand_present(frames, off)
        if not present.any():
            continue
        blk = frames[np.ix_(present, np.arange(off + HAND_SHAPE_START,
                                               off + HAND_SHAPE_END))]
        pts = blk.reshape(blk.shape[0], -1, 3)
        x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0] = x * cos_t - y * sin_t
        pts[:, :, 1] = x * sin_t + y * cos_t
        frames[np.ix_(present, np.arange(off + HAND_SHAPE_START,
                                         off + HAND_SHAPE_END))] = pts.reshape(blk.shape)
    return out


# ── Jitter ───────────────────────────────────────────────────────────────────

def jitter(sample: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add small Gaussian noise to the landmarks.

    MediaPipe output is never perfectly steady, and two people never form a
    sign identically. Noise teaches the model to rely on the overall shape
    rather than on exact coordinates it will never see reproduced.

    Absent hands stay exactly zero — see _hand_present.
    """
    out = sample.copy()
    frames = _frames_view(out)
    for off in (0, VALS_PER_HAND):
        present = _hand_present(frames, off)
        if not present.any():
            continue
        cols = np.arange(off, off + HAND_SHAPE_END)
        sub = frames[np.ix_(present, cols)]
        frames[np.ix_(present, cols)] = sub + rng.normal(0.0, sigma, sub.shape)
    return out


# ── Time warp ────────────────────────────────────────────────────────────────

def time_warp(sample: np.ndarray, factor: float) -> np.ndarray:
    """
    Sign the same gesture faster or slower.

    The per-frame block is already resampled to a fixed SEQUENCE_LENGTH, so
    speed lives entirely in the global features — which is exactly why those
    globals exist. Warping therefore rescales the globals rather than the
    frames: `factor` > 1 means the gesture took longer.

    Some signs in this vocabulary differ from their neighbours ONLY by tempo,
    so this is kept deliberately gentle. Stretched too far it would turn an
    urgent sign into its calm counterpart and mislabel it.
    """
    out = sample.copy()
    g = out[FRAME_BLOCK:]
    g[_G["duration_s"]] *= factor
    for k in ("mean_speed", "peak_speed"):
        g[_G[k]] /= factor
    g[_G["speed_variance"]] /= factor * factor
    # path_length, ranges and net displacement describe WHERE the hand went,
    # which does not change when the same path is walked at another pace.
    return out


# ── Composite ────────────────────────────────────────────────────────────────

def augment_batch(X: np.ndarray, y: np.ndarray, *, n_copies: int = 2,
                  do_mirror: bool = True, seed: int = 42,
                  scale_range=(0.88, 1.12), rotate_deg: float = 10.0,
                  jitter_sigma: float = 0.006, warp_range=(0.85, 1.18)):
    """
    Expand a TRAINING set. Returns (X_aug, y_aug) including the originals.

    Must only ever be applied to training data. An augmented copy of a test
    sample sitting in the training set is textbook leakage: the model would be
    scored on a near-duplicate of something it had already memorised, and the
    reported accuracy would be meaningless. Keeping augmentation inside the
    training fold — rather than writing expanded rows to the CSV — makes that
    mistake structurally impossible.
    """
    if X.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"expected {TOTAL_FEATURES} features, got {X.shape[1]}")

    rng = np.random.default_rng(seed)
    outX, outY = [X], [y]

    if do_mirror:
        outX.append(np.stack([mirror(x) for x in X]))
        outY.append(y.copy())

    for _ in range(n_copies):
        batch = np.empty_like(X)
        for i, x in enumerate(X):
            s = x
            if do_mirror and rng.random() < 0.5:
                s = mirror(s)
            s = scale(s, rng.uniform(*scale_range))
            s = rotate(s, rng.uniform(-rotate_deg, rotate_deg))
            s = time_warp(s, rng.uniform(*warp_range))
            s = jitter(s, jitter_sigma, rng)
            batch[i] = s
        outX.append(batch)
        outY.append(y.copy())

    return np.concatenate(outX, axis=0), np.concatenate(outY, axis=0)
