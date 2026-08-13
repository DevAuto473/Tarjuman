"""
dtw_matcher.py — Dynamic Time Warping similarity for "تعلم مع ترجمان"
======================================================================
Scores how closely a user's attempt at a sign matches a reference recording,
returning a continuous 0-100 percentage instead of a hard class label.

Why DTW and not the classifier
------------------------------
The recognition model answers "which of my N known signs is this?" — it must
pick one, and it says nothing about *quality*. Teaching needs a different
question: "how close was this attempt to the correct sign?" DTW answers that
directly by aligning the two sequences in time and measuring the residual
distance.

It also sidesteps the classifier's biggest practical limitation for a
dictionary: adding a new word requires retraining the whole model. With DTW you
record ONE reference clip and the word is immediately usable.

Why "time warping" specifically
-------------------------------
A learner performs a sign more slowly and unevenly than the reference — pausing
mid-gesture, rushing the ending. A frame-by-frame comparison would punish that
harshly even when the movement is correct. DTW finds the best non-linear
alignment between the two timelines first, so it grades the SHAPE of the
movement rather than its exact tempo.
"""

import csv
import os

import numpy as np

from feature_extractor import (
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    VALS_PER_FRAME,
    VALS_PER_HAND,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Tuning
# ─────────────────────────────────────────────────────────────────────────────

# Sakoe-Chiba band: how far the alignment may drift from the diagonal, as a
# fraction of sequence length. Without a band, DTW can align a wild attempt to
# a reference by warping absurdly, inflating the score. It also makes the
# algorithm meaningfully faster.
WARP_BAND_RATIO = 0.25

# Maps raw DTW distance → 0-100 score. Distance at which the score hits ~37 %.
# Calibrate against real recordings of correct vs. deliberately wrong attempts.
SCORE_DECAY = 0.55

# Weighting between the two halves of the feature vector. Hand SHAPE is what a
# learner most needs to get right; absolute position matters but is noisier
# (it depends where they stand), so it is weighted lower.
WRIST_WEIGHT = 0.4
SHAPE_WEIGHT = 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Core DTW
# ─────────────────────────────────────────────────────────────────────────────

def _frame_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Weighted Euclidean distance between every frame of `a` and every frame of
    `b`, shape (len(a), len(b)).

    Per hand the 63 values are [wrist(3) | shape(60)]; the two groups are
    weighted differently (see WRIST_WEIGHT / SHAPE_WEIGHT).
    """
    weights = np.ones(VALS_PER_FRAME, dtype=np.float32)
    for hand_idx in range(2):
        base = hand_idx * VALS_PER_HAND
        weights[base:base + 3] = WRIST_WEIGHT
        weights[base + 3:base + VALS_PER_HAND] = SHAPE_WEIGHT

    aw = a * weights
    bw = b * weights

    # (len(a), 1, F) - (1, len(b), F) → pairwise differences
    diff = aw[:, None, :] - bw[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def dtw_distance(a: np.ndarray, b: np.ndarray,
                 band_ratio: float = WARP_BAND_RATIO) -> float:
    """
    Length-normalised DTW distance between two (T, VALS_PER_FRAME) sequences.

    Returns
    -------
    float — mean distance along the optimal alignment path. Lower is better;
    0.0 means identical.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    cost = _frame_distance_matrix(a, b)

    band = max(1, int(round(max(n, m) * band_ratio)))

    # acc[i, j] = cheapest total cost to align a[:i+1] with b[:j+1]
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        # Only evaluate cells near the diagonal (Sakoe-Chiba band)
        j_lo = max(1, int(i * m / n) - band)
        j_hi = min(m, int(i * m / n) + band)
        for j in range(j_lo, j_hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j],      # insertion
                acc[i, j - 1],      # deletion
                acc[i - 1, j - 1],  # match
            )

    total = acc[n, m]
    if not np.isfinite(total):
        return float("inf")

    # Normalise by path length so long sequences are not penalised
    return float(total / (n + m))


def similarity_score(distance: float, decay: float = SCORE_DECAY) -> float:
    """
    Convert a DTW distance into a friendly 0-100 score.

    Exponential decay keeps small errors forgiving while still separating a
    good attempt from a wrong one.
    """
    if not np.isfinite(distance):
        return 0.0
    return float(round(100.0 * np.exp(-distance / decay), 1))


# ─────────────────────────────────────────────────────────────────────────────
#  Reference library
# ─────────────────────────────────────────────────────────────────────────────

class SignReferenceLibrary:
    """
    Holds one canonical reference sequence per sign label.

    Built from the same CSV the classifier trains on, so no extra recording
    session is required to start using practice mode.
    """

    def __init__(self):
        self.references: dict[str, np.ndarray] = {}

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str) -> "SignReferenceLibrary":
        """
        Load every sample, then keep the MEDOID of each label — the recording
        with the smallest total distance to all other recordings of that sign.

        The medoid is used rather than a frame-wise average because averaging
        several takes of a gesture smears the movement into something no human
        actually performed. The medoid is a real, clean recording.
        """
        lib = cls()
        if not os.path.isfile(path):
            print(f"⚠️  DTW reference CSV not found: {path}")
            return lib

        by_label: dict[str, list[np.ndarray]] = {}

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)                       # header
            for row in reader:
                if not row or len(row) != TOTAL_FEATURES + 1:
                    continue
                label = row[0]
                try:
                    values = np.asarray(row[1:], dtype=np.float32)
                except ValueError:
                    continue
                by_label.setdefault(label, []).append(
                    values.reshape(SEQUENCE_LENGTH, VALS_PER_FRAME)
                )

        for label, samples in by_label.items():
            lib.references[label] = _medoid(samples)

        print(f"✅  DTW references loaded: {len(lib.references)} sign(s) "
              f"from {os.path.basename(path)}")
        return lib

    # ── Scoring ─────────────────────────────────────────────────────────────

    def score(self, label: str, attempt: np.ndarray) -> dict | None:
        """
        Grade one attempt against the reference for `label`.

        Returns None when the label has no reference recording.
        """
        reference = self.references.get(label)
        if reference is None:
            return None

        distance = dtw_distance(attempt, reference)
        score = similarity_score(distance)

        if   score >= 80: verdict, quality = "ممتاز! إشارة صحيحة", "great"
        elif score >= 60: verdict, quality = "جيد — اقتربتَ كثيراً", "good"
        elif score >= 40: verdict, quality = "مقبول — راجع شكل اليد", "fair"
        else:             verdict, quality = "أعد المحاولة — الحركة مختلفة", "poor"

        return {
            "label":    label,
            "score":    score,
            "distance": round(distance, 4),
            "verdict":  verdict,
            "quality":  quality,
        }

    def best_match(self, attempt: np.ndarray) -> dict | None:
        """Score the attempt against EVERY reference and return the closest."""
        results = [
            r for r in (self.score(lbl, attempt) for lbl in self.references)
            if r is not None
        ]
        return max(results, key=lambda r: r["score"]) if results else None

    @property
    def labels(self) -> list[str]:
        return sorted(self.references)


def _medoid(samples: list[np.ndarray]) -> np.ndarray:
    """Return the sample with the smallest summed DTW distance to the others."""
    if len(samples) == 1:
        return samples[0]

    # Cap the comparison set: medoid search is O(n²) in DTW calls, and beyond a
    # couple of dozen takes the extra precision is not worth the startup cost.
    pool = samples[:24]

    best_idx, best_total = 0, float("inf")
    for i, candidate in enumerate(pool):
        total = sum(
            dtw_distance(candidate, other)
            for j, other in enumerate(pool) if i != j
        )
        if total < best_total:
            best_idx, best_total = i, total

    return pool[best_idx]
