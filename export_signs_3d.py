"""
export_signs_3d.py — turn recorded signs into robot animations
===============================================================
Reads the dataset you recorded and writes `trained_signs.json`, which the 3D
avatar plays back. Every word you teach the recogniser becomes a word the robot
can perform — no Blender, no hand-authored poses.

    python export_signs_3d.py
    npm run export3d

How it works
------------
The dataset stores, per frame, the hand landmarks that MediaPipe saw. A robot
skeleton needs bone ROTATIONS instead. The two are related by geometry:

  • Finger bones — MediaPipe gives four points per finger, which form three
    segments, exactly matching the rig's three bones per finger. The angle
    between consecutive segments IS the joint's bend angle.

  • Arm bones  — the hand's position in body coordinates is a reach target.
    Two-bone inverse kinematics recovers the shoulder and elbow angles that
    put the hand there. The target is the palm CENTRE, not landmark 0: that
    landmark is the heel of the palm, so aiming at it placed the robot's arm
    a hand's length short of where the real hand was.

Which take is used
------------------
For each word, the MEDOID recording (the one closest to all the others) is
chosen rather than an average. Averaging several takes of a gesture smears the
motion into something no human actually performed; the medoid is a real, clean
recording.

Limitations, stated plainly
---------------------------
  • Depth from a single camera is unreliable, so the arm solve works in the
    image plane. Signs whose meaning depends on moving toward or away from the
    body will look flatter than they are.
  • The rig's rest orientation is assumed to be arms-down. If the robot's poses
    come out rotated, adjust ARM_BASE_* below rather than the maths.
"""

import csv
import json
import os
import sys

import numpy as np

from dtw_matcher import SignReferenceLibrary
from feature_extractor import SEQUENCE_LENGTH

INPUT_CSV   = os.environ.get("TARJUMAN_CSV", "dynamic_gestures_v4.csv")
OUTPUT_JSON = os.path.join("tarjuman", "public", "trained_signs.json")

# How many key poses to keep per sign. The player interpolates between them, so
# a handful of well-chosen moments reproduce the movement; storing all 30 frames
# would bloat the file and add nothing visible.
KEYFRAMES_PER_SIGN = 6

# ------------------------------------------------------------------------------
#  Geometry                                                                      
# ------------------------------------------------------------------------------
# The maths lives in pose_to_bones.py, shared with the live server. Two copies
# would drift, and a drift there means recorded signs and live mirroring move
# differently with nothing in the logs to explain it.

from pose_to_bones import frame_to_bone_dirs   # noqa: E402


def frame_to_pose(frame) -> dict:
    """One 140-value frame -> {boneName: [dx, dy, dz]} directions."""
    return frame_to_bone_dirs(frame)


# -----------------------------------------------------------------------------
#  Export
# -----------------------------------------------------------------------------

def sequence_to_sign(seq: np.ndarray, duration: float = 1.4) -> dict:
    """Pick evenly spaced key poses from a 30-frame sequence."""
    idxs = np.linspace(0, seq.shape[0] - 1, KEYFRAMES_PER_SIGN).astype(int)
    keys = []
    for k, i in enumerate(idxs):
        keys.append({
            "t": round(k / (KEYFRAMES_PER_SIGN - 1), 3),
            "pose": frame_to_pose(seq[i]),
        })
    # `format` tells the player these are DIRECTIONS, not Euler angles — the
    # hand-authored dictionary still uses angles, so both must coexist.
    return {"duration": round(duration, 2), "keys": keys,
            "source": "recorded", "format": "directions"}


def arabic_names() -> dict:
    """id -> Arabic word, taken from vocabulary.py when available."""
    try:
        from vocabulary import as_dicts
        return {e["id"]: e["arabic"] for e in as_dicts()}
    except Exception:
        return {}


def mean_duration(csv_path: str) -> dict:
    """Average recorded duration per label, so playback matches real timing."""
    from feature_extractor import TOTAL_FEATURES
    sums, counts = {}, {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return {}
            dur_col = header.index("g_duration_s") if "g_duration_s" in header else None
            if dur_col is None:
                return {}
            for row in reader:
                if not row or len(row) != TOTAL_FEATURES + 1:
                    continue
                try:
                    d = float(row[dur_col])
                except ValueError:
                    continue
                sums[row[0]] = sums.get(row[0], 0.0) + d
                counts[row[0]] = counts.get(row[0], 0) + 1
    except OSError:
        return {}
    return {k: sums[k] / counts[k] for k in sums if counts[k]}


def main() -> int:
    print("=" * 64)
    print("  EXPORT RECORDED SIGNS -> 3D ROBOT ANIMATIONS")
    print("=" * 64)

    if not os.path.isfile(INPUT_CSV):
        print(f"\n[FAIL] Dataset not found: {INPUT_CSV}")
        print("       Record something first:  npm run collect")
        return 1

    print(f"\n  reading  : {INPUT_CSV}")
    library = SignReferenceLibrary.from_csv(INPUT_CSV)
    if not library.labels:
        print("[FAIL] No usable rows found.")
        return 1

    names = arabic_names()
    durations = mean_duration(INPUT_CSV)

    signs = {}
    print(f"\n  converting {len(library.labels)} sign(s):")
    for term_id in library.labels:
        seq = library.references[term_id]              # medoid, (30, 140)
        dur = durations.get(term_id, 1.4)
        entry = sequence_to_sign(seq, dur)
        entry["id"] = term_id
        entry["label"] = names.get(term_id, term_id)

        moved = len({b for k in entry["keys"] for b in k["pose"]})
        signs[term_id] = entry
        print(f"    {term_id:<18s} {entry['label']:<16s} "
              f"{dur:4.2f}s  {moved:2d} bones")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    payload = {
        "version": 1,
        "generated_from": os.path.basename(INPUT_CSV),
        "sequence_length": SEQUENCE_LENGTH,
        "signs": signs,
    }
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
    except OSError as exc:
        print(f"\n[FAIL] Could not write {OUTPUT_JSON}: {exc}")
        return 1

    kb = os.path.getsize(OUTPUT_JSON) / 1024
    print(f"\n  [OK] wrote {OUTPUT_JSON}  ({kb:.1f} KB)")
    print("       The app picks it up automatically — reload the page.")
    print("       Open the 'الإشارات المدرَّبة' panel to play them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
