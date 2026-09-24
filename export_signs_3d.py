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

# -- Import bootstrap ---------------------------------------------------------
# Puts src/ on the path so `tarjuman_core` resolves when this file is run
# directly (`python export_signs_3d.py`). Running through `npm run ...` sets PYTHONPATH
# instead, and `pip install -e .` makes both unnecessary - this is the belt to
# those braces, so a plain `python` invocation never fails with ImportError.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.basename(_os.path.dirname(_os.path.abspath(__file__))) == "scripts"
    else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "src"))

import csv
import json
import os
import sys

import numpy as np

from tarjuman_core.paths import data, dataset_csv, FRONTEND_DIR
from tarjuman_core.dtw_matcher import SignReferenceLibrary
from tarjuman_core.feature_extractor import SEQUENCE_LENGTH

INPUT_CSV   = os.environ.get("TARJUMAN_CSV", dataset_csv())
OUTPUT_JSON = str(FRONTEND_DIR / "public" / "trained_signs.json")

# How many key poses to keep per sign. The player interpolates between them, so
# a handful of well-chosen moments reproduce the movement; storing all 30 frames
# would bloat the file and add nothing visible.
# كم إطاراً مفتاحياً يُصدَّر لكلّ إشارة.
#
# كان ستّة، تُنتقى بتباعدٍ متساوٍ من الثلاثين ثمّ تُوزَّع بتباعدٍ متساوٍ في
# الزمن. وهذا يُتلف شيئين معاً:
#
#   • السكنات. الوقفة في لغة الإشارة معنى لا فراغ — والوقفةُ الممتدّة على
#     ثمانية إطارات كانت تُختزل إلى إطارٍ واحد، فلا يتوقّف الروبوت أصلاً.
#   • الإيقاع. خمسةٌ وعشرون إطاراً من كل ثلاثين تُرمى، فيمرّ كلُّ شيء
#     بالسرعة نفسها: لا اندفاعَ في بداية الحركة ولا تمهّلَ في نهايتها.
#
# القيمة 0 تعني: صدِّر كل إطار بتوقيته الحقيقي. الثلاثون إطاراً هي أصلاً
# نتيجةُ إعادة تشكيلٍ لِما سُجِّل، فهي تحمل السكنات كما أدّيتها — تظهر
# الوقفةُ فيها إطاراتٍ متتابعة شبه متطابقة، فيقف الروبوت فعلاً.
KEYFRAMES_PER_SIGN = 0

# أقلّ نسبة حضورٍ تجعل العظمة جزءاً من الإشارة لا رصداً عابراً.
MIN_BONE_PRESENCE = 0.40

# ------------------------------------------------------------------------------
#  Geometry                                                                      
# ------------------------------------------------------------------------------
# The maths lives in pose_to_bones.py, shared with the live server. Two copies
# would drift, and a drift there means recorded signs and live mirroring move
# differently with nothing in the logs to explain it.

from tarjuman_core.pose_to_bones import (                      # noqa: E402
    anchors_from_sequence, frame_to_bone_dirs)


def frame_to_pose(frame, anchors=None) -> dict:
    """One feature frame -> {boneName: [dx, dy, dz]} directions."""
    return frame_to_bone_dirs(frame, anchors=anchors)


# -----------------------------------------------------------------------------
#  Export
# -----------------------------------------------------------------------------

def signer_anchors(library) -> dict:
    """
    Where the signer's own landmarks sit, recovered once from the whole dataset.

    The landmarks belong to the PERSON, not to any one sign, so every recording
    is evidence about the same five points. Pooling them is not just tidier, it
    is what makes the recovery possible at all: a single sign's wrist usually
    travels in a near-straight line, and circles centred along a line meet in a
    ridge instead of a point - `anchors_from_sequence` detects that and refuses.
    Forty-three signs reaching in forty-three directions do not have that
    problem.
    """
    seqs = [library.references[t] for t in library.labels]
    if not seqs:
        return {}
    pooled = np.vstack(seqs)
    found = anchors_from_sequence(pooled)
    if found:
        print(f"\n  landmarks: recovered {len(found)} of 5 from "
              f"{pooled.shape[0]} frames")
        for name, (x, y) in found.items():
            print(f"     {name:<9s} ({x:+.3f}, {y:+.3f})")
    else:
        print("\n  [!] Could not recover the signer's landmarks from the data.")
        print("      Hands will be placed at their raw recorded coordinates,")
        print("      so signs that touch the face will land at YOUR proportions")
        print("      rather than the avatar's.")
    return found


def sequence_to_sign(seq: np.ndarray, duration: float = 1.4,
                     anchors: dict = None) -> dict:
    """Pick evenly spaced key poses from a 30-frame sequence."""
    total = seq.shape[0]
    idxs = (range(total) if KEYFRAMES_PER_SIGN <= 0
            else np.linspace(0, total - 1, KEYFRAMES_PER_SIGN).astype(int))
    idxs = list(idxs)
    last = max(1, len(idxs) - 1)

    poses = [frame_to_pose(seq[i], anchors) for i in idxs]

    # ── عظمةٌ تظهر في إطارين ثمّ تختفي ليست إشارة، بل رصدٌ عابر ───────────
    # بتصدير كل الإطارات صارت اليد التي رُصدت في 1% منها تدخل الملفّ، فترتفع
    # الذراع اليسرى إطاراً واحداً ثمّ تسقط — نتوءٌ لم يكن موجوداً حين كانت
    # المفاتيح ستّة. تُسقَط العظام النادرة، وتُملأ فجوات الباقية بآخر قيمة
    # معروفة حتى لا يقفز شيء.
    counts = {}
    for pose in poses:
        for bone in pose:
            counts[bone] = counts.get(bone, 0) + 1
    keep = {b for b, n in counts.items() if n >= MIN_BONE_PRESENCE * len(poses)}

    filled, last = [], {}
    for pose in poses:
        frame = {}
        for bone in keep:
            if bone in pose:
                last[bone] = pose[bone]
            if bone in last:
                frame[bone] = last[bone]
        filled.append(frame)
    # فجوةٌ في البداية: املأها بأوّل قيمة ظهرت، لا بلا شيء.
    first = {}
    for frame in reversed(filled):
        first.update(frame)
    for frame in filled:
        for bone in keep:
            frame.setdefault(bone, first.get(bone))
        for bone in [b for b, v in frame.items() if v is None]:
            del frame[bone]

    keys = []
    for k, i in enumerate(idxs):
        keys.append({
            # التوقيت من موضع الإطار في التسجيل لا من ترتيبه بين المفاتيح،
            # فيبقى الإيقاع الحقيقي حتى لو حُذفت مفاتيح لاحقاً.
            "t": round(int(i) / (total - 1), 4) if total > 1 else 0.0,
            "pose": filled[k],
        })

    # `format` tells the player these are DIRECTIONS, not Euler angles — the
    # hand-authored dictionary still uses angles, so both must coexist.
    #
    # و`easing` يقول للمشغّل كيف يصل بين مفتاحين. المفاتيح هنا كثيفة (كلّ
    # 1/30 من الإشارة) وتحمل تسارعها الطبيعي، فالوصلُ بينها خطّي. أمّا منحنى
    # التنعيم فيُبطئ عند كل مفتاح ويُسرع بينه وبين تاليه — وهو مناسبٌ لثلاثة
    # مفاتيح مكتوبة يدوياً، لكنّه مع ثلاثين مفتاحاً يُنتج نبضاً ميكانيكياً
    # ثلاثين مرّة في الثانية والنصف.
    return {"duration": round(duration, 2), "keys": keys,
            "source": "recorded", "format": "directions", "easing": "linear"}


AR_LABELS_JSON = data("sign_labels_ar.json")


def arabic_names() -> dict:
    """
    id -> Arabic word, for the label the app shows under each sign.

    Two sources, in this order:

      1. `vocabulary.py`   — the planned vocabulary. Each row carries the sign's
                             attributes (handshape, location, movement) as well
                             as its name, which is what the collision analysis
                             needs. A word here is fully described.
      2. `sign_labels_ar.json` — names only, for words recorded before they were
                             added to the vocabulary. This file exists so a new
                             recording shows a real Arabic name immediately,
                             without inventing Stokoe attributes for it that
                             nobody has verified.

    A word missing from BOTH keeps its id as its label - readable, and visibly
    unfinished, which is better than a guess.
    """
    names = {}
    try:
        from tarjuman_core.vocabulary import as_dicts
        names.update({e["id"]: e["arabic"] for e in as_dicts()})
    except Exception as exc:
        print(f"  [!] vocabulary.py unreadable ({exc}) - falling back to ids")

    # Only fills gaps. vocabulary.py stays authoritative for the words it
    # describes, so this file can never silently rename a planned sign.
    try:
        with open(AR_LABELS_JSON, encoding="utf-8") as f:
            extra = json.load(f)
        added = 0
        for term_id, word in extra.items():
            if term_id.startswith("_") or not isinstance(word, str):
                continue
            if term_id not in names:
                names[term_id] = word
                added += 1
        if added:
            print(f"  labels   : +{added} Arabic name(s) from "
                  f"{os.path.basename(AR_LABELS_JSON)}")
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        print(f"  [!] {os.path.basename(AR_LABELS_JSON)} unreadable ({exc})")

    return names


def durations_by_label(csv_path: str) -> dict:
    """
    كل المدد المسجَّلة لكل تسمية، بترتيب الملفّ.

    بالترتيب تحديداً: المصدِّر يأخذ وضعياتِ التسجيلة الوسيطة، فيجب أن يأخذ
    مدّتها هي أيضاً. وكان يأخذ المتوسّط — فتُمدّ حركةٌ سريعة على زمنٍ أبطأ أو
    تُضغط بطيئةٌ في زمنٍ أسرع، وكلاهما يُفسد الإيقاع الذي حُفظ بعناية.
    """
    from tarjuman_core.feature_extractor import TOTAL_FEATURES
    sums = {}
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
                # طولٌ مختلفٌ عن المتوقَّع ليس سبباً للرفض: أعمدة الوصف في
                # مقدّمة الملفّ تغيّره. يكفي أن يبلغ الصفُّ عمودَ المدّة.
                if not row or len(row) <= dur_col:
                    continue
                try:
                    d = float(row[dur_col])
                except ValueError:
                    continue
                sums.setdefault(row[0], []).append(d)
    except OSError:
        return {}
    return sums


def warn_if_stale(csv_path: str) -> None:
    """
    Say so if a NEWER dataset is sitting next to the one being read.

    The avatar's signs were exported from v5 for weeks while the recogniser had
    moved to v6 - so the model answered words the avatar could not perform, and
    nothing anywhere said so. The default comes from TARJUMAN_DATASET, which is
    easy to forget. This does not choose for you; it just refuses to let the
    mismatch pass in silence.
    """
    import glob
    here = os.path.basename(csv_path)
    folder = os.path.dirname(csv_path) or "."
    try:
        mine = os.path.getmtime(csv_path)
    except OSError:
        return
    newer = [(os.path.getmtime(f), os.path.basename(f))
             for f in glob.glob(os.path.join(folder, "dynamic_gestures*.csv"))
             if os.path.basename(f) != here and os.path.getsize(f) > 0
             and os.path.getmtime(f) > mine]
    if not newer:
        return
    newest = max(newer)[1]
    print(f"\n  [!] {newest} is NEWER than {here}.")
    print(f"      The avatar will perform the vocabulary in {here}, which may")
    print(f"      not be the one the model was trained on.")
    print(f"      To use it:  set TARJUMAN_CSV=data/{newest}   (or run "
          f"npm run trainstill)\n")


def main() -> int:
    print("=" * 64)
    print("  EXPORT RECORDED SIGNS -> 3D ROBOT ANIMATIONS")
    print("=" * 64)

    if not os.path.isfile(INPUT_CSV):
        print(f"\n[FAIL] Dataset not found: {INPUT_CSV}")
        print("       Record something first:  npm run collect")
        return 1

    print(f"\n  reading  : {INPUT_CSV}")
    warn_if_stale(INPUT_CSV)
    library = SignReferenceLibrary.from_csv(INPUT_CSV)
    if not library.labels:
        print("[FAIL] No usable rows found.")
        return 1

    names = arabic_names()
    all_durations = durations_by_label(INPUT_CSV)
    anchors = signer_anchors(library)

    signs = {}
    print(f"\n  converting {len(library.labels)} sign(s):")
    for term_id in library.labels:
        seq = library.references[term_id]              # medoid, (30, 140)
        # مدّة التسجيلة الوسيطة نفسها — هي صاحبة هذه الوضعيات.
        takes = all_durations.get(term_id, [])
        index = library.medoid_index.get(term_id, 0)
        dur = takes[index] if 0 <= index < len(takes) else (
            sum(takes) / len(takes) if takes else 1.4)
        entry = sequence_to_sign(seq, dur, anchors)
        entry["id"] = term_id
        entry["label"] = names.get(term_id, term_id)

        moved = len({b for k in entry["keys"] for b in k["pose"]})
        signs[term_id] = entry
        avg = (sum(takes) / len(takes)) if takes else dur
        drift = f"  (mean {avg:.2f})" if abs(avg - dur) > 0.05 else ""
        print(f"    {term_id:<18s} {entry['label']:<16s} "
              f"{dur:4.2f}s  {moved:2d} bones{drift}")

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
    print("       Open the trained-signs panel in the app to play them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
