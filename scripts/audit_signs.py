"""
audit_signs.py — لماذا تتشابه إشارتان على الروبوت؟
================================================================================
الروبوت لا «يخلط» بين الكلمات: هو يؤدّي ما سُجِّل بأمانة. فإن بدت كلمتان
متشابهتين على الشاشة، فالسبب أنّ التسجيلتين متشابهتان فعلاً — ولا يصلح ذلك
كودٌ، بل إعادةُ تسجيل.

يفحص هذا التقرير ثلاثة أشياء، ولكلٍّ منها أثرٌ بصريّ مباشر:

  ١. أيّ يدٍ رُصدت فعلاً. الإشارة ثنائية اليد المسجَّلة بيدٍ واحدة تفقد نصفها،
     ولا يبقى ما يفرّقها عن أخرى.

  ٢. مسار الكفّ: من أين يبدأ، وكم يرتفع، وأين ينتهي، وكم يقطع. إشارةٌ تبدأ
     واليدُ مرفوعةٌ أصلاً تعني أنّ المقطِّع فاتته بدايةُ الحركة.

  ٣. أقرب الأزواج. المسافة بين مسارَي الكفّ هي أقرب ما يكون إلى ما تراه العين.

    npm run audit
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import numpy as np

from tarjuman_core.paths import data, dataset_csv
from tarjuman_core.dataset import iter_samples
from tarjuman_core.dtw_matcher import SignReferenceLibrary, dtw_distance, similarity_score
from tarjuman_core.feature_extractor import (
    FRAME_FEATURES, SEQUENCE_LENGTH, VALS_PER_FRAME, VALS_PER_HAND,
)
from tarjuman_core.pose_to_bones import hand_centre

CSV = _os.environ.get("TARJUMAN_CSV", dataset_csv())

# عتبات يُبنى عليها الحكم، مجموعة في مكان واحد ليسهل ضبطها.
LOW_HAND_PCT = 15.0      # حضورُ يدٍ دون هذا يعني أنّها غائبة عملياً
SHORT_PATH = 1.00        # مسارٌ أقصر من هذا حركةٌ باهتة
HIGH_START = 0.35        # بدايةٌ أعلى من هذا تعني أنّ اليد كانت مرفوعة سلفاً
CLOSE_PAIR = 0.45        # مسافةُ مسارٍ أقلّ من هذا = زوجٌ متشابه


def vocabulary_info() -> dict:
    """id -> (النصّ العربي، عدد اليدين المطلوب). فارغة إن تعذّر الاستيراد."""
    try:
        from tarjuman_core.vocabulary import as_dicts
        return {e["id"]: (e["arabic"], int(e.get("hands", 1))) for e in as_dicts()}
    except Exception:
        return {}


def hand_presence(csv_path):
    """لكل تسمية: نسبة الإطارات التي رُصدت فيها كل يد."""
    out = {}
    counts = {}
    for label, values in iter_samples(csv_path):
        frames = values[:FRAME_FEATURES].reshape(SEQUENCE_LENGTH, VALS_PER_FRAME)
        acc = out.setdefault(label, [0, 0, 0, 0])   # يسرى، يمنى، معاً، الكلّ
        counts[label] = counts.get(label, 0) + 1
        for f in frames:
            left = bool(np.any(f[0:VALS_PER_HAND]))
            right = bool(np.any(f[VALS_PER_HAND:2 * VALS_PER_HAND]))
            acc[0] += left
            acc[1] += right
            acc[2] += left and right
            acc[3] += 1
    return {k: (100 * v[0] / v[3], 100 * v[1] / v[3], 100 * v[2] / v[3], counts[k])
            for k, v in out.items()}


def hand_path(reference, side_index):
    """مسار مركز الكفّ عبر إطارات المرجع. NaN حيث لا يدَ مرصودة."""
    pts = []
    for frame in reference:
        block = frame[side_index * VALS_PER_HAND:(side_index + 1) * VALS_PER_HAND]
        pts.append(hand_centre(block) if np.any(block) else np.full(3, np.nan))
    return np.asarray(pts, dtype=float)


def main() -> int:
    print("=" * 70)
    print("  RECORDED SIGN AUDIT")
    print("=" * 70)

    if not _os.path.isfile(CSV):
        print(f"\n[FAIL] no dataset: {CSV}")
        print("       record first:  npm run collect")
        return 1

    vocab = vocabulary_info()
    name = lambda k: vocab.get(k, (k, 1))[0]
    hands_needed = lambda k: vocab.get(k, (k, 1))[1]

    # ── ١. حضور اليدين ────────────────────────────────────────────────────
    presence = hand_presence(CSV)
    if not presence:
        print(f"\n[FAIL] no usable rows in {CSV}")
        return 1

    print(f"\n1. WHICH HAND WAS ACTUALLY SEEN   ({CSV})\n")
    print(f"   {'word':<16s}{'takes':>7s}{'left':>8s}{'right':>8s}{'both':>8s}")
    print("   " + "-" * 62)
    # يدٌ غائبة ليست خطأً في ذاتها: معظم الإشارات بيدٍ واحدة أصلاً. الخطأ أن
    # تكون المفردة معرَّفةً بيدين في vocabulary.py ثمّ تُسجَّل بيدٍ واحدة.
    missing_hand = []
    for k in sorted(presence, key=lambda k: presence[k][0]):
        l, r, b, n = presence[k]
        need = hands_needed(k)
        mark = "  <- needs two hands" if need == 2 and b < LOW_HAND_PCT else ""
        print(f"   {name(k):<16s}{n:>7d}{l:>7.0f}%{r:>7.0f}%{b:>7.0f}%{mark}")
        if need == 2 and b < LOW_HAND_PCT:
            missing_hand.append((k, b))

    # ── ٢. مسار الكفّ ─────────────────────────────────────────────────────
    library = SignReferenceLibrary.from_csv(CSV)
    if not library.references:
        print("\n[FAIL] could not build references.")
        return 1

    paths, stats = {}, {}
    for k, ref in library.references.items():
        # اليد اليمنى هي العاملة في معظم التسجيلات؛ إن غابت فاليسرى.
        p = hand_path(ref, 1)
        if np.isnan(p).all():
            p = hand_path(ref, 0)
        paths[k] = p
        up = -p[:, 1]                       # محور الملفّ +y لأسفل
        travel = float(np.nansum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        stats[k] = (float(up[0]), float(np.nanmax(up)), float(up[-1]), travel)

    print("\n2. HAND PATH  (height relative to the shoulder line, + is up)\n")
    print(f"   {'word':<16s}{'starts at':>10s}{'peak':>8s}{'ends at':>9s}{'travel':>9s}   note")
    print("   " + "-" * 70)
    for k in sorted(stats, key=lambda k: -stats[k][1]):
        s0, peak, s1, travel = stats[k]
        notes = []
        if s0 > HIGH_START:
            notes.append("starts with hand raised")
        if travel < SHORT_PATH:
            notes.append("faint motion")
        if abs(s1) > HIGH_START:
            notes.append("does not return to rest")
        print(f"   {name(k):<16s}{s0:>10.2f}{peak:>8.2f}{s1:>9.2f}{travel:>9.2f}   "
              + ", ".join(notes))

    # ── ٣. أقرب الأزواج ───────────────────────────────────────────────────
    keys = sorted(paths)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pa, pb = paths[a], paths[b]
            ok = ~(np.isnan(pa).any(1) | np.isnan(pb).any(1))
            if ok.sum() < 5:
                continue
            visual = float(np.linalg.norm(pa[ok] - pb[ok], axis=1).mean())
            score = similarity_score(
                dtw_distance(library.references[a], library.references[b]))
            pairs.append((visual, score, a, b))
    pairs.sort()

    print("\n3. CLOSEST PAIRS  (smaller = look more alike)\n")
    print(f"   {'pair':<34s}{'path':>8s}{'DTW':>8s}")
    print("   " + "-" * 50)
    for visual, score, a, b in pairs[:10]:
        flag = "  <- look alike" if visual < CLOSE_PAIR else ""
        print(f"   {name(a) + ' ↔ ' + name(b):<34s}{visual:>8.3f}{score:>8.1f}{flag}")

    # ── الخلاصة ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WHAT TO DO")
    print("=" * 70)
    todo = []
    for k, both in missing_hand:
        todo.append(f"'{name(k)}': declared two-handed in vocabulary.py, but both "
                    f"hands appear together in only {both:.0f}% of frames - "
                    "re-record with both hands inside the frame.")
    for k, (s0, _, s1, travel) in stats.items():
        if s0 > HIGH_START:
            todo.append(f"'{name(k)}': takes begin with the hand already raised - "
                        "start still at your side, sign, then lower it again.")
        elif travel < SHORT_PATH:
            todo.append(f"'{name(k)}': short path ({travel:.2f}) - sign it wider and "
                        "a little slower so the segmenter catches all of it.")
    for visual, _, a, b in pairs[:4]:
        if visual < CLOSE_PAIR:
            todo.append(f"'{name(a)}' and '{name(b)}' are very close ({visual:.2f}) - "
                        "exaggerate the difference in position or direction.")

    if todo:
        for i, line in enumerate(dict.fromkeys(todo), 1):
            print(f"  {i}. {line}")
    else:
        print("  Nothing - every sign is distinct and complete.")
    print()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
