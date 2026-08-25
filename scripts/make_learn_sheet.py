"""
make_learn_sheet.py — build learn.csv, the recording progress sheet
====================================================================
Produces a printable / Excel-friendly checklist of every term the model has to
be taught, with a ✔ column that fills itself in from the data you have actually
recorded so far.

Why generate it rather than hand-maintain it
--------------------------------------------
A hand-kept checklist drifts from reality the moment a recording session runs
long. This reads `dynamic_gestures_v3.csv` directly, so the ✔ column is always
the truth. Re-run it any time to refresh:

    python make_learn_sheet.py
    npm run sheet

Excel note: the file is written with a UTF-8 BOM. Without it, Excel on Windows
opens Arabic CSVs as mojibake — the single most common reason a perfectly good
export "doesn't work".
"""

# -- Import bootstrap ---------------------------------------------------------
# Puts src/ on the path so `tarjuman_core` resolves when this file is run
# directly (`python make_learn_sheet.py`). Running through `npm run ...` sets PYTHONPATH
# instead, and `pip install -e .` makes both unnecessary - this is the belt to
# those braces, so a plain `python` invocation never fails with ImportError.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.basename(_os.path.dirname(_os.path.abspath(__file__))) == "scripts"
    else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "src"))

import csv
import os
import sys

from tarjuman_core.paths import data
from tarjuman_core.vocabulary import (
    LOCATION_AR, MOTION_AR, ORIENT_AR, SHAPE_AR,
    as_dicts, category_of, signature, validate,
)

DATA_CSV   = data("dynamic_gestures_v4.csv")
OUTPUT_CSV = data("learn.csv")
TARGET_PER_LABEL = 30

DONE_MARK    = "✔"
PARTIAL_MARK = "…"
TODO_MARK    = ""


def recorded_counts(path: str) -> dict:
    """How many samples exist per label, read straight from the dataset."""
    counts = {}
    if not os.path.isfile(path):
        return counts
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)                 # header
            for row in reader:
                if row and row[0]:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    except OSError as exc:
        print(f"[WARN] could not read {path}: {exc}")
    return counts


def main() -> int:
    errors = validate()
    if errors:
        print("[FAIL] vocabulary contains uncontrolled values — fix these first:")
        for e in errors[:10]:
            print("   ", e)
        return 1

    vocab = as_dicts()
    counts = recorded_counts(DATA_CSV)

    # Flag collisions so the sheet warns you BEFORE you record them
    seen = {}
    collides = set()
    for e in vocab:
        sig = signature(e)
        if sig in seen:
            collides.add(e["id"])
            collides.add(seen[sig])
        seen[sig] = e["id"]

    # Column headers in English so the file is safe to grep, diff and open
    # anywhere; the DATA columns stay Arabic because that is the content.
    header = [
        "done", "#", "term_ar", "id", "category",
        "handshape", "location", "motion", "orientation", "hands",
        "target", "recorded", "remaining", "warning", "notes",
    ]

    rows = []
    done = partial = 0
    for i, e in enumerate(vocab, 1):
        n = counts.get(e["id"], 0)
        remaining = max(0, TARGET_PER_LABEL - n)

        if n >= TARGET_PER_LABEL:
            mark = DONE_MARK; done += 1
        elif n > 0:
            mark = PARTIAL_MARK; partial += 1
        else:
            mark = TODO_MARK

        warn = []
        if e["id"] in collides:
            warn.append("تصادم — راجع الوصف")
        if e["speed_critical"]:
            warn.append("أدِّها بسرعة — السرعة جزء من المعنى")
        if e["motion"] == "static":
            warn.append("ثابتة — احملها ثانية كاملة")

        rows.append([
            mark, i, e["arabic"], e["id"], category_of(e["id"]),
            SHAPE_AR.get(e["shape"], e["shape"]),
            LOCATION_AR.get(e["location"], e["location"]),
            MOTION_AR.get(e["motion"], e["motion"]),
            ORIENT_AR.get(e["orient"], e["orient"]),
            e["hands"],
            TARGET_PER_LABEL, n, remaining,
            " + ".join(warn), "",
        ])

    # utf-8-sig -> Excel on Windows renders the Arabic correctly
    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    except PermissionError:
        # Excel holds an exclusive lock on an open file. Say so plainly —
        # a bare traceback here looks like a bug in the tool.
        print(f"[FAIL] Cannot write {OUTPUT_CSV} — the file is open in another")
        print( "       program (usually Excel). Close it and run again.")
        return 1
    except OSError as exc:
        print(f"[FAIL] Cannot write {OUTPUT_CSV}: {exc}")
        return 1

    total = len(vocab)
    recorded_total = sum(counts.get(e["id"], 0) for e in vocab)
    needed_total = total * TARGET_PER_LABEL

    print(f"[OK] wrote {OUTPUT_CSV}")
    print(f"     terms        : {total}")
    print(f"     complete     : {done}")
    print(f"     in progress  : {partial}")
    print(f"     not started  : {total - done - partial}")
    print(f"     samples      : {recorded_total} / {needed_total} "
          f"({recorded_total / needed_total * 100:.0f}%)")

    # Show what is still missing, using ids (terminals cannot shape Arabic)
    missing = [(e["id"], TARGET_PER_LABEL - counts.get(e["id"], 0))
               for e in vocab if counts.get(e["id"], 0) < TARGET_PER_LABEL]
    if missing and recorded_total:
        print(f"\n     next up ({len(missing)} terms still short):")
        for term_id, need in missing[:10]:
            print(f"       {term_id:<16s} needs {need} more")
        if len(missing) > 10:
            print(f"       ... and {len(missing) - 10} more")

    if collides:
        print(f"\n[WARN] {len(collides)} terms flagged as colliding — "
              f"review before recording.")
    if not os.path.isfile(DATA_CSV):
        print(f"\n[INFO] {DATA_CSV} does not exist yet, so all counts are zero.")
        print(f"       Start recording with:  npm run collect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
