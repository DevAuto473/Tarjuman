"""
add_provenance.py — tag the existing dataset with who recorded it
=================================================================
    npm run provenance                    show what would change
    npm run provenance -- --apply         do it

Adds signer / camera / session / recorded_at columns to a dataset that predates
them, and restores the header row that a rewrite dropped.

Nothing is destroyed. The samples are copied verbatim, a timestamped backup is
written before the swap, and the result is only kept if the row count and
column width both check out. Re-running it is a no-op.

Defaults to dry-run, because the user's standing instruction on this project is
that nothing touching their recordings happens without an explicit go-ahead.
"""

# -- Import bootstrap ---------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import argparse
import collections
import csv
import os
import sys

from tarjuman_core.paths import data
from tarjuman_core.provenance import (
    N_PROVENANCE, PROVENANCE_COLUMNS, TOTAL_COLUMNS, detect_layout, migrate,
    slugify,
)


def summarise(path: str):
    counts = collections.Counter()
    widths = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        rows = [] if (first and first[0] == "label") else ([first] if first else [])
        for row in list(rows) + list(reader):
            if row:
                counts[row[0]] += 1
                widths.add(len(row))
    return counts, widths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=data("dynamic_gestures_v4.csv"))
    ap.add_argument("--signer", default="yazid",
                    help="who recorded the EXISTING samples")
    ap.add_argument("--camera", default="logi-c310")
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default is a dry run)")
    args = ap.parse_args()

    print("=" * 66)
    print("   ADD PROVENANCE COLUMNS")
    print("=" * 66)

    if not os.path.isfile(args.csv):
        print(f"   [FAIL] not found: {args.csv}")
        return 1

    layout = detect_layout(args.csv)
    counts, widths = summarise(args.csv)
    signer = slugify(args.signer, "unknown")
    camera = slugify(args.camera, "unknown")

    print(f"   file    : {args.csv}")
    print(f"   layout  : {layout}")
    print(f"   samples : {sum(counts.values()):,}   column widths: {sorted(widths)}")
    for k, v in sorted(counts.items()):
        print(f"      {k:<18s} {v}")

    if layout == "provenance":
        print("\n   Already has provenance columns — nothing to do.")
        return 0
    if layout == "missing":
        print("\n   Empty dataset — nothing to do.")
        return 0

    print(f"\n   Will add: {', '.join(PROVENANCE_COLUMNS)}")
    print(f"   Existing samples tagged  signer={signer}  camera={camera}")
    print(f"   Width {sorted(widths)[0]} -> {TOTAL_COLUMNS}"
          f"  (+{N_PROVENANCE} columns, features untouched)")

    result = migrate(args.csv, signer=signer, camera=camera,
                     session="legacy", dry_run=not args.apply)

    if result.get("rescued_first_row"):
        print("\n   NOTE: this file has no header, so its first line is a DATA row.")
        print("   pandas has been reading that row as column names and dropping")
        print("   the sample on every training run. It is recovered here.")

    if not args.apply:
        print(f"\n   DRY RUN — would write {result['rows']:,} rows. Nothing changed.")
        print("   Run again with --apply when you are ready:")
        print("      npm run provenance -- --apply")
        return 0

    if result["status"] != "ok":
        print(f"\n   [FAIL] {result['status']}: {result.get('reason', '')}")
        print("   The original file was left exactly as it was.")
        return 1

    after, widths_after = summarise(args.csv)
    print(f"\n   Migrated {result['rows']:,} rows.")
    print(f"   backup   : {os.path.basename(result['backup'])}")
    print(f"   widths   : {sorted(widths_after)}")
    print(f"   counts unchanged: {after == counts}")
    print("\n   Next:")
    print("      npm run loso          measure across signers")
    print("      npm run collect       new recordings now ask who is signing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
