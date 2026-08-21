"""
migrate_dataset.py — Convert the legacy Holistic dataset to the v2 hands-only
============================================================================
Rewrites `dynamic_gestures.csv` (old, 9 000 features) into
`dynamic_gestures_v2.csv` (new, 3 780 features) WITHOUT re-recording anything.

This is possible because the old rows already contain the hand landmarks —
they were simply stored alongside Pose data that the new pipeline discards.

Old layout (per frame, 300 values)
----------------------------------
    [   0 : 132 ]   Pose        33 landmarks × (x, y, z, visibility)   ← dropped
    [ 132 : 216 ]   Left hand   21 landmarks × (x, y, z, visibility)
    [ 216 : 300 ]   Right hand  21 landmarks × (x, y, z, visibility)

The 4th value of every hand landmark is always 0.0 — MediaPipe hand landmarks
carry no visibility score, and the old collector wrote a
`getattr(lm, "visibility", 0.0)` fallback. It is dropped here.

New layout (per frame, 126 values) — see feature_extractor.py
-------------------------------------------------------------
    [  0 :  63 ]  Left hand   -> raw wrist (3) + 20 landmarks relative (60)
    [ 63 : 126 ]  Right hand  -> same

A hand that was absent in the old row (all 84 values zero) stays absent:
it is written as 63 zeros rather than being fabricated from zero coordinates.

Usage
-----
    python migrate_dataset.py
    python migrate_dataset.py --input other.csv --output other_v2.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

from feature_extractor import (
    N_HAND_LANDMARKS,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    VALS_PER_FRAME,
    VALS_PER_HAND,
    hand_features_from_array,
)


# -----------------------------------------------------------------------------
#  Legacy geometry (the format we are migrating FROM)
# -----------------------------------------------------------------------------

OLD_N_POSE          = 33
OLD_COORDS_PER_LM   = 4                       # x, y, z, visibility
OLD_POSE_VALS       = OLD_N_POSE * OLD_COORDS_PER_LM              # 132
OLD_HAND_VALS       = N_HAND_LANDMARKS * OLD_COORDS_PER_LM        # 84
OLD_VALS_PER_FRAME  = OLD_POSE_VALS + OLD_HAND_VALS * 2           # 300
OLD_TOTAL_FEATURES  = OLD_VALS_PER_FRAME * SEQUENCE_LENGTH        # 9 000

DEFAULT_INPUT  = "dynamic_gestures.csv"
DEFAULT_OUTPUT = "dynamic_gestures_v2.csv"


# -----------------------------------------------------------------------------
#  Conversion helpers
# -----------------------------------------------------------------------------

def convert_hand_block(block: list[float]) -> list[float]:
    """
    Convert one legacy 84-value hand block into the new 63-value hybrid block.

    Legacy : 21 landmarks × (x, y, z, visibility)
    New    : raw wrist (x, y, z) + 20 landmarks expressed relative to the wrist

    An all-zero legacy block means "hand not detected in this frame" and is
    passed through as 63 zeros — computing offsets from a zero wrist would
    inject a fake hand sitting at the frame origin.

    The actual feature math is NOT implemented here: it delegates to
    feature_extractor.hand_features_from_array(), the same function the live
    server calls. That is what guarantees migrated data and live inference
    are numerically identical.
    """
    if not any(block):
        return [0.0] * VALS_PER_HAND

    # Drop the always-zero visibility column -> (21, 3)
    coords = np.asarray(block, dtype=np.float64).reshape(
        N_HAND_LANDMARKS, OLD_COORDS_PER_LM
    )[:, :3]

    return hand_features_from_array(coords)


def convert_row(values: list[float]) -> list[float]:
    """Convert one legacy 9 000-value feature row into 3 780 new values."""
    out: list[float] = []

    for frame_idx in range(SEQUENCE_LENGTH):
        start = frame_idx * OLD_VALS_PER_FRAME
        frame = values[start:start + OLD_VALS_PER_FRAME]

        lh_start = OLD_POSE_VALS                       # skip Pose entirely
        rh_start = lh_start + OLD_HAND_VALS

        out.extend(convert_hand_block(frame[lh_start:rh_start]))
        out.extend(convert_hand_block(frame[rh_start:rh_start + OLD_HAND_VALS]))

    return out


def build_header() -> list[str]:
    """`label, f0_v0 … f29_v125` — same naming scheme as the collector."""
    header = ["label"]
    for frame_idx in range(SEQUENCE_LENGTH):
        for val_idx in range(VALS_PER_FRAME):
            header.append(f"f{frame_idx}_v{val_idx}")
    return header


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate the legacy Holistic dataset to the v2 hands-only format."
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("     Tarjuman — Dataset Migration  (Holistic 9 000 -> Hands 3 780)")
    print("=" * 68)

    if not os.path.isfile(args.input):
        print(f"\n[FAIL]  Input file not found: {args.input}")
        sys.exit(1)

    if os.path.exists(args.output) and not args.force:
        print(f"\n[FAIL]  Output file already exists: {args.output}")
        print("    -> Re-run with --force to overwrite it.")
        sys.exit(1)

    print(f"\n  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"    {OLD_TOTAL_FEATURES} features/row  ->  {TOTAL_FEATURES} features/row")

    rows_written  = 0
    rows_skipped  = 0
    label_counts: dict[str, int] = {}
    # Track how often each hand was actually present, per label
    hand_presence: dict[str, list[int]] = {}

    with open(args.input, "r", newline="", encoding="utf-8") as fin, \
         open(args.output, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader, None)
        if header is None:
            print("\n[FAIL]  Input file is empty.")
            sys.exit(1)

        expected_cols = OLD_TOTAL_FEATURES + 1
        if len(header) != expected_cols:
            print(f"\n[!]  Warning: header has {len(header)} columns, "
                  f"expected {expected_cols}. Rows will still be validated individually.")

        writer.writerow(build_header())

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue

            label  = row[0]
            values = row[1:]

            if len(values) != OLD_TOTAL_FEATURES:
                print(f"   [!]  Line {line_no}: expected {OLD_TOTAL_FEATURES} "
                      f"features, found {len(values)} — skipped.")
                rows_skipped += 1
                continue

            try:
                numeric = [float(v) for v in values]
            except ValueError as exc:
                print(f"   [!]  Line {line_no}: non-numeric value ({exc}) — skipped.")
                rows_skipped += 1
                continue

            converted = convert_row(numeric)
            assert len(converted) == TOTAL_FEATURES, (
                f"internal error: produced {len(converted)} values, "
                f"expected {TOTAL_FEATURES}"
            )

            writer.writerow([label] + converted)
            rows_written += 1
            label_counts[label] = label_counts.get(label, 0) + 1

            # Diagnostics: how many frames had each hand present
            presence = hand_presence.setdefault(label, [0, 0])
            for f in range(SEQUENCE_LENGTH):
                base = f * VALS_PER_FRAME
                if any(converted[base:base + VALS_PER_HAND]):
                    presence[0] += 1
                if any(converted[base + VALS_PER_HAND:base + VALS_PER_FRAME]):
                    presence[1] += 1

    # -- Summary --------------------------------------------------------------
    print(f"\n[OK]  Migration complete")
    print(f"    |-- rows written : {rows_written:,}")
    print(f"    `-- rows skipped : {rows_skipped:,}")

    if label_counts:
        print(f"\n  Samples per label:")
        for label in sorted(label_counts):
            total_frames = label_counts[label] * SEQUENCE_LENGTH
            lh, rh = hand_presence[label]
            print(f"    |-- {label:<20} {label_counts[label]:>4} samples   "
                  f"(left hand in {lh / total_frames * 100:5.1f}% of frames, "
                  f"right hand in {rh / total_frames * 100:5.1f}%)")

    print(f"\n->   Next: retrain on the new file")
    print(f"    1. point INPUT_CSV in train_model.py to '{args.output}'")
    print(f"    2. python train_model.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
