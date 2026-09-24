"""
provenance.py — remember WHO recorded each sample, and with what
================================================================

Why this exists
---------------
Every row in the dataset was a label and 4212 numbers, and nothing else. That
is enough to train a model and not nearly enough to trust one.

Two things forced the change:

  * A second signer is about to record. Merged into an anonymous file, his
    samples become indistinguishable from the existing ones — and the single
    most valuable measurement in this project, "how well does a model trained
    on one person work on another", becomes permanently impossible. You cannot
    hold out a signer you cannot identify.

  * Samples were recorded on a laptop webcam and will now also be recorded on
    a Raspberry Pi camera with a different lens and field of view. If that
    turns out to hurt, the only remedy without provenance is to discard
    everything and start again.

Columns, in order, after the label:

    signer        who performed the sign            e.g. yazid, ahmed
    camera        which lens                        e.g. logi-c310, picamera3
    session       one recording sitting             e.g. 20260909-1430
    recorded_at   ISO-8601 timestamp

Signer is the one that matters for evaluation; the rest cost nothing to keep
and answer questions that otherwise require guesswork.

This module also repairs a real defect. `_ensure_csv_header` only wrote a
header when the file did not exist, and the file was later rewritten in place
by a filtering step — which dropped the header. `pandas.read_csv` then ate the
first DATA row as column names, silently discarding one sample from every
training run and naming the columns "salam", "0.0", "0.0"...
"""

from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime

from .feature_extractor import (
    GLOBAL_FEATURE_NAMES, SEQUENCE_LENGTH, TOTAL_FEATURES, VALS_PER_FRAME,
)

PROVENANCE_COLUMNS = ("signer", "camera", "session", "recorded_at")
N_PROVENANCE = len(PROVENANCE_COLUMNS)

LABEL_COLUMN = "label"
UNKNOWN_SIGNER = "unknown"

# label + provenance + features
TOTAL_COLUMNS = 1 + N_PROVENANCE + TOTAL_FEATURES
# The pre-provenance layout, kept so a legacy file can be recognised for what
# it is rather than rejected as corrupt.
LEGACY_COLUMNS = 1 + TOTAL_FEATURES


def feature_column_names() -> list[str]:
    names = []
    for frame in range(SEQUENCE_LENGTH):
        for val in range(VALS_PER_FRAME):
            names.append(f"f{frame}_v{val}")
    names += [f"g_{n}" for n in GLOBAL_FEATURE_NAMES]
    return names


def header_row() -> list[str]:
    return [LABEL_COLUMN, *PROVENANCE_COLUMNS, *feature_column_names()]


def new_session_id() -> str:
    """One id per recording sitting, so a bad session can be found later."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(text: str, fallback: str) -> str:
    """
    Reduce a typed name to something safe to group by.

    'Ahmed Al-Otaibi' and 'ahmed al otaibi' must not become two different
    signers, or a leave-one-signer-out split would train and test on the same
    person while believing otherwise.
    """
    cleaned = "".join(c.lower() if (c.isalnum() or c in "-_") else "-"
                      for c in (text or "").strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or fallback


def detect_layout(path: str) -> str:
    """
    'missing' | 'legacy' | 'provenance' — what is actually in this file.

    Determined by reading the first row, not by trusting a filename or an
    assumption about what wrote it.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return "missing"
    with open(path, newline="", encoding="utf-8") as f:
        first = next(csv.reader(f), None)
    if not first:
        return "missing"
    if first[0] == LABEL_COLUMN:
        return "provenance" if first[1:1 + N_PROVENANCE] == list(PROVENANCE_COLUMNS) \
               else "legacy"
    # No header at all: classify by width.
    return "provenance" if len(first) == TOTAL_COLUMNS else "legacy"


def ensure_header(path: str) -> None:
    """Create the file with a header if it does not exist yet."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header_row())


def append_sample(path: str, label: str, features, *, signer: str,
                  camera: str, session: str) -> None:
    """Append one recorded sample together with where it came from."""
    ensure_header(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label, signer, camera, session,
                                datetime.now().isoformat(timespec="seconds"),
                                *features])


def migrate(path: str, *, signer: str, camera: str = "unknown",
            session: str = "legacy", dry_run: bool = False) -> dict:
    """
    Add provenance columns to an existing dataset.

    Writes to a temporary file and swaps it in only after the row count and
    column width of the result have been verified, and takes a timestamped
    backup first. The samples themselves are copied verbatim — this only
    prepends columns and restores the missing header.

    Returns a summary; with dry_run=True it reports what WOULD happen and
    writes nothing.
    """
    layout = detect_layout(path)
    if layout == "missing":
        return {"status": "missing", "rows": 0}
    if layout == "provenance":
        return {"status": "already-migrated", "rows": _count_rows(path)}

    rows_in = rows_out = 0
    widths = set()
    tmp = path + ".migrating"
    backup = f"{path}.pre-provenance-{datetime.now():%Y%m%d-%H%M%S}.bak"

    with open(path, newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        first = next(reader, None)
        if first is None:
            return {"status": "missing", "rows": 0}

        # The legacy file may or may not have a header. If the first cell is
        # not the literal "label" it is a DATA row and must be kept — this is
        # exactly the sample that pandas has been silently discarding.
        rescued_first_row = None if first[0] == LABEL_COLUMN else first

        out = None if dry_run else open(tmp, "w", newline="", encoding="utf-8")
        try:
            writer = None if dry_run else csv.writer(out)
            if writer:
                writer.writerow(header_row())

            def emit(row):
                nonlocal rows_out
                widths.add(len(row))
                if writer:
                    writer.writerow([row[0], signer, camera, session, "", *row[1:]])
                rows_out += 1

            if rescued_first_row:
                rows_in += 1
                emit(rescued_first_row)
            for row in reader:
                if not row:
                    continue
                rows_in += 1
                emit(row)
        finally:
            if out:
                out.close()

    result = {"status": "ok", "rows": rows_out, "rows_read": rows_in,
              "widths": sorted(widths), "signer": signer,
              "rescued_first_row": bool(rescued_first_row), "backup": backup}

    if dry_run:
        result["status"] = "dry-run"
        return result

    if rows_out != rows_in or widths != {LEGACY_COLUMNS}:
        os.remove(tmp)
        result["status"] = "aborted"
        result["reason"] = (f"expected {rows_in} rows of {LEGACY_COLUMNS} columns, "
                            f"got {rows_out} rows of widths {sorted(widths)}")
        return result

    shutil.copy2(path, backup)      # backup BEFORE the swap, never after
    os.replace(tmp, path)
    return result


def _count_rows(path: str) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        n = 0 if (first and first[0] == LABEL_COLUMN) else (1 if first else 0)
        return n + sum(1 for r in reader if r)


def read_dataset(path: str):
    """
    Load a dataset as (labels, signers, features) regardless of layout.

    A legacy file still loads, with every signer reported as UNKNOWN_SIGNER,
    so nothing breaks before migration — it just cannot be split by signer.
    """
    import numpy as np

    layout = detect_layout(path)
    labels, signers, feats = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        rows = ([] if (first and first[0] == LABEL_COLUMN) else [first]) if first else []
        for row in list(rows) + list(reader):
            if not row:
                continue
            if layout == "provenance":
                labels.append(row[0])
                signers.append(row[1] or UNKNOWN_SIGNER)
                feats.append(row[1 + N_PROVENANCE:])
            else:
                labels.append(row[0])
                signers.append(UNKNOWN_SIGNER)
                feats.append(row[1:])
    return (np.array(labels),
            np.array(signers),
            np.asarray(feats, dtype=np.float32))
