"""
evaluate_signers.py — how well does this work on someone it has never seen?
===========================================================================
    npm run loso                    leave-one-signer-out across the dataset
    npm run loso -- --augment       same, with augmentation on the training side

Why this is the number that matters
-----------------------------------
Ordinary cross-validation shuffles samples. When every sample comes from one
person, each fold trains on that person and tests on that same person — so a
score of 100% means "the model recognises Yazid", which nobody doubted. It
says nothing whatever about the next person to stand in front of the camera,
and that is the only question a sign-language translator has to answer.

Leave-one-signer-out removes an entire PERSON from training and tests on them:

    train on {everyone else}  ->  test on {held-out signer}

The result is an honest estimate of performance on an unseen signer, and it is
the figure to put in the report. Expect it to be lower than the shuffled
score. That drop is not a regression — it is the shuffled score having been
measuring the wrong thing.

Needs at least two signers in the data. Until a second person records, this
will say so rather than print a meaningless number.
"""

# -- Import bootstrap ---------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import argparse
import collections
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tarjuman_core.paths import data
from tarjuman_core.provenance import UNKNOWN_SIGNER, read_dataset

# Kept in step with train_model.py so the number describes the real model.
N_ESTIMATORS = 150
MAX_DEPTH = 20
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                       max_depth=MAX_DEPTH,
                                       random_state=RANDOM_STATE,
                                       n_jobs=-1)),
    ])


def bar(fraction: float, width: int = 22) -> str:
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.environ.get(
        "TARJUMAN_CSV", data("dynamic_gestures_v4.csv")))
    ap.add_argument("--augment", action="store_true",
                    help="augment the training side of each fold")
    args = ap.parse_args()

    print("=" * 68)
    print("   LEAVE-ONE-SIGNER-OUT EVALUATION")
    print("=" * 68)

    labels, signers, X = read_dataset(args.csv)
    print(f"   dataset : {args.csv}")
    print(f"   samples : {len(labels):,}   features : {X.shape[1]:,}")

    counts = collections.Counter(signers)
    print(f"   signers : {len(counts)}")
    for s, n in counts.most_common():
        words = len(set(labels[signers == s]))
        print(f"      {s:<16s} {n:>5,} samples across {words} words")

    if len(counts) < 2:
        only = next(iter(counts))
        print("\n" + "=" * 68)
        print("   NOT ENOUGH SIGNERS TO MEASURE THIS")
        print("=" * 68)
        if only == UNKNOWN_SIGNER:
            print("   The dataset has no signer column yet. Add it with:")
            print("      npm run provenance")
        else:
            print(f"   Every sample is from '{only}'. Leave-one-signer-out needs")
            print("   at least two people, because the whole point is to test on")
            print("   someone absent from training.")
        print("\n   Until then the honest statement about this model is that its")
        print("   accuracy on an unseen signer is UNMEASURED — not that it is high.")
        print("   Ten samples per word from one other person is enough to find out.")
        return 1

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)

    print("\n" + "-" * 68)
    print("   PER-SIGNER RESULTS  (each row: that person removed from training)")
    print("-" * 68)

    per_word_hits = collections.defaultdict(lambda: [0, 0])
    fold_scores = []

    for held in counts:
        test = signers == held
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        if len(set(y[train])) < 2:
            print(f"   {held:<16s} skipped — training side has one class only")
            continue

        Xtr, ytr = X[train], y[train]
        if args.augment:
            # Augment the TRAINING side only. Doing it before the split would
            # put a transformed copy of a test sample into training, and the
            # score would then be measuring memorisation.
            from tarjuman_core.augment import augment_batch
            Xtr, ytr = augment_batch(Xtr.astype(np.float64), ytr, n_copies=2)

        model = build_pipeline().fit(Xtr, ytr)
        pred = model.predict(X[test])
        acc = float((pred == y[test]).mean())
        fold_scores.append((held, acc, int(test.sum())))

        for true_i, pred_i in zip(y[test], pred):
            w = class_names[true_i]
            per_word_hits[w][1] += 1
            if true_i == pred_i:
                per_word_hits[w][0] += 1

        print(f"   {held:<16s} train {train.sum():>5,} -> test {test.sum():>4,}   "
              f"accuracy {acc * 100:5.1f}%  {bar(acc)}")

    if not fold_scores:
        print("\n   No usable folds.")
        return 1

    accs = np.array([a for _, a, _ in fold_scores])
    weights = np.array([n for _, _, n in fold_scores], dtype=float)
    overall = float((accs * weights).sum() / weights.sum())

    print("\n" + "-" * 68)
    print("   PER-WORD RECALL ON UNSEEN SIGNERS")
    print("-" * 68)
    for w in sorted(per_word_hits, key=lambda k: per_word_hits[k][0] / max(1, per_word_hits[k][1])):
        hit, total = per_word_hits[w]
        r = hit / max(1, total)
        flag = "  <-- weakest" if r < 0.6 else ""
        print(f"   {w:<18s} {hit:>4}/{total:<4} {r * 100:5.1f}%  {bar(r)}{flag}")

    print("\n" + "=" * 68)
    print("   HEADLINE")
    print("=" * 68)
    print(f"   Accuracy on an UNSEEN signer : {overall * 100:.1f}%"
          f"   (mean {accs.mean() * 100:.1f}%, spread {accs.min() * 100:.1f}-{accs.max() * 100:.1f}%)")
    print(f"   Augmentation                 : {'on' if args.augment else 'off'}")
    print(f"   Signers                      : {len(fold_scores)}")
    print()
    print("   This is the figure for the report. With two signers it is a")
    print("   coarse estimate built on two folds — worth far more than a")
    print("   shuffled score that never tested generalisation at all, and")
    print("   worth stating with its limits rather than rounding up.")
    if not args.augment:
        print("\n   Compare against:  npm run loso -- --augment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
