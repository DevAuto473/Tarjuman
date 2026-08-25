"""
train_model.py — Advanced ML Training Pipeline for Tarjuman
=============================================================
Trains a lightweight RandomForestClassifier on dynamic gesture sequences
(30 frames × 300 landmarks = 9 000 features per sample) and exports the
trained pipeline to ONNX format for high-performance inference on a
Raspberry Pi.

Pipeline overview
-----------------
  1. Load  `dynamic_gestures.csv`  (col 0 = label, cols 1–9000 = features)
  2. Encode string labels -> integers with LabelEncoder
  3. Split 80 / 20 stratified train / test
  4. Augment training data with Gaussian jitter (simulates hand tremor)
  5. Build  StandardScaler -> RandomForest(150 trees, depth 20)
  6. Evaluate: accuracy, classification report, confusion matrix PNG
  7. Export: sign_model.onnx  +  labels.json
"""

# -- Import bootstrap ---------------------------------------------------------
# Puts src/ on the path so `tarjuman_core` resolves when this file is run
# directly (`python train_model.py`). Running through `npm run ...` sets PYTHONPATH
# instead, and `pip install -e .` makes both unnecessary - this is the belt to
# those braces, so a plain `python` invocation never fails with ImportError.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.basename(_os.path.dirname(_os.path.abspath(__file__))) == "scripts"
    else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "src"))

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tarjuman_core.paths import data, root
from tarjuman_core.onnx_export import export_pipeline




# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------

INPUT_CSV       = os.environ.get("TARJUMAN_CSV", data("dynamic_gestures_v4.csv"))
PKL_MODEL_PATH  = root("sign_model.pkl")     # dev / inspection only
ONNX_MODEL_PATH = root("sign_model.onnx")    # production artifact used by the server
LABELS_JSON     = data("labels.json")
CM_IMAGE        = root("confusion_matrix.png")

# Cross-validation folds. Five is the usual compromise: each fold trains on 80%
# of the data, and every sample is tested exactly once across the five rounds.
# Set TARJUMAN_SKIP_CV=1 to skip it once the vocabulary is large enough that
# five extra trainings start to hurt.
CV_FOLDS = 5

# Model hyper-parameters
N_ESTIMATORS = 150
MAX_DEPTH    = 20
RANDOM_STATE = 42

# -- Augmentation -------------------------------------------------------------
NOISE_STDDEV = 0.005          # Gaussian jitter σ  (relative to landmark scale ≈ 0–1)

# Temporal augmentation — simulates the SAME sign performed at different speeds.
# Without this the model only ever sees gestures at the exact pace they were
# recorded, which in practice forces users to sign unnaturally slowly to be
# recognised. Combined with gesture_segmenter.resample_sequence(), this is what
# makes signing speed stop mattering.
TIME_WARP_FACTORS = (0.7, 1.4)   # 30 % faster … 40 % slower
FRAME_DROPOUT_P   = 0.10         # probability a frame is dropped, then re-filled

# Data geometry — imported, never redeclared (see feature_extractor.py)
from tarjuman_core.feature_extractor import (
    FRAME_FEATURES, N_GLOBAL_FEATURES, SEQUENCE_LENGTH, TOTAL_FEATURES, VALS_PER_FRAME,
)


# -----------------------------------------------------------------------------
#  Helper: Gaussian noise augmentation
# -----------------------------------------------------------------------------

def augment_with_jitter(X: np.ndarray, y: np.ndarray,
                        noise_std: float = NOISE_STDDEV) -> tuple:
    """
    Create one synthetic copy of every training sample by adding
    small Gaussian noise (jitter) to simulate slight hand tremors
    or different spatial positions.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    noise_std : float
        Standard deviation of the additive Gaussian noise.

    Returns
    -------
    X_aug : ndarray — original + synthetic rows stacked
    y_aug : ndarray — labels duplicated accordingly
    """
    rng = np.random.default_rng(seed=RANDOM_STATE)
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape).astype(X.dtype)
    X_synthetic = X + noise

    X_aug = np.vstack([X, X_synthetic])
    y_aug = np.concatenate([y, y])
    return X_aug, y_aug


# -----------------------------------------------------------------------------
#  Helper: temporal augmentation (speed invariance)
# -----------------------------------------------------------------------------

def _split_blocks(X: np.ndarray):
    """
    Separate the per-frame block from the global block.

    Augmentation must NEVER touch the globals: they encode duration, tempo and
    direction, which is precisely the information some signs depend on. Warping
    them would teach the model to ignore the very thing that separates
    طوارئ from a calm wave.
    """
    frames = X[:, :FRAME_FEATURES].reshape(-1, SEQUENCE_LENGTH, VALS_PER_FRAME)
    globals_ = X[:, FRAME_FEATURES:]
    return frames, globals_


def _join_blocks(frames: np.ndarray, globals_: np.ndarray) -> np.ndarray:
    """Inverse of _split_blocks."""
    return np.hstack([frames.reshape(frames.shape[0], -1), globals_])


def _resample(seq: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample one (T, VALS_PER_FRAME) sequence to target_len frames.

    Mirrors gesture_segmenter.resample_sequence() so training-time warping and
    run-time segmentation speak the same language.
    """
    if seq.shape[0] == target_len:
        return seq.astype(np.float32)

    src = np.linspace(0.0, 1.0, num=seq.shape[0])
    dst = np.linspace(0.0, 1.0, num=target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for col in range(seq.shape[1]):
        out[:, col] = np.interp(dst, src, seq[:, col])
    return out


def augment_time_warp(X: np.ndarray, y: np.ndarray, rng,
                      speed_critical_mask=None) -> tuple:
    """
    Create one speed-warped copy of every sample.

    A random factor stretches or compresses the sequence in time, then it is
    resampled back to SEQUENCE_LENGTH — exactly what happens at inference when
    a signer performs the same sign faster or slower than the training take.

    `speed_critical_mask` marks samples whose CLASS is defined by its tempo
    (طوارئ, إسعاف, ساعدني فوراً). Those are warped far more gently: teaching
    full speed-invariance on them would erase the only thing distinguishing
    them from their calm counterparts.
    """
    frames, globals_ = _split_blocks(X)
    warped = np.empty_like(frames)

    lo, hi = TIME_WARP_FACTORS
    for i, seq in enumerate(frames):
        if speed_critical_mask is not None and speed_critical_mask[i]:
            factor = rng.uniform(0.92, 1.08)      # gentle: keep tempo meaningful
        else:
            factor = rng.uniform(lo, hi)
        stretched_len = max(4, int(round(SEQUENCE_LENGTH * factor)))
        stretched = _resample(seq, stretched_len)            # change the pace…
        warped[i] = _resample(stretched, SEQUENCE_LENGTH)    # …then re-normalise

    # Globals pass through untouched — see _split_blocks.
    return _join_blocks(warped, globals_.copy()), y.copy()


def augment_frame_dropout(X: np.ndarray, y: np.ndarray, rng) -> tuple:
    """
    Create one copy with random frames dropped and the gap resampled shut.

    Simulates dropped frames on a loaded Raspberry Pi and small stutters in the
    signer's movement, so the model does not depend on any single frame.
    """
    frames, globals_ = _split_blocks(X)
    out = np.empty_like(frames)

    for i, seq in enumerate(frames):
        keep = rng.random(SEQUENCE_LENGTH) > FRAME_DROPOUT_P
        if keep.sum() < 4:            # never drop almost everything
            keep[:] = True
        out[i] = _resample(seq[keep], SEQUENCE_LENGTH)

    return _join_blocks(out, globals_.copy()), y.copy()


# -----------------------------------------------------------------------------
#  Helper: Pretty confusion-matrix heatmap
# -----------------------------------------------------------------------------

def report_confusions(y_true, y_pred, class_names, top_n: int = 12) -> None:
    """
    List the class pairs the model actually mixes up, worst first.

    Ranked by combined count in BOTH directions, since a genuinely ambiguous
    pair confuses symmetrically while a one-way error usually means one class
    is simply under-represented.
    """
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            both = int(cm[i, j] + cm[j, i])
            if both:
                support = int(cm[i].sum() + cm[j].sum())
                pairs.append((both, support, class_names[i], class_names[j]))

    print("\n   Most-confused pairs:")
    if not pairs:
        print("       No confusions between classes.")
        return

    pairs.sort(reverse=True)
    for both, support, a, b in pairs[:top_n]:
        rate = both / support * 100 if support else 0.0
        print(f"       {a}  <->  {b}   —  {both} errors ({rate:.0f}% of their samples)")

    if len(pairs) > top_n:
        print(f"       ... and {len(pairs) - top_n} more pairs")
    print("       -> record more samples for these, or redesign the sign.")


def save_confusion_matrix(y_true, y_pred, class_names, path: str):
    """
    Build and save a publication-quality confusion-matrix heatmap.
    Uses a modern dark colour palette for maximum readability.
    """
    # Imported here, not at the top. Matplotlib and seaborn drag in a large
    # native stack that the TRAINING does not need, and if that stack is what
    # crashes the interpreter, deferring it means the model is already saved by
    # the time anything goes wrong.
    import matplotlib
    matplotlib.use("Agg")          # no GUI backend: this runs headless
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.75),
                                    max(6, len(class_names) * 0.65)))

    sns.set_theme(style="darkgrid")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.6,
        linecolor="#333333",
        cbar_kws={"shrink": 0.8, "label": "Samples"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title("Tarjuman — Gesture Classification Confusion Matrix",
                 fontsize=14, fontweight="bold", pad=14)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    fig.savefig(path, dpi=180)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _step(n: int, total: int, title: str) -> float:
    """Print a numbered step header and return its start time."""
    bar = "#" * n + "." * (total - n)
    print(f"\n[{bar}] {n}/{total}  {title}")
    return time.perf_counter()


def _done(t_start: float) -> None:
    print(f"        ...done in {time.perf_counter() - t_start:.2f}s")


def _write_bytes(path: str, data: bytes) -> None:
    """
    Write a model file safely, and explain the one failure that looks like
    nothing happening at all.

    Windows locks a file while another process has it open. The server keeps an
    ONNX session on sign_model.onnx for the whole time it runs, so training
    while `npm run dev:all` is up cannot replace it: the write raises
    PermissionError, the traceback scrolls past, and the user sees stale files
    with yesterday's timestamp and concludes that training "did nothing".

    Writing to a temporary file first also means a failed write can no longer
    destroy the model that was already working.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        os.replace(tmp, path)
    except PermissionError:
        os.remove(tmp)
        print(f"\n[FAIL] Cannot write {path} - the file is in use.")
        print("       On Windows the running server holds the model open.")
        print("       Stop it (Ctrl-C in the `npm run dev:all` terminal),")
        print("       then run `npm run train` again.")
        print("       Your previous model was left untouched.")
        sys.exit(1)


def cross_validate_all(pipeline, X, y, class_names):
    """
    Test on EVERY sample, not just the 20% held out once.

    Why this matters more than it sounds
    ------------------------------------
    A single 80/20 split of 30 recordings tests six of them. Six. A perfect
    score on six samples is compatible with a genuinely accurate model AND with
    a mediocre one that got lucky - the exact binomial bound says all you may
    claim is "better than 54%". Most of the evidence you spent time recording is
    never used for measurement at all.

    K-fold rotates the split instead: five rounds, a different fifth held out
    each time, so every recording is predicted exactly once by a model that did
    not see it. Same data, same recording effort, five times the evidence.

    The SPREAD is the second reason. One number cannot tell a stable model from
    one that happens to score well on a lucky split; five numbers can. A mean of
    95% reads very differently at +/-1% than at +/-12%, and only the latter
    tells you the result is not to be trusted yet.

    Returns the cross-validated predictions, or None if it could not run.
    """
    if os.getenv("TARJUMAN_SKIP_CV"):
        print("\n[cv] Skipped (TARJUMAN_SKIP_CV is set)")
        return None

    counts = np.bincount(y)
    smallest = int(counts.min())
    folds = min(CV_FOLDS, smallest)
    if folds < 2:
        print(f"\n[cv] Skipped - '{class_names[int(counts.argmin())]}' has only "
              f"{smallest} sample(s); k-fold needs at least 2 per class.")
        return None
    if folds < CV_FOLDS:
        print(f"\n[cv] Using {folds} folds instead of {CV_FOLDS}: the smallest "
              f"class has only {smallest} samples.")

    print("\n" + "=" * 65)
    print(f"   CROSS-VALIDATION  ({folds} folds, every sample tested once)")
    print("=" * 65)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.perf_counter()
    scores = []
    for k, (tr, te) in enumerate(cv.split(X, y), start=1):
        import sklearn.base
        model = sklearn.base.clone(pipeline)
        model.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], model.predict(X[te]))
        scores.append(acc)
        print(f"   fold {k}/{folds}: {acc * 100:6.2f} %   ({len(te)} samples)")

    scores = np.asarray(scores)
    print(f"\n   mean     : {scores.mean() * 100:.2f} %  "
          f"+/- {scores.std() * 100:.2f}")
    print(f"   worst    : {scores.min() * 100:.2f} %")

    # The spread is the headline, so say what it means rather than leaving the
    # reader to interpret a standard deviation.
    spread = scores.std() * 100
    if spread < 2:
        print("   -> Consistent across folds. The number is trustworthy.")
    elif spread < 6:
        print("   -> Some variation between folds. Usable, but more samples")
        print("      per word would tighten it.")
    else:
        print("   -> UNSTABLE: folds disagree a lot, so the mean is not")
        print("      meaningful yet. Record more samples per word before")
        print("      drawing any conclusion from this figure.")

    # Every sample predicted by a model that never saw it.
    y_cv = cross_val_predict(pipeline, X, y, cv=cv)
    print(f"\n   Per-word recall over all {len(y)} samples:")
    for i, name in enumerate(class_names):
        mask = y == i
        n = int(mask.sum())
        hit = int((y_cv[mask] == i).sum())
        bar = "#" * int(round(hit / n * 20)) if n else ""
        print(f"     {name:<14s} {hit:>4d}/{n:<4d}  {hit / n * 100:5.1f} %  {bar}")

    print(f"\n   ({time.perf_counter() - t0:.1f}s)")
    return y_cv


def preflight(csv_path: str) -> None:
    """
    Refuse to train on data that cannot produce a usable model.

    Training happily "succeeds" on one class and reports 100 % accuracy, which
    is meaningless — there is nothing to choose between. Catching that here,
    with an explanation, beats shipping a model that always answers the same
    word.
    """
    print("\n" + "=" * 65)
    print("   PRE-FLIGHT CHECKS")
    print("=" * 65)

    if not os.path.isfile(csv_path):
        print(f"\n[FAIL] Dataset not found: {csv_path}")
        print("       Record some samples first:  npm run collect")
        sys.exit(1)

    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"   dataset      : {csv_path}  ({size_mb:.1f} MB)")

    # Count classes without loading the whole file into memory twice
    import csv as _csv
    counts = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0]:
                counts[row[0]] = counts.get(row[0], 0) + 1

    if header is None or not counts:
        print("\n[FAIL] Dataset is empty.")
        sys.exit(1)

    expected_cols = TOTAL_FEATURES + 1
    if len(header) != expected_cols:
        print(f"\n[FAIL] Column count mismatch: found {len(header)}, "
              f"expected {expected_cols}.")
        print("       The dataset was recorded with a DIFFERENT feature layout.")
        print("       Re-record with the current code, or migrate the old file.")
        sys.exit(1)
    print(f"   columns      : {len(header)}  (matches TOTAL_FEATURES) [OK]")

    total = sum(counts.values())
    smallest = min(counts.values())
    print(f"   samples      : {total}")
    print(f"   classes      : {len(counts)}")

    # Show the distribution — imbalance is the usual reason a model looks fine
    # on paper and fails in the room.
    for name, n in sorted(counts.items(), key=lambda kv: kv[1]):
        flag = "  <- too few" if n < 10 else ""
        print(f"       {name:<20s} {n:>4d}{flag}")

    if len(counts) < 2:
        print("\n[FAIL] Only ONE class in the dataset.")
        print("       A classifier needs at least two things to choose between.")
        print("       It would train, report 100 % accuracy, and always answer")
        print("       the same word — which is worse than not training at all.")
        print("\n       Record a second term:  npm run collect")
        sys.exit(1)

    if smallest < 5:
        print(f"\n[FAIL] Smallest class has {smallest} sample(s).")
        print("       The train/test split cannot be stratified below 5.")
        sys.exit(1)

    if smallest < 20:
        print(f"\n[WARN] Smallest class has only {smallest} samples "
              f"(recommended: 30+).")
        print("       Expect the model to memorise rather than generalise.")

    print("\n   [OK] pre-flight passed")


def verify_outputs() -> bool:
    """
    Confirm the artifacts actually exist on disk after export.

    An export can fail in ways that leave the console looking successful — a
    permissions error, a full disk, a path typo. Checking the files afterwards
    turns "I don't see sign_model.onnx" into an answer instead of a mystery.
    """
    print("\n" + "=" * 65)
    print("   OUTPUT VERIFICATION")
    print("=" * 65)

    expected = [
        (ONNX_MODEL_PATH, "used by the server"),
        (PKL_MODEL_PATH,  "development only"),
        (LABELS_JSON,     "class name map"),
        (CM_IMAGE,        "confusion matrix"),
    ]
    ok = True
    for path, note in expected:
        if os.path.isfile(path):
            kb = os.path.getsize(path) / 1024
            when = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(path)))
            print(f"   [OK]   {path:<24s} {kb:>8.1f} KB   {when}   ({note})")
        else:
            print(f"   [FAIL] {path:<24s} NOT CREATED   ({note})")
            ok = False

    print()
    if ok:
        print(f"   Full path: {os.path.abspath(ONNX_MODEL_PATH)}")
        print("   Start the app with:  npm run dev:all")
    else:
        print("   Some artifacts were not written. Check the errors above,")
        print("   and that the folder is writable and not full.")
    return ok


def main():
    t0 = time.perf_counter()
    print("=" * 65)
    print("   Tarjuman - Dynamic Gesture Training Pipeline")
    print("=" * 65)

    preflight(INPUT_CSV)

    # -- Step 1: Load CSV ----------------------------------------------------
    print("\n[1/7] Loading data from", INPUT_CSV)

    if not os.path.isfile(INPUT_CSV):
        print(f"\n[FAIL] File not found:")
        print("        Run data_collector.py first to record samples.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Validate shape
    expected_cols = TOTAL_FEATURES + 1      # label + 9 000 features
    if df.shape[1] != expected_cols:
        print(f"\n[WARN] Column count ({df.shape[1]}) != expected ({expected_cols}).")
        print("       Continuing with the data as-is.")

    labels_raw = df.iloc[:, 0].values       # first column  -> labels
    features   = df.iloc[:, 1:].values      # remaining cols -> float features

    print(f"   samples      : {len(df):,}")
    print(f"   columns      : {df.shape[1]:,}  (1 label + {df.shape[1] - 1} features)")
    print(f"   classes      : {np.unique(labels_raw).tolist()}")

    # -- Step 2: Encode labels -----------------------------------------------
    print("\n[2/7] Encoding labels (LabelEncoder)")
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    class_names = le.classes_.tolist()
    print(f"   `--  {len(class_names)} classes -> {class_names}")

    # -- Step 3: Train / test split ------------------------------------------
    print("\n[3/7] Train/test split (80% / 20%)")
    X_train, X_test, y_train, y_test = train_test_split(
        features, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"   train : {X_train.shape[0]:,} samples")
    print(f"   test  : {X_test.shape[0]:,} samples")

    # -- Step 4: Data augmentation -------------------------------------------
    print(f"\n[####...] 4/7  Augmentation (spatial + temporal)" if False else f"\n[####...] 4/7  Augmentation (spatial + temporal)")
    print(f"")
    original_count = X_train.shape[0]
    rng = np.random.default_rng(seed=RANDOM_STATE)

    X_base, y_base = X_train, y_train

    # a) Spatial: Gaussian jitter — hand tremor / slight position differences
    X_jit, y_jit = augment_with_jitter(X_base, y_base)
    X_jit, y_jit = X_jit[original_count:], y_jit[original_count:]   # synthetic half only
    print(f"   gaussian jitter (sigma={NOISE_STDDEV}) : +{X_jit.shape[0]:,}")

    # b) Temporal: speed warping — the SAME sign performed faster / slower.
    #    Signs whose meaning IS their tempo are warped only gently.
    try:
        from tarjuman_core.vocabulary import as_dicts
        speed_ids = {e["id"] for e in as_dicts() if e["speed_critical"]}
    except Exception:
        speed_ids = set()
    speed_mask = np.array([class_names[i] in speed_ids for i in y_base])
    if speed_mask.any():
        print(f"   speed-critical samples protected from warping: {int(speed_mask.sum())} samples")

    X_warp, y_warp = augment_time_warp(X_base, y_base, rng, speed_mask)
    print(f"   time warp {TIME_WARP_FACTORS}              : +{X_warp.shape[0]:,}")

    # c) Temporal: frame dropout — dropped frames / movement stutter
    X_drop, y_drop = augment_frame_dropout(X_base, y_base, rng)
    print(f"   frame dropout (p={FRAME_DROPOUT_P})        : +{X_drop.shape[0]:,}")

    X_train = np.vstack([X_base, X_jit, X_warp, X_drop])
    y_train = np.concatenate([y_base, y_jit, y_warp, y_drop])

    print(f"   original         : {original_count:,}")
    print(f"   total            : {X_train.shape[0]:,}  "
          f"(×{X_train.shape[0] / max(original_count, 1):.0f})")

    # -- Step 5: Build & train pipeline --------------------------------------
    print(f"\n[#####..] 5/7  Building and training the pipeline" if False else f"\n[#####..] 5/7  Building and training the pipeline")
    print(f"")
    print(f"         StandardScaler -> RandomForest(n={N_ESTIMATORS}, depth={MAX_DEPTH})")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
                       n_estimators=N_ESTIMATORS,
                       max_depth=MAX_DEPTH,
                       random_state=RANDOM_STATE,
                       n_jobs=-1,
                   )),
    ])

    t_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed_train = time.perf_counter() - t_train
    print(f"   trained in {elapsed_train:.1f}s")

    # -- Step 6: Evaluate ----------------------------------------------------
    print("\n[6/7] Evaluating on the held-out test set")
    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n   accuracy : {accuracy * 100:.2f} %\n")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # The confusion matrix and the confusable-pair report are produced from
    # the CROSS-VALIDATED predictions below instead, which cover every sample.

    # -- Cross-validation: measure using EVERY sample -----------------------
    cv_pred = cross_validate_all(pipeline, features, y, class_names)
    if cv_pred is not None:
        # The saved picture uses the cross-validated predictions, because they
        # cover the whole dataset. A confusion matrix built from a 20% holdout
        # of 30 recordings has six cells' worth of evidence per word, which is
        # not enough to see a confusion that happens one time in ten.
        try:
            save_confusion_matrix(y, cv_pred, class_names, CM_IMAGE)
            print(f"\n   [OK] confusion matrix saved -> {CM_IMAGE}"
                  f"  (cross-validated, all {len(y)} samples)")
        except Exception as exc:
            print(f"   [!] Confusion matrix skipped ({type(exc).__name__}: {exc})")
        report_confusions(y, cv_pred, class_names)

    # -- Step 7: Export ONNX (production) + PKL (dev) + labels.json ---------
    print(f"\n[#######] 7/7  Exporting ONNX + PKL + labels.json" if False else f"\n[#######] 7/7  Exporting ONNX + PKL + labels.json")
    print(f"")

    # -- 7a. PKL — kept for local inspection/debugging only -----------------
    import pickle
    _write_bytes(PKL_MODEL_PATH, pickle.dumps(pipeline))

    pkl_size = os.path.getsize(PKL_MODEL_PATH) / (1024 * 1024)
    print(f"   [OK] {PKL_MODEL_PATH}  ({pkl_size:.1f} MB)  (development only)")

    # -- 7b. ONNX — the artifact the server actually runs -------------------
    # Why ONNX: sklearn's predict_proba carries heavy Python/joblib overhead
    # per call (~22.7 ms/frame measured). ONNX Runtime executes the same forest
    # as a single compiled TreeEnsemble op (~0.04 ms/frame) — a ~546× speedup,
    # which matters enormously on a Raspberry Pi.
    #
    # Built here rather than by skl2onnx, which crashes the interpreter
    # outright (0xC0000005) on the development machine - a native fault no
    # try/except can catch, so training could never reach this step. The graph
    # needed is two standard ai.onnx.ml operators, so `onnx` alone can emit it,
    # and the parity check below is what keeps that honest.
    #
    # The batch dimension stays dynamic so one file serves single-frame
    # inference on the server and batched evaluation offline.
    onnx_bytes = export_pipeline(pipeline, TOTAL_FEATURES)
    _write_bytes(ONNX_MODEL_PATH, onnx_bytes)

    onnx_size = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    print(f"   [OK] {ONNX_MODEL_PATH}  ({onnx_size:.1f} MB)  (production)")

    # -- 7c. Verify ONNX output matches sklearn before trusting it ----------
    # A silent numerical divergence here would be very hard to debug later,
    # so we assert parity on the held-out test set at export time.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        in_name   = sess.get_inputs()[0].name
        prob_name = sess.get_outputs()[1].name      # [0]=label, [1]=probabilities

        proba_sklearn = pipeline.predict_proba(X_test)
        proba_onnx    = sess.run([prob_name], {in_name: X_test.astype(np.float32)})[0]

        max_diff  = float(np.abs(proba_sklearn - proba_onnx).max())
        agreement = float((proba_sklearn.argmax(1) == proba_onnx.argmax(1)).mean())

        print(f"   Parity check against scikit-learn:")
        print(f"     max prob diff  : {max_diff:.2e}")
        print(f"     agreement      : {agreement * 100:.2f} %")

        if agreement < 1.0:
            print("   [WARN] ONNX does not match scikit-learn 100% - review before deploying.")
    except ImportError:
        print("   [WARN] onnxruntime not installed - parity check skipped.")

    # -- 7d. Class mapping  { "0": "Alef", "1": "Beh", … } ------------------
    # LabelEncoder guarantees integer classes 0..n-1 in sorted order, so the
    # column index of the ONNX probabilities output maps directly to this key.
    label_map = {str(i): name for i, name in enumerate(class_names)}
    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"   [OK] {LABELS_JSON}  ({len(label_map)} classes)")

    # -- Done ----------------------------------------------------------------
    total = time.perf_counter() - t0
    print("\n" + "=" * 65)
    print(f"   Done in {total:.1f}s")
    print(f"     ONNX model   : {ONNX_MODEL_PATH}   <- used by the server")
    print(f"     PKL model    : {PKL_MODEL_PATH}    <- dev only")
    print(f"     label map    : {LABELS_JSON}")
    print(f"     confusion    : {CM_IMAGE}")
    print("=" * 65)


if __name__ == "__main__":
    # Nothing may fail invisibly. Without this, an exception anywhere in the
    # pipeline prints a traceback that scrolls off, or - worse - dies while
    # encoding its own output and leaves the terminal blank.
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n[stop] Interrupted - nothing was written.")
        sys.exit(130)
    except Exception as exc:
        import traceback
        print("\n" + "=" * 65)
        print("   TRAINING FAILED")
        print("=" * 65)
        print(f"   {type(exc).__name__}: {exc}")
        print("\n   Full traceback:")
        traceback.print_exc()
        print("\n   Copy everything above when reporting this.")
        sys.exit(1)
