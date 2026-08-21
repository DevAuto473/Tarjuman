"""
train_model_cnn.py — 1D-CNN trainer for Tarjuman dynamic gestures
==================================================================
Alternative to train_model.py's RandomForest, for when a real dataset exists.

Why a 1D-CNN instead of RandomForest
------------------------------------
RandomForest sees the gesture as 3 780 unrelated numbers. Column 1 400 means
nothing to it beyond "the 1 400th feature" — it has no notion that the value
sits at frame 11 of a movement. So a signer who performs the sign slightly
faster shifts every value into a different column and the forest loses the
thread.

A 1D convolution slides across the TIME axis. It learns motion patterns
(this finger rises while the wrist travels left) that are recognised wherever
in the sequence they occur. That is the property the data actually has and the
forest structurally cannot use.

Drop-in ONNX contract
---------------------
The exported graph deliberately mirrors the sklearn export exactly:
    input  : "input"          float32 [batch, TOTAL_FEATURES]   (flat!)
    outputs: "label"          int64   [batch]
             "probabilities"  float32 [batch, n_classes]
The reshape from flat -> (channels, time) happens INSIDE the graph, so
websocket_server.py needs no changes at all: point it at this .onnx and it
works. `_validate_onnx_session()` accepts it unchanged.

Requirements
------------
PyTorch is a TRAINING-only dependency and is intentionally NOT in
requirements.txt — the Raspberry Pi only ever runs onnxruntime.

    pip install -r requirements-train.txt
    python train_model_cnn.py

[!]  With a tiny dataset this network will simply memorise it. Collect a real,
    balanced dataset first (see the project checklist, item 12) — a CNN needs
    considerably more data than a forest to beat it.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[FAIL]  PyTorch is not installed (training-only dependency).")
    print("    -> pip install -r requirements-train.txt")
    sys.exit(1)

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_extractor import (
    FRAME_FEATURES, N_GLOBAL_FEATURES, SEQUENCE_LENGTH, TOTAL_FEATURES, VALS_PER_FRAME,
)


# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------

INPUT_CSV       = os.environ.get("TARJUMAN_CSV", "dynamic_gestures_v4.csv")
ONNX_MODEL_PATH = "sign_model_cnn.onnx"
LABELS_JSON     = "labels.json"

EPOCHS        = 120
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
RANDOM_STATE  = 42

# Temporal augmentation, same idea as train_model.py
TIME_WARP_FACTORS = (0.7, 1.4)
FRAME_DROPOUT_P   = 0.10
NOISE_STDDEV      = 0.005


# -----------------------------------------------------------------------------
#  Model
# -----------------------------------------------------------------------------

class GestureCNN(nn.Module):
    """
    Compact 1D-CNN over the time axis.

    Input is accepted FLAT — (batch, TOTAL_FEATURES) — and reshaped internally,
    so the ONNX signature matches the sklearn pipeline it replaces.

        (B, 3780) -> (B, 30, 126) -> transpose -> (B, 126, 30)
                  -> conv blocks -> global average pool -> linear -> softmax

    Global average pooling (rather than flatten+dense) keeps the parameter
    count small and makes the network tolerant of *where* in the window the
    distinctive movement happens.
    """

    def __init__(self, n_classes: int, in_channels: int = VALS_PER_FRAME):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),                       # 30 -> 15 frames

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),               # -> (B, 256, 1)
        )
        # 256 pooled conv features + the global block appended in forward()
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(256 + N_GLOBAL_FEATURES, n_classes),
        )

    def forward(self, x):
        # Input is [per-frame block | global block]. Only the per-frame block
        # has a time axis; the globals are whole-gesture summaries and would
        # corrupt the reshape (and the convolution) if folded in.
        frames = x[:, :FRAME_FEATURES]
        globals_ = x[:, FRAME_FEATURES:]

        # (B, FRAME_FEATURES) -> (B, TIME, FEATURES) -> (B, FEATURES, TIME)
        seq = frames.reshape(-1, SEQUENCE_LENGTH, VALS_PER_FRAME).transpose(1, 2)
        pooled = self.features(seq).flatten(1)          # (B, 256)

        # Globals join the classifier directly — they carry duration and tempo,
        # which is exactly what separates طوارئ from a calm wave.
        return self.classifier(torch.cat([pooled, globals_], dim=1))


class ExportWrapper(nn.Module):
    """Adds softmax + argmax so the ONNX graph emits (label, probabilities)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        probabilities = torch.softmax(self.model(x), dim=1)
        label = torch.argmax(probabilities, dim=1)
        return label, probabilities


# -----------------------------------------------------------------------------
#  Augmentation (mirrors train_model.py / gesture_segmenter.resample_sequence)
# -----------------------------------------------------------------------------

def _resample(seq: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    if seq.shape[0] == target_len:
        return seq.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=seq.shape[0])
    dst = np.linspace(0.0, 1.0, num=target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for col in range(seq.shape[1]):
        out[:, col] = np.interp(dst, src, seq[:, col])
    return out


def augment(X: np.ndarray, y: np.ndarray, rng) -> tuple:
    """Original + jitter + time-warp + frame-dropout -> 4× the training data.

    The global block is carried through untouched: warping duration would erase
    the very feature that makes tempo-defined signs distinguishable.
    """
    seqs = X[:, :FRAME_FEATURES].reshape(-1, SEQUENCE_LENGTH, VALS_PER_FRAME)
    globals_ = X[:, FRAME_FEATURES:]

    jitter = seqs + rng.normal(0.0, NOISE_STDDEV, seqs.shape).astype(np.float32)

    warped = np.empty_like(seqs)
    lo, hi = TIME_WARP_FACTORS
    for i, seq in enumerate(seqs):
        stretched = _resample(seq, max(4, int(round(SEQUENCE_LENGTH * rng.uniform(lo, hi)))))
        warped[i] = _resample(stretched, SEQUENCE_LENGTH)

    dropped = np.empty_like(seqs)
    for i, seq in enumerate(seqs):
        keep = rng.random(SEQUENCE_LENGTH) > FRAME_DROPOUT_P
        if keep.sum() < 4:
            keep[:] = True
        dropped[i] = _resample(seq[keep], SEQUENCE_LENGTH)

    X_aug = np.vstack([
        np.hstack([s.reshape(s.shape[0], -1), globals_])
        for s in (seqs, jitter, warped, dropped)
    ])
    y_aug = np.concatenate([y] * 4)
    return X_aug.astype(np.float32), y_aug


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    print("=" * 68)
    print("     Tarjuman — 1D-CNN Gesture Training")
    print("=" * 68)

    if not os.path.isfile(INPUT_CSV):
        print(f"\n[FAIL]  '{INPUT_CSV}' not found. Run migrate_dataset.py or "
              f"data_collector.py first.")
        sys.exit(1)

    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    # -- Load ----------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    if df.shape[1] != TOTAL_FEATURES + 1:
        print(f"[!]  Expected {TOTAL_FEATURES + 1} columns, found {df.shape[1]}.")

    labels_raw = df.iloc[:, 0].values
    features   = df.iloc[:, 1:].values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)

    print(f"\n  samples: {len(df):,} | classes: {n_classes} -> {class_names}")

    counts = np.bincount(y)
    if counts.min() < 10:
        print(f"\n[!]  WARNING: smallest class has only {counts.min()} sample(s).")
        print("    A CNN will memorise a dataset this small. Any accuracy")
        print("    figure below is meaningless — collect real data first.")

    # -- Split & augment -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, y_train = augment(X_train, y_train, rng)
    print(f"    train (augmented): {X_train.shape[0]:,} | test: {X_test.shape[0]:,}")

    # Normalise using TRAIN statistics only (no test leakage)
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test - mean) / std

    # -- Train ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNN(n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n   1D-CNN on {device} — {n_params:,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_train_n).to(device)
    yt = torch.from_numpy(y_train).long().to(device)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(Xt.size(0), device=device)
        epoch_loss = 0.0
        for start in range(0, Xt.size(0), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if idx.numel() < 2:        # BatchNorm needs ≥2 samples
                continue
            optimiser.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"    epoch {epoch:>4}/{EPOCHS}  loss {epoch_loss:.4f}")

    # -- Evaluate ------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_n).to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    print(f"\n  accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %\n")
    print(classification_report(y_test, y_pred, target_names=class_names,
                                zero_division=0))

    # -- Export ONNX (drop-in for the sklearn model) -------------------------
    # Fold normalisation into the graph so the server keeps sending RAW
    # features and needs no knowledge of the training statistics.
    class Normalised(nn.Module):
        def __init__(self, inner, mean, std):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer("std",  torch.from_numpy(std.astype(np.float32)))

        def forward(self, x):
            return self.inner((x - self.mean) / self.std)

    export_model = ExportWrapper(Normalised(model, mean, std)).to("cpu").eval()
    dummy = torch.zeros(1, TOTAL_FEATURES, dtype=torch.float32)

    torch.onnx.export(
        export_model, dummy, ONNX_MODEL_PATH,
        input_names=["input"],
        output_names=["label", "probabilities"],
        dynamic_axes={"input": {0: "batch"},
                      "label": {0: "batch"},
                      "probabilities": {0: "batch"}},
        opset_version=13,
    )
    size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    print(f"\n  {ONNX_MODEL_PATH}  ({size_mb:.1f} MB)")

    # -- Verify the export matches PyTorch -----------------------------------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        onnx_probs = sess.run(["probabilities"], {"input": X_test.astype(np.float32)})[0]
        with torch.no_grad():
            torch_probs = torch.softmax(
                model(torch.from_numpy(X_test_n)), dim=1
            ).numpy()
        print(f"     max |ONNX - PyTorch| : {np.abs(onnx_probs - torch_probs).max():.2e}")
        print(f"       prediction agreement : "
              f"{(onnx_probs.argmax(1) == torch_probs.argmax(1)).mean() * 100:.2f} %")
    except ImportError:
        print("   [!]  onnxruntime missing — skipped parity check.")

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump({str(i): n for i, n in enumerate(class_names)}, f,
                  ensure_ascii=False, indent=2)
    print(f"   [OK]  {LABELS_JSON} ({n_classes} classes)")

    print("\n" + "=" * 68)
    print(f"   done in {time.perf_counter() - t0:.1f}s")
    print(f"   -> to use it: set ONNX_MODEL_PATH in websocket_server.py")
    print(f"     to '{ONNX_MODEL_PATH}' (no other change needed)")
    print("=" * 68)


if __name__ == "__main__":
    main()
