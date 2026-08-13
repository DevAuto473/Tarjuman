"""
data_collector.py — Dynamic Gesture Sequence Recorder for Tarjuman
===================================================================
Captures sliding-window gesture sequences and saves them to
`dynamic_gestures.csv` for training the hybrid DTW/Random Forest model.

Recording architecture
----------------------
  • One sequence  = SEQUENCE_LENGTH consecutive frames  (default 30)
  • One CSV row   = label  +  flattened features of all 30 frames
  • Feature set per frame (hands only — see feature_extractor.py):
        - Left hand  : 63 values (raw wrist x,y,z + 20 landmarks relative)
        - Right hand : 63 values (same layout)
        = 126 values / frame
        × 30 frames  = 3 780 values per row  (+1 label column)

  The geometry above is NOT defined here — it is imported from
  feature_extractor.py so the collector and the live server can never drift.

Controls
--------
  r   →  Start recording the next 30 frames
  q   →  Quit and save any buffered sequences
"""

import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# Import our smart camera abstraction (works on laptop and Raspberry Pi)
from camera_manager import (
    DROIDCAM_IP, DROIDCAM_PORT, SmartCamera, droidcam_url, list_local_cameras,
)

# Feature geometry + extraction — single source of truth shared with the server
from feature_extractor import (
    GLOBAL_FEATURE_NAMES,
    N_GLOBAL_FEATURES,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    VALS_PER_FRAME,
    PoseTracker,
    compute_global_features,
    extract_frame_features,
    prepare_frame,
    split_hands,
)
from gesture_segmenter import resample_sequence


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_CSV = "dynamic_gestures_v4.csv"   # v4 = body-anchored features

# Samples per label before a class is considered "done".
# 30+ per class is the realistic floor for a 100-class problem; below that the
# model memorises rather than generalises.
TARGET_PER_LABEL = 30


# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe Hands setup
# ─────────────────────────────────────────────────────────────────────────────
# Hands ONLY. Holistic additionally ran BlazePose (33 pts) and Face Mesh
# (468 pts) every frame — the face landmarks were never used at all.

mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Body anchors. Location is a defining parameter of a sign (forehead vs. chin),
# so the hands are recorded relative to the body rather than to the picture.
# Pose is throttled internally — see PoseTracker.
#
# Created lazily: building it loads a MediaPipe model, and doing that at import
# time means merely importing this module for a helper function pays the cost —
# and fails outright if the model cannot be loaded.
_pose_tracker = None


def get_pose_tracker() -> PoseTracker:
    global _pose_tracker
    if _pose_tracker is None:
        _pose_tracker = PoseTracker()
    return _pose_tracker


# ─────────────────────────────────────────────────────────────────────────────
#  Landmark extraction
# ─────────────────────────────────────────────────────────────────────────────
# Deliberately NOT implemented here — see feature_extractor.py. Duplicating
# the layout between collector and server is exactly how a silent
# train/inference mismatch gets introduced.

def extract_frame_landmarks(results, anchors=None) -> np.ndarray:
    """Thin wrapper: shared extractor → float32 array of (VALS_PER_FRAME,)."""
    return np.array(extract_frame_features(results, anchors), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_csv_header(filepath: str) -> None:
    """
    Create the CSV with a header row if it does not exist.

    Layout:  label, f0_v0 … f29_v125, g_duration_s … g_openness_change
    The trailing global columns are what let duration and tempo survive the
    fixed-length resampling — without them, signs that differ only by speed
    become identical rows.
    """
    if os.path.exists(filepath):
        return   # Append mode: header already written

    header = ["label"]
    for frame_idx in range(SEQUENCE_LENGTH):
        for val_idx in range(VALS_PER_FRAME):
            header.append(f"f{frame_idx}_v{val_idx}")
    header += [f"g_{name}" for name in GLOBAL_FEATURE_NAMES]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    print(f"[Collector] Created CSV: {filepath}  ({len(header)} columns)")


def save_sequence(label: str, sequence: list, duration: float, filepath: str) -> None:
    """
    Write one recorded gesture as a CSV row.

    The raw capture is used TWICE, deliberately:
      • resampled to SEQUENCE_LENGTH → the per-frame block
      • measured as-is                → the global block (duration, tempo,
                                        direction, openness)

    Order matters: globals must be computed BEFORE resampling, because
    resampling is exactly what destroys them.
    """
    globals_ = compute_global_features(sequence, duration)
    frames = resample_sequence(sequence)

    row = [label] + frames.reshape(-1).tolist() + list(globals_)

    assert len(row) == TOTAL_FEATURES + 1, (
        f"row has {len(row)} values, expected {TOTAL_FEATURES + 1}"
    )

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
#  Visual overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette
_GREEN  = (50,  220,  50)
_RED    = (30,   30, 220)
_WHITE  = (240, 240, 240)
_SHADOW = (20,   20,  20)
_AMBER  = (30,  180, 230)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.65
_THICKNESS  = 2


def _put_text_with_shadow(frame, text: str, pos: tuple, color: tuple,
                           scale=_FONT_SCALE, thickness=_THICKNESS) -> None:
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), _FONT, scale, _SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),          _FONT, scale, color,   thickness,     cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, label: str, is_recording: bool,
                 current_frame: int, sequences_saved: int) -> None:
    """
    Render the status HUD onto the frame in-place.

    Layout (top of frame)
    ---------------------
      Line 1 — Gesture label
      Line 2 — Recording status / progress bar
      Line 3 — Total sequences saved so far
    """
    h, w = frame.shape[:2]

    # ── Translucent top banner ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Line 1: Label ────────────────────────────────────────────────────────
    _put_text_with_shadow(frame, f"Gesture: {label}", (12, 28), _WHITE, scale=0.7)

    # ── Line 2: Status ───────────────────────────────────────────────────────
    if is_recording:
        status_text  = f"Status: RECORDING... [{current_frame} / {SEQUENCE_LENGTH}]"
        status_color = _RED

        # Progress bar
        bar_x1, bar_y1 = 12, 38
        bar_x2, bar_y2 = w - 12, 52
        filled_x = bar_x1 + int((bar_x2 - bar_x1) * current_frame / SEQUENCE_LENGTH)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (filled_x, bar_y2), _RED, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), _WHITE, 1)

    else:
        status_text  = "Status: READY  (Press 'r' to record)"
        status_color = _GREEN

    _put_text_with_shadow(frame, status_text, (12, 72), status_color, scale=0.6)

    # ── Line 3: Sequences saved counter ─────────────────────────────────────
    saved_text = f"Saved: {sequences_saved} sequence(s)"
    _put_text_with_shadow(frame, saved_text, (w - 220, 28), _AMBER, scale=0.58)


def draw_landmarks_on_frame(frame: np.ndarray, results) -> None:
    """Draw MediaPipe hand skeletons on the frame (in-place)."""
    left, right = split_hands(results)
    for landmarks in (left, right):
        if landmarks is not None:
            mp_drawing.draw_landmarks(
                frame, landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw_spec, mp_draw_spec,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def _choose_camera():
    """
    Ask which camera to record with.

    A phone camera over DroidCam is usually much sharper than a laptop webcam,
    and sharper frames give cleaner MediaPipe landmarks — the raw material the
    whole model is built from. Worth the extra prompt.

    Returns a source string (or int for USB index) understood by SmartCamera.
    """
    preset = os.getenv("CAMERA_SOURCE")
    if preset and preset.lower() != "auto":
        print(f"\nCamera: {preset}  (from CAMERA_SOURCE)")
        return preset

    print("\n" + "=" * 60)
    print("  Select camera")
    print("=" * 60)
    print("  1. Built-in laptop webcam        (index 0)")
    print("  2. External USB webcam           (pick from a scan)")
    print("  3. DroidCam / phone")
    print("  4. Other stream URL")
    print()

    while True:
        choice = input("Choice [1-4, default 2]: ").strip() or "2"

        if choice == "1":
            # Index 0 is effectively always the built-in camera on a laptop.
            return "laptop"

        if choice == "2":
            # An external webcam is NOT index 0, so "laptop" would silently
            # open the built-in one instead. Scan and let the user choose.
            print("\n  [External USB webcam]")
            print("  Scanning camera devices...")
            cams = list_local_cameras()

            if not cams:
                print("   [!] No camera devices responded.")
                print("       - Is the webcam plugged in and its light on?")
                print("       - Close Zoom / Teams / Windows Camera / OBS.")
                print("       - Try a different USB port.")
                continue

            for idx, w, h in cams:
                note = "   <- built-in (usually index 0)" if idx == 0 else ""
                print(f"     index {idx}:  {w}x{h}{note}")

            external = [c for c in cams if c[0] != 0]
            default = str(external[0][0]) if external else str(cams[0][0])
            if not external:
                print("\n   [!] Only index 0 found — the external webcam is not")
                print("       being detected. Check the cable/port, then re-run.")

            print("\n   Not sure which is which?  npm run cameras -- --preview")
            raw = input(f"\n   Camera index [default {default}]: ").strip() or default
            if raw.isdigit():
                return f"index:{raw}"
            print("   [!] Enter one of the numbers listed above.")
            continue

        if choice == "3":
            print("\n  [DroidCam Setup]")
            print("  Wi-Fi MJPEG has no flow control: when bandwidth dips it")
            print("  silently DROPS frames, and lost frames distort the exact")
            print("  timing the model learns from. Prefer USB for recording.")
            print()
            print("  1. USB - virtual webcam    <- iPhone AND Android")
            print("  2. USB - adb tunnel        <- ANDROID ONLY (no PC client)")
            print("  3. Wi-Fi                   (requires IP, may stutter)")
            print()
            print("  iPhone: adb cannot talk to iOS at all — use option 1.")
            dc_mode = input("  Choose mode [1-3, default 1]: ").strip() or "1"

            if dc_mode == "3":
                print("\n  Before continuing:")
                print("   - DroidCam app is OPEN on the phone")
                print("   - Phone and PC are on the same Wi-Fi")
                print(f"   - The app shows {DROIDCAM_IP}:{DROIDCAM_PORT}")
                print("   (change via DROIDCAM_IP / DROIDCAM_PORT in .env)")
                return "droidcam"

            if dc_mode == "2":
                print("\n  [adb tunnel - ANDROID ONLY]")
                print("   This will NOT work with an iPhone.")
                print("   1. Platform Tools folder set in .env as ADB_PATH")
                print("   2. Developer Options ON (tap Build number 7 times)")
                print("   3. USB debugging ON")
                print("   4. DroidCam app OPEN on the phone")
                print("   5. Cable connected, 'Allow USB debugging' accepted")
                return "droidcam-usb"

            # Default: virtual webcam — works for iPhone and Android alike
            print("\n  [USB - virtual webcam]")
            print("   1. DroidCam (or iVCam / Camo) PC client is OPEN")
            print("   2. Mode set to USB, 'Connect' pressed")
            print("   3. Phone connected by cable, app open on the phone")
            print("   iPhone also needs Apple's iTunes / Apple Devices installed")
            print("   (it provides the USB drivers Windows needs).")
            print("\n  Scanning local camera devices...")
            cams = list_local_cameras()
            if cams:
                for idx, w, h in cams:
                    hint = "   <- likely the phone (higher resolution)" if w >= 1280 else ""
                    print(f"     index {idx}:  {w}x{h}{hint}")
            else:
                print("     [!] No devices responded. Is the PC client connected?")
            cam_idx = input("\n   Camera index [default 1]: ").strip() or "1"
            try:
                return int(cam_idx)
            except ValueError:
                return "usb_dshow"

        if choice == "4":
            url = input("  Stream URL: ").strip()
            if "://" in url:
                return url
            print("  [!] Must be a full URL, e.g. http://192.168.8.177:4747/video")
            continue

        print("  [!] Enter 1, 2, 3 or 4.")


def _existing_counts(filepath: str) -> dict:
    """How many samples each label already has, so you can see what's missing."""
    counts = {}
    if not os.path.exists(filepath):
        return counts
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    except OSError as exc:
        print(f"Could not read existing samples: {exc}")
    return counts


def _choose_label() -> str:
    """Interactive picker over vocabulary.py, with per-label progress."""
    try:
        from vocabulary import as_dicts
        vocab = as_dicts()
    except Exception as exc:
        print(f"⚠️  vocabulary.py unavailable ({exc}) — falling back to free text.")
        label = input("\nEnter the gesture label: ").strip()
        if not label:
            print("❌ Label cannot be empty. Exiting.")
            sys.exit(1)
        return label

    counts = _existing_counts(OUTPUT_CSV)
    total = len(vocab)
    done = sum(1 for e in vocab if counts.get(e["id"], 0) >= TARGET_PER_LABEL)

    print(f"\nVocabulary: {total} terms | complete: {done} | "
          f"target {TARGET_PER_LABEL} samples each")
    print("Arabic words are in learn.csv — terminals cannot render them correctly.\n")

    # Ids only: Windows consoles do not shape or bidi-order Arabic, so printing
    # the words here produces reversed, unreadable text.
    for i, e in enumerate(vocab, 1):
        n = counts.get(e["id"], 0)
        mark = "[x]" if n >= TARGET_PER_LABEL else ("[~]" if n else "[ ]")
        print(f"  {mark} {i:3d}. {e['id']:<17s} {n:>2d}/{TARGET_PER_LABEL}", end="")
        if i % 3 == 0:
            print()
    print("\n")

    while True:
        raw = input("Pick a number (or type the id): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= total:
            return vocab[int(raw) - 1]["id"]
        for e in vocab:
            if raw == e["id"] or raw == e["arabic"]:
                return e["id"]
        print("   [!] Not found - try again.")


def main() -> None:
    # ── Terminal prompt: get gesture label before opening camera window ──────
    print("=" * 60)
    print("  Tarjuman — Dynamic Gesture Sequence Recorder")
    print("=" * 60)
    print(f"  Output file     : {OUTPUT_CSV}")
    print(f"  Sequence length : {SEQUENCE_LENGTH} frames")
    print(f"  Values per row  : 1 (label) + {SEQUENCE_LENGTH * VALS_PER_FRAME} (coords)")
    print("=" * 60)

    # Camera first: an unreachable phone stream should be discovered before
    # you have picked a label and mentally prepared to sign.
    camera_source = _choose_camera()

    # ── Pick the label from the official vocabulary ─────────────────────────
    # Typing labels freehand is how a dataset ends up with "test1", "test_1"
    # and "test 2" as three separate classes. Choosing from the list keeps the
    # ids consistent with vocabulary.py and shows progress as you go.
    label = _choose_label()

    target_sequences = input(
        f"📦 How many sequences to record for '{label}'? [default: 30]: "
    ).strip()
    target_sequences = int(target_sequences) if target_sequences.isdigit() else 30

    auto_input = input("⚡ Auto-record mode? Type 'y' for Dynamic (1s pause), 'h' for Static Hold (0s pause), or Enter for Manual: ").strip().lower()
    is_auto = (auto_input in ['y', 'h'])
    auto_pause = 0.0 if auto_input == 'h' else 1.0

    print(f"\n✅ Recording {target_sequences} sequences for label: '{label}'")
    if is_auto:
        print("   Auto-mode is ON. Press 'r' ONCE to start the chain.")
    else:
        print("   Press 'r' inside the camera window to start each sequence.")
    print("   Press 'q' to quit early.\n")

    # ── Ensure CSV exists with the correct header ────────────────────────────
    _ensure_csv_header(OUTPUT_CSV)

    # ── State variables ──────────────────────────────────────────────────────
    is_recording     = False
    current_sequence = []       # Accumulates frame landmark arrays
    record_started_at = 0.0     # wall clock — real duration, not frame count
    sequences_saved  = 0
    next_auto_start_time = float('inf')

    # ── Camera + main loop ───────────────────────────────────────────────────
    pose_tracker = get_pose_tracker()
    cam = SmartCamera(source=camera_source)
    cam.start()

    if not cam.is_running:
        print("[FAIL] Camera did not open. See the checklist above. Exiting.")
        sys.exit(1)

    print(f"[Collector] Camera backend: {cam.backend}")
    print("[Collector] Camera window open. Follow the on-screen instructions.\n")

    try:
        while sequences_saved < target_sequences:
            ret, frame = cam.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror the frame (shared switch, identical to the live server)
            frame = prepare_frame(frame)

            # ── MediaPipe processing ─────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            # Same RGB frame for both models, so they agree on the world.
            anchors = pose_tracker.update(rgb)
            rgb.flags.writeable = True

            # ── Draw skeleton on the display frame ───────────────────────────
            draw_landmarks_on_frame(frame, results)

            # ── Recording state machine ──────────────────────────────────────
            if is_recording:
                # Extract landmarks for this frame (zero-padded if not detected)
                frame_data = extract_frame_landmarks(results, anchors)
                current_sequence.append(frame_data)

                if len(current_sequence) == SEQUENCE_LENGTH:
                    # Sequence complete → save to CSV.
                    # Duration is MEASURED, not assumed: the same 30 frames can
                    # span 1 s or 2 s depending on camera load, and several
                    # signs mean what they mean because of their tempo.
                    duration = max(time.time() - record_started_at, 1e-3)
                    save_sequence(label, current_sequence, duration, OUTPUT_CSV)
                    sequences_saved += 1
                    print(
                        f"  ✅ Sequence {sequences_saved}/{target_sequences} "
                        f"saved for '{label}'  ({duration:.2f}s)"
                    )

                    # Reset for next sequence
                    current_sequence = []
                    is_recording     = False

                    if sequences_saved >= target_sequences:
                        print(f"\n🎉 All {target_sequences} sequences recorded! Closing.")
                        break
                    
                    if is_auto:
                        next_auto_start_time = time.time() + auto_pause

            # Auto-record trigger
            if is_auto and not is_recording and time.time() >= next_auto_start_time:
                is_recording = True
                current_sequence = []
                record_started_at = time.time()
                next_auto_start_time = float('inf')
                print(f"  🔴 Auto-Recording sequence {sequences_saved + 1}/{target_sequences}...")

            # ── HUD overlay ──────────────────────────────────────────────────
            draw_overlay(
                frame, label, is_recording,
                len(current_sequence), sequences_saved,
            )

            cv2.imshow(f"Tarjuman Recorder — '{label}'", frame)

            # ── Keyboard handling ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r") and not is_recording:
                is_recording      = True
                current_sequence  = []
                record_started_at = time.time()
                print(f"  🔴 Recording sequence {sequences_saved + 1}/{target_sequences}...")

            elif key == ord("q"):
                print("\n⏹  Quit key pressed. Stopping early.")
                break

    finally:
        # ── Graceful teardown ────────────────────────────────────────────────
        hands.close()
        pose_tracker.close()
        cam.release()
        cv2.destroyAllWindows()

        print(f"\n📊 Session summary for '{label}':")
        print(f"   Sequences recorded : {sequences_saved}")
        print(f"   Output file        : {os.path.abspath(OUTPUT_CSV)}")

        if sequences_saved > 0:
            print("✅ Data saved successfully.")
        else:
            print("⚠️  No sequences were saved in this session.")


if __name__ == "__main__":
    main()