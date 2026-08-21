"""
test_model.py — try the trained model live, without launching the whole app
============================================================================
Opens the camera, runs the exact same pipeline the server runs, and prints
what the model thinks you signed — with its confidence and the runners-up.

    python test_model.py
    npm run test

Why a separate tool
-------------------
Debugging recognition through the full Tauri app means every experiment costs
a rebuild and a UI round trip. This is the same code path (same features, same
segmenter, same ONNX session) with nothing else in the way, so a bad result
points at the model rather than at the plumbing.

Controls:  q = quit   ·   r = reset the segmenter   ·   space = pause
"""

import json
import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[FAIL] onnxruntime is not installed.")
    print("       venv\\Scripts\\pip install -r requirements.txt")
    sys.exit(1)

from camera_manager import SmartCamera, choose_camera_interactive
from feature_extractor import (
    TOTAL_FEATURES, PoseTracker, extract_frame_features, prepare_frame,
    split_hands,
)
from gesture_segmenter import GestureSegmenter

ONNX_MODEL_PATH = "sign_model.onnx"
LABELS_JSON     = "labels.json"
TOP_K           = 3          # how many candidates to show
HISTORY_LEN     = 6          # recent results kept on screen

# Colours (BGR)
C_GREEN  = (80, 220, 100)
C_AMBER  = (40, 190, 240)
C_RED    = (60, 60, 240)
C_WHITE  = (240, 240, 240)
C_GREY   = (150, 150, 150)
C_SHADOW = (20, 20, 20)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------------------------------------------------------
#  Model loading
# -----------------------------------------------------------------------------

def load_model():
    """Load the ONNX graph and validate its shape against the current layout."""
    if not os.path.isfile(ONNX_MODEL_PATH):
        print(f"[FAIL] {ONNX_MODEL_PATH} not found.")
        print("       Train a model first:  npm run train")
        sys.exit(1)
    if not os.path.isfile(LABELS_JSON):
        print(f"[FAIL] {LABELS_JSON} not found — it is written by training.")
        sys.exit(1)

    labels = json.load(open(LABELS_JSON, encoding="utf-8"))
    sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    n_features = inp.shape[1] if len(inp.shape) == 2 else None

    if n_features != TOTAL_FEATURES:
        # The single most common cause of "it recognises nothing": the model
        # was trained on an older feature layout.
        print(f"[FAIL] Model expects {n_features} features, current layout "
              f"produces {TOTAL_FEATURES}.")
        print("       The model is from an older feature version.")
        print("       Re-train:  npm run train")
        sys.exit(1)

    prob_out = next((o for o in sess.get_outputs()
                     if "prob" in o.name.lower()), sess.get_outputs()[-1])

    print(f"[OK] model  : {ONNX_MODEL_PATH}")
    print(f"     input  : {inp.name} [batch, {n_features}]")
    print(f"     classes: {len(labels)}  -> {', '.join(labels.values())}")
    if len(labels) < 2:
        print("\n[WARN] Only one class. The model can only ever answer that one")
        print("       word, no matter what you sign. Record another term.")
    return sess, inp.name, prob_out.name, labels


# -----------------------------------------------------------------------------
#  Drawing
# -----------------------------------------------------------------------------

def text(img, s, pos, colour=C_WHITE, scale=0.6, thick=2):
    x, y = pos
    cv2.putText(img, s, (x + 1, y + 1), FONT, scale, C_SHADOW, thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def draw_hud(frame, *, hands_ok, body_ok, capturing, captured, last, history,
             fps, threshold):
    h, w = frame.shape[:2]

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 96), (12, 12, 12), -1)
    cv2.rectangle(panel, (0, h - 118), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)

    # -- Top: tracking state -------------------------------------------------
    text(frame, f"Hands: {'YES' if hands_ok else 'no '}",
         (12, 28), C_GREEN if hands_ok else C_GREY, 0.6)
    # Without the body reference, location-based signs cannot be told apart.
    text(frame, f"Body: {'YES' if body_ok else 'NO'}",
         (170, 28), C_GREEN if body_ok else C_AMBER, 0.6)
    text(frame, f"{fps:4.1f} fps", (w - 110, 28), C_GREY, 0.55)

    if capturing:
        text(frame, f"RECORDING GESTURE  [{captured} frames]", (12, 60), C_RED, 0.7)
        cv2.circle(frame, (w - 30, 60), 10, C_RED, -1)
    else:
        text(frame, "waiting for a sign...", (12, 60), C_GREY, 0.6)

    # -- Bottom: the prediction ----------------------------------------------
    if last:
        top = last["top"]
        conf = last["conf"]
        accepted = conf >= threshold
        colour = C_GREEN if accepted else C_AMBER
        text(frame, f"{top}", (12, h - 78), colour, 1.1, 3)
        text(frame, f"confidence {conf * 100:5.1f}%"
                    f"{'' if accepted else '   (below threshold)'}",
             (12, h - 48), colour, 0.6)

        # Runners-up matter: a 51/49 split is a very different situation from
        # 95/2, even though both would show the same winning label.
        runners = "   ".join(f"{n} {p * 100:.0f}%" for n, p in last["others"])
        if runners:
            text(frame, f"then: {runners}", (12, h - 22), C_GREY, 0.5, 1)
    else:
        text(frame, "no gesture detected yet", (12, h - 60), C_GREY, 0.7)

    # -- Right: recent history -----------------------------------------------
    for i, item in enumerate(list(history)[-HISTORY_LEN:]):
        text(frame, item, (w - 230, h - 96 + i * 17), C_GREY, 0.45, 1)


# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 62)
    print("  TARJUMAN — LIVE MODEL TEST")
    print("=" * 62)

    sess, in_name, prob_name, labels = load_model()

    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    print(f"     threshold: {threshold:.2f}")

    # Ask, rather than silently defaulting to "auto". Testing the model on a
    # different camera than it was TRAINED on is a quiet way to conclude the
    # model is bad: a different lens shifts every landmark, and the recogniser
    # was fitted to the one you recorded with.
    source = choose_camera_interactive()
    cam = SmartCamera(source=source)
    cam.start()
    if not cam.is_running:
        print("[FAIL] Camera did not open.  Try:  npm run cameras")
        return 1

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    spec       = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           model_complexity=0,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    pose = PoseTracker()
    seg  = GestureSegmenter()

    print("\n  Controls:  q = quit   r = reset   space = pause")
    print("  Perform a sign and watch the bottom of the window.\n")

    history = deque(maxlen=HISTORY_LEN)
    last = None
    paused = False
    frame_times = deque(maxlen=30)

    try:
        while True:
            t_frame = time.time()
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame = prepare_frame(frame)

            if not paused:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                anchors = pose.update(rgb)
                rgb.flags.writeable = True

                left, right = split_hands(results)
                hands_ok = (left is not None) or (right is not None)

                feats = extract_frame_features(results, anchors)
                captured = seg.update(feats, hands_ok, now=time.monotonic())

                if captured is not None:
                    x = np.concatenate([
                        captured["sequence"].reshape(-1),
                        np.asarray(captured["globals"], dtype=np.float32),
                    ]).reshape(1, TOTAL_FEATURES).astype(np.float32)

                    probs = sess.run([prob_name], {in_name: x})[0][0]
                    order = np.argsort(probs)[::-1]

                    top_name = labels.get(str(int(order[0])), str(order[0]))
                    top_conf = float(probs[order[0]])
                    others = [(labels.get(str(int(i)), str(i)), float(probs[i]))
                              for i in order[1:1 + TOP_K - 1]]

                    last = {"top": top_name, "conf": top_conf, "others": others}
                    mark = "OK " if top_conf >= threshold else "LOW"
                    line = f"[{mark}] {top_name} {top_conf*100:.0f}%"
                    history.append(line)
                    print(f"  {line}   ({captured['frames']} frames, "
                          f"{captured['duration']:.2f}s)"
                          + ("   -> " + ", ".join(f"{n} {p*100:.0f}%" for n, p in others)
                             if others else ""))

                # Draw the skeleton so tracking problems are visible
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, lm,
                                                  mp_hands.HAND_CONNECTIONS,
                                                  spec, spec)

                frame_times.append(time.time() - t_frame)
                fps = 1.0 / max(1e-6, float(np.mean(frame_times)))

                draw_hud(frame, hands_ok=hands_ok, body_ok=anchors.valid,
                         capturing=seg.is_capturing, captured=seg.captured_frames,
                         last=last, history=history, fps=fps, threshold=threshold)
            else:
                text(frame, "PAUSED", (12, 40), C_AMBER, 1.0, 3)

            cv2.imshow("Tarjuman — live model test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                seg.reset()
                last = None
                print("  [reset]")
            if key == ord(" "):
                paused = not paused

    finally:
        hands.close()
        pose.close()
        cam.release()
        cv2.destroyAllWindows()

    print("\n  Session ended.")
    if history:
        print("  Last results:")
        for h in history:
            print(f"    {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
