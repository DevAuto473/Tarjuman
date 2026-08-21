"""
mirror_test.py — you on the right, the derived skeleton on the left
====================================================================
Live side-by-side: the camera feed next to a stick figure driven by exactly the
same maths `export_signs_3d.py` uses. Move, and the figure moves with you.

    python mirror_test.py
    npm run mirror

Why this exists
---------------
When the 3D robot performed a sign wrongly there was no way to tell whether the
derivation was wrong or the three.js mapping was. Two separate suspects, no way
to question them apart, and every experiment cost an export plus a page reload.

This removes three.js from the picture. The figure is drawn straight from
`arm_directions()` and `finger_directions()`, so:

    figure mirrors you correctly  ->  the maths is right,
                                      the bug is in the 3D mapping
    figure moves wrongly          ->  the maths is wrong, fix it here

and the feedback loop is a frame instead of a rebuild.

Controls:  q = quit   ·   m = mirror the figure   ·   s = save a frame
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from camera_manager import SmartCamera, choose_camera_interactive
from pose_to_bones import (
    ARM_SPAN, FINGER_CHAINS, LOWER_FRAC, TORSO_HALF_D, TORSO_HALF_W, UPPER_FRAC,
    arm_directions, finger_directions, hand_centre, hand_points,
)
from feature_extractor import (
    PoseTracker, VALS_PER_HAND, extract_frame_features, prepare_frame,
    split_hands,
)

PANEL_W, PANEL_H = 480, 620

# Stick-figure proportions, in the same shoulder-width units the maths uses.
SHOULDER_HALF = 0.5
# Taken from the solver, not guessed again here. When the diagnostic draws with
# different proportions than the maths it is checking, it reports faults that
# only exist in the drawing.
UPPER_LEN     = ARM_SPAN * UPPER_FRAC
LOWER_LEN     = ARM_SPAN * LOWER_FRAC
HEAD_R        = 0.30
FINGER_LEN    = 0.11

C_BONE   = (235, 235, 235)
C_JOINT  = (90, 200, 255)
C_LEFT   = (255, 170, 90)
C_RIGHT  = (120, 230, 140)
C_BODY   = (150, 150, 150)
C_DIM    = (90, 90, 90)
C_WARN   = (60, 190, 250)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, pos, colour=(230, 230, 230), scale=0.5, thick=1):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (15, 15, 15), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


class StickCanvas:
    """Draws in body-space units, where the origin sits between the shoulders."""

    def __init__(self, w=PANEL_W, h=PANEL_H, mirror=False):
        self.w, self.h = w, h
        self.scale = h * 0.22          # pixels per shoulder-width
        self.cx, self.cy = w // 2, int(h * 0.36)
        self.mirror = mirror
        self.img = None

    def clear(self):
        self.img = np.full((self.h, self.w, 3), 18, np.uint8)
        for gx in range(0, self.w, 40):
            cv2.line(self.img, (gx, 0), (gx, self.h), (28, 28, 28), 1)
        for gy in range(0, self.h, 40):
            cv2.line(self.img, (0, gy), (self.w, gy), (28, 28, 28), 1)

    def px(self, p):
        """Body units -> pixels. Body +y is up, screen +y is down."""
        x = -p[0] if self.mirror else p[0]
        return (int(self.cx + x * self.scale), int(self.cy - p[1] * self.scale))

    def bone(self, a, b, colour=C_BONE, thick=5):
        cv2.line(self.img, self.px(a), self.px(b), colour, thick, cv2.LINE_AA)

    def joint(self, p, colour=C_JOINT, r=5):
        cv2.circle(self.img, self.px(p), r, colour, -1, cv2.LINE_AA)

    def torso(self, anchors=None):
        """
        Draw the body using the anchors Pose ACTUALLY found.

        An earlier version drew an idealised head at assumed proportions. That
        made a correct skeleton look wrong: the hand sat exactly where the data
        said, but the reference head it was compared against was invented. Any
        mismatch was in the drawing, not in the maths — the worst kind of
        diagnostic, one that accuses the wrong suspect.
        """
        l = np.array([-SHOULDER_HALF, 0.0])
        r = np.array([SHOULDER_HALF, 0.0])
        self.bone(l, r, C_BODY, 6)
        self.bone(np.array([0.0, 0.0]), np.array([0.0, -1.1]), C_BODY, 4)

        if anchors is None or not anchors.valid:
            # No pose: say so rather than drawing a body that is not there.
            label(self.img, "no body reference", (self.cx - 60, self.cy - 40), C_WARN, 0.45)
            return

        pts = {}
        for name in ("nose", "mouth", "ear", "shoulder", "chest"):
            p = anchors.points.get(name)
            if p is None:
                continue
            # Anchors are stored in image orientation (y down); flip for drawing.
            pts[name] = np.array([float(p[0]), float(-p[1])])

        if "nose" in pts:
            self.bone(np.array([0.0, 0.0]), pts["nose"] * 0.75, C_BODY, 4)
            head_r = 0.35
            if "ear" in pts:
                head_r = max(0.18, float(np.linalg.norm(pts["ear"] - pts["nose"])) * 1.1)
            cv2.circle(self.img, self.px(pts["nose"]),
                       int(head_r * self.scale), C_BODY, 2, cv2.LINE_AA)

        for name, p in pts.items():
            cv2.circle(self.img, self.px(p), 4, C_DIM, -1, cv2.LINE_AA)
            x, y = self.px(p)
            label(self.img, name, (x + 7, y + 4), C_DIM, 0.36)

    def arm(self, side, dirs, hand_pts, true_wrist=None, anchors=None):
        """
        Draw one arm from the SAME directions the exporter produces.

        `true_wrist` is drawn as a hollow ring: it is where the data says the
        hand is, independent of the arm solve. If the solid joint and the ring
        do not coincide, the IK is failing — usually because the target is out
        of the assumed arm reach.
        """
        colour = C_RIGHT if side == "R" else C_LEFT
        sign = 1.0 if side == "R" else -1.0
        shoulder = np.array([sign * SHOULDER_HALF, 0.0])

        du = dirs.get(f"UpperArm.{side}")
        dl = dirs.get(f"LowerArm.{side}")
        if du is None or dl is None:
            return

        elbow = shoulder + np.array([du[0], du[1]]) * UPPER_LEN
        wrist = elbow + np.array([dl[0], dl[1]]) * LOWER_LEN

        self.bone(shoulder, elbow, colour, 6)
        self.bone(elbow, wrist, colour, 6)
        self.joint(shoulder, C_JOINT, 6)
        self.joint(elbow, C_JOINT, 5)
        self.joint(wrist, colour, 7)

        # Fingers: chain each phalanx along its own derived direction.
        for finger in FINGER_CHAINS:
            p = wrist.copy()
            for seg in range(1, 4):
                d = hand_pts.get(f"{finger}.0{seg}.{side}")
                if d is None:
                    break
                nxt = p + np.array([d[0], d[1]]) * FINGER_LEN
                self.bone(p, nxt, colour, 2)
                p = nxt

        wx, wy = self.px(wrist)
        label(self.img, side, (wx - 4, wy - 14), colour, 0.5)

        if true_wrist is not None:
            tw = np.array([float(true_wrist[0]), float(-true_wrist[1])])
            cv2.circle(self.img, self.px(tw), 10, (255, 255, 255), 1, cv2.LINE_AA)
            gap = float(np.linalg.norm(tw - wrist))
            if gap > 0.08:
                # Solid joint and ring have parted: the reach was clamped.
                cv2.line(self.img, self.px(tw), self.px(wrist), C_WARN, 1, cv2.LINE_AA)
                label(self.img, f"reach clamped {gap:.2f}",
                      (self.px(tw)[0] + 12, self.px(tw)[1]), C_WARN, 0.38)

            # Naming the closest anchor turns coordinates into meaning:
            # "the hand is at the mouth" is checkable at a glance.
            if anchors is not None and anchors.valid:
                best, bestd = None, 1e9
                for nm, p in anchors.points.items():
                    d = float(np.linalg.norm(
                        np.array([float(p[0]), float(-p[1])]) - tw))
                    if d < bestd:
                        best, bestd = nm, d
                if best:
                    label(self.img, f"near {best} ({bestd:.2f})",
                          (wx - 30, wy + 22), colour, 0.42)

        return wrist


def main() -> int:
    print("=" * 62)
    print("  TARJUMAN — LIVE MIRROR  (camera right, derived skeleton left)")
    print("=" * 62)
    print("  q = quit   m = mirror the figure   s = save a frame\n")

    cam = SmartCamera(source=choose_camera_interactive())
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
    pose = PoseTracker(every_n=1)          # every frame: this is a live mirror

    canvas = StickCanvas()
    saved = 0
    fps_hist = []

    try:
        while True:
            t0 = time.time()
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame = prepare_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            anchors = pose.update(rgb)
            rgb.flags.writeable = True

            feats = np.asarray(extract_frame_features(results, anchors), dtype=np.float32)

            # -- Left panel: the derivation ----------------------------------
            canvas.clear()
            readouts = []
            canvas.torso(anchors)

            active_sides = []
            for hand_idx, side in ((0, "L"), (1, "R")):
                base = hand_idx * VALS_PER_HAND
                block = feats[base:base + VALS_PER_HAND]
                if not np.any(block):
                    continue
                active_sides.append(side)
                pts = hand_points(block)
                canvas.arm(side,
                           arm_directions(hand_centre(block), side),
                           finger_directions(pts, side),
                           true_wrist=hand_centre(block),
                           anchors=anchors)
                readouts.append(
                    f"{side}: hand=({hand_centre(block)[0]:+.2f},{hand_centre(block)[1]:+.2f})")

            label(canvas.img, "DERIVED SKELETON", (12, 24), (200, 220, 255), 0.6, 2)
            label(canvas.img, "same maths as export_signs_3d.py", (12, 42), C_DIM, 0.4)
            label(canvas.img, "ring = where the data says the hand is",
                  (12, 58), C_DIM, 0.36)
            label(canvas.img,
                  f"body: {'LOCKED' if anchors.valid else 'NOT FOUND'}",
                  (12, canvas.h - 40),
                  (120, 230, 140) if anchors.valid else C_WARN, 0.5)
            label(canvas.img, f"hands: {', '.join(active_sides) or 'none'}",
                  (12, canvas.h - 20), C_DIM, 0.45)
            for i, line in enumerate(readouts):
                label(canvas.img, line, (12, canvas.h - 62 - i * 16), C_DIM, 0.42)
            if anchors.valid:
                label(canvas.img,
                      f"shoulder width={anchors.scale:.3f}  face={anchors.face_size:.2f}",
                      (12, canvas.h - 96), C_DIM, 0.4)
            if canvas.mirror:
                label(canvas.img, "[mirrored]", (canvas.w - 90, 24), C_WARN, 0.45)

            # -- Right panel: what the camera sees ---------------------------
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                              spec, spec)
            cam_panel = cv2.resize(frame, (PANEL_W, PANEL_H))
            label(cam_panel, "CAMERA", (12, 24), (200, 220, 255), 0.6, 2)

            fps_hist.append(time.time() - t0)
            fps_hist = fps_hist[-30:]
            label(cam_panel, f"{1.0 / max(1e-6, np.mean(fps_hist)):.0f} fps",
                  (PANEL_W - 80, 24), C_DIM, 0.5)

            combined = np.hstack([canvas.img, cam_panel])
            cv2.line(combined, (PANEL_W, 0), (PANEL_W, PANEL_H), (60, 60, 60), 2)
            cv2.imshow("Tarjuman — live mirror", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                canvas.mirror = not canvas.mirror
                print(f"  mirror = {canvas.mirror}"
                      f"   -> if this is what fixes it, set MIRROR_X = "
                      f"{canvas.mirror} in export_signs_3d.py")
            if key == ord("s"):
                saved += 1
                name = f"mirror_{saved:02d}.png"
                cv2.imwrite(name, combined)
                print(f"  saved {name}")

    finally:
        hands.close()
        pose.close()
        cam.release()
        cv2.destroyAllWindows()

    print("\n  Done.")
    print("  Figure matched you      -> maths is fine, the bug is in the 3D mapping")
    print("  Figure moved wrongly    -> fix arm_directions/finger_directions here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
