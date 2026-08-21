"""
pose_to_bones.py — landmarks to robot bone directions
======================================================
Shared by the offline exporter and the live server, so the robot is driven by
exactly one implementation. Two copies of this maths would drift, and a drift
here shows up as a robot that performs recorded signs differently from live
ones — with nothing in the logs to say why.

Why DIRECTIONS and not Euler angles
-----------------------------------
A rig's bone axes are arbitrary: "rotate 40° about X" bends a finger on one
skeleton and twists it on another. Angles therefore have to be guessed, and a
wrong guess produces splayed, stretched poses. "This bone points there" is true
regardless of how the rig was built; the player converts it using the
skeleton's own rest pose.

A note on the wrist
-------------------
MediaPipe's landmark 0 is the HEEL of the palm, not the middle of the hand.
With a hand held upright — fingers up, as in many signs — the heel sits near the
chest while the fingertips reach the mouth. Anchoring purely on landmark 0
therefore reports "hand at chest" for a sign made at the mouth.
`hand_centre()` exists for that reason: it is what a person would point at and
call "the hand".

Three things this module guarantees
-----------------------------------
1. The arm never passes through the torso. A single camera gives no usable
   depth for the arm, so everything used to be solved flat at z = 0 — in the
   same plane as the chest. A hand held in front of the body therefore ended up
   INSIDE it. Anatomy settles it without needing depth: a hand over the chest
   must be in FRONT of the chest, so the target is pushed forward and the elbow
   is swung outward.

2. Hand-to-shoulder geometry matches yours proportionally. Positions are
   expressed as a fraction of ARM LENGTH rather than in raw units, so "my hand
   is at 70% of my reach, up and to the left" becomes the same 70% on a robot
   whose arms are a different length. That is what makes the robot's pose read
   as the same pose rather than a vaguely similar one.

3. The palm faces where your palm faces. A single direction vector cannot
   express twist — every rotation that points a bone the right way is equally
   valid, and the minimal one is chosen arbitrarily. Hand bones therefore carry
   SIX numbers: a direction plus the palm normal, which pins the roll down.
"""

import numpy as np

from feature_extractor import HAND_SHAPE_VALS, N_HAND_LANDMARKS

# Finger chains as MediaPipe landmark indices.
FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}

# Knuckle landmarks — the flat of the hand, a stabler "where is the hand" than
# either the wrist heel or the fingertips.
KNUCKLES = [5, 9, 13, 17]

INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17

# -- Axis convention ----------------------------------------------------------
# MediaPipe is y-DOWN and its landmark z grows AWAY from the camera; the robot's
# world is y-up with z toward the camera. Both axes therefore flip, which is a
# 180° rotation about X.
#
# This used to flip y alone. That is a REFLECTION (determinant -1), not a
# rotation, and a reflection silently reverses handedness: every cross product
# computed downstream came out pointing the wrong way, so fingers bent backwards
# in depth and a palm normal would have faced behind the hand. Flipping both
# axes makes it a proper rotation and the handedness survives.
_TO_WORLD = np.diag([1.0, -1.0, -1.0])

MIRROR_X = False
FLIP_Y = False
SWAP_YZ = False

# -- Body geometry ------------------------------------------------------------
# Everything below is in SHOULDER-WIDTH units with the origin between the
# shoulders, and +y pointing DOWN to match the stored landmark convention.
# -z is toward the camera. Conversion to the robot's world happens once, in
# unit(), at the very end.
SHOULDER_HALF_W = 0.50

# How long an arm is, shoulder joint to hand centre, fully extended.
#
# The two differ, and that difference is the whole reason this section exists.
# The solver used to work in HUMAN proportions and emit directions, which the
# robot then applied with its own shorter arms — so every position the solver
# had carefully placed arrived 16% closer to the body than intended. A hand
# cleared to just in front of the chest ended up inside it.
HUMAN_ARM_SPAN = 1.60      # people cluster tightly around this
RIG_ARM_SPAN = 1.35        # measured from TarjumanRobot2.glb (1.130 / 0.840)
ARM_SPAN = RIG_ARM_SPAN    # what the IK actually solves in

# Upper-arm / forearm split, read from the rig itself (0.580 and 0.550 units).
# Taking it from the rig rather than assuming 50/50 is what keeps the ELBOW in
# the same relative place as yours, not just the hand.
UPPER_FRAC = 0.513
LOWER_FRAC = 0.487

# -- The robot's own body, measured from the GLB ------------------------------
# Measured, not guessed. Guessing here is what put the hand behind the head:
# the head reaches 0.59 forward and the keep-out volume allowed 0.30, so a hand
# raised to the temple was placed a third of a head INSIDE the skull.
TORSO_HALF_W = 0.46
TORSO_HALF_D = 0.46
TORSO_TOP = -0.38          # the chest rises above the shoulder line
TORSO_BOTTOM = 0.83

HEAD_CENTRE_Y = -1.04      # above the shoulders (remember: +y is down)
HEAD_HALF_W = 0.75         # includes the ears
HEAD_HALF_H = 0.51
HEAD_HALF_D = 0.59         # the face/screen juts well forward

BODY_CLEARANCE = 0.12      # gap so limbs never graze the surface

# Minimum forward offset for a hand, even out at the sides. Signing happens in
# the space in FRONT of the signer; a hand level with the middle of the body
# would be half-buried whenever it drifted inward.
ARM_FORWARD_BIAS = 0.14

MIRROR_X = False
FLIP_Y = False
SWAP_YZ = False

# MediaPipe is y-DOWN and its landmark z grows AWAY from the camera; the robot's
# world is y-up with z toward the camera. Both axes therefore flip, which is a
# 180° rotation about X.
#
# This used to flip y alone. That is a REFLECTION (determinant -1), not a
# rotation, and a reflection silently reverses handedness: every cross product
# computed downstream came out pointing the wrong way, so fingers bent backwards
# in depth and a palm normal would have faced behind the hand. Flipping both
# axes makes it a proper rotation and the handedness survives.
def to_world(v) -> np.ndarray:
    """Landmark space -> robot world space, as a proper rotation."""
    v = np.asarray(v, dtype=np.float64)
    if v.size < 3:
        v = np.append(v, [0.0] * (3 - v.size))
    w = _TO_WORLD @ v[:3]
    if MIRROR_X:
        w[0] = -w[0]
    if FLIP_Y:
        w[1] = -w[1]
    if SWAP_YZ:
        w[1], w[2] = w[2], w[1]
    return w


def _round3(w) -> list:
    n = float(np.linalg.norm(w))
    if n < 1e-8:
        return [0.0, -1.0, 0.0]
    w = w / n
    return [round(float(w[0]), 4), round(float(w[1]), 4), round(float(w[2]), 4)]


def unit(v) -> list:
    """Normalise a landmark-space vector into the robot's space."""
    return _round3(to_world(v))


def unit_world(w) -> list:
    """Normalise a vector that is ALREADY in world space.

    Palm normals must be built by crossing two world-space vectors, never by
    converting a normal computed in landmark space — a normal is a pseudovector
    and does not survive an axis flip the way a direction does.
    """
    return _round3(np.asarray(w, dtype=np.float64))


def hand_points(hand_block) -> np.ndarray:
    """Rebuild the 21 landmarks from one 68-value hand block (wrist at origin)."""
    pts = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)
    pts[1:] = np.asarray(hand_block[3:3 + HAND_SHAPE_VALS],
                         dtype=np.float32).reshape(N_HAND_LANDMARKS - 1, 3)
    return pts


def hand_centre(hand_block) -> np.ndarray:
    """
    Where a person would say the hand IS, in body coordinates (3D).

    The mean of the knuckles, offset from the stored wrist. With the hand held
    upright this sits a palm-length above landmark 0 — which is the difference
    between reporting a sign at the mouth and reporting it at the chest.
    """
    wrist = np.asarray(hand_block[0:3], dtype=np.float32).copy()
    pts = hand_points(hand_block)
    knuckles = pts[KNUCKLES].mean(axis=0)

    # The stored shape is palm-length normalised, so scale it back down before
    # adding it to a body-space position.
    return wrist + knuckles * 0.16


# -----------------------------------------------------------------------------
#  Torso avoidance
# -----------------------------------------------------------------------------

def _ellipsoid_front_z(x, y, cx, cy, hw, hh, hd):
    """
    How far forward this solid reaches at (x, y), or None if it is not there.

    Returns the frontmost z of an ellipsoid, so the surface is followed rather
    than approximated by a slab: at the edge of the head a hand needs almost no
    clearance, at the centre of the face it needs all of it.
    """
    u = (float(x) - cx) / hw
    v = (float(y) - cy) / hh
    r = u * u + v * v
    if r >= 1.0:
        return None
    return -hd * float(np.sqrt(1.0 - r))          # -z is toward the camera


def body_front_z(x, y) -> float:
    """
    The frontmost surface of the robot at (x, y) — head and torso together.

    This is the single rule that keeps arms visible: a hand is never allowed
    behind this value, so wherever you put your hand relative to your own body,
    the robot's hand appears in FRONT of the corresponding part of its body
    rather than buried in it.
    """
    front = -ARM_FORWARD_BIAS
    for z in (_ellipsoid_front_z(x, y, 0.0, (TORSO_TOP + TORSO_BOTTOM) / 2,
                                 TORSO_HALF_W, (TORSO_BOTTOM - TORSO_TOP) / 2,
                                 TORSO_HALF_D),
              _ellipsoid_front_z(x, y, 0.0, HEAD_CENTRE_Y,
                                 HEAD_HALF_W, HEAD_HALF_H, HEAD_HALF_D)):
        if z is not None:
            front = min(front, z - BODY_CLEARANCE)
    return front


def _in_ellipsoid(p, cy, hw, hh, hd) -> bool:
    u = float(p[0]) / hw
    v = (float(p[1]) - cy) / hh
    w = float(p[2]) / hd
    return (u * u + v * v + w * w) < 1.0


def _inside_body(p) -> bool:
    """
    Is this point buried in the robot's head or torso?

    Strictly a SOLID test, deliberately separate from body_front_z(). Those are
    two different rules and conflating them cost a debugging round: using the
    front surface as the collision test made the entire half-space behind the
    robot count as solid, so every pose scored as a collision, the retry loop
    ran to its most extreme attempt every time, and hands were flung a metre
    toward the camera. A hand beside or behind the body is unusual; it is not
    a hand inside the body.
    """
    return (_in_ellipsoid(p, (TORSO_TOP + TORSO_BOTTOM) / 2, TORSO_HALF_W,
                          (TORSO_BOTTOM - TORSO_TOP) / 2, TORSO_HALF_D)
            or _in_ellipsoid(p, HEAD_CENTRE_Y, HEAD_HALF_W, HEAD_HALF_H,
                             HEAD_HALF_D))


# Kept under the old names so the diagnostic tools keep working.
def _inside_torso(p) -> bool:
    return _inside_body(p)


def _lift_clear_of_body(p) -> np.ndarray:
    """
    Move a point out of the body by bringing it FORWARD, never sideways.

    Sideways would be a lie: a sign made at the centre of the chest is made
    there, and moving it to the shoulder changes its meaning — location is one
    of the parameters that distinguishes one sign from another. Depth is the one
    axis a single camera could not measure, so depth is the one free to correct.
    """
    p = np.asarray(p, dtype=np.float64).copy()
    front = body_front_z(p[0], p[1])
    if p[2] > front:
        p[2] = front
    return p


_lift_clear_of_torso = _lift_clear_of_body


def _torso_overlap(a, b, samples: int = 9, skip_start: float = 0.0) -> int:
    """
    How many sample points along a bone fall inside the body.

    `skip_start` ignores the first fraction of the bone. The head of the humerus
    sits INSIDE the shoulder joint, which is part of the torso — so the top of
    every upper arm ever is "inside the body" and testing it reports a collision
    for poses that are simply someone standing there.
    """
    lo = max(1, int(round(skip_start * samples)))
    return sum(1 for i in range(lo, samples)
               if _inside_body(a + (b - a) * (i / samples)))


def _segment_hits_torso(a, b, samples: int = 9, skip_start: float = 0.0) -> bool:
    """Does the straight bone from a to b pass through the body?"""
    return _torso_overlap(a, b, samples, skip_start) > 0


def arm_directions(target, side: str,
                   upper_len: float = None, lower_len: float = None) -> dict:
    """
    Upper-arm and forearm directions that reach `target` (body coordinates).

    Two-bone IK in 3D. The elbow is free to rotate anywhere on a circle around
    the shoulder-to-hand axis, so a POLE vector picks the point on that circle:
    out to the side, down, and slightly back — where a human elbow actually
    goes. Aiming the pole outward is also what keeps the forearm clear of the
    chest, so avoidance falls out of the anatomy rather than being bolted on.

    If a solution still clips the torso, the HAND is nudged forward and the pole
    re-aimed, and the first clean result wins. Nudging depth is the honest knob:
    it is the axis a single camera never measured, so correcting it invents
    nothing, whereas moving the hand sideways or vertically would corrupt the
    location parameter that distinguishes one sign from another.
    """
    upper_len = ARM_SPAN * UPPER_FRAC if upper_len is None else upper_len
    lower_len = ARM_SPAN * LOWER_FRAC if lower_len is None else lower_len

    sign = 1.0 if side == "R" else -1.0
    shoulder = np.array([sign * SHOULDER_HALF_W, 0.0, 0.0], dtype=np.float64)

    target = np.asarray(target, dtype=np.float64)
    if target.size < 3:
        target = np.append(target, [0.0] * (3 - target.size))
    # Human reach -> robot reach. Both hands end up at the same FRACTION of
    # their own arm's length, which is what makes the pose read as the same
    # pose; skipping it silently shrank every position by 16%.
    base_target = shoulder + (target[:3] - shoulder) * (RIG_ARM_SPAN / HUMAN_ARM_SPAN)
    base_target = _lift_clear_of_body(base_target)

    if float(np.linalg.norm(base_target - shoulder)) < 1e-4:
        down = np.array([0.0, 1.0, 0.0])
        return {f"UpperArm.{side}": unit(down), f"LowerArm.{side}": unit(down)}

    # (extra forward push, outward pole weight, backward/forward pole lean)
    attempts = ((0.00, 0.75, 0.30), (0.10, 1.40, -0.20), (0.22, 2.20, -0.55),
                (0.36, 3.20, -0.90), (0.52, 4.50, -1.30), (0.70, 2.00, -2.20),
                (0.90, 0.60, -3.00))

    # Keep the BEST attempt, not the first. A cross-body reach — left hand to
    # the right hip — genuinely cannot avoid passing in front of the torso, so
    # some poses have no perfectly clean answer; returning the least-bad one
    # beats returning the first one tried, which was usually the worst.
    best = None
    for push, out_w, lean in attempts:
        t = base_target.copy()
        t[2] -= push                       # -z is toward the camera
        elbow, t = _solve_arm(shoulder, t, sign, upper_len, lower_len, out_w, lean)
        score = (_torso_overlap(shoulder, elbow, skip_start=0.35)
                 + _torso_overlap(elbow, t)
                 + (4 if _inside_body(t) else 0))
        if best is None or score < best[0]:
            best = (score, elbow, t)
        if score == 0:
            break

    _, elbow, t = best
    return {
        f"UpperArm.{side}": unit(elbow - shoulder),
        f"LowerArm.{side}": unit(t - elbow),
    }


def _solve_arm(shoulder, target, sign, upper_len, lower_len, out_w, lean):
    """Two-bone IK; returns (elbow, reachable_target)."""
    to_target = target - shoulder
    dist = float(np.linalg.norm(to_target))
    reach = upper_len + lower_len

    # Never fully straighten: at exactly full reach the elbow circle collapses
    # to a point and the solve loses all control of where the elbow goes.
    if dist > reach * 0.985:
        to_target = to_target * (reach * 0.985 / dist)
        dist = reach * 0.985
        target = shoulder + to_target
    dist = max(dist, 1e-4)

    axis = to_target / dist
    a = (upper_len ** 2 - lower_len ** 2 + dist ** 2) / (2 * dist)
    h = float(np.sqrt(max(0.0, upper_len ** 2 - a ** 2)))
    mid = shoulder + axis * a

    # The outward pull is strengthened when the hand is near the body's midline,
    # which is exactly when a weakly-placed elbow would clip the ribs.
    closeness = max(0.0, 1.0 - abs(float(target[0])) / TORSO_HALF_W)  # noqa: E501
    pole = np.array([sign * (out_w + 0.75 * closeness), 0.60, lean])
    return _solve_elbow(mid, axis, pole, h), target


def _solve_elbow(mid, axis, pole, h):
    """Point on the elbow circle that lies closest to the pole direction."""
    pole = np.asarray(pole, dtype=np.float64)
    perp = pole - axis * float(np.dot(pole, axis))
    n = float(np.linalg.norm(perp))
    if n < 1e-6:
        # Pole parallel to the arm axis: fall back to any perpendicular.
        alt = np.array([0.0, 0.0, 1.0])
        perp = alt - axis * float(np.dot(alt, axis))
        n = float(np.linalg.norm(perp)) or 1.0
    return mid + (perp / n) * h


# -----------------------------------------------------------------------------
#  Hand and fingers
# -----------------------------------------------------------------------------

def palm_normal(pts: np.ndarray) -> np.ndarray:
    """
    Outward normal of the palm, in WORLD space.

    Built by crossing two world-space edges of the palm, in the same landmark
    order the player uses on the rig (index knuckle, then pinky knuckle). Same
    order on both sides means the two normals agree on which face is the palm,
    whatever the rig's own axes happen to be.
    """
    v_index = to_world(pts[INDEX_MCP] - pts[0])
    v_pinky = to_world(pts[PINKY_MCP] - pts[0])
    return np.cross(v_index, v_pinky)


def hand_directions(pts: np.ndarray, side: str) -> dict:
    """
    Hand bone as SIX numbers: long axis, then palm normal.

    The extra three are what make "palm toward the camera" reproducible. With a
    direction alone the player picks the smallest rotation that aims the bone,
    and the twist it happens to land on is arbitrary — the hand points correctly
    while facing anywhere.
    """
    axis = pts[MIDDLE_MCP] - pts[0]
    return {f"Hand.{side}": unit(axis) + unit_world(palm_normal(pts))}


def finger_directions(pts: np.ndarray, side: str) -> dict:
    """Unit direction for each of the three bones in every finger."""
    out = {}
    for finger, chain in FINGER_CHAINS.items():
        for seg in range(3):
            a = chain[seg + 1]
            b = chain[seg + 2] if seg + 2 < len(chain) else None
            v = (pts[b] - pts[a]) if b is not None else (pts[chain[seg + 1]] - pts[chain[seg]])
            out[f"{finger}.0{seg + 1}.{side}"] = unit(v)
    return out


def frame_to_bone_dirs(frame) -> dict:
    """
    One feature frame -> {boneName: [...]} for every bone it drives.

    Values are three numbers (a direction) except hand bones, which carry six
    (direction + palm normal). The player tells them apart by length.
    """
    from feature_extractor import VALS_PER_HAND

    pose = {}
    frame = np.asarray(frame, dtype=np.float32)
    for hand_idx, side in ((0, "L"), (1, "R")):
        base = hand_idx * VALS_PER_HAND
        block = frame[base:base + VALS_PER_HAND]
        if not np.any(block):
            continue
        pts = hand_points(block)
        pose.update(arm_directions(hand_centre(block), side))
        pose.update(hand_directions(pts, side))
        pose.update(finger_directions(pts, side))
    return pose
