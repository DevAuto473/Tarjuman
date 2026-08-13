/**
 * poses.js — reusable hand shapes for the Tarjuman robot
 * =======================================================
 * Bone names come straight from the rig inside TarjumanRobot2.glb
 * (skin "Robot_Rig", 57 joints, standard Blender .L/.R naming).
 *
 * Why a pose LIBRARY instead of one animation per word
 * ----------------------------------------------------
 * Sign language reuses a small set of hand shapes over and over — a fist, a
 * flat palm, a pointing index. Authoring a separate Blender animation per word
 * would re-model those same shapes hundreds of times. Here each shape is
 * defined ONCE as data, and a word becomes "hand shape X carried along arm
 * path Y". Adding a word is then a few lines of JSON, not a modelling session.
 *
 * Rotation convention
 * -------------------
 * Every value is [x, y, z] Euler angles in DEGREES, applied in the bone's own
 * local space, relative to its rest pose. Degrees (not radians) because these
 * numbers are meant to be hand-tuned by a human.
 *
 * ⚠️  The numeric values below are STARTING POINTS, not finished poses. A
 *     Blender rig's bone axes are arbitrary, so the only reliable way to get
 *     real angles is to tune them visually — that is what the calibration
 *     panel is for. Treat these as a skeleton to correct, not as truth.
 */

// ── Bone name constants ──────────────────────────────────────────────────────
// Centralised so a rig rename breaks in ONE place instead of silently
// producing a robot that simply never moves.
export const BONES = {
  root: 'Root',
  hips: 'Hips',
  spine: 'Spine',
  chest: 'Chest',
  neck: 'Neck',
  head: 'Head',
  jaw: 'Jaw',
  shoulder: (s) => `Shoulder.${s}`,
  upperArm: (s) => `UpperArm.${s}`,
  lowerArm: (s) => `LowerArm.${s}`,
  hand: (s) => `Hand.${s}`,
  finger: (name, seg, s) => `${name}.0${seg}.${s}`,
};

export const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
export const SIDES = ['L', 'R'];

/**
 * Build the finger-bone rotations for one hand from a per-finger curl amount.
 *
 * @param {Object} curls  e.g. { index: 0, middle: 1, thumb: 0.5 }
 *                        0 = fully extended, 1 = fully curled
 * @param {string} side   'L' or 'R'
 */
export function fingerPose(curls, side) {
  const pose = {};
  for (const finger of FINGERS) {
    const curl = curls[finger] ?? 0;
    // Each successive joint bends a little more, which is what a real finger
    // does — the tip curls further than the knuckle.
    const perSegment = finger === 'thumb'
      ? [-45 * curl, -35 * curl, -30 * curl]
      : [-80 * curl, -70 * curl, -55 * curl];

    for (let seg = 1; seg <= 3; seg++) {
      pose[BONES.finger(finger, seg, side)] = [perSegment[seg - 1], 0, 0];
    }
  }
  return pose;
}

// ── Hand shape library ───────────────────────────────────────────────────────
// Curl amounts per finger: 0 = straight, 1 = fully closed.
export const HAND_SHAPES = {
  flat:    { thumb: 0.0, index: 0.0, middle: 0.0, ring: 0.0, pinky: 0.0 },
  fist:    { thumb: 0.9, index: 1.0, middle: 1.0, ring: 1.0, pinky: 1.0 },
  point:   { thumb: 0.8, index: 0.0, middle: 1.0, ring: 1.0, pinky: 1.0 },
  peace:   { thumb: 0.9, index: 0.0, middle: 0.0, ring: 1.0, pinky: 1.0 },
  thumbUp: { thumb: 0.0, index: 1.0, middle: 1.0, ring: 1.0, pinky: 1.0 },
  pinch:   { thumb: 0.5, index: 0.5, middle: 1.0, ring: 1.0, pinky: 1.0 },
  cup:     { thumb: 0.4, index: 0.4, middle: 0.4, ring: 0.4, pinky: 0.4 },
  open5:   { thumb: 0.1, index: 0.1, middle: 0.1, ring: 0.1, pinky: 0.1 },
};

/** Convenience: full finger rotation map for a named shape. */
export function handShape(shapeName, side) {
  const curls = HAND_SHAPES[shapeName];
  if (!curls) {
    console.warn(`[signing] unknown hand shape: ${shapeName}`);
    return {};
  }
  return fingerPose(curls, side);
}

// ── Arm positions ────────────────────────────────────────────────────────────
// Coarse placements for where the hand sits in space. Sign language is highly
// location-dependent (chest vs. face vs. neutral space), so these are named by
// LOCATION rather than by angle.
export const ARM_POSITIONS = {
  rest: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 0],
    [BONES.upperArm(s)]: [0, 0, 0],
    [BONES.lowerArm(s)]: [0, 0, 0],
  }),
  chest: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 5],
    [BONES.upperArm(s)]: [-45, 0, s === 'R' ? 25 : -25],
    [BONES.lowerArm(s)]: [-60, 0, 0],
  }),
  face: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 10],
    [BONES.upperArm(s)]: [-70, 0, s === 'R' ? 20 : -20],
    [BONES.lowerArm(s)]: [-80, 0, 0],
  }),
  forehead: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 12],
    [BONES.upperArm(s)]: [-85, 0, s === 'R' ? 15 : -15],
    [BONES.lowerArm(s)]: [-90, 0, 0],
  }),
  outward: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 8],
    [BONES.upperArm(s)]: [-50, 0, s === 'R' ? 45 : -45],
    [BONES.lowerArm(s)]: [-20, 0, 0],
  }),
  side: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 0],
    [BONES.upperArm(s)]: [-30, 0, s === 'R' ? 50 : -50],
    [BONES.lowerArm(s)]: [-30, 0, 0],
  }),
};

/**
 * Compose one full keyframe pose.
 *
 * @param {Object} spec
 *   arm   : { R: 'chest', L: 'rest' }      — position name per side
 *   hand  : { R: 'flat',  L: 'fist' }      — hand shape per side
 *   head  : [x, y, z]                      — optional head tilt
 *   extra : { 'BoneName': [x,y,z] }        — manual overrides, applied last
 */
export function composePose({ arm = {}, hand = {}, head, extra = {} } = {}) {
  let pose = {};

  for (const side of SIDES) {
    const position = arm[side];
    if (position && ARM_POSITIONS[position]) {
      pose = { ...pose, ...ARM_POSITIONS[position](side) };
    }
    const shape = hand[side];
    if (shape) pose = { ...pose, ...handShape(shape, side) };
  }

  if (head) pose[BONES.head] = head;

  return { ...pose, ...extra };
}
