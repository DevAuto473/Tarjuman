/**
 * useSignPlayer.js — drives the robot's bones from pose data
 * ===========================================================
 * Turns dictionary entries into actual skeleton motion at runtime, so new
 * vocabulary never requires a Blender round-trip.
 *
 * How it works each frame
 * -----------------------
 *   1. Find which two keyframes the playhead sits between.
 *   2. Slerp every bone's quaternion between them.
 *   3. Write the result onto the bone, on top of its REST rotation.
 *
 * Why quaternions and not Euler angles
 * ------------------------------------
 * Interpolating Euler angles directly produces gimbal artefacts and takes
 * visibly wrong paths through space — an arm that should sweep sideways
 * instead corkscrews. Poses are AUTHORED in degrees because humans think in
 * degrees, then converted to quaternions for the actual blending.
 *
 * Why the idle clip is faded out
 * ------------------------------
 * The GLTF's `Robot_Idle` action writes to the same bones every frame. If it
 * kept running, it would overwrite the sign a moment after it was applied and
 * the robot would appear frozen. It is faded out while signing and faded back
 * in afterwards, which also gives a natural settle at the end.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const DEG2RAD = Math.PI / 180;

// Blend time into and out of the signing state, in seconds.
const ENTER_BLEND = 0.25;
const EXIT_BLEND = 0.35;
// Pause inserted between consecutive words so a sentence does not run together.
const INTER_SIGN_GAP = 0.18;

/** Convert an authored [x°, y°, z°] triple into a quaternion. */
function eulerDegToQuat(deg, target = new THREE.Quaternion()) {
  return target.setFromEuler(
    new THREE.Euler(deg[0] * DEG2RAD, deg[1] * DEG2RAD, deg[2] * DEG2RAD, 'XYZ')
  );
}

export function useSignPlayer({ scene, actions, enabled = true }) {
  const bonesRef = useRef(new Map());       // boneName -> THREE.Bone
  const restRef = useRef(new Map());        // boneName -> rest quaternion
  const queueRef = useRef([]);              // pending signs
  const currentRef = useRef(null);          // { sign, startedAt }
  const blendRef = useRef(0);               // 0 = idle, 1 = fully signing
  const scratchA = useRef(new THREE.Quaternion());
  const scratchB = useRef(new THREE.Quaternion());
  const scratchOut = useRef(new THREE.Quaternion());

  const [signing, setSigning] = useState(false);
  const [currentWord, setCurrentWord] = useState(null);

  // ── Bind bones once the model exists ──────────────────────────────────────
  useEffect(() => {
    if (!scene) return;
    const bones = new Map();
    const rest = new Map();

    scene.traverse((obj) => {
      if (obj.isBone) {
        bones.set(obj.name, obj);
        rest.set(obj.name, obj.quaternion.clone());
      }
    });

    bonesRef.current = bones;
    restRef.current = rest;

    if (bones.size === 0) {
      console.warn('[signing] no bones found — is the GLB actually rigged?');
    } else {
      console.log(`[signing] bound ${bones.size} bones`);
    }
  }, [scene]);

  // ── Public API ────────────────────────────────────────────────────────────

  /** Queue one or more { word, sign } entries for playback. */
  const playSigns = useCallback((entries) => {
    const playable = entries.filter((e) => e.sign);
    if (playable.length === 0) return false;
    queueRef.current.push(...playable);
    return true;
  }, []);

  const stop = useCallback(() => {
    queueRef.current = [];
    currentRef.current = null;
    setCurrentWord(null);
  }, []);

  // ── Per-frame update (call from useFrame) ─────────────────────────────────
  const update = useCallback((delta) => {
    const bones = bonesRef.current;
    const rest = restRef.current;
    if (!enabled || bones.size === 0) return;

    // Pull the next sign off the queue when idle
    if (!currentRef.current && queueRef.current.length > 0) {
      const next = queueRef.current.shift();
      currentRef.current = { ...next, elapsed: -INTER_SIGN_GAP };
      setCurrentWord(next.word);
      setSigning(true);
    }

    const active = currentRef.current;

    // ── Blend weight: ramp toward signing or back toward idle ───────────────
    const targetBlend = active ? 1 : 0;
    const rate = targetBlend > blendRef.current
      ? delta / ENTER_BLEND
      : delta / EXIT_BLEND;
    blendRef.current = THREE.MathUtils.clamp(
      blendRef.current + Math.sign(targetBlend - blendRef.current) * rate,
      0, 1
    );

    // Keep the idle clip out of the way while the sign is driving the bones
    const idle = actions?.Robot_Idle;
    if (idle) idle.setEffectiveWeight(1 - blendRef.current);

    if (!active) {
      if (blendRef.current <= 0.001 && signing) {
        setSigning(false);
        setCurrentWord(null);
      }
      // Ease bones back to rest while blending out
      if (blendRef.current > 0.001) {
        for (const [name, bone] of bones) {
          const restQ = rest.get(name);
          if (restQ) bone.quaternion.slerp(restQ, 1 - blendRef.current);
        }
      }
      return;
    }

    // ── Advance the playhead ────────────────────────────────────────────────
    active.elapsed += delta;
    if (active.elapsed < 0) return;           // still in the inter-sign gap

    const { duration, keys } = active.sign;
    const t = THREE.MathUtils.clamp(active.elapsed / duration, 0, 1);

    // Find the surrounding keyframes
    let a = keys[0], b = keys[keys.length - 1];
    for (let i = 0; i < keys.length - 1; i++) {
      if (t >= keys[i].t && t <= keys[i + 1].t) { a = keys[i]; b = keys[i + 1]; break; }
    }
    const span = Math.max(1e-6, b.t - a.t);
    const localT = THREE.MathUtils.clamp((t - a.t) / span, 0, 1);
    // Smoothstep: real gestures accelerate and decelerate, they do not move
    // at a constant speed between poses.
    const eased = localT * localT * (3 - 2 * localT);

    // ── Apply every bone mentioned by either keyframe ───────────────────────
    const touched = new Set([...Object.keys(a.pose), ...Object.keys(b.pose)]);

    for (const name of touched) {
      const bone = bones.get(name);
      const restQ = rest.get(name);
      if (!bone || !restQ) continue;

      const from = a.pose[name] ? eulerDegToQuat(a.pose[name], scratchA.current)
                                : scratchA.current.identity();
      const to = b.pose[name] ? eulerDegToQuat(b.pose[name], scratchB.current)
                              : scratchB.current.identity();

      // Interpolate the OFFSET, then apply it on top of the rest rotation.
      scratchOut.current.copy(from).slerp(to, eased);
      bone.quaternion.copy(restQ).multiply(scratchOut.current);
    }

    // Bones this sign never mentions drift back to rest, so a curled finger
    // from a previous sign cannot leak into this one.
    for (const [name, bone] of bones) {
      if (touched.has(name)) continue;
      const restQ = rest.get(name);
      if (restQ) bone.quaternion.slerp(restQ, Math.min(1, delta * 8));
    }

    if (t >= 1) {
      currentRef.current = null;
      if (queueRef.current.length === 0) setCurrentWord(null);
    }
  }, [actions, enabled, signing]);

  return {
    playSigns,
    stop,
    update,
    signing,
    currentWord,
    queueLength: queueRef.current.length,
    boneCount: bonesRef.current.size,
  };
}
