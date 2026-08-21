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

/**
 * Rotation that makes a bone point along `worldDir`, given its rest pose.
 *
 * Why this exists
 * ---------------
 * Recorded signs are exported as DIRECTIONS, not angles. A rig's bone axes are
 * arbitrary — "rotate 40° about X" might bend a finger on one skeleton and
 * twist it on another — so authored angles are a guess, and a wrong guess
 * produces splayed, stretched poses.
 *
 * A direction is unambiguous. Here it is converted into the bone's own space
 * using the skeleton's actual rest orientation, so the maths adapts to whatever
 * rig is loaded rather than assuming one.
 */
const _restDir = new THREE.Vector3();
const _targetDir = new THREE.Vector3();
const _parentQuat = new THREE.Quaternion();
const _invParent = new THREE.Quaternion();

function quatFromDirection(bone, restDirLocal, worldDir, out = new THREE.Quaternion()) {
  if (!restDirLocal) return out.identity();

  // Bring the world-space target into the bone's parent space, because a
  // bone's local rotation is expressed relative to its parent.
  _targetDir.set(worldDir[0], worldDir[1], worldDir[2]).normalize();
  if (bone.parent) {
    bone.parent.getWorldQuaternion(_parentQuat);
    _invParent.copy(_parentQuat).invert();
    _targetDir.applyQuaternion(_invParent);
  }

  _restDir.copy(restDirLocal).normalize();
  return out.setFromUnitVectors(_restDir, _targetDir);
}

/**
 * Rotation that points a bone along `worldDir` AND rolls it so its palm faces
 * `worldNormal`.
 *
 * Why a direction alone is not enough
 * -----------------------------------
 * Infinitely many rotations point a bone the right way — they differ by a spin
 * about the bone's own axis. `setFromUnitVectors` picks the smallest one, which
 * is an arbitrary choice, so a hand would point correctly while its palm faced
 * anywhere at all. Supplying a second, non-parallel vector removes that freedom
 * and pins the roll down completely.
 *
 * Both pairs are turned into orthonormal frames and the rotation between the
 * frames is read off, which needs no assumption about which rig axis is "up".
 */
const _restX = new THREE.Vector3();
const _restY = new THREE.Vector3();
const _restZ = new THREE.Vector3();
const _tgtX = new THREE.Vector3();
const _tgtY = new THREE.Vector3();
const _tgtZ = new THREE.Vector3();
const _mRest = new THREE.Matrix4();
const _mTgt = new THREE.Matrix4();
const _nrm = new THREE.Vector3();

function orthonormal(dir, ref, ex, ey, ez) {
  ex.copy(dir).normalize();
  ey.copy(ref).addScaledVector(ex, -ref.dot(ex));
  if (ey.lengthSq() < 1e-10) {
    // Reference parallel to the bone: any perpendicular will do, and the roll
    // is genuinely undetermined in this frame anyway.
    ey.set(0, 0, 1).addScaledVector(ex, -ex.z);
    if (ey.lengthSq() < 1e-10) ey.set(0, 1, 0).addScaledVector(ex, -ex.y);
  }
  ey.normalize();
  ez.crossVectors(ex, ey);
}

function quatFromDirectionAndNormal(bone, restDirLocal, restNormalLocal,
  worldDir, worldNormal, out = new THREE.Quaternion()) {
  if (!restDirLocal || !restNormalLocal) return out.identity();

  _targetDir.set(worldDir[0], worldDir[1], worldDir[2]).normalize();
  _nrm.set(worldNormal[0], worldNormal[1], worldNormal[2]).normalize();
  if (bone.parent) {
    bone.parent.getWorldQuaternion(_parentQuat);
    _invParent.copy(_parentQuat).invert();
    _targetDir.applyQuaternion(_invParent);
    _nrm.applyQuaternion(_invParent);
  }

  orthonormal(restDirLocal, restNormalLocal, _restX, _restY, _restZ);
  orthonormal(_targetDir, _nrm, _tgtX, _tgtY, _tgtZ);

  _mRest.makeBasis(_restX, _restY, _restZ);
  _mTgt.makeBasis(_tgtX, _tgtY, _tgtZ);
  // rotation = target frame * inverse(rest frame); both are orthonormal, so the
  // inverse is the transpose and no general matrix inversion is needed.
  _mRest.transpose();
  _mTgt.multiply(_mRest);
  return out.setFromRotationMatrix(_mTgt);
}

/**
 * Reduce a bone name to a form that survives exporters and loaders.
 *
 * three.js strips characters reserved by its animation-binding syntax — dots
 * above all — so Blender's `thumb.01.R` arrives in the scene as `thumb01R`.
 * Pose data written against the Blender names then matches nothing, and the
 * robot silently holds its bind pose: every bone "missing", no error anywhere.
 *
 * Comparing on a normalised key sidesteps the whole class of problem, whatever
 * separator a future rig happens to use.
 */
function normaliseBoneName(name) {
  return String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
}

/** How deep a bone sits in the skeleton, used to solve parents before children. */
function depthOf(bone) {
  let d = 0;
  let n = bone;
  while (n && n.parent) { d += 1; n = n.parent; }
  return d;
}

export function useSignPlayer({ scene, actions, enabled = true }) {
  const bonesRef = useRef(new Map());       // boneName -> THREE.Bone
  const restRef = useRef(new Map());        // boneName -> rest quaternion
  // boneName -> unit vector the bone points along in its OWN local space.
  // Taken from the child's rest offset, which IS the bone's direction.
  const restDirRef = useRef(new Map());
  // boneName -> palm normal in the bone's own local space, for bones that have
  // a meaningful roll (the hands). Everything else is roll-free.
  const restNormalRef = useRef(new Map());
  const queueRef = useRef([]);              // pending signs
  const currentRef = useRef(null);          // { sign, startedAt }
  const blendRef = useRef(0);               // 0 = idle, 1 = fully signing
  const signingRef = useRef(false);         // guards the setSigning setter
  const clipsPausedRef = useRef(null);      // tracks the mixer pause state
  const liveRef = useRef(false);            // live mirror is driving the bones
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
    const restDirs = new Map();
    const restNormals = new Map();     // hand bones only: the palm's facing

    scene.traverse((obj) => {
      if (obj.isBone) {
        bones.set(obj.name, obj);
        rest.set(obj.name, obj.quaternion.clone());
        // Second key so pose data written with Blender's dotted names
        // ("thumb.01.R") still finds the loader's stripped name ("thumb01R").
        const key = normaliseBoneName(obj.name);
        if (key !== obj.name) {
          bones.set(key, obj);
          rest.set(key, obj.quaternion.clone());
        }
      }
    });

    // A bone's direction is where its child sits. Bones with no child (finger
    // tips) inherit their parent's direction, so they still animate sensibly.
    //
    // A hand is the exception: it has five children fanning out, so the first
    // one found would aim the palm at the index finger. The MEAN of the finger
    // roots is the hand's real long axis.
    for (const [name, bone] of bones) {
      const boneKids = bone.children.filter((c) => c.isBone);
      if (boneKids.length >= 3) {
        const mean = new THREE.Vector3();
        boneKids.forEach((c) => mean.add(c.position));
        mean.divideScalar(boneKids.length);
        if (mean.lengthSq() > 1e-10) restDirs.set(name, mean.normalize());
        continue;
      }
      const child = boneKids[0];
      if (child && child.position.lengthSq() > 1e-10) {
        restDirs.set(name, child.position.clone().normalize());
      }
    }

    // Rest palm normal, measured from the RIG rather than assumed. The index
    // and pinky knuckles span the palm, so their cross product is its normal —
    // the same construction, in the same order, that the backend runs on the
    // MediaPipe landmarks. Deriving both sides the same way is what lets them
    // agree on which face is the palm without hard-coding either rig's axes.
    const findKid = (bone, prefix) => bone.children.find(
      (c) => c.isBone && normaliseBoneName(c.name).startsWith(prefix));

    for (const [name, bone] of bones) {
      const idx = findKid(bone, 'index01');
      const pky = findKid(bone, 'pinky01');
      if (!idx || !pky) continue;
      const n = new THREE.Vector3().crossVectors(idx.position, pky.position);
      if (n.lengthSq() > 1e-10) restNormals.set(name, n.normalize());
    }
    for (const [name, bone] of bones) {
      if (restDirs.has(name)) continue;
      const parentDir = bone.parent && restDirs.get(bone.parent.name);
      restDirs.set(name, (parentDir || new THREE.Vector3(0, 1, 0)).clone());
    }

    bonesRef.current = bones;
    restRef.current = rest;
    restDirRef.current = restDirs;
    restNormalRef.current = restNormals;

    if (bones.size === 0) {
      console.warn('[signing] no bones found — is the GLB actually rigged?');
    } else {
      console.log(`[signing] bound ${bones.size} bones, `
        + `${restNormals.size} with a palm normal`);
    }
  }, [scene]);

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Solve one bone's rotation from its pose value.
   *
   * Three numbers mean a direction; SIX mean direction plus palm normal, which
   * additionally fixes the roll. Distinguishing them by length keeps signs
   * exported before hands were driven loading unchanged.
   */
  const solveBone = useCallback((bone, name, value, out) => {
    if (!value || value.length < 3) return false;
    const key = normaliseBoneName(name);
    const restDir = restDirRef.current.get(bone.name) || restDirRef.current.get(key);
    if (!restDir) return false;

    if (value.length >= 6) {
      const restNormal = restNormalRef.current.get(bone.name)
        || restNormalRef.current.get(key);
      if (restNormal) {
        quatFromDirectionAndNormal(bone, restDir, restNormal,
          value, [value[3], value[4], value[5]], out);
        return true;
      }
    }
    quatFromDirection(bone, restDir, value, out);
    return true;
  }, []);

  /** Queue one or more { word, sign } entries for playback. */
  const playSigns = useCallback((entries) => {
    const playable = entries.filter((e) => e.sign);
    if (playable.length === 0) return false;
    queueRef.current.push(...playable);
    return true;
  }, []);

  /**
   * Drive the skeleton straight from a live pose, bypassing the queue.
   *
   * Used by the live mirror: there is no timeline to interpolate, just the
   * newest frame. Bones are solved parents-first for the same reason the
   * keyframe path does it — a child converts into its parent's space.
   */
  const applyLivePose = useCallback((pose) => {
    const bones = bonesRef.current;
    if (!pose || bones.size === 0) return;

    const resolve = (n) => bones.get(n) || bones.get(normaliseBoneName(n));
    const names = Object.keys(pose)
      .filter((n) => resolve(n))
      .sort((a, b) => depthOf(resolve(a)) - depthOf(resolve(b)));

    for (const name of names) {
      const bone = resolve(name);
      if (!solveBone(bone, name, pose[name], scratchOut.current)) continue;
      // Smooth a little: raw landmark jitter would otherwise read as a tremor.
      bone.quaternion.slerp(scratchOut.current, 0.5);
      bone.updateMatrixWorld(true);
    }
    liveRef.current = true;
  }, [solveBone]);

  /** Leave live mode and let the idle clip take the skeleton back. */
  const clearLivePose = useCallback(() => { liveRef.current = false; }, []);

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

    // Accept either the authored name or its normalised form.
    const resolve = (n) => bones.get(n) || bones.get(normaliseBoneName(n));

    // Pull the next sign off the queue when idle
    if (!currentRef.current && queueRef.current.length > 0) {
      const next = queueRef.current.shift();
      currentRef.current = { ...next, elapsed: -INTER_SIGN_GAP };
      // State is written through a ref-guard, never unconditionally: calling a
      // setter inside useFrame re-renders the component on every single frame,
      // which recreates this very callback mid-animation.
      if (!signingRef.current) { signingRef.current = true; setSigning(true); }
      setCurrentWord(next.word);

      const boneNames = new Set(next.sign.keys.flatMap((k) => Object.keys(k.pose)));
      const missing = [...boneNames].filter((n) => !resolve(n));
      console.log(
        `[signing] "${next.word}" format=${next.sign.format || 'euler'} `
        + `keys=${next.sign.keys.length} bones=${boneNames.size} `
        + `missing=${missing.length}`
        + (missing.length ? ` -> ${missing.slice(0, 5).join(', ')}` : '')
      );
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

    // Keep the idle clip out of the way while the sign drives the bones.
    // Weight alone is not enough — a running action still writes to the
    // skeleton, and whichever runs last wins. Every clip is paused outright
    // while signing, then resumed.
    const clips = actions ? Object.values(actions) : [];
    const wantPaused = !!active || liveRef.current;
    if (clipsPausedRef.current !== wantPaused) {
      clipsPausedRef.current = wantPaused;
      for (const a of clips) {
        if (!a) continue;
        a.paused = wantPaused;
        a.setEffectiveWeight(wantPaused ? 0 : 1);
        if (!wantPaused && !a.isRunning()) a.play();
      }
      console.log(`[signing] clips ${wantPaused ? 'paused' : 'resumed'} (${clips.length})`);
    }

    // While the mirror is driving the bones, the keyframe path must keep its
    // hands off them — including the "drift back to rest" pass below.
    if (liveRef.current && !active) return;

    if (!active) {
      if (blendRef.current <= 0.001 && signingRef.current) {
        signingRef.current = false;
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
    const isDirections = active.sign.format === 'directions';
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

    // Hierarchy order matters: quatFromDirection converts into PARENT space, so
    // a parent solved after its child would leave the child using a stale
    // reference. Sorting by depth guarantees parents go first.
    const ordered = [...touched]
      .filter((n) => resolve(n))
      .sort((n1, n2) => depthOf(resolve(n1)) - depthOf(resolve(n2)));

    for (const name of ordered) {
      const bone = resolve(name);
      const restQ = rest.get(name) || rest.get(normaliseBoneName(name));
      if (!bone || !restQ) continue;

      if (isDirections) {
        // Recorded sign: blend the two DIRECTIONS, then solve the rotation
        // once. Interpolating directions and solving afterwards avoids the
        // shortest-arc artefacts of blending two separately-solved rotations.
        const da = a.pose[name];
        const db = b.pose[name] || da;
        if (!da) continue;
        // Blends all components, so a hand's palm normal (elements 3-5) is
        // carried through the interpolation exactly like its direction.
        const n = Math.min(da.length, db.length);
        const dir = new Array(n);
        for (let i = 0; i < n; i += 1) dir[i] = da[i] + (db[i] - da[i]) * eased;

        if (!solveBone(bone, name, dir, scratchOut.current)) continue;
        bone.quaternion.copy(scratchOut.current);
        bone.updateMatrixWorld(true);   // children resolve against the new pose
      } else {
        const from = a.pose[name] ? eulerDegToQuat(a.pose[name], scratchA.current)
                                  : scratchA.current.identity();
        const to = b.pose[name] ? eulerDegToQuat(b.pose[name], scratchB.current)
                                : scratchB.current.identity();

        // Interpolate the OFFSET, then apply it on top of the rest rotation.
        scratchOut.current.copy(from).slerp(to, eased);
        bone.quaternion.copy(restQ).multiply(scratchOut.current);
      }
    }

    // Bones this sign never mentions drift back to rest, so a curled finger
    // from a previous sign cannot leak into this one.
    // `bones` holds each bone under two keys (raw + normalised), so compare on
    // the resolved object rather than the key to avoid touching one twice.
    const touchedBones = new Set(ordered.map((n) => resolve(n)).filter(Boolean));
    for (const [name, bone] of bones) {
      if (touchedBones.has(bone)) continue;
      const restQ = rest.get(name);
      if (restQ) bone.quaternion.slerp(restQ, Math.min(1, delta * 8));
    }

    if (t >= 1) {
      currentRef.current = null;
      if (queueRef.current.length === 0) setCurrentWord(null);
    }
  }, [actions, enabled, signing, solveBone]);

  return {
    playSigns,
    applyLivePose,
    clearLivePose,
    stop,
    update,
    signing,
    currentWord,
    queueLength: queueRef.current.length,
    boneCount: bonesRef.current.size,
  };
}
