/**
 * dictionary.js — the sign vocabulary, expressed as data
 * =======================================================
 * Every entry is a short timeline of key poses. The player interpolates
 * between them, so a sign needs only its 2-4 defining moments, not a
 * frame-by-frame animation.
 *
 * Shape of an entry
 * -----------------
 *   duration : seconds the whole sign takes
 *   keys     : [{ t: 0..1 normalised time, pose: composePose({...}) }]
 *
 * Bones you do NOT mention keep their idle/rest rotation, so each sign only
 * describes what actually moves.
 *
 * Adding a word
 * -------------
 * Add one object here. No Blender, no re-export, no rebuild of the model.
 * That is the whole point of driving the rig from data.
 *
 * ⚠️  Angles are first-draft estimates — tune them in the calibration panel.
 */

import { composePose, BONES } from './poses';

export const SIGNS = {
  // ── Greetings ──────────────────────────────────────────────────────────────
  'السلام عليكم': {
    duration: 1.6,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'rest' },     hand: { R: 'flat' } }) },
      { t: 0.4, pose: composePose({ arm: { R: 'forehead' }, hand: { R: 'flat' } }) },
      { t: 0.75, pose: composePose({ arm: { R: 'forehead' }, hand: { R: 'flat' }, head: [5, 0, 0] }) },
      { t: 1.0, pose: composePose({ arm: { R: 'outward' },  hand: { R: 'flat' } }) },
    ],
  },

  'مرحبا': {
    duration: 1.2,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'rest' },  hand: { R: 'open5' } }) },
      { t: 0.4, pose: composePose({ arm: { R: 'face' },  hand: { R: 'open5' }, extra: { [BONES.hand('R')]: [0, 0, -20] } }) },
      { t: 0.7, pose: composePose({ arm: { R: 'face' },  hand: { R: 'open5' }, extra: { [BONES.hand('R')]: [0, 0, 20] } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'face' },  hand: { R: 'open5' }, extra: { [BONES.hand('R')]: [0, 0, -20] } }) },
    ],
  },

  'شكرا': {
    duration: 1.3,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'rest' },    hand: { R: 'flat' } }) },
      { t: 0.45, pose: composePose({ arm: { R: 'face' },   hand: { R: 'flat' } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'outward' }, hand: { R: 'flat' }, head: [8, 0, 0] }) },
    ],
  },

  // ── Common words ───────────────────────────────────────────────────────────
  'نعم': {
    duration: 0.9,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'fist' } }) },
      { t: 0.5, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'fist' }, extra: { [BONES.hand('R')]: [-30, 0, 0] } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'fist' }, extra: { [BONES.hand('R')]: [0, 0, 0] } }) },
    ],
  },

  'لا': {
    duration: 0.9,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'peace' } }) },
      { t: 0.5, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'pinch' }, head: [0, -10, 0] }) },
      { t: 1.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'pinch' }, head: [0, 10, 0] }) },
    ],
  },

  'أنا': {
    duration: 0.8,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'rest' },  hand: { R: 'point' } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'point' } }) },
    ],
  },

  'أنت': {
    duration: 0.8,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'rest' },    hand: { R: 'point' } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'outward' }, hand: { R: 'point' } }) },
    ],
  },

  'من فضلك': {
    duration: 1.4,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'flat' } }) },
      { t: 0.5, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'flat' }, extra: { [BONES.upperArm('R')]: [-35, 0, 20] } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'flat' } }) },
    ],
  },

  'جيد': {
    duration: 1.0,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'fist' } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'thumbUp' }, head: [6, 0, 0] }) },
    ],
  },

  'كيف حالك': {
    duration: 1.5,
    keys: [
      { t: 0.0,  pose: composePose({ arm: { R: 'rest',  L: 'rest' },  hand: { R: 'cup', L: 'cup' } }) },
      { t: 0.45, pose: composePose({ arm: { R: 'chest', L: 'chest' }, hand: { R: 'cup', L: 'cup' } }) },
      { t: 1.0,  pose: composePose({ arm: { R: 'outward', L: 'outward' }, hand: { R: 'open5', L: 'open5' }, head: [0, 0, 5] }) },
    ],
  },

  // ── Demo entries matching the current test dataset ─────────────────────────
  'ليمونة': {
    duration: 1.1,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'cup' } }) },
      { t: 0.5, pose: composePose({ arm: { R: 'face' },  hand: { R: 'fist' } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'face' },  hand: { R: 'cup' } }) },
    ],
  },

  'بابايا': {
    duration: 1.2,
    keys: [
      { t: 0.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'open5' } }) },
      { t: 0.5, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'cup' }, extra: { [BONES.hand('R')]: [0, 25, 0] } }) },
      { t: 1.0, pose: composePose({ arm: { R: 'chest' }, hand: { R: 'cup' }, extra: { [BONES.hand('R')]: [0, -25, 0] } }) },
    ],
  },
};

/**
 * Neutral pose the robot returns to between signs.
 * Kept explicit so a sign can never "leak" a bent finger into the next one.
 */
export const REST_POSE = composePose({
  arm:  { L: 'rest', R: 'rest' },
  hand: { L: 'flat', R: 'flat' },
  head: [0, 0, 0],
});

/** Longest-match lookup: prefers "كيف حالك" over "كيف" when both exist. */
export function findSign(text) {
  const normalised = normaliseArabic(text);
  for (const key of Object.keys(SIGNS)) {
    if (normaliseArabic(key) === normalised) return SIGNS[key];
  }
  return null;
}

/**
 * Strip diacritics and unify alef/ya/ta-marbuta variants.
 *
 * Without this, "أنا" typed with a different hamza, or a word carrying tashkeel
 * from the AI assistant, would silently fail to match its dictionary entry.
 */
export function normaliseArabic(text) {
  return String(text)
    .replace(/[ً-ْـ]/g, '')   // tashkeel + tatweel
    .replace(/[أإآ]/g, 'ا')
    .replace(/ى/g, 'ي')
    .replace(/ة/g, 'ه')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Split a sentence into the longest dictionary phrases it contains.
 * Returns [{ word, sign|null }] so the caller can report what is unavailable
 * instead of silently skipping it.
 */
export function tokenise(sentence, maxPhraseWords = 3) {
  const words = normaliseArabic(sentence).split(' ').filter(Boolean);
  const out = [];
  let i = 0;

  while (i < words.length) {
    let matched = null;
    let span = 0;

    for (let n = Math.min(maxPhraseWords, words.length - i); n >= 1; n--) {
      const phrase = words.slice(i, i + n).join(' ');
      const sign = findSign(phrase);
      if (sign) { matched = { word: phrase, sign }; span = n; break; }
    }

    if (matched) { out.push(matched); i += span; }
    else { out.push({ word: words[i], sign: null }); i += 1; }
  }
  return out;
}

export const AVAILABLE_SIGNS = Object.keys(SIGNS);
