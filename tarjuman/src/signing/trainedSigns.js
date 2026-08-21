/**
 * trainedSigns.js — signs derived from your own recordings
 * =========================================================
 * Loads `public/trained_signs.json`, produced by `npm run export3d` from the
 * dataset you recorded. Every word taught to the recogniser becomes a word the
 * robot can perform, without anyone authoring poses by hand.
 *
 * Two dictionaries, deliberately
 * ------------------------------
 *   dictionary.js   — hand-authored poses. Clean and deliberate, but every new
 *                     word costs manual tuning.
 *   trained_signs   — derived from real recordings. Free for every word you
 *                     teach, and it moves the way a person actually moved.
 *
 * Recorded entries win when both exist: a real performance beats an estimate.
 */

let cache = null;
let loadPromise = null;

/**
 * Fetch the exported signs once and keep them.
 *
 * A missing file is normal — it simply means nothing has been exported yet —
 * so this resolves to an empty set rather than throwing and breaking the UI.
 */
export function loadTrainedSigns() {
  if (cache) return Promise.resolve(cache);
  if (loadPromise) return loadPromise;

  loadPromise = fetch('/trained_signs.json')
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      cache = data?.signs ?? {};
      console.log(`[signing] loaded ${Object.keys(cache).length} recorded sign(s)`);
      return cache;
    })
    .catch((err) => {
      // Not an error worth surfacing: it just means `npm run export3d`
      // has not been run yet.
      console.info('[signing] no trained_signs.json yet —', err.message);
      cache = {};
      return cache;
    });

  return loadPromise;
}

/** All recorded signs as [{ id, label, duration, keys }]. */
export function trainedList() {
  return Object.values(cache ?? {}).sort((a, b) =>
    String(a.label).localeCompare(String(b.label), 'ar')
  );
}

/** Look a recorded sign up by its id or its Arabic label. */
export function findTrainedSign(query) {
  const signs = cache ?? {};
  if (signs[query]) return signs[query];
  const q = String(query).trim();
  return Object.values(signs).find((s) => s.label === q) ?? null;
}

export function trainedCount() {
  return Object.keys(cache ?? {}).length;
}
