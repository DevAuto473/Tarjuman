/**
 * scripts/trainstill.mjs — from recorded videos to a trained model, in one command
 * ================================================================================
 *   npm run trainstill
 *   npm run trainstill -- --dry-run       # extract and check, don't train
 *   npm run trainstill -- --no-mirror     # the video is already mirrored
 *   npm run trainstill -- --skip-extract  # train on the existing v6, no re-extract
 *   npm run trainstill -- --drop-rare     # set aside words with too few samples
 *   npm run trainstill -- --jobs 4        # parallel workers
 *   npm run trainstill -- --no-export     # train only, leave the avatar alone
 *
 * Three steps:
 *   1. video_to_csv.py     data/DATA/*  ->  data/dynamic_gestures_v6.csv
 *   2. train_model.py      that file    ->  sign_model.onnx + data/labels.json
 *   3. export_signs_3d.py  that file    ->  public/trained_signs.json
 *
 * Why the export belongs here
 * ---------------------------
 * The recogniser and the avatar read the SAME dataset: one learns to read each
 * sign, the other learns to perform it. Run separately they drift — the model
 * answers a word the avatar cannot show, because trained_signs.json was left
 * behind on an older CSV. That is exactly what happened: it was still built
 * from v5 while the model had moved on. Exporting in the same command is what
 * keeps the two in step.
 *
 * Why v6 is rebuilt from scratch every run
 * ----------------------------------------
 * `video_to_csv.py` appends rather than replaces — which is right when you add a
 * new batch. But a single command run twice must give the same result, otherwise
 * every sample silently doubles and training accuracy rises on a lie (the same
 * row lands in both the train and the test split). The videos are the source and
 * the CSV is derived from them: the old file is moved aside and rebuilt.
 */

import { spawn } from 'node:child_process';
import { existsSync, renameSync, statSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const runPython = join(root, 'scripts', 'run-python.mjs');

const VIDEOS = join('data', 'DATA');
const LABELS = join('data', 'DATA', 'labels_map.json');
const OUT    = join('data', 'dynamic_gestures_v6.csv');

const argv = process.argv.slice(2);
const has = (f) => argv.includes(f);
// --drop-rare is for the trainer, not the extractor; passing it to
// video_to_csv.py would just be an unknown argument.
const trainOnly = ['--skip-extract', '--drop-rare', '--no-export'];
const passthrough = argv.filter((a) => !trainOnly.includes(a));
const trainEnv = has('--drop-rare') ? { TARJUMAN_DROP_RARE: '1' } : {};

function step(args, env = {}) {
  return new Promise((done) => {
    const child = spawn(process.execPath, [runPython, ...args], {
      cwd: root, stdio: 'inherit', env: { ...process.env, ...env },
    });
    for (const s of ['SIGINT', 'SIGTERM']) process.on(s, () => child.kill(s));
    child.on('exit', (code) => done(code ?? 0));
    child.on('error', (e) => { console.error(`FAILED  ${e.message}`); done(1); });
  });
}

/**
 * Rows in the file that did NOT come from a video, by their `source` column.
 *
 * `data_collector.py` records straight from the camera and appends to the same
 * CSV. This command rebuilds that CSV from the videos, so those rows would be
 * rebuilt away — recoverable from the .bak, but only by someone who knew to
 * look. Counting them first means the loss is announced instead of discovered.
 *
 * Only the first few fields are parsed: a row is 4,338 columns wide and
 * `source` sits near the front, so splitting the whole line would cost seconds
 * per megabyte for six fields.
 */
async function liveRows(path) {
  if (!existsSync(path)) return 0;
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  let col = -1, n = 0, first = true;
  for await (const line of rl) {
    if (first) {
      first = false;
      col = line.slice(0, 400).split(',').indexOf('source');
      if (col < 0) break;                 // older file, no provenance recorded
      continue;
    }
    const head = line.slice(0, 400).split(',');
    if (head.length > col && head[col] && head[col] !== 'video') n++;
  }
  rl.close();
  return n;
}

/** Number of data rows in the file, excluding the header. */
async function countRows(path) {
  if (!existsSync(path)) return 0;
  let n = 0;
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const _ of rl) n++;
  return Math.max(0, n - 1);
}

const banner = (t) => console.log(`\n${'='.repeat(70)}\n   ${t}\n${'='.repeat(70)}`);

(async () => {
  if (!has('--skip-extract')) {
    if (!existsSync(join(root, VIDEOS))) {
      console.error(`FAILED  No video folder: ${VIDEOS}`);
      process.exit(1);
    }
    if (!existsSync(join(root, LABELS))) {
      console.error(`FAILED  No label map: ${LABELS}`);
      console.error('        Without it one word enters the dataset under two labels.');
      process.exit(1);
    }

    // Move the previous file aside so rows do not double on a re-run.
    const outAbs = join(root, OUT);
    if (existsSync(outAbs) && statSync(outAbs).size > 0) {
      const live = await liveRows(outAbs);
      if (live > 0) {
        console.log(`\n   [!] ${OUT} holds ${live} row(s) recorded live from the`);
        console.log('       camera, not from a video. Rebuilding from data/DATA');
        console.log('       leaves them behind — they survive only in the .bak.');
        console.log('       To keep them, stop now and run with --skip-extract.\n');
      }
      renameSync(outAbs, `${outAbs}.bak`);
      console.log(`   Previous file moved to ${OUT}.bak`);
    }

    banner('1/3  Extracting features from the videos');
    const code = await step([
      'scripts/video_to_csv.py',
      '--videos', VIDEOS, '--labels-map', LABELS, '--out', OUT,
      ...passthrough,
    ]);
    if (code) process.exit(code);
  }

  if (has('--dry-run')) {
    console.log('\n   (Dry run - nothing trained)');
    process.exit(0);
  }

  const rows = await countRows(join(root, OUT));
  if (rows < 2) {
    console.error(`\nFAILED  ${OUT} has ${rows} row(s) - nothing to train on.`);
    console.error('        Check the rejection reasons in the report above.');
    process.exit(1);
  }

  banner(`2/3  Training on ${rows} samples`);
  const trained = await step(['train_model.py'], { TARJUMAN_CSV: OUT, ...trainEnv });
  if (trained) process.exit(trained);

  if (has('--no-export')) {
    console.log('\n   (--no-export: the avatar keeps its previous signs)');
    process.exit(0);
  }

  // The SAME csv the trainer used. Left to itself, export_signs_3d.py falls back
  // to TARJUMAN_DATASET, which still names v5 — which is how the avatar came to
  // perform an older vocabulary than the model could recognise.
  banner('3/3  Exporting the signs the avatar performs');
  const exported = await step(['export_signs_3d.py'], { TARJUMAN_CSV: OUT, ...trainEnv });
  if (exported) {
    console.error('\n[!] The model is trained and saved. Only the avatar export failed.');
    console.error('    Retry that step alone:  npm run export3d');
  }
  process.exit(exported);
})();
