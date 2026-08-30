/**
 * scripts/setup.mjs — one command to take a fresh clone to a working install
 * ==========================================================================
 *     npm run setup
 *
 * Why this exists
 * ---------------
 * Setting this project up on a second machine took an entire afternoon and
 * three separate failures, none of which named its own cause:
 *
 *   1. Python 3.8 was on PATH, so pip reported "no version satisfies
 *      absl-py==2.4.0" — which reads as a missing package, not a wrong
 *      interpreter. pip only ever lists versions compatible with the running
 *      Python, and never says so.
 *
 *   2. requirements.txt pinned opencv-python-headless 4.9, which predates
 *      NumPy 2 and has no GUI at all. It died on `import cv2` with
 *      "_ARRAY_API not found" and would have had no window even if it loaded.
 *      The dev machine had a newer OpenCV installed by hand on top, so it
 *      worked there and nowhere else.
 *
 *   3. A pip upgrade had pulled in protobuf 7, which removed two APIs
 *      MediaPipe calls. `import mediapipe` still succeeded; it only failed
 *      later, when a camera actually started.
 *
 * The common thread is that every check reported success right up until the
 * moment something real was attempted. So this script does not trust version
 * numbers or imports: it BUILDS a MediaPipe graph, DECODES with OpenCV, and
 * only then calls the install good.
 *
 * Safe to run repeatedly. It never overwrites an existing .env — losing your
 * API keys to a setup script is its own kind of bad afternoon.
 */

import { spawnSync } from 'node:child_process';
import { existsSync, copyFileSync, rmSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WIN = process.platform === 'win32';
const VENV = join(ROOT, 'venv');
const VENV_PY = WIN ? join(VENV, 'Scripts', 'python.exe')
                    : join(VENV, 'bin', 'python');

// MediaPipe 0.10.14 publishes wheels for 3.8-3.12; numpy 2.4.x needs >= 3.11.
// The intersection is the only range where a clean install is possible.
const PY_MIN = [3, 11];
const PY_MAX = [3, 12];

let failed = false;

const c = {
  bold: s => `\x1b[1m${s}\x1b[0m`,
  dim: s => `\x1b[2m${s}\x1b[0m`,
  red: s => `\x1b[31m${s}\x1b[0m`,
  green: s => `\x1b[32m${s}\x1b[0m`,
  yellow: s => `\x1b[33m${s}\x1b[0m`,
};

const step = t => console.log(`\n${c.bold(t)}\n${'-'.repeat(t.length)}`);
const ok = m => console.log(`  ${c.green('OK')}    ${m}`);
const warn = m => console.log(`  ${c.yellow('WARN')}  ${m}`);
const bad = m => { failed = true; console.log(`  ${c.red('FAIL')}  ${m}`); };
const note = m => console.log(`        ${c.dim(m)}`);

function run(cmd, args, opts = {}) {
  return spawnSync(cmd, args, { encoding: 'utf8', cwd: ROOT, ...opts });
}
function runLoud(cmd, args) {
  return spawnSync(cmd, args, { stdio: 'inherit', cwd: ROOT });
}

// ── 1. Interpreter ───────────────────────────────────────────────────────────
// Finding a USABLE python, not just any python. This is the check that would
// have saved the most time: the failure it prevents disguises itself as a
// missing package.

function parseVersion(out) {
  const m = /Python (\d+)\.(\d+)\.(\d+)/.exec(out || '');
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}
const cmp = (a, b) => a[0] - b[0] || a[1] - b[1];

function findPython() {
  const candidates = WIN
    ? [['py', ['-3.11']], ['py', ['-3.12']], ['python', []], ['python3', []]]
    : [['python3.11', []], ['python3.12', []], ['python3', []], ['python', []]];

  const seen = [];
  for (const [cmd, pre] of candidates) {
    const r = run(cmd, [...pre, '--version']);
    if (r.error || r.status !== 0) continue;
    const v = parseVersion((r.stdout || '') + (r.stderr || ''));
    if (!v) continue;
    seen.push(`${cmd} ${pre.join(' ')}`.trim() + ` -> ${v.join('.')}`);
    if (cmp(v, PY_MIN) >= 0 && cmp(v, PY_MAX) <= 0) return { cmd, pre, v };
  }
  return { cmd: null, seen };
}

step('1. Python interpreter');
const py = findPython();
if (!py.cmd) {
  bad(`No Python between ${PY_MIN.join('.')} and ${PY_MAX.join('.')} found.`);
  if (py.seen?.length) {
    note('Interpreters detected:');
    py.seen.forEach(s => note('  ' + s));
  }
  note('');
  note('Install Python 3.11 and re-run `npm run setup`:');
  note(WIN ? '  winget install Python.Python.3.11'
           : '  sudo apt install python3.11 python3.11-venv');
  note('On Windows, tick "Add python.exe to PATH" in the installer.');
  note('');
  note('Why this is strict: on an older Python, pip hides every wheel it');
  note('cannot use and then reports "no matching distribution" — which looks');
  note('like a broken requirements file rather than a wrong interpreter.');
  process.exit(1);
}
ok(`Python ${py.v.join('.')} (${py.cmd} ${py.pre.join(' ')}`.trim() + ')');

// ── 2. Virtual environment ───────────────────────────────────────────────────
step('2. Virtual environment');

// On a Raspberry Pi, picamera2 and libcamera are installed by apt into the
// SYSTEM python and cannot be pip-installed. A plain venv is sealed off from
// them, so the CSI camera silently reports itself as unavailable.
const onPi = process.platform === 'linux' && /arm|aarch/.test(process.arch);

if (!existsSync(VENV_PY)) {
  const args = [...py.pre, '-m', 'venv'];
  if (onPi) args.push('--system-site-packages');
  args.push('venv');
  console.log(`  creating venv${onPi ? ' (--system-site-packages, for picamera2)' : ''}...`);
  const r = runLoud(py.cmd, args);
  if (r.status !== 0 || !existsSync(VENV_PY)) {
    bad('Could not create the virtual environment.');
    process.exit(1);
  }
  ok('venv created');
} else {
  const r = run(VENV_PY, ['--version']);
  const v = parseVersion((r.stdout || '') + (r.stderr || ''));
  if (!v || cmp(v, PY_MIN) < 0 || cmp(v, PY_MAX) > 0) {
    bad(`Existing venv runs Python ${v ? v.join('.') : 'unknown'} — out of range.`);
    note('It was built by the wrong interpreter. Delete it and re-run:');
    note(WIN ? '  rmdir /s /q venv' : '  rm -rf venv');
    process.exit(1);
  }
  ok(`venv present, Python ${v.join('.')}`);
}

// ── 3. Dependencies ──────────────────────────────────────────────────────────
step('3. Python dependencies');
run(VENV_PY, ['-m', 'pip', 'install', '--upgrade', 'pip', '--quiet']);
console.log('  installing requirements.txt (this takes a few minutes)...');
if (runLoud(VENV_PY, ['-m', 'pip', 'install', '-r', 'requirements.txt']).status !== 0) {
  bad('pip install failed — see the output above.');
  process.exit(1);
}
ok('requirements installed');

// ── 4. Untangle OpenCV ───────────────────────────────────────────────────────
// Every OpenCV distribution writes the SAME cv2/ directory, so installing two
// of them leaves whichever was written last in charge. pip list then shows a
// package that is not the one Python imports, and uninstalling one can leave
// the other's binaries behind. That mismatch is invisible until import fails.
step('4. OpenCV consistency');

const pinned = (readFileSync(join(ROOT, 'requirements.txt'), 'utf8')
  .split('\n').find(l => /^opencv[-\w]*==/.test(l.trim())) || '').trim();

const freeze = run(VENV_PY, ['-m', 'pip', 'list', '--format=freeze']).stdout || '';
const installed = freeze.split('\n')
  .map(l => l.trim()).filter(l => /^opencv/i.test(l));

if (installed.length > 1) {
  warn(`${installed.length} OpenCV packages installed — they overwrite each other:`);
  installed.forEach(p => note('  ' + p));
  console.log('  removing all, then installing only the pinned one...');
  runLoud(VENV_PY, ['-m', 'pip', 'uninstall', '-y',
    'opencv-python', 'opencv-python-headless',
    'opencv-contrib-python', 'opencv-contrib-python-headless']);
  // pip leaves files it did not register. Whatever survives here would still
  // shadow the fresh install, so the directory goes too.
  const cv2dir = join(VENV, WIN ? 'Lib' : `lib/python${py.v[0]}.${py.v[1]}`,
                      'site-packages', 'cv2');
  try { rmSync(cv2dir, { recursive: true, force: true }); } catch { /* absent */ }
  if (pinned) runLoud(VENV_PY, ['-m', 'pip', 'install', pinned]);
  ok('OpenCV reinstalled cleanly');
} else if (installed.length === 1) {
  ok(installed[0]);
} else {
  bad('No OpenCV installed.');
}

// ── 5. Prove it actually works ───────────────────────────────────────────────
// The important part. Imports succeeding proved nothing in any of the three
// failures this script exists to catch.
step('5. Verification (building real objects, not just importing)');

const PROBE = `
import sys, json
r = {}
try:
    import numpy; r['numpy'] = numpy.__version__
except Exception as e: r['numpy_error'] = str(e)
try:
    import cv2; r['cv2'] = cv2.__version__
    import numpy as np
    r['cv2_decode'] = bool(cv2.cvtColor(np.zeros((4,4,3), np.uint8), cv2.COLOR_BGR2RGB).shape)
    r['cv2_gui'] = hasattr(cv2, 'imshow')
except Exception as e: r['cv2_error'] = f'{type(e).__name__}: {e}'
try:
    import google.protobuf as p; r['protobuf'] = p.__version__
except Exception as e: r['protobuf_error'] = str(e)
try:
    import mediapipe as mp
    r['mediapipe'] = mp.__version__
    h = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1); h.close()
    r['mediapipe_graph'] = True
except Exception as e: r['mediapipe_error'] = f'{type(e).__name__}: {e}'
try:
    import onnxruntime; r['onnxruntime'] = onnxruntime.__version__
except Exception as e: r['onnxruntime_error'] = str(e)
print('<<<' + json.dumps(r) + '>>>')
`;

const probe = run(VENV_PY, ['-c', PROBE]);
const raw = ((probe.stdout || '') + (probe.stderr || ''));
const m = /<<<(.*)>>>/s.exec(raw);
const R = m ? JSON.parse(m[1]) : {};

if (!m) {
  bad('The verification probe did not complete.');
  note(raw.trim().split('\n').slice(-6).join('\n        '));
} else {
  R.numpy ? ok(`numpy ${R.numpy}`) : bad(`numpy: ${R.numpy_error}`);

  if (R.cv2_error) {
    bad(`cv2: ${R.cv2_error}`);
    if (/_ARRAY_API|multiarray/.test(R.cv2_error)) {
      note('This OpenCV build predates NumPy 2. Fix with:');
      note(`  ${pinned || 'opencv-contrib-python'}`);
    }
  } else {
    ok(`cv2 ${R.cv2} (decode verified)`);
    R.cv2_gui ? ok('cv2 has a GUI — collector/test windows will open')
              : bad('cv2 is a HEADLESS build: cv2.imshow does not exist, so ' +
                    '`npm run collect` and `npm run test` cannot show a window.');
  }

  if (R.mediapipe_error) {
    bad(`mediapipe: ${R.mediapipe_error}`);
    if (/label|GetPrototype|Descriptor/.test(R.mediapipe_error)) {
      note(`protobuf ${R.protobuf || '?'} removed APIs MediaPipe still calls.`);
      note('Fix with:  npm run fix');
    }
  } else {
    ok(`mediapipe ${R.mediapipe} (built a real Hands graph)`);
  }

  R.onnxruntime ? ok(`onnxruntime ${R.onnxruntime}`)
                : bad(`onnxruntime: ${R.onnxruntime_error}`);
}

// ── 6. Node packages ─────────────────────────────────────────────────────────
step('6. Node packages');
if (!existsSync(join(ROOT, 'node_modules'))) {
  runLoud(WIN ? 'npm.cmd' : 'npm', ['install']);
}
existsSync(join(ROOT, 'node_modules')) ? ok('node_modules present')
                                       : warn('npm install did not complete');

// ── 7. .env — create, NEVER overwrite ────────────────────────────────────────
step('7. Configuration');
const ENV = join(ROOT, '.env');
if (existsSync(ENV)) {
  // Deliberate: a `copy .env.example .env` prompt was answered "yes" during
  // the real setup and silently destroyed a working key file. A setup script
  // must never be able to do that.
  ok('.env exists — left untouched');
  const body = readFileSync(ENV, 'utf8');
  const empty = ['OPENROUTER_API_KEY', 'GROQ_API_KEY']
    .filter(k => !new RegExp(`^${k}=.+`, 'm').test(body));
  if (empty.length) {
    warn(`No value set for: ${empty.join(', ')}`);
    note('collect / train / test work without these.');
    note('The server (npm run dev:browser) will refuse to start.');
  }
} else if (existsSync(join(ROOT, '.env.example'))) {
  copyFileSync(join(ROOT, '.env.example'), ENV);
  ok('.env created from .env.example');
  warn('Add your OPENROUTER_API_KEY and GROQ_API_KEY before running the server.');
}

// ── Summary ──────────────────────────────────────────────────────────────────
console.log('\n' + '='.repeat(64));
if (failed) {
  console.log(c.red(c.bold('  SETUP INCOMPLETE — see the FAIL lines above')));
  console.log('='.repeat(64));
  process.exit(1);
}
console.log(c.green(c.bold('  READY')));
console.log('='.repeat(64));
console.log(`
  npm run collect   record signs
  npm run train     train the model
  npm run test      try it live on camera

  Full command reference: COMMANDS.md
`);
