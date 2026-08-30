/**
 * scripts/setup.mjs — scan first, then install only what is missing
 * =================================================================
 *     npm run setup           scan, show the plan, then fix
 *     npm run setup -- --check    scan only, change nothing
 *
 * Two phases, deliberately separated:
 *
 *   PHASE 1  reads the machine and prints exactly what is missing, wrong or
 *            already fine. It writes nothing. You see the whole picture
 *            before a single byte is downloaded.
 *
 *   PHASE 2  fixes only what phase 1 flagged. An existing venv is reused, a
 *            satisfied requirements.txt is skipped, node_modules is left
 *            alone if present, and .env is NEVER overwritten.
 *
 * Why it verifies by BUILDING things
 * ----------------------------------
 * Setting this project up on a second machine cost an afternoon and three
 * failures, none of which named its own cause:
 *
 *   Python 3.8 on PATH   -> pip said "no version satisfies absl-py==2.4.0",
 *                           which reads as a missing package. pip only lists
 *                           wheels the running Python can use, and never
 *                           mentions that it is filtering.
 *   opencv-headless 4.9  -> predates NumPy 2, so `import cv2` died with
 *                           "_ARRAY_API not found"; and being headless it had
 *                           no cv2.imshow, which the whole collector needs.
 *   protobuf 7           -> removed two APIs MediaPipe calls. `import
 *                           mediapipe` still SUCCEEDED and only failed later,
 *                           when a camera actually started.
 *
 * Every one of those passed a naive check. So the final step here builds a
 * real MediaPipe graph and decodes a real frame instead of trusting imports.
 */

import { spawnSync } from 'node:child_process';
import { existsSync, copyFileSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WIN = process.platform === 'win32';
const VENV = join(ROOT, 'venv');
const VENV_PY = WIN ? join(VENV, 'Scripts', 'python.exe')
                    : join(VENV, 'bin', 'python');
const CHECK_ONLY = process.argv.includes('--check');
// Opt-ins for version drift. Neither happens by default: see the drift block.
const SYNC_VERSIONS  = process.argv.includes('--sync-versions');   // machine -> pins
const ADOPT_VERSIONS = process.argv.includes('--adopt-versions');  // pins -> machine

// mediapipe 0.10.14 ships wheels for 3.8-3.12; numpy 2.4.x needs >= 3.11.
// That intersection is the only range where a clean install is possible.
const PY_MIN = [3, 11], PY_MAX = [3, 12];

const C = {
  b: s => `\x1b[1m${s}\x1b[0m`,   d: s => `\x1b[2m${s}\x1b[0m`,
  r: s => `\x1b[31m${s}\x1b[0m`,  g: s => `\x1b[32m${s}\x1b[0m`,
  y: s => `\x1b[33m${s}\x1b[0m`,  c: s => `\x1b[36m${s}\x1b[0m`,
};
const run  = (cmd, args, o = {}) => spawnSync(cmd, args, { encoding: 'utf8', cwd: ROOT, ...o });
const loud = (cmd, args, o = {}) => spawnSync(cmd, args, { stdio: 'inherit', cwd: ROOT, ...o });

// npm on Windows is npm.cmd, and since the CVE-2024-27980 fix Node REFUSES to
// spawn .cmd/.bat files unless shell:true is set. Without it this reported
// "Node / npm not found" on a machine that had just launched this very script
// through npm — a false blocker, which is the worst kind for a setup tool.
const npm = (args, o = {}) =>
  spawnSync(WIN ? 'npm.cmd' : 'npm', args,
            { encoding: 'utf8', cwd: ROOT, shell: WIN, ...o });
const norm = s => s.toLowerCase().replace(/[_.]/g, '-');
const parseVer = o => { const m = /Python (\d+)\.(\d+)\.(\d+)/.exec(o || ''); return m ? [+m[1], +m[2], +m[3]] : null; };
const cmp = (a, b) => a[0] - b[0] || a[1] - b[1];

// ═══════════════════════════════════════════════════════════════════════════
//  PHASE 1 — SCAN.  Reads only. Changes nothing.
// ═══════════════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(66));
console.log(C.b('  TARJUMAN SETUP') + C.d('   —  scanning this machine'));
console.log('='.repeat(66));

const S = {};   // scan results

// -- interpreter --------------------------------------------------------------
(() => {
  const cands = WIN
    ? [['py', ['-3.11']], ['py', ['-3.12']], ['python', []], ['python3', []]]
    : [['python3.11', []], ['python3.12', []], ['python3', []], ['python', []]];
  const seen = [];
  for (const [cmd, pre] of cands) {
    const r = run(cmd, [...pre, '--version']);
    if (r.error || r.status !== 0) continue;
    const v = parseVer((r.stdout || '') + (r.stderr || ''));
    if (!v) continue;
    seen.push(`${[cmd, ...pre].join(' ')} -> ${v.join('.')}`);
    if (cmp(v, PY_MIN) >= 0 && cmp(v, PY_MAX) <= 0) { S.py = { cmd, pre, v, seen }; return; }
  }
  S.py = { cmd: null, seen };
})();

// -- node ---------------------------------------------------------------------
S.node = (() => {
  const r = npm(['--version']);
  // If npm cannot be probed we are still RUNNING under node, so node itself is
  // certainly present. Treat only npm as unknown rather than declaring the
  // whole toolchain missing.
  return { ok: r.status === 0, version: (r.stdout || '').trim() || 'unknown',
           nodeVersion: process.version };
})();

// -- venv ---------------------------------------------------------------------
S.venv = (() => {
  if (!existsSync(VENV_PY)) return { state: 'missing' };
  const r = run(VENV_PY, ['--version']);
  const v = parseVer((r.stdout || '') + (r.stderr || ''));
  if (!v) return { state: 'broken' };
  if (cmp(v, PY_MIN) < 0 || cmp(v, PY_MAX) > 0) return { state: 'wrong-python', v };
  return { state: 'ok', v };
})();

// -- python packages: compare requirements.txt against what is installed ------
// Done by reading pip's own list rather than by attempting an install, so the
// scan stays offline and instant.
S.pkgs = { missing: [], drift: [], opencv: [], satisfied: 0, checked: false };
if (S.venv.state === 'ok') {
  const freeze = run(VENV_PY, ['-m', 'pip', 'list', '--format=freeze']).stdout || '';
  const have = new Map();
  for (const line of freeze.split('\n')) {
    const [n, v] = line.trim().split('==');
    if (n) have.set(norm(n), v);
  }
  S.pkgs.opencv = [...have.keys()].filter(k => k.startsWith('opencv'));
  const req = readFileSync(join(ROOT, 'requirements.txt'), 'utf8').split('\n');
  for (const raw of req) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const m = /^([A-Za-z0-9._-]+)==([^\s;]+)/.exec(line);
    if (!m) continue;
    const [, name, want] = m;
    const got = have.get(norm(name));
    if (!got) S.pkgs.missing.push(`${name}==${want}`);
    else if (got !== want) S.pkgs.drift.push({ name, got, want });
    else S.pkgs.satisfied++;
  }
  S.pkgs.checked = true;
}
S.pinnedCv = (readFileSync(join(ROOT, 'requirements.txt'), 'utf8')
  .split('\n').map(l => l.trim()).find(l => /^opencv[\w-]*==/.test(l))) || '';

// -- node_modules -------------------------------------------------------------
S.nodeModules = existsSync(join(ROOT, 'node_modules'));

// -- .env ---------------------------------------------------------------------
S.env = (() => {
  if (!existsSync(join(ROOT, '.env'))) return { state: 'missing' };
  const body = readFileSync(join(ROOT, '.env'), 'utf8');
  const empty = ['OPENROUTER_API_KEY', 'GROQ_API_KEY']
    .filter(k => !new RegExp(`^${k}=.+`, 'm').test(body));
  return { state: 'present', empty };
})();

// -- camera -------------------------------------------------------------------
// A dependency check that passes and then leaves you with no camera is only
// half an answer: the first thing anyone runs is `npm run collect`, which
// needs a working lens before it needs anything else. Probed here so the gap
// shows up during setup rather than in the middle of a recording session.
S.camera = { state: 'unknown', detail: '' };
if (S.venv.state === 'ok' && !S.pkgs.missing.length) {
  const CAM = `
import json, sys
out = {"backends": [], "picamera2": False}
try:
    from picamera2 import Picamera2
    out["picamera2"] = len(Picamera2.global_camera_info()) > 0
except Exception:
    pass
try:
    import cv2, os
    try: cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception: pass
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ok, f = cap.read()
                if ok and f is not None:
                    out["backends"].append([i, int(f.shape[1]), int(f.shape[0])])
        finally:
            cap.release()
except Exception as e:
    out["error"] = str(e)
print('<<<' + json.dumps(out) + '>>>')
`;
  const r = run(VENV_PY, ['-c', CAM], { timeout: 45000 });
  const cm = /<<<(.*)>>>/s.exec((r.stdout || '') + (r.stderr || ''));
  if (cm) {
    const cams = JSON.parse(cm[1]);
    const n = (cams.backends || []).length;
    if (cams.picamera2) { S.camera = { state: 'ok', detail: 'Raspberry Pi CSI camera detected' }; }
    else if (n) {
      S.camera = { state: 'ok',
        detail: `${n} camera(s): ` + cams.backends.map(([i, w, h]) => `index ${i} ${w}x${h}`).join(', ') };
    } else S.camera = { state: 'none', detail: 'no camera responded' };
  }
}

// -- Rust (desktop app only) --------------------------------------------------
S.rust = run('cargo', ['--version']).status === 0;

// -- Raspberry Pi system packages ---------------------------------------------
// picamera2 and libcamera come from apt, never from pip. `npm run setup`
// cannot install them (they need sudo), so it names them precisely instead.
S.pi = null;
if (process.platform === 'linux' && /arm|aarch/.test(process.arch)) {
  const hasPicam = S.venv.state === 'ok' &&
    run(VENV_PY, ['-c', 'import picamera2']).status === 0;
  S.pi = { picamera2: hasPicam, rpicam: run('which', ['rpicam-hello']).status === 0,
           display: !!process.env.DISPLAY };
}

// ── report ───────────────────────────────────────────────────────────────────
const row = (label, status, detail = '') => {
  const tag = { ok: C.g('  OK  '), fix: C.y(' FIX  '), block: C.r('BLOCK '), skip: C.d(' --   ') }[status];
  console.log(`  ${tag} ${label.padEnd(22)} ${detail}`);
};

console.log();
const todo = [];
const blockers = [];

if (S.py.cmd) row('Python', 'ok', `${S.py.v.join('.')}  (${[S.py.cmd, ...S.py.pre].join(' ')})`);
else { row('Python', 'block', `need ${PY_MIN.join('.')}–${PY_MAX.join('.')}`); blockers.push('python'); }

if (S.node.ok) row('Node / npm', 'ok', `node ${S.node.nodeVersion}, npm ${S.node.version}`);
else { row('Node / npm', 'block', 'not found'); blockers.push('node'); }

if (S.venv.state === 'ok') row('venv', 'ok', `Python ${S.venv.v.join('.')}`);
else if (S.venv.state === 'missing') { row('venv', 'fix', 'not created yet'); todo.push('venv'); }
else if (S.venv.state === 'wrong-python') { row('venv', 'fix', `built with Python ${S.venv.v.join('.')} — will rebuild`); todo.push('venv-rebuild'); }
else { row('venv', 'fix', 'unreadable — will rebuild'); todo.push('venv-rebuild'); }

if (!S.pkgs.checked) row('Python packages', 'skip', 'needs a venv first');
else if (S.pkgs.missing.length) {
  row('Python packages', 'fix', `${S.pkgs.satisfied} ok, ${S.pkgs.missing.length} missing`);
  S.pkgs.missing.slice(0, 6).forEach(p => console.log(C.d(`           missing: ${p}`)));
  if (S.pkgs.missing.length > 6) console.log(C.d(`           ...and ${S.pkgs.missing.length - 6} more`));
  todo.push('pip');
} else {
  row('Python packages', 'ok', `${S.pkgs.satisfied} pinned packages satisfied`);
}

// Version DRIFT is reported but never "fixed" on its own.
//
// Downgrading packages that are installed and working is how you break a
// machine that was fine a minute ago - and on this project that machine may
// be the one that trained the model. Newer-than-pinned is usually harmless;
// silently rolling scikit-learn back three minor versions is not. So drift is
// surfaced, explained, and left to an explicit decision.
if (S.pkgs.drift.length) {
  const newer = S.pkgs.drift.filter(d => d.got > d.want).length;
  row('Version drift', 'skip', `${S.pkgs.drift.length} package(s) differ from requirements.txt`);
  S.pkgs.drift.slice(0, 8).forEach(d =>
    console.log(C.d(`           ${d.name}: installed ${d.got}, pinned ${d.want}`)));
  console.log(C.d(`           Not changed automatically — downgrading working packages`));
  console.log(C.d(`           is riskier than the drift itself${newer ? ' (most are NEWER than pinned)' : ''}.`));
  console.log(C.d(`           npm run setup -- --sync-versions   to force the pins`));
  console.log(C.d(`           npm run setup -- --adopt-versions  to update requirements.txt to match`));
}

if (S.pkgs.checked) {
  if (S.pkgs.opencv.length > 1) {
    row('OpenCV', 'fix', `${S.pkgs.opencv.length} packages installed — they overwrite each other`);
    S.pkgs.opencv.forEach(p => console.log(C.d(`           ${p}`)));
    todo.push('opencv');
  } else if (S.pkgs.opencv.length === 1) row('OpenCV', 'ok', S.pkgs.opencv[0]);
  else { row('OpenCV', 'fix', 'not installed'); }
}

if (S.nodeModules) row('node_modules', 'ok', 'present — will not reinstall');
else { row('node_modules', 'fix', 'missing'); todo.push('npm'); }

if (S.camera.state === 'ok') row('Camera', 'ok', S.camera.detail);
else if (S.camera.state === 'none') {
  row('Camera', 'skip', C.y('no camera responded'));
  console.log(C.d('           collect / test need one. Check it is plugged in and that'));
  console.log(C.d('           Zoom / Teams / OBS are not holding it open.'));
} else if (S.venv.state === 'ok' && !S.pkgs.missing.length) row('Camera', 'skip', 'could not probe');

if (S.pi) {
  S.pi.picamera2 ? row('picamera2 (Pi)', 'ok', 'importable from the venv')
                 : row('picamera2 (Pi)', 'skip', C.y('missing — CSI camera option will not appear'));
  if (!S.pi.picamera2) {
    console.log(C.d('           sudo apt install -y python3-picamera2 python3-libcamera'));
    console.log(C.d('           apt, NOT pip: it is built against the system libcamera.'));
    console.log(C.d('           The venv must also exist with --system-site-packages.'));
  }
  if (!S.pi.rpicam) console.log(C.d('           sudo apt install -y rpicam-apps   (for rpicam-hello --list-cameras)'));
  if (!S.pi.display) {
    row('Display (Pi)', 'skip', C.y('$DISPLAY not set — cv2.imshow cannot open a window'));
    console.log(C.d('           Pi OS Lite has no GUI. Minimal X:'));
    console.log(C.d('           sudo apt install -y --no-install-recommends xserver-xorg xinit openbox'));
    console.log(C.d('           then:  startx &'));
  }
}

if (S.env.state === 'present') {
  if (S.env.empty.length) row('.env', 'ok', C.y(`present (no value for ${S.env.empty.join(', ')})`));
  else row('.env', 'ok', 'present, keys set');
} else { row('.env', 'fix', 'will create from .env.example'); todo.push('env'); }

// ── blockers stop everything ─────────────────────────────────────────────────
if (blockers.length) {
  console.log('\n' + '='.repeat(66));
  console.log(C.r(C.b('  CANNOT CONTINUE — install these first')));
  console.log('='.repeat(66));
  if (blockers.includes('python')) {
    console.log(`\n  ${C.b('Python 3.11')}`);
    if (S.py.seen?.length) { console.log(C.d('    detected instead:')); S.py.seen.forEach(s => console.log(C.d('      ' + s))); }
    console.log(C.c(WIN ? '    winget install Python.Python.3.11'
                        : '    sudo apt install python3.11 python3.11-venv'));
    if (WIN) console.log(C.d('    Tick "Add python.exe to PATH" in the installer.'));
    console.log(C.d('    Needed because an older Python makes pip hide every wheel it'));
    console.log(C.d('    cannot use, then report a "missing package" instead.'));
  }
  if (blockers.includes('node')) {
    console.log(`\n  ${C.b('Node.js LTS')}`);
    console.log(C.c(WIN ? '    winget install OpenJS.NodeJS.LTS'
                        : '    sudo apt install nodejs npm'));
  }
  console.log(C.d('\n  Then open a NEW terminal and run  npm run setup  again.'));
  console.log(C.d('  (PATH changes are not visible to an already-open window.)\n'));
  process.exit(1);
}

// ── nothing to do? ───────────────────────────────────────────────────────────
console.log();
if (!todo.length) {
  console.log(C.g('  Everything is already in place — nothing to install.'));
} else if (CHECK_ONLY) {
  console.log(C.b(`  ${todo.length} thing(s) need attention.`) + C.d('  Run  npm run setup  to fix.'));
  process.exit(0);
} else {
  console.log(C.b('  PLAN') + C.d('  (only what is missing above)'));
  const plan = {
    'venv': 'create the virtual environment',
    'venv-rebuild': 'rebuild the virtual environment with the correct Python',
    'pip': 'install the missing Python packages',
    'opencv': 'remove the duplicate OpenCV packages and reinstall the pinned one',
    'npm': 'npm install',
    'env': 'create .env from .env.example (existing files are never touched)',
  };
  todo.forEach(t => console.log(`    • ${plan[t]}`));
}

// ═══════════════════════════════════════════════════════════════════════════
//  PHASE 2 — APPLY
// ═══════════════════════════════════════════════════════════════════════════

const onPi = process.platform === 'linux' && /arm|aarch/.test(process.arch);
let failed = false;
const fail = m => { failed = true; console.log(`  ${C.r('FAIL')}  ${m}`); };
const done = m => console.log(`  ${C.g('OK')}    ${m}`);
const head = t => console.log(`\n${C.b(t)}\n${'-'.repeat(t.length)}`);

if (todo.includes('venv') || todo.includes('venv-rebuild')) {
  head('Creating the virtual environment');
  if (todo.includes('venv-rebuild')) rmSync(VENV, { recursive: true, force: true });
  // On a Pi, picamera2 and libcamera come from apt into the SYSTEM python and
  // cannot be pip-installed. A sealed venv makes the CSI camera look absent.
  const args = [...S.py.pre, '-m', 'venv', ...(onPi ? ['--system-site-packages'] : []), 'venv'];
  if (onPi) console.log(C.d('  using --system-site-packages so picamera2 stays visible'));
  loud(S.py.cmd, args);
  existsSync(VENV_PY) ? done('venv ready') : fail('could not create venv');
  if (failed) process.exit(1);
  todo.push('pip');            // a new venv is empty by definition
}

if (ADOPT_VERSIONS && S.pkgs.drift.length) {
  head('Updating requirements.txt to match this machine');
  console.log(C.d('  Recording what actually works here, so other machines match it.'));
  console.log(C.d('  Use this on the machine that TRAINED the model — it is the one'));
  console.log(C.d('  whose environment the model and dataset were produced under.\n'));
  const path = join(ROOT, 'requirements.txt');
  let text = readFileSync(path, 'utf8');
  for (const d of S.pkgs.drift) {
    const re = new RegExp(`^${d.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}==\\S+`, 'mi');
    if (re.test(text)) {
      text = text.replace(re, `${d.name}==${d.got}`);
      console.log(`  ${C.g('OK')}    ${d.name}: ${d.want} -> ${d.got}`);
    }
  }
  writeFileSync(path, text);
  console.log(C.d('\n  Commit this change so the other machines pick it up.'));
}

if (SYNC_VERSIONS && S.pkgs.drift.length) todo.push('pip');

if (todo.includes('pip')) {
  head('Installing Python packages');
  if (SYNC_VERSIONS) console.log(C.y('  --sync-versions: pinned versions will be forced, including downgrades.\n'));
  run(VENV_PY, ['-m', 'pip', 'install', '--upgrade', 'pip', '--quiet']);
  console.log(C.d('  this is the slow part — a few minutes on a fresh venv\n'));
  if (loud(VENV_PY, ['-m', 'pip', 'install', '-r', 'requirements.txt']).status !== 0) {
    fail('pip install failed — see above'); process.exit(1);
  }
  done('packages installed');
}

if (todo.includes('opencv')) {
  head('Untangling OpenCV');
  console.log(C.d('  Every OpenCV distribution writes the same cv2/ directory, so'));
  console.log(C.d('  the last one installed wins and pip list stops matching what'));
  console.log(C.d('  Python imports. Removing all, then installing one.\n'));
  loud(VENV_PY, ['-m', 'pip', 'uninstall', '-y', 'opencv-python',
    'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless']);
  // pip leaves behind files it never registered; those would shadow the new
  // install, so the directory itself goes.
  const site = join(VENV, WIN ? 'Lib' : `lib/python${S.py.v[0]}.${S.py.v[1]}`, 'site-packages', 'cv2');
  try { rmSync(site, { recursive: true, force: true }); } catch { /* already gone */ }
  if (S.pinnedCv) loud(VENV_PY, ['-m', 'pip', 'install', S.pinnedCv]);
  done('OpenCV reinstalled cleanly');
}

if (todo.includes('npm')) {
  head('Installing Node packages');
  npm(['install'], { stdio: 'inherit', encoding: undefined });
  existsSync(join(ROOT, 'node_modules')) ? done('node_modules ready') : fail('npm install did not complete');
}

if (todo.includes('env')) {
  head('Configuration');
  // Never overwrite: a `copy .env.example .env` prompt answered "yes" during a
  // real setup destroyed a working key file. A setup script must not be able
  // to do that, so this branch only runs when .env does not exist at all.
  if (existsSync(join(ROOT, '.env.example'))) {
    copyFileSync(join(ROOT, '.env.example'), join(ROOT, '.env'));
    done('.env created from .env.example');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  VERIFY — build real objects, because imports proved nothing
// ═══════════════════════════════════════════════════════════════════════════
head('Verifying (building real objects, not just importing)');

const PROBE = `
import json
r = {}
try:
    import numpy; r['numpy'] = numpy.__version__
except Exception as e: r['numpy_error'] = str(e)
try:
    import cv2, numpy as np; r['cv2'] = cv2.__version__
    cv2.cvtColor(np.zeros((4,4,3), np.uint8), cv2.COLOR_BGR2RGB)
    r['cv2_decode'] = True
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
const raw = (probe.stdout || '') + (probe.stderr || '');
const mm = /<<<(.*)>>>/s.exec(raw);
const R = mm ? JSON.parse(mm[1]) : null;

if (!R) {
  fail('the verification probe did not complete');
  console.log(C.d('        ' + raw.trim().split('\n').slice(-6).join('\n        ')));
} else {
  R.numpy ? done(`numpy ${R.numpy}`) : fail(`numpy: ${R.numpy_error}`);

  if (R.cv2_error) {
    fail(`cv2: ${R.cv2_error}`);
    if (/_ARRAY_API|multiarray/.test(R.cv2_error))
      console.log(C.d(`        This OpenCV predates NumPy 2. Fix:  pip install ${S.pinnedCv}`));
  } else {
    done(`cv2 ${R.cv2} (decoded a frame)`);
    R.cv2_gui ? done('cv2 has a GUI — collector and test windows will open')
              : fail('cv2 is a HEADLESS build: no cv2.imshow, so npm run collect / test cannot show a window');
  }

  if (R.mediapipe_error) {
    fail(`mediapipe: ${R.mediapipe_error}`);
    if (/label|GetPrototype|Descriptor/.test(R.mediapipe_error)) {
      console.log(C.d(`        protobuf ${R.protobuf || '?'} removed APIs MediaPipe calls.`));
      console.log(C.d('        Fix:  npm run fix'));
    }
  } else done(`mediapipe ${R.mediapipe} (built a real Hands graph)`);

  R.onnxruntime ? done(`onnxruntime ${R.onnxruntime}`) : fail(`onnxruntime: ${R.onnxruntime_error}`);
}

// ── summary ──────────────────────────────────────────────────────────────────
console.log('\n' + '='.repeat(66));
if (failed) {
  console.log(C.r(C.b('  SETUP INCOMPLETE — see the FAIL lines above')));
  console.log('='.repeat(66) + '\n');
  process.exit(1);
}
console.log(C.g(C.b('  READY')));
console.log('='.repeat(66));
if (S.env.state === 'missing' || S.env.empty?.length)
  console.log(C.y('\n  Add OPENROUTER_API_KEY and GROQ_API_KEY to .env before running the server.') +
              C.d('\n  (collect / train / test work without them)'));
if (S.camera.state === 'none')
  console.log(C.y('\n  No camera was detected — connect one before npm run collect.'));
if (!S.rust)
  console.log(C.d('\n  Rust is not installed, so the DESKTOP app (npm run dev:all / build)') +
              C.d('\n  will not compile. Everything else works. Install from https://rustup.rs') +
              C.d('\n  if you want it; npm run dev:browser needs no Rust at all.'));
console.log(`
  npm run collect   record signs
  npm run train     train the model
  npm run test      try it live on camera

  Reference: COMMANDS.md
`);
