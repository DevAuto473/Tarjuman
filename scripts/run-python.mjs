/**
 * scripts/run-python.mjs — cross-platform Python launcher for Tarjuman
 * =====================================================================
 * Replaces the hardcoded `./venv/bin/python` that used to live in
 * package.json: a Unix-only path, which meant the one-command dev script
 * never worked on Windows — the very platform this project is developed on.
 *
 * Resolution order:
 *   1. venv\Scripts\python.exe   (Windows virtualenv)
 *   2. venv/bin/python           (Linux / macOS / Raspberry Pi virtualenv)
 *   3. .venv equivalents
 *   4. python3 / python on PATH  (last resort, with a warning)
 *
 * Usage:
 *   node scripts/run-python.mjs websocket_server.py
 *   node scripts/run-python.mjs train_model.py
 *   node scripts/run-python.mjs migrate_dataset.py --force
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// Scripts that need API keys present before they can do anything useful
const NEEDS_ENV = new Set(['websocket_server.py']);

const VENV_CANDIDATES = [
  join(projectRoot, 'venv', 'Scripts', 'python.exe'), // Windows
  join(projectRoot, 'venv', 'bin', 'python'),         // Unix
  join(projectRoot, '.venv', 'Scripts', 'python.exe'),
  join(projectRoot, '.venv', 'bin', 'python'),
];

function resolvePython() {
  const found = VENV_CANDIDATES.find(existsSync);
  if (found) return { cmd: found, fromVenv: true };

  console.warn(
    '⚠️  No virtualenv found at venv/ or .venv/ — falling back to the system\n' +
    '    Python on PATH. Dependencies may be missing. Create one with:\n' +
    '        python -m venv venv\n' +
    '        venv\\Scripts\\pip install -r requirements.txt   (Windows)\n' +
    '        venv/bin/pip install -r requirements.txt        (Linux/macOS)\n'
  );
  return {
    cmd: process.platform === 'win32' ? 'python' : 'python3',
    fromVenv: false,
  };
}

// ── Arguments ────────────────────────────────────────────────────────────────

const [scriptName, ...scriptArgs] = process.argv.slice(2);

if (!scriptName) {
  console.error('❌  Usage: node scripts/run-python.mjs <script.py> [args...]');
  process.exit(1);
}

const scriptPath = join(projectRoot, scriptName);
if (!existsSync(scriptPath)) {
  console.error(`❌  Script not found: ${scriptPath}`);
  process.exit(1);
}

// ── Pre-flight: secrets ──────────────────────────────────────────────────────

if (NEEDS_ENV.has(scriptName) && !existsSync(join(projectRoot, '.env'))) {
  console.error(
    '❌  .env not found. The server needs OPENROUTER_API_KEY and GROQ_API_KEY.\n' +
    '    Copy .env.example to .env and fill in your keys.'
  );
  process.exit(1);
}

// ── Launch ───────────────────────────────────────────────────────────────────

const { cmd, fromVenv } = resolvePython();
console.log(`[py] ${scriptName} → ${cmd}${fromVenv ? '' : '  (system Python)'}`);

const child = spawn(cmd, [scriptPath, ...scriptArgs], {
  cwd: projectRoot,   // model / CSV / label files resolve relative to the root
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHONUNBUFFERED: '1',       // stream logs immediately
    // Windows consoles default to a legacy code page (cp1252 / cp437 / cp1256)
    // that cannot encode arrows, box-drawing characters or Arabic. Python then
    // raises UnicodeEncodeError mid-print and the script dies part-way through
    // its own progress output — which reads as "it printed nothing and made no
    // files". 'replace' guarantees a substituted character instead of a crash.
    PYTHONIOENCODING: 'utf-8:replace',
  },
});

child.on('error', (err) => {
  console.error(`❌  Failed to start Python (${cmd}): ${err.message}`);
  process.exit(1);
});

// Forward Ctrl-C so the server's finally block releases the camera cleanly
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}

child.on('exit', (code) => {
  // Say so out loud. A silent non-zero exit is the hardest kind of failure to
  // report, because there is nothing for the user to copy back.
  if (code) console.error(`\n[FAIL] ${scriptName} exited with code ${code}`);
  process.exit(code ?? 0);
});
