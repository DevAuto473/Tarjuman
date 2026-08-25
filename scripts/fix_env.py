"""
fix_env.py - repair the dependency versions Tarjuman actually needs
==================================================================
    npm run fix

Why this exists
---------------
`npm run test` died with:

    AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    AttributeError: 'FieldDescriptor' object has no attribute 'label'

Neither line mentions the real problem. protobuf 5 removed both of those
APIs, and mediapipe 0.10.14 calls them directly when it builds a solution
graph. requirements.txt pins protobuf==4.25.9 for exactly this reason, but a
`pip install --force-reinstall` on some other package had quietly pulled
protobuf 7 in behind it.

That failure mode is nasty in a specific way: the mismatch does NOT show up
at import time, so `npm run doctor` reported `mediapipe ok` and looked
healthy. It only fires when Hands()/Pose() is CONSTRUCTED. So this script
does not trust a version number or a bare import - it constructs a real
MediaPipe solution in a subprocess and checks that it survives.

Nothing is installed unless a version is genuinely wrong, and every install
is pinned, so running this repeatedly is safe.
"""

import importlib.metadata as md
import subprocess
import sys

PY = sys.executable

# (package, required spec, why it is pinned)
# Only the packages whose versions are load-bearing. Everything else is
# whatever requirements.txt resolved to and does not need policing.
PINS = [
    ("protobuf", "4.25.9",
     "protobuf 5+ removed FieldDescriptor.label and "
     "MessageFactory.GetPrototype, which mediapipe 0.10.14 calls directly."),
    ("mediapipe", "0.10.14",
     "the hand/pose landmark models the dataset was recorded with."),
]

# The check that actually matters: does a solution graph BUILD? An import
# succeeding proves nothing here - that is what made the original failure so
# confusing to diagnose.
SMOKE = (
    "import mediapipe as mp\n"
    "h = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)\n"
    "h.close()\n"
    "p = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0)\n"
    "p.close()\n"
    "print('SMOKE OK')\n"
)


def version_of(pkg: str):
    try:
        return md.version(pkg)
    except Exception:
        return None


def smoke_test() -> tuple:
    """Build a real MediaPipe graph in a fresh interpreter. (ok, output)."""
    try:
        p = subprocess.run([PY, "-c", SMOKE], capture_output=True, timeout=180)
        out = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
        return p.returncode == 0 and "SMOKE OK" in out, out.strip()
    except subprocess.TimeoutExpired:
        return False, "timed out building a MediaPipe solution"


def pip_install(spec: str) -> bool:
    print(f"   installing {spec} ...")
    p = subprocess.run([PY, "-m", "pip", "install", "--no-cache-dir", spec],
                       capture_output=True)
    if p.returncode != 0:
        print((p.stderr or b"").decode("utf-8", "replace").strip()[-1500:])
        return False
    return True


def main() -> int:
    print("=" * 68)
    print("   TARJUMAN ENVIRONMENT REPAIR")
    print("=" * 68)
    print(f"   python : {sys.version.split()[0]}")
    print(f"   venv   : {PY}")

    print("\n" + "-" * 68)
    print("   VERSION CHECK")
    print("-" * 68)

    wrong = []
    for pkg, want, why in PINS:
        have = version_of(pkg)
        if have == want:
            print(f"   {pkg:<12s} {have:<12s} ok")
        else:
            print(f"   {pkg:<12s} {str(have):<12s} WRONG (need {want})")
            print(f"      {why}")
            wrong.append((pkg, want))

    if wrong:
        print("\n" + "-" * 68)
        print("   REPAIRING")
        print("-" * 68)
        for pkg, want in wrong:
            if not pip_install(f"{pkg}=={want}"):
                print(f"\n   [FAIL] could not install {pkg}=={want}.")
                print("   Close any running Tarjuman process (the server or a")
                print("   camera window can hold a DLL open on Windows) and retry.")
                return 1
    else:
        print("\n   All pinned versions are correct.")

    print("\n" + "-" * 68)
    print("   SMOKE TEST - can MediaPipe actually BUILD a graph?")
    print("-" * 68)
    print("   (a successful import proves nothing here - the original failure")
    print("    imported fine and then died on Hands())")
    ok, out = smoke_test()

    if ok:
        print("\n   MediaPipe Hands and Pose both constructed cleanly.")
        print("\n" + "=" * 68)
        print("   READY - run:  npm run test")
        print("=" * 68)
        return 0

    print("\n   [FAIL] MediaPipe still cannot build a solution graph.")
    print("\n   output:")
    for line in out.splitlines()[-15:]:
        print("     " + line)
    print("\n   Next step: restore every pinned version at once with")
    print(f"     {PY} -m pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
