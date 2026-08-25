"""
diagnose.py - find the import that is killing Python
=====================================================
    python diagnose.py
    npm run doctor

Why this exists
---------------
`train_model.py` died with exit code 3221225477 (0xC0000005, ACCESS_VIOLATION)
before printing a single line. That is not a Python exception - it is a crash
inside a native library, so no try/except can catch it and no traceback is
produced. The only evidence is that the process is gone.

Crashes like this are almost never caused by one library alone. They come from
two libraries loading incompatible copies of the same native dependency into one
process - typically OpenMP (libiomp5md / libgomp / vcomp) or protobuf, both of
which ship inside several wheels. Import either library on its own and all is
well; import both, in that order, and the process dies.

That is why this script tests CUMULATIVELY, and why every probe runs in its own
subprocess: a segfault takes the whole interpreter with it, so the only way to
learn which import caused it is to not be in the same process when it happens.
The last probe that succeeds, followed by the first that dies, names the pair.
"""

import os
import subprocess
import sys

PY = sys.executable

# The order train_model.py imports things, which is the order that matters.
CUMULATIVE = [
    ("numpy", "import numpy"),
    ("pandas", "import pandas"),
    ("matplotlib", "import matplotlib; matplotlib.use('Agg')"),
    ("matplotlib.pyplot", "import matplotlib.pyplot"),
    ("seaborn", "import seaborn"),
    ("sklearn", "import sklearn.ensemble, sklearn.pipeline, sklearn.preprocessing"),
    ("skl2onnx", "from skl2onnx import convert_sklearn"),
    ("cv2", "import cv2"),
    ("feature_extractor", "from tarjuman_core import feature_extractor"),
]

# Individually, to tell "this package is broken" apart from "these two clash".
SOLO = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn",
        "skl2onnx", "onnx", "onnxruntime", "cv2", "mediapipe", "joblib"]

VERSIONS = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn",
            "skl2onnx", "onnx", "onnxruntime", "cv2", "mediapipe", "protobuf",
            "google.protobuf", "joblib", "threadpoolctl"]


def run(code: str, timeout: int = 180) -> tuple:
    """
    Run a snippet in a fresh interpreter. Returns (returncode, stderr).

    `-X faulthandler` is the important part. A native crash produces no Python
    traceback by default, so all you learn is that the process died. With the
    fault handler armed, the interpreter dumps the Python stack it was executing
    at the moment the signal arrived - which names the exact module and line,
    turning "skl2onnx crashes" into "skl2onnx crashes while importing X".
    """
    try:
        p = subprocess.run([PY, "-X", "faulthandler", "-c", code],
                           capture_output=True, timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8:replace",
                                "PYTHONFAULTHANDLER": "1"})
        return p.returncode, (p.stderr or b"").decode("utf-8", "replace").strip()
    except subprocess.TimeoutExpired:
        return -999, "timed out"


def describe(code: int) -> str:
    if code == 0:
        return "ok"
    if code in (3221225477, -1073741819):
        return "CRASH 0xC0000005 access violation"
    if code in (3221225725, -1073741571):
        return "CRASH stack overflow"
    if code == -999:
        return "hung"
    return f"exit {code}"


def main() -> int:
    print("=" * 68)
    print("   TARJUMAN IMPORT DOCTOR")
    print("=" * 68)
    print(f"   python   : {sys.version.split()[0]}  ({PY})")
    print(f"   platform : {sys.platform}")

    print("\n" + "-" * 68)
    print("   INSTALLED VERSIONS")
    print("-" * 68)
    probe = ("import importlib.metadata as m\n"
             "for name in %r:\n"
             "    try: print(name, m.version(name))\n"
             "    except Exception: print(name, '-')\n" % (VERSIONS,))
    out = subprocess.run([PY, "-c", probe], capture_output=True)
    for line in out.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            print(f"   {parts[0]:<20s} {parts[1]}")

    print("\n" + "-" * 68)
    print("   EACH PACKAGE ON ITS OWN")
    print("-" * 68)
    broken = []
    for mod in SOLO:
        rc, err = run(f"import {mod}")
        status = describe(rc)
        print(f"   {mod:<20s} {status}")
        if rc != 0:
            broken.append((mod, status, err))
            for line in err.splitlines()[:14]:
                print("        " + line)

    if any(m == "skl2onnx" for m, _, _ in broken):
        print("\n" + "-" * 68)
        print("   WALKING INTO skl2onnx")
        print("-" * 68)
        print("   skl2onnx is pure Python, so it cannot crash by itself - one of")
        print("   the libraries it pulls in must be doing it. Loading them one at")
        print("   a time says which.")
        print()
        chain = [
            ("scipy.sparse",       "import scipy.sparse"),
            ("onnx",               "import onnx"),
            ("onnx.helper",        "import onnx; import onnx.helper"),
            ("onnx.numpy_helper",  "import onnx.numpy_helper"),
            ("onnx.defs",          "import onnx.defs; onnx.defs.get_all_schemas()"),
            ("onnxconverter_common", "import onnxconverter_common"),
            ("sklearn+onnx",       "import sklearn.ensemble; import onnx"),
            ("onnx+sklearn",       "import onnx; import sklearn.ensemble"),
            ("skl2onnx.common",    "import skl2onnx.common"),
            ("skl2onnx.algebra",   "import skl2onnx.algebra.onnx_ops"),
            ("skl2onnx (full)",    "from skl2onnx import convert_sklearn"),
        ]
        first_bad = None
        for name, stmt in chain:
            rc, err = run(stmt)
            print(f"   {name:<24s} {describe(rc)}")
            if rc != 0 and first_bad is None:
                first_bad = (name, stmt, err)
        if first_bad:
            name, stmt, err = first_bad
            print(f"\n   First to die: {name}")
            print(f"   Reproduce with:  python -X faulthandler -c \"{stmt}\"")
            if err:
                print("\n   Crash stack:")
                for line in err.splitlines()[:25]:
                    print("     " + line)

    print("\n" + "-" * 68)
    print("   CUMULATIVE, IN THE ORDER train_model.py USES THEM")
    print("-" * 68)
    print("   (the first line that CRASHES names the culprit)")
    print()
    so_far = []
    culprit = None
    for name, stmt in CUMULATIVE:
        so_far.append(stmt)
        rc, err = run("\n".join(so_far))
        status = describe(rc)
        print(f"   +{name:<28s} {status}")
        if rc != 0:
            culprit = (name, status, err)
            break

    print("\n" + "=" * 68)
    print("   VERDICT")
    print("=" * 68)

    if culprit is None and not broken:
        print("   Every import succeeded. The crash is NOT at import time -")
        print("   re-run `npm run train` and note how far the progress output")
        print("   gets before it dies.")
        return 0

    if culprit:
        name, status, err = culprit
        print(f"   Python dies when {name} is added to the imports before it.")
        print(f"   Status: {status}")
        if err:
            print("\n   stderr:")
            for line in err.splitlines()[-12:]:
                print("     " + line)
        print("\n   What this usually means")
        print("   -----------------------")
        print("   Two wheels are shipping different builds of the same native")
        print("   library. The usual suspects, in order of likelihood:")
        print()
        print("   1. OpenMP  - numpy/scipy/sklearn built against MKL loading")
        print("      libiomp5md.dll while another loads vcomp/libgomp.")
        print(f"        {PY} -m pip install --force-reinstall --no-cache-dir numpy scipy scikit-learn")
        print()
        print("   2. protobuf - mediapipe and onnx bundle different versions.")
        print(f"        {PY} -m pip install \"protobuf<5\"")
        print()
        print("   3. A half-installed wheel from an interrupted pip run.")
        print(f"        {PY} -m pip install --force-reinstall --no-cache-dir {name}")

    if broken:
        print("\n   These packages crash or fail on their own:")
        for mod, status, err in broken:
            print(f"     {mod:<20s} {status}")
            if err:
                print("       " + err.splitlines()[-1][:100])

    return 1


if __name__ == "__main__":
    sys.exit(main())
