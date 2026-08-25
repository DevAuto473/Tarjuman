"""
tarjuman_core.paths - where things live, resolved from the repo, not the cwd
============================================================================
Every dataset and model path used to be a bare filename like
"dynamic_gestures_v4.csv", which resolves against the CURRENT WORKING DIRECTORY.
That worked only because everything happened to be launched from the repo root;
running the same script from anywhere else would create a second, empty dataset
beside you instead of reading the real one, with no error to explain it.

Anchoring on this file's own location removes the assumption entirely.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "tarjuman"


def data(name: str) -> str:
    """Absolute path to a file in data/."""
    return str(DATA_DIR / name)


def root(name: str) -> str:
    """Absolute path to a file at the repo root (models, reports)."""
    return str(PROJECT_ROOT / name)
