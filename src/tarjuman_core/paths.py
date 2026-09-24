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


# -- اسم مجموعة البيانات الحالية ----------------------------------------------
# كان الاسم مكتوباً حرفياً في خمسة ملفّات، فبدءُ مجموعةٍ جديدة يعني تعديلها
# كلّها — ونسيانُ واحدٍ منها يعني أن تُدرَّب على مجموعة وتُصدَّر من أخرى بصمت.
# هنا اسمٌ واحد، ويُغيَّر من متغيّر البيئة عند الحاجة دون لمس أيّ ملفّ.
#
# v6 = المجموعة الحالية: مبنيّة من الفيديوهات المسجَّلة عبر `npm run trainstill`،
# ٤٣ كلمة. وما قبلها (v5، v4) باقٍ في مكانه — لم يُحذَف، فالرجوع إليه ممكن
# بضبط TARJUMAN_DATASET دون لمس أيّ ملفّ.
#
# ولماذا تأخّر تحديث هذا السطر مشكلةً: بقي الاسم عند v5 بعد أن انتقل التدريب
# إلى v6، فصار النموذج يعرف ٤٣ كلمة بينما يقرأ `export_signs_3d.py`
# و`websocket_server.py` مرجعَ ٢٧ كلمة قديمة — وهو الانحراف الصامت الذي
# وُجد هذا الملفّ أصلاً لمنعه.
import os as _os

DATASET_NAME = _os.getenv("TARJUMAN_DATASET", "dynamic_gestures_v6.csv")


def dataset_csv() -> str:
    """المسار الكامل لمجموعة البيانات الحالية."""
    return str(DATA_DIR / DATASET_NAME)


def data(name: str) -> str:
    """Absolute path to a file in data/."""
    return str(DATA_DIR / name)


def root(name: str) -> str:
    """Absolute path to a file at the repo root (models, reports)."""
    return str(PROJECT_ROOT / name)
