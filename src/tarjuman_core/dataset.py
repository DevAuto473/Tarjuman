"""
dataset.py — قراءة مجموعة البيانات بحسب ترويسة الملفّ، لا بحسب موضع الأعمدة.
================================================================================
لماذا يوجد هذا الملفّ
---------------------
كان كل قارئ يفترض أنّ الصفّ هو `[label] + features` تماماً، ويرفض أيّ صفّ
طوله مختلف. ثمّ اكتسب الملفّ أعمدةَ وصفٍ في المقدّمة:

    label, signer, camera, session, recorded_at, f0_v0, …, g_openness_change

فصار طول الصفّ 4217 بدل 4213، فرُفض كلُّ صفّ في المجموعة بصمت:
«No usable rows found» عند التصدير، و«DTW references loaded: 0 sign(s)» عند
التدريب التفاعلي — أي أنّ وضع التدرّب كان بلا مراجع إطلاقاً.

الحلّ أن تُحدَّد كتلةُ السمات بالاسم: أوّل عمود اسمه `f0_v0`. عندئذٍ يستطيع
أيُّ أحدٍ إضافةَ أعمدة وصفٍ جديدة متى شاء دون أن يكسر شيئاً.
"""

import csv
import os

import numpy as np

from tarjuman_core.feature_extractor import TOTAL_FEATURES

FIRST_FEATURE_COLUMN = "f0_v0"
LABEL_COLUMN = "label"


def describe(path: str) -> dict:
    """يقرأ الترويسة ويصف تخطيط الملفّ دون تحميل الصفوف."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if not header:
        return {"header": None, "start": 1, "meta": [], "columns": 0}

    if header[0] != LABEL_COLUMN:
        # ملفّ قديم بلا ترويسة: الصفّ الأوّل بيانات، والتخطيط هو القديم.
        return {"header": None, "start": 1, "meta": [], "columns": len(header)}

    start = (header.index(FIRST_FEATURE_COLUMN)
             if FIRST_FEATURE_COLUMN in header else 1)
    return {"header": header, "start": start,
            "meta": header[1:start], "columns": len(header)}


def iter_samples(path: str):
    """
    يولّد (label, features) لكلّ صفّ صالح.

    `features` مصفوفة float32 طولها TOTAL_FEATURES بالضبط. تُتخطّى الصفوف
    القصيرة أو التي تحوي قيمةً غير رقمية بصمت، كما كان الحال.
    """
    if not os.path.isfile(path):
        return

    layout = describe(path)
    start = layout["start"]
    needed = start + TOTAL_FEATURES

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if layout["header"] is not None:
            next(reader, None)
        for row in reader:
            if not row or len(row) < needed:
                continue
            try:
                values = np.asarray(row[start:needed], dtype=np.float32)
            except ValueError:
                continue
            yield row[0], values


def column(path: str, name: str) -> int | None:
    """موضع عمودٍ بالاسم، أو None إن لم يكن للملفّ ترويسة."""
    layout = describe(path)
    header = layout["header"]
    if not header or name not in header:
        return None
    return header.index(name)
