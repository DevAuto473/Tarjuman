"""
take_quality.py — هل خرجت هذه التسجيلة كاملةً وصالحة؟
================================================================================
يُستعمل في موضعين، وهذا سببُ وجوده مستقلّاً:

  • `data_collector.py` عند المعايرة — قبل أن تسجّل خمسين عيّنة، تُؤدّى الكلمة
    مرّةً واحدة فيُقاس طولها الحقيقي وتُضبط نافذة التسجيل عليه، وتُرفض المعايرة
    إن كانت معيبة.

  • `scripts/audit_signs.py` بعد التسجيل — نفس المقاييس على المجموعة كلّها.

وكونُهما يستعملان نفس العتبات هو المقصود: لا معنى لأن يقبل المسجِّل تسجيلةً
يرفضها الفاحص بعد ساعة.
"""

import numpy as np

from tarjuman_core.feature_extractor import VALS_PER_HAND
from tarjuman_core.pose_to_bones import hand_centre

# ── العتبات ──────────────────────────────────────────────────────────────────
# كلّها مقيسة على مجموعة v4، حيث كانت «أهلاً» أنظف تسجيلة (مسار 2.62، تبدأ
# من الجنب وتعود إليه) و«لا أفهم» أسوأها (مسار 0.69، تبدأ واليد مرفوعة).

MIN_HAND_FRAMES = 0.50      # نسبة الإطارات التي يجب أن تُرى فيها يدٌ واحدة على الأقلّ

# -- كم مساراً يُعَدّ حركةً لا ارتعاشاً --------------------------------------
# كان هذا رقماً واحداً (1.00) مفروضاً على كلّ الكلمات، وهو الخطأ نفسه الذي
# كانت ترتكبه النافذة الزمنية الثابتة: الإشاراتُ لا تتساوى في السَّعة.
#
# قِيس على v4: 'salam' مسارها الوسيط 1.99، و'sorry' مسارها 0.68 — لأنّها
# إشارةٌ صغيرة على الصدر بطبعها، لا لأنّها رديئة. فكانت العتبةُ الواحدة تردّ
# 60 تسجيلة من 60 لكلمة 'sorry'، أي أنّها تمنع تعلُّمَ الكلمة بالكامل.
#
# فصارت العتبةُ نسبةً من مسار الكلمة نفسها كما قِيس في معايرتها، وتحتها أرضيةٌ
# مطلقة تمسك التسجيلة الميتة حقاً (يدٌ واقفة، أو فُقد التتبّع). أثرُ ذلك على
# v4: المرورُ الإجمالي 62% ← 87%، و'sorry' من 0% إلى 50% (والنصف الباقي
# يسقط لانقطاع التتبّع، وهو ردٌّ في محلّه).
PATH_FLOOR = 0.25           # أقلّ مسارٍ يُعَدّ حركةً أصلاً، أيّاً كانت الكلمة
PATH_FRACTION = 0.45        # ونسبةُ ما قِيس للكلمة في معايرتها
REST_SPEED = 0.25           # سرعة الحدّ ÷ الذروة: أقلّ من هذا يعني «ساكن»
REST_FRAMES = 3             # كم إطاراً يُفحَص عند كل حدّ
MOTION_ON = 0.20            # سرعة ÷ ذروة تُعَدّ بدايةَ الحركة

# هامشٌ يُضاف إلى المدّة المقيسة عند ضبط النافذة، فلا تُقتطع المحاولةُ التالية
# لمجرّد أنّها جاءت أبطأ قليلاً من معايرتها.
WINDOW_MARGIN = 1.35
MIN_WINDOW = 1.20
MAX_WINDOW = 4.00


def _hand_path(sequence):
    """مسار مركز الكفّ عبر إطارات التسجيلة. NaN حيث لا يدَ مرصودة."""
    pts, seen_right, seen_left = [], 0, 0
    for frame in sequence:
        frame = np.asarray(frame, dtype=np.float32)
        right = frame[VALS_PER_HAND:2 * VALS_PER_HAND]
        left = frame[0:VALS_PER_HAND]
        seen_right += bool(np.any(right))
        seen_left += bool(np.any(left))
        block = right if np.any(right) else (left if np.any(left) else None)
        pts.append(hand_centre(block) if block is not None else np.full(3, np.nan))
    return np.asarray(pts, dtype=float), seen_right, seen_left


def analyse(sequence, duration: float, segmented: bool = False,
            expect_path: float | None = None) -> dict:
    """
    يقيس تسجيلةً واحدة ويعيد وصفاً لها.

    المفاتيح المهمّة:
        ok              — صالحة للاعتماد عليها في المعايرة

    `expect_path` مسارُ الكلمة كما قِيس في معايرتها. حين يُمرَّر، تُقاس سَعةُ
    التسجيلة إليه لا إلى رقمٍ عامّ — فلا تُردّ إشارةٌ صغيرةُ السَّعة بطبعها.
    وحين لا يُمرَّر (أثناء المعايرة نفسها، إذ لم يُقَس بعد) تُطبَّق الأرضية
    المطلقة وحدها: مهمّةُ المعايرة أن تقيس لا أن تحكم على السَّعة.

    `segmented=True` حين يكون المقطِّع هو من حدّد بداية التسجيلة ونهايتها:
    عندئذٍ لا معنى لفحص «تبدأ ساكنة / تنتهي ساكنة»، لأنّ التسجيلة مقصوصةٌ
    على الحركة نفسها بالتعريف — وحدودها مسؤولية المقطِّع لا المستخدم.
        problems        — قائمة رسائل عربية بما فيها من عيب
        gesture_seconds — زمن الحركة نفسها داخل النافذة (لا طول النافذة)
        window          — النافذة المقترَحة لبقيّة عيّنات هذه الكلمة
        path            — طول مسار الكفّ
        hands           — نسبة حضور كل يد
    """
    n = len(sequence)
    out = {"ok": False, "problems": [], "frames": n, "seconds": float(duration),
           "gesture_seconds": float(duration), "window": MIN_WINDOW,
           "path": 0.0, "need_path": PATH_FLOOR, "hands": {"left": 0.0, "right": 0.0}, "two_handed": False}

    if n < 5:
        out["problems"].append("Take too short - the camera did not deliver enough frames.")
        return out

    pts, seen_right, seen_left = _hand_path(sequence)
    out["hands"] = {"left": seen_left / n, "right": seen_right / n}
    out["two_handed"] = (min(seen_left, seen_right) / n) >= 0.40

    seen_any = float(np.mean(~np.isnan(pts).any(axis=1)))
    if seen_any < MIN_HAND_FRAMES:
        out["problems"].append(
            f"A hand was seen in only {seen_any * 100:.0f}% of frames - "
            "move closer to the camera or improve the lighting.")
        return out

    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path = float(np.nansum(step))
    out["path"] = path

    # المرجع هو المئين التسعون لا القيمة العظمى: قفزةٌ واحدة — تحدث كلّما
    # فُقدت اليد لإطارٍ ثمّ عادت — تكفي لتجعل الذروةَ عشرةَ أضعاف السرعة
    # الحقيقية، فتصير كلُّ الحركة «ساكنة» بالقياس إليها.
    valid = step[~np.isnan(step)]
    peak = float(np.percentile(valid, 90)) if valid.size else 0.0

    need_path = PATH_FLOOR
    if expect_path:
        need_path = max(PATH_FLOOR, PATH_FRACTION * float(expect_path))
    out["need_path"] = need_path

    if path < need_path:
        out["problems"].append(
            f"Motion too faint (path {path:.2f}, needs above {need_path:.2f}) - "
            "sign it wider.")

    def _edge(slice_):
        """متوسّط سرعة طرفٍ من التسجيلة، أو None إن لم تُرصد يدٌ هناك."""
        valid_edge = slice_[~np.isnan(slice_)]
        return float(valid_edge.mean()) if valid_edge.size else None

    if peak > 0:
        head = _edge(step[:REST_FRAMES])
        tail = _edge(step[-REST_FRAMES:])

        # لا يدَ عند الطرف أصلاً: عيبٌ في ذاته، لا مجرّد تعذّرٍ في القياس.
        if head is None:
            out["problems"].append("No hand detected at the start of the take.")
            head = 0.0
        else:
            head /= peak
        if tail is None:
            out["problems"].append("No hand detected at the end of the take.")
            tail = 0.0
        else:
            tail /= peak

        if not segmented:
            if head > REST_SPEED:
                out["problems"].append(
                    "Take begins with the hand moving - start still, hands at your side.")
            if tail > REST_SPEED:
                out["problems"].append(
                    "Take ends with the hand moving - the window is shorter than the "
                    "sign, or you did not return your hand.")

        # ── زمن الحركة نفسها داخل النافذة ──────────────────────────────────
        moving = np.where(np.nan_to_num(step) > MOTION_ON * peak)[0]
        if moving.size:
            span = (moving[-1] - moving[0] + 1) / max(1, len(step))
            out["gesture_seconds"] = max(0.2, span * duration)

    out["window"] = float(np.clip(out["gesture_seconds"] * WINDOW_MARGIN,
                                  MIN_WINDOW, MAX_WINDOW))
    out["ok"] = not out["problems"]
    return out


def suggest_samples(report: dict, base: int) -> int:
    """
    كم عيّنة تستحقّ هذه الكلمة.

    إشارةٌ بيدين أو ذاتُ مسارٍ طويل فيها تنوّعٌ أكثر، فتحتاج أمثلةً أكثر كي
    يتعلّم النموذج حدودها؛ وإشارةٌ قصيرة ثابتة تشبع أسرع. الزيادة محدودة عمداً
    (±٢٠٪) — هذا ترجيحٌ لا حساب.
    """
    factor = 1.0
    if report.get("two_handed"):
        factor += 0.12
    if report.get("path", 0) > 2.20:
        factor += 0.08
    elif report.get("path", 0) < 1.40:
        factor -= 0.08
    return int(round(base * factor / 5.0) * 5)


# ═════════════════════════════════════════════════════════════════════════════
#  التشابه بين التسجيلات — للشذوذ والإشباع
# ═════════════════════════════════════════════════════════════════════════════
#
# المسافة بين تسجيلتين تُحسب بـDTW، وهي نفس الدالة التي يقيس بها وضعُ التدرّب
# أداءَ المستخدم. استعمالها هنا يعني أنّ «شاذّة» في المسجِّل و«بعيدة» في
# التدرّب يقيسان الشيء ذاته.


def take_distance(a, b) -> float:
    """مسافة DTW بين تسجيلتين، كلٌّ منهما (SEQUENCE_LENGTH, VALS_PER_FRAME)."""
    from tarjuman_core.dtw_matcher import dtw_distance
    return float(dtw_distance(np.asarray(a, dtype=np.float32),
                              np.asarray(b, dtype=np.float32)))


def spread(takes) -> dict:
    """
    كم تتباعد التسجيلات عن بعضها.

    يُعيد وسيطَها (medoid) ومتوسّطَ بُعد التسجيلات عنه. هذا الرقم هو ما يُقاس
    عليه شيئان:

      • الشذوذ — تسجيلةٌ أبعد عن الوسيط من أضعاف هذا المتوسّط ليست مثالاً
        للكلمة، بل خطأً في الأداء.

      • الإشباع — حين يتوقّف هذا المتوسّط عن التغيّر مع إضافة تسجيلات جديدة،
        فقد صارت الجديدة لا تضيف شيئاً، وتوقّف التسجيل قرارٌ مبنيّ على قياس
        لا على عدّ.
    """
    takes = [np.asarray(t, dtype=np.float32) for t in takes]
    if len(takes) < 2:
        return {"medoid": 0, "mean": 0.0, "distances": [0.0] * len(takes)}

    totals = []
    matrix = [[0.0] * len(takes) for _ in takes]
    for i, a in enumerate(takes):
        for j in range(i + 1, len(takes)):
            d = take_distance(a, takes[j])
            matrix[i][j] = matrix[j][i] = d
        totals.append(sum(matrix[i]))

    medoid = int(np.argmin(totals))
    distances = matrix[medoid]
    others = [d for k, d in enumerate(distances) if k != medoid]
    return {"medoid": medoid,
            "mean": float(np.mean(others)) if others else 0.0,
            "distances": distances}
