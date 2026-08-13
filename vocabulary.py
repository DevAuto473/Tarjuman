"""
vocabulary.py — Tarjuman target vocabulary + separability analysis
===================================================================
100 everyday terms, described with a CLOSED set of attribute values so that
"do these two signs collide?" is a real question with a real answer.

Why the closed vocabulary matters
---------------------------------
An earlier version of this file let each sign carry a free-text description of
its movement. 79 of 100 signs ended up with a movement string no other sign
used — so "zero collisions" was guaranteed by the way the descriptions were
written, not by the signs being different. The check was circular and worthless.

Now every attribute must come from a fixed list below. If two signs really are
made the same way, they are forced to carry the same values, and the collision
is exposed instead of hidden behind a different choice of words.

The parameters are the classic sign-language ones (Stokoe): handshape,
location, movement, orientation — plus how many hands. Two signs identical in
all five are indistinguishable from landmarks, full stop.

⚠️  What this file is NOT
-------------------------
The WORD LIST is a usage judgement. The ATTRIBUTE values are engineering
placeholders so collisions can be found BEFORE recording. They are not an
authoritative description of Saudi Sign Language. Verify each sign against:
    • القاموس السعودي الإشاري الموحد — https://saudisla.org/programs/dictionary/
    • القاموس الإشاري العربي للصم    — https://selaa.org/ar/node/204
then correct the rows here and re-run. Recording against a wrong description
means recording everything twice.

Run:  python vocabulary.py
"""

# ═════════════════════════════════════════════════════════════════════════════
#  Controlled vocabularies — adding a value here is a deliberate act
# ═════════════════════════════════════════════════════════════════════════════

SHAPES = {
    "flat",        # كف مسطّح، أصابع مضمومة وممدودة
    "open5",       # كف مفتوح، أصابع مفرودة متباعدة
    "fist",        # قبضة
    "index",       # سبابة فقط
    "two",         # سبابة + وسطى
    "three",       # ثلاثة أصابع
    "four",        # أربعة أصابع
    "thumb",       # إبهام فقط
    "pinch",       # إبهام + سبابة متلامسان
    "cup",         # كف مقوّس كالكوب
    "claw",        # أصابع مقوّسة متباعدة
    "tips",        # أطراف الأصابع مجتمعة
    "hook",        # سبابة معقوفة
    "bent",        # كف منثنٍ عند المفاصل
}

LOCATIONS = {
    "neutral",     # الفضاء المحايد أمام الجسم
    "chest", "shoulder", "belly", "waist",
    "forehead", "temple", "eyes", "ear", "cheek", "chin", "mouth", "throat",
    "outward",     # باتجاه المخاطَب
    "other_palm",  # على كف اليد الأخرى
    "other_wrist", # على معصم اليد الأخرى
}

MOTIONS = {
    "static",      # بلا حركة (وضعية محفوظة)
    "tap",         # لمس متكرر
    "forward", "backward", "up", "down", "side",
    "circular",    # دوران
    "arc",         # قوس
    "wave",        # تذبذب جانبي
    "alternate",   # تبادل بين اليدين
    "open",        # انفتاح
    "close",       # انغلاق
    "contact",     # التقاء اليدين
    "shake",       # اهتزاز سريع قصير
}

ORIENTATIONS = {
    "fwd",   # الكف للأمام
    "in",    # الكف نحو الجسم
    "up",
    "down",
    "side",
}

FIELDS = ("id", "arabic", "shape", "location", "motion", "orient", "hands", "speed_critical")

# (id, arabic, shape, location, motion, orient, hands, speed_critical)
VOCABULARY = [
    # ── تحيّات ومجاملات ──────────────────────────────────────────────────────
    ("salam",         "السلام عليكم", "flat",  "forehead",   "arc",       "fwd",  1, False),
    ("hello",         "أهلاً",         "open5", "shoulder",   "wave",      "fwd",  1, False),
    ("morning",       "صباح الخير",    "flat",  "neutral",    "up",        "up",   2, False),
    ("evening",       "مساء الخير",    "flat",  "neutral",    "down",      "down", 2, False),
    ("goodbye",       "مع السلامة",    "open5", "outward",    "wave",      "fwd",  1, False),
    ("thanks",        "شكراً",         "tips",  "chin",       "forward",   "in",   1, False),
    ("sorry",         "آسف",           "fist",  "chest",      "circular",  "in",   1, False),
    ("excuse",        "عفواً",         "flat",  "chest",      "forward",   "up",   1, False),
    ("please",        "لو سمحت",       "flat",  "chest",      "circular",  "up",   1, False),
    ("welcome",       "تشرفنا",        "open5", "chest",      "open",      "up",   2, False),

    # ── أسئلة ───────────────────────────────────────────────────────────────
    ("q_who",         "مَن؟",          "index", "chin",       "circular",  "in",   1, False),
    ("q_what",        "ماذا؟",         "open5", "neutral",    "shake",     "up",   2, False),
    ("q_where",       "أين؟",          "index", "neutral",    "side",      "fwd",  1, False),
    ("q_when",        "متى؟",          "index", "other_wrist","tap",       "down", 1, False),
    ("q_why",         "لماذا؟",        "index", "temple",     "forward",   "in",   1, False),
    ("q_how",         "كيف؟",          "cup",   "neutral",    "circular",  "up",   2, False),
    ("q_howmuch",     "كم؟",           "pinch", "neutral",    "shake",     "up",   1, False),

    # ── ضمائر وأشخاص ────────────────────────────────────────────────────────
    ("i_me",          "أنا",           "index", "chest",      "tap",       "in",   1, False),
    ("you",           "أنت",           "index", "outward",    "static",    "fwd",  1, False),
    ("he_she",        "هو/هي",         "index", "neutral",    "side",      "side", 1, False),
    ("we",            "نحن",           "index", "chest",      "arc",       "in",   1, False),
    ("father",        "أب",            "thumb", "forehead",   "tap",       "side", 1, False),
    ("mother",        "أم",            "thumb", "chin",       "tap",       "side", 1, False),
    ("brother",       "أخ",            "index", "forehead",   "down",      "in",   2, False),
    ("sister",        "أخت",           "index", "cheek",      "down",      "in",   2, False),
    ("son",           "ابن",           "flat",  "waist",      "arc",       "up",   2, False),
    ("daughter",      "بنت",           "index", "cheek",      "down",      "side", 1, False),
    ("friend",        "صديق",          "hook",  "chest",      "contact",   "in",   2, False),
    ("neighbour",     "جار",           "flat",  "neutral",    "side",      "in",   2, False),

    # ── الصمم والتواصل ──────────────────────────────────────────────────────
    ("deaf",          "أصم",           "index", "ear",        "arc",       "in",   1, False),
    ("hearing",       "سامع",          "index", "mouth",      "circular",  "in",   1, False),
    ("sign_lang",     "لغة إشارة",     "open5", "neutral",    "alternate", "fwd",  2, False),
    ("interpreter",   "مترجم",         "flat",  "mouth",      "circular",  "fwd",  1, False),
    ("name",          "اسم",           "two",   "neutral",    "contact",   "down", 2, False),
    ("understand",    "أفهم",          "index", "temple",     "up",        "in",   1, False),
    ("not_understand","لا أفهم",       "index", "temple",     "side",      "in",   1, False),
    ("repeat",        "أعد",           "bent",  "other_palm", "arc",       "up",   2, False),

    # ── احتياجات يومية ──────────────────────────────────────────────────────
    ("water",         "ماء",           "three", "mouth",      "tap",       "side", 1, False),
    ("eat",           "أكل",           "tips",  "mouth",      "tap",       "in",   1, False),
    ("drink",         "أشرب",          "cup",   "mouth",      "arc",       "in",   1, False),
    ("sleep",         "نوم",           "flat",  "cheek",      "static",    "side", 1, False),
    ("toilet",        "دورة مياه",     "fist",  "neutral",    "shake",     "fwd",  1, False),
    ("home",          "بيت",           "flat",  "neutral",    "contact",   "down", 2, False),
    ("help",          "مساعدة",        "fist",  "other_palm", "up",        "up",   2, False),
    ("want",          "أريد",          "cup",   "chest",      "backward",  "up",   2, False),
    ("dont_want",     "لا أريد",       "cup",   "chest",      "forward",   "down", 2, False),
    ("hungry",        "جوعان",         "cup",   "belly",      "down",      "in",   1, False),
    ("thirsty",       "عطشان",         "index", "throat",     "down",      "in",   1, False),
    ("wait",          "انتظر",         "open5", "neutral",    "wave",      "up",   2, False),

    # ── مشاعر وحالة ─────────────────────────────────────────────────────────
    ("fine",          "بخير",          "thumb", "chest",      "up",        "in",   1, False),
    ("happy",         "سعيد",          "flat",  "chest",      "up",        "in",   2, False),
    ("sad",           "حزين",          "open5", "eyes",       "down",      "in",   2, False),
    ("angry",         "غاضب",          "claw",  "chest",      "up",        "in",   1, False),
    ("afraid",        "خائف",          "open5", "chest",      "backward",  "fwd",  2, False),
    ("tired",         "تعبان",         "bent",  "shoulder",   "down",      "in",   2, False),
    ("sick",          "مريض",          "bent",  "forehead",   "static",    "in",   1, False),
    ("love",          "أحب",           "fist",  "chest",      "contact",   "in",   2, False),
    ("good",          "جيد",           "flat",  "chin",       "forward",   "up",   1, False),
    ("bad",           "سيئ",           "flat",  "chin",       "down",      "down", 1, False),

    # ── الوقت ───────────────────────────────────────────────────────────────
    ("today",         "اليوم",         "index", "neutral",    "down",      "down", 1, False),
    ("yesterday",     "أمس",           "thumb", "shoulder",   "backward",  "in",   1, False),
    ("tomorrow",      "غداً",          "thumb", "cheek",      "forward",   "side", 1, False),
    ("now",           "الآن",          "bent",  "neutral",    "down",      "up",   2, False),
    ("hour",          "ساعة",          "index", "other_wrist","circular",  "down", 1, False),
    ("minute",        "دقيقة",         "index", "other_palm", "forward",   "side", 2, False),
    ("day",           "يوم",           "index", "neutral",    "arc",       "fwd",  1, False),
    ("week",          "أسبوع",         "index", "other_palm", "side",      "down", 2, False),
    ("month",         "شهر",           "index", "neutral",    "down",      "side", 2, False),
    ("year",          "سنة",           "fist",  "neutral",    "circular",  "side", 2, False),

    # ── أرقام ١-١٠ (وضعيات ثابتة — مسار المصنّف الثابت) ──────────────────────
    ("num_1",         "واحد",          "index", "neutral",    "static",    "fwd",  1, False),
    ("num_2",         "اثنان",         "two",   "neutral",    "static",    "fwd",  1, False),
    ("num_3",         "ثلاثة",         "three", "neutral",    "static",    "fwd",  1, False),
    ("num_4",         "أربعة",         "four",  "neutral",    "static",    "fwd",  1, False),
    ("num_5",         "خمسة",          "open5", "neutral",    "static",    "fwd",  1, False),
    ("num_6",         "ستة",           "pinch", "neutral",    "static",    "fwd",  1, False),
    ("num_7",         "سبعة",          "tips",  "neutral",    "static",    "fwd",  1, False),
    ("num_8",         "ثمانية",        "claw",  "neutral",    "static",    "fwd",  1, False),
    ("num_9",         "تسعة",          "hook",  "neutral",    "static",    "fwd",  1, False),
    ("num_10",        "عشرة",          "fist",  "neutral",    "static",    "fwd",  1, False),

    # ── أماكن وتنقّل ────────────────────────────────────────────────────────
    ("hospital",      "مستشفى",        "two",   "shoulder",   "contact",   "in",   1, False),
    ("pharmacy",      "صيدلية",        "pinch", "other_palm", "tap",       "down", 2, False),
    ("school",        "مدرسة",         "flat",  "other_palm", "tap",       "down", 2, False),
    ("work",          "عمل",           "fist",  "other_wrist","tap",       "down", 2, False),
    ("market",        "سوق",           "flat",  "neutral",    "alternate", "up",   2, False),
    ("mosque",        "مسجد",          "flat",  "neutral",    "arc",       "down", 2, False),
    ("car",           "سيارة",         "fist",  "neutral",    "alternate", "in",   2, False),
    ("road",          "طريق",          "flat",  "neutral",    "forward",   "side", 2, False),
    ("go",            "أذهب",          "index", "neutral",    "forward",   "fwd",  1, False),

    # ── طوارئ وصحة ─────────────────────────────────────────────────────────
    ("emergency",     "طوارئ",         "open5", "shoulder",   "shake",     "fwd",  2, True),
    ("ambulance",     "إسعاف",         "fist",  "shoulder",   "circular",  "fwd",  1, True),
    ("police",        "شرطة",          "flat",  "chest",      "tap",       "in",   1, False),
    ("fire",          "حريق",          "claw",  "neutral",    "up",        "in",   2, False),
    ("doctor",        "طبيب",          "two",   "other_wrist","static",    "down", 2, False),
    ("medicine",      "دواء",          "bent",  "other_palm", "circular",  "down", 2, False),
    ("pain",          "ألم",           "index", "neutral",    "shake",     "in",   2, False),
    ("help_now",      "ساعدني فوراً",  "fist",  "outward",    "backward",  "up",   2, True),

    # ── أفعال وأوامر شائعة ─────────────────────────────────────────────────
    ("come",          "تعال",          "index", "outward",    "backward",  "up",   1, False),
    ("sit",           "اجلس",          "two",   "other_palm", "down",      "down", 2, False),
    ("stop",          "توقف",          "flat",  "other_palm", "contact",   "side", 2, False),
    ("give_me",       "أعطني",         "flat",  "outward",    "backward",  "up",   1, False),
]


# Section each term belongs to — used for grouping in the recording sheet.
# Kept separate from the tuples so adding it did not have to touch every row.
CATEGORIES = [
    ("تحيّات ومجاملات",   "salam",     "welcome"),
    ("أسئلة",             "q_who",     "q_howmuch"),
    ("ضمائر وأشخاص",      "i_me",      "neighbour"),
    ("الصمم والتواصل",    "deaf",      "repeat"),
    ("احتياجات يومية",    "water",     "wait"),
    ("مشاعر وحالة",       "fine",      "bad"),
    ("الوقت",             "today",     "year"),
    ("أرقام",             "num_1",     "num_10"),
    ("أماكن وتنقّل",      "hospital",  "go"),
    ("طوارئ وصحة",        "emergency", "help_now"),
    ("أفعال وأوامر",      "come",      "give_me"),
]


def category_of(term_id: str) -> str:
    """Which section a term belongs to."""
    ids = [row[0] for row in VOCABULARY]
    try:
        pos = ids.index(term_id)
    except ValueError:
        return "غير مصنّف"
    for name, first, last in CATEGORIES:
        if ids.index(first) <= pos <= ids.index(last):
            return name
    return "غير مصنّف"


# Human-readable Arabic for the controlled values, for the recording sheet.
SHAPE_AR = {
    "flat": "كف مسطّح", "open5": "كف مفتوح", "fist": "قبضة", "index": "سبابة",
    "two": "سبابة+وسطى", "three": "ثلاثة أصابع", "four": "أربعة أصابع",
    "thumb": "إبهام", "pinch": "إبهام+سبابة", "cup": "كف مقوّس",
    "claw": "أصابع مخلبية", "tips": "أطراف الأصابع", "hook": "سبابة معقوفة",
    "bent": "كف منثنٍ",
}
LOCATION_AR = {
    "neutral": "أمام الجسم", "chest": "الصدر", "shoulder": "الكتف",
    "belly": "البطن", "waist": "الخصر", "forehead": "الجبهة",
    "temple": "الصدغ", "eyes": "العينان", "ear": "الأذن", "cheek": "الخد",
    "chin": "الذقن", "mouth": "الفم", "throat": "الحلق",
    "outward": "نحو المخاطَب", "other_palm": "كف اليد الأخرى",
    "other_wrist": "معصم اليد الأخرى",
}
MOTION_AR = {
    "static": "ثابتة", "tap": "نقر", "forward": "للأمام", "backward": "للخلف",
    "up": "لأعلى", "down": "لأسفل", "side": "جانبية", "circular": "دائرية",
    "arc": "قوسية", "wave": "تلويح", "alternate": "تبادل اليدين",
    "open": "انفتاح", "close": "انغلاق", "contact": "التقاء", "shake": "اهتزاز",
}
ORIENT_AR = {"fwd": "للأمام", "in": "نحو الجسم", "up": "لأعلى",
             "down": "لأسفل", "side": "جانبي"}


def as_dicts():
    return [dict(zip(FIELDS, row)) for row in VOCABULARY]


def signature(entry):
    """The five parameters a landmark-based recogniser can actually observe."""
    return (entry["shape"], entry["location"], entry["motion"],
            entry["orient"], entry["hands"])


# ═════════════════════════════════════════════════════════════════════════════
#  Validation + separability report
# ═════════════════════════════════════════════════════════════════════════════

def validate() -> list[str]:
    """
    Reject any value outside the controlled lists.

    This is the guard that keeps the collision check honest: without it, an
    inconvenient collision can always be "fixed" by inventing a new movement
    name, which changes nothing in the real world.
    """
    errors = []
    seen_ids, seen_words = set(), set()

    for e in as_dicts():
        for field, allowed in (("shape", SHAPES), ("location", LOCATIONS),
                               ("motion", MOTIONS), ("orient", ORIENTATIONS)):
            if e[field] not in allowed:
                errors.append(f"{e['id']}: {field}={e[field]!r} not in the controlled list")
        if e["hands"] not in (1, 2):
            errors.append(f"{e['id']}: hands={e['hands']} (must be 1 or 2)")
        if e["id"] in seen_ids:
            errors.append(f"duplicate id: {e['id']}")
        if e["arabic"] in seen_words:
            errors.append(f"duplicate term: {e['id']}")
        seen_ids.add(e["id"])
        seen_words.add(e["arabic"])

    return errors


# Terminal output is ENGLISH and refers to signs by their ASCII `id`.
# Windows terminals do not shape or bidi-order Arabic, so printing the Arabic
# words there produces unreadable reversed text. The Arabic lives in learn.csv,
# which opens in Excel where it renders correctly.

def report() -> int:
    from collections import Counter, defaultdict

    V = as_dicts()
    print("=" * 74)
    print(f"  SEPARABILITY REPORT — {len(V)} terms")
    print("=" * 74)

    errors = validate()
    if errors:
        print("\n[FAIL] values outside the controlled lists:")
        for msg in errors[:20]:
            print("   ", msg)
        return len(errors)
    print("\n[OK] every attribute value comes from the controlled lists")

    # ── How reusable are the values? (guards against circular checks) ───────
    print()
    print("Description density — the fewer unique values, the more honest the check:")
    for f in ("shape", "location", "motion", "orient"):
        c = Counter(e[f] for e in V)
        once = sum(1 for v in c.values() if v == 1)
        print(f"   {f:9s}: {len(c):2d} distinct | used only once: {once}"
              f" ({once / len(V) * 100:.0f}% of terms)")

    # ── Real collisions ─────────────────────────────────────────────────────
    groups = defaultdict(list)
    for e in V:
        groups[signature(e)].append(e)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}

    print()
    if collisions:
        involved = sum(len(v) for v in collisions.values())
        print(f"[BLOCKING] exact collisions: {len(collisions)} group(s), "
              f"{involved} terms ({involved / len(V) * 100:.0f}%)")
        print("   Indistinguishable from landmarks — redesign or drop them.")
        for sig, ents in sorted(collisions.items(), key=lambda x: -len(x[1])):
            print(f"\n   [{len(ents)}] {', '.join(e['id'] for e in ents)}")
            print(f"        shape={sig[0]} location={sig[1]} motion={sig[2]} "
                  f"orient={sig[3]} hands={sig[4]}")
    else:
        print("[OK] no exact collisions — every term has a unique signature")

    # ── One-parameter separations ───────────────────────────────────────────
    keys = ("shape", "location", "motion", "orient", "hands")
    fragile = defaultdict(list)
    for i, a in enumerate(V):
        for b in V[i + 1:]:
            diffs = [k for k, x, y in zip(keys, signature(a), signature(b)) if x != y]
            if len(diffs) == 1:
                fragile[diffs[0]].append((a["id"], b["id"]))

    total_fragile = sum(len(v) for v in fragile.values())
    print()
    print(f"[WARN] pairs separated by ONE parameter only: {total_fragile}")
    print("   These are the pairs most likely to be confused. Record extra")
    print("   samples for them and watch the confusion report after training.")
    for attr, pairs in sorted(fragile.items(), key=lambda x: -len(x[1])):
        print(f"   differ only in '{attr}' — {len(pairs)} pair(s):")
        for a, b in pairs[:5]:
            print(f"       {a:<16s} <-> {b}")
        if len(pairs) > 5:
            print(f"       ... and {len(pairs) - 5} more")

    speed = [e for e in V if e["speed_critical"]]
    static = [e for e in V if e["motion"] == "static"]
    print()
    print(f"[SPEED]  meaning depends on tempo: {len(speed)}")
    print(f"         {', '.join(e['id'] for e in speed)}")
    print(f"         -> perform these FAST when recording, or the meaning is lost")
    print(f"[STATIC] no movement: {len(static)}")
    print(f"         {', '.join(e['id'] for e in static)}")
    print(f"         -> hold each for a full second so the segmenter fires")

    print()
    print("=" * 74)
    print("  NOTE: attribute values are engineering estimates, NOT authoritative")
    print("  Saudi Sign Language. Verify before recording:")
    print("  https://saudisla.org/programs/dictionary/")
    print("=" * 74)
    return len(collisions)


if __name__ == "__main__":
    import sys
    sys.exit(1 if report() else 0)
