/**
 * poses.js — reusable hand shapes for the Tarjuman robot
 * =======================================================
 * Bone names come straight from the rig inside Tarjuman_Signer.glb
 * (skin "Tarjuman_Rig", 54 joints, standard Blender .L/.R naming).
 *
 * Why a pose LIBRARY instead of one animation per word
 * ----------------------------------------------------
 * Sign language reuses a small set of hand shapes over and over — a fist, a
 * flat palm, a pointing index. Authoring a separate Blender animation per word
 * would re-model those same shapes hundreds of times. Here each shape is
 * defined ONCE as data, and a word becomes "hand shape X carried along arm
 * path Y". Adding a word is then a few lines of JSON, not a modelling session.
 *
 * Rotation convention
 * -------------------
 * Every value is [x, y, z] Euler angles in DEGREES, applied in the bone's own
 * local space, relative to its rest pose. Degrees (not radians) because these
 * numbers are meant to be hand-tuned by a human.
 *
 * ⚠️  انحناءات الأصابع أدناه تقديرات، تُضبط بالعين من لوحة المعايرة.
 *     أمّا زوايا الذراع فلم تعد كذلك: حُلَّت عددياً على هذا الهيكل بعينه
 *     (انظر ARM_POSITIONS). محاور عظام Blender اعتباطية، ولهذا كان الحلّ
 *     العددي هو السبيل الوحيد إلى أرقام صحيحة.
 */

// ── Bone name constants ──────────────────────────────────────────────────────
// Centralised so a rig rename breaks in ONE place instead of silently
// producing a robot that simply never moves.
export const BONES = {
  root: 'Root',
  hips: 'Hips',
  spine: 'Spine',
  chest: 'Chest',
  neck: 'Neck',
  head: 'Head',
  jaw: 'Jaw',
  shoulder: (s) => `Shoulder.${s}`,
  upperArm: (s) => `UpperArm.${s}`,
  lowerArm: (s) => `LowerArm.${s}`,
  hand: (s) => `Hand.${s}`,
  finger: (name, seg, s) => `${name}.0${seg}.${s}`,
};

export const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
export const SIDES = ['L', 'R'];

/* وضعيّتا الكفّ: مفتوح ومقبوض — أربعة أرقام (w,x,y,z) لكل سلامية.
 *
 * لماذا وضعيّتان لا زاوية واحدة
 * -----------------------------
 * وضعية السكون في هذا الهيكل ليست كفّاً مبسوطاً: الأصابع فيها منفرجة ومنحنية
 * قليلاً. فكان «انثناء صفر» يعني «اتركها كما وُلدت» لا «افردها»، ولهذا كان
 * الكفّ المفتوح يخرج مفرّج الأصابع نصف مقبوض.
 *
 * الآن الطرفان محسوبان عددياً على الهيكل نفسه:
 *   • المفتوح: حُلّ لكل سلامية دورانٌ يوجّهها إلى امتداد الكفّ (محور اليد
 *     الطولي) مع تفريجٍ جانبي صغير متدرّج (سبّابة ‑0.03 … خنصر +0.12 من
 *     خطّ المفاصل)، بالتسلسل من المفصل إلى الطرف. والإبهام إلى جانب الكفّ
 *     مع ميلٍ خفيف خارج مستواه.
 *   • المقبوض: مبنيّ على تشريح القبضة لا على تخمين زاوية — السلامية الأولى
 *     تدور ٩٠° فتصير عموديّة على الكفّ، والثانية ١٠٢° فترجع نحو الرسغ،
 *     والثالثة ٧٢° فتغوص في الكفّ. المجموع ~٢٦٤°، وهو ما يجعل أطراف
 *     الأصابع تلامس الكفّ (قِيس: ١–٥ مم فوق مستواه). وكان المجموع ٢٢٠°
 *     فتبقى الأطراف ١٣–١٦ مم في الهواء — «مخلب» لا قبضة.
 *     والإبهام محلولٌ باتّجاهات صريحة: يعبُر فوق السلاميات الوسطى كما في
 *     القبضة الحقيقية، لا ينثني في مستواه.
 *
 * وما بينهما slerp حقيقي، لا ضربٌ في نسبة: ضربُ زوايا أويلر يغيّر المحور
 * في المنتصف، فتلتوي الأصابع في الطريق.
 */
const HAND_OPEN = {
  'thumb.01.R': [  0.96979,  -0.05754,   0.00000,  -0.23704],
  'thumb.02.R': [  0.96268,   0.07076,  -0.00000,  -0.26122],
  'thumb.03.R': [  0.98348,   0.15704,   0.00000,   0.09007],
  'index.01.R': [  0.99814,  -0.05337,  -0.00000,   0.02935],
  'index.02.R': [  0.98836,  -0.15107,   0.00000,  -0.01819],
  'index.03.R': [  0.99751,  -0.07012,   0.00000,  -0.00691],
  'middle.01.R': [  0.99229,  -0.12396,   0.00000,  -0.00198],
  'middle.02.R': [  0.98829,  -0.15131,   0.00000,  -0.01969],
  'middle.03.R': [  0.98701,  -0.16003,   0.00000,  -0.01440],
  'ring.01.R': [  0.98523,  -0.17065,   0.00000,  -0.01383],
  'ring.02.R': [  0.98319,  -0.18127,   0.00000,  -0.02205],
  'ring.03.R': [  0.98778,  -0.15513,   0.00000,  -0.01480],
  'pinky.01.R': [  0.97473,  -0.22113,   0.00000,  -0.03183],
  'pinky.02.R': [  0.98832,  -0.15108,   0.00000,  -0.02012],
  'pinky.03.R': [  0.98955,  -0.14335,   0.00000,  -0.01574],
  'thumb.01.L': [  0.96599,  -0.05366,   0.00000,   0.25296],
  'thumb.02.L': [  0.97059,   0.08115,   0.00000,   0.22667],
  'thumb.03.L': [  0.98857,   0.12261,  -0.00000,  -0.08772],
  'index.01.L': [  0.99805,  -0.05319,  -0.00000,  -0.03273],
  'index.02.L': [  0.98831,  -0.15039,   0.00000,   0.02510],
  'index.03.L': [  0.99750,  -0.06980,   0.00000,   0.01140],
  'middle.01.L': [  0.99244,  -0.12263,   0.00000,   0.00381],
  'middle.02.L': [  0.98831,  -0.15058,  -0.00000,   0.02403],
  'middle.03.L': [  0.98702,  -0.15924,  -0.00000,   0.02063],
  'ring.01.L': [  0.98545,  -0.16891,   0.00000,   0.01885],
  'ring.02.L': [  0.98329,  -0.18027,  -0.00000,   0.02557],
  'ring.03.L': [  0.98775,  -0.15425,   0.00000,   0.02343],
  'pinky.01.L': [  0.97511,  -0.21873,   0.00000,   0.03626],
  'pinky.02.L': [  0.98832,  -0.15031,   0.00000,   0.02514],
  'pinky.03.L': [  0.98952,  -0.14261,  -0.00000,   0.02246],
};

const HAND_FIST = {
  'thumb.01.R': [  0.97437,   0.22492,  -0.00000,  -0.00364],
  'thumb.02.R': [  0.93358,   0.31084,   0.00000,  -0.17835],
  'thumb.03.R': [  0.94552,   0.32486,  -0.00001,   0.02144],
  'index.01.R': [  0.77162,   0.63489,   0.02043,   0.03297],
  'index.02.R': [  0.77931,   0.62653,  -0.01152,   0.00007],
  'index.03.R': [  0.81940,   0.57318,  -0.00361,   0.00509],
  'middle.01.R': [  0.81534,   0.57877,   0.00078,   0.01546],
  'middle.02.R': [  0.77957,   0.62619,  -0.01169,   0.00511],
  'middle.03.R': [  0.86795,   0.49660,  -0.00661,   0.00391],
  'ring.01.R': [  0.84193,   0.53941,  -0.00541,   0.01250],
  'ring.02.R': [  0.79835,   0.60200,  -0.01171,   0.00967],
  'ring.03.R': [  0.86553,   0.50073,  -0.00603,   0.00941],
  'pinky.01.R': [  0.86870,   0.49508,  -0.01625,  -0.00038],
  'pinky.02.R': [  0.77954,   0.62615,  -0.01092,   0.01199],
  'pinky.03.R': [  0.85952,   0.51097,  -0.00679,   0.00923],
  'thumb.01.L': [  0.97437,   0.22485,   0.00000,   0.00673],
  'thumb.02.L': [  0.93358,   0.31685,   0.00000,   0.16743],
  'thumb.03.L': [  0.94552,   0.32443,   0.00000,  -0.02725],
  'index.01.L': [  0.77111,   0.63476,  -0.02316,  -0.04394],
  'index.02.L': [  0.77907,   0.62673,   0.01519,  -0.00477],
  'index.03.L': [  0.81932,   0.57323,   0.00588,  -0.00963],
  'middle.01.L': [  0.81459,   0.57955,  -0.00069,  -0.02357],
  'middle.02.L': [  0.77932,   0.62636,   0.01334,  -0.01242],
  'middle.03.L': [  0.86770,   0.49694,   0.00909,  -0.00802],
  'ring.01.L': [  0.84116,   0.54038,   0.00695,  -0.01974],
  'ring.02.L': [  0.79798,   0.60225,   0.01215,  -0.01917],
  'ring.03.L': [  0.86532,   0.50094,   0.00980,  -0.01331],
  'pinky.01.L': [  0.86770,   0.49675,   0.01737,  -0.00490],
  'pinky.02.L': [  0.77928,   0.62627,   0.01317,  -0.01815],
  'pinky.03.L': [  0.85932,   0.51119,   0.00984,  -0.01224],
};

/** slerp بين وضعيّتين ثمّ تحويلٌ إلى أويلر XYZ بترتيب three.js (R = Rx·Ry·Rz). */
function slerpToEulerDeg(a, b, t) {
  let [aw, ax, ay, az] = a;
  let [bw, bx, by, bz] = b;
  let dot = aw * bw + ax * bx + ay * by + az * bz;
  if (dot < 0) { bw = -bw; bx = -bx; by = -by; bz = -bz; dot = -dot; }

  let w, x, y, z;
  if (dot > 0.9995) {
    w = aw + (bw - aw) * t; x = ax + (bx - ax) * t;
    y = ay + (by - ay) * t; z = az + (bz - az) * t;
  } else {
    const th = Math.acos(Math.min(1, dot)), s = Math.sin(th);
    const k0 = Math.sin((1 - t) * th) / s, k1 = Math.sin(t * th) / s;
    w = aw * k0 + bw * k1; x = ax * k0 + bx * k1;
    y = ay * k0 + by * k1; z = az * k0 + bz * k1;
  }
  const n = Math.hypot(w, x, y, z) || 1;
  w /= n; x /= n; y /= n; z /= n;

  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const m00 = 1 - 2 * (yy + zz), m01 = 2 * (xy - wz), m02 = 2 * (xz + wy);
  const m11 = 1 - 2 * (xx + zz), m12 = 2 * (yz - wx);
  const m21 = 2 * (yz + wx),     m22 = 1 - 2 * (xx + yy);

  const D = 180 / Math.PI;
  const ey = Math.asin(Math.max(-1, Math.min(1, m02)));
  let ex, ez;
  if (Math.abs(m02) < 0.9999999) {
    ex = Math.atan2(-m12, m22);
    ez = Math.atan2(-m01, m00);
  } else {
    ex = Math.atan2(m21, m11);
    ez = 0;
  }
  return [ex * D, ey * D, ez * D];
}

/**
 * Build the finger-bone rotations for one hand from a per-finger curl amount.
 *
 * @param {Object} curls  e.g. { index: 0, middle: 1, thumb: 0.5 }
 *                        0 = كفّ مبسوط تماماً، 1 = مقبوض تماماً
 * @param {string} side   'L' or 'R'
 */
export function fingerPose(curls, side) {
  const pose = {};
  for (const finger of FINGERS) {
    const curl = Math.max(0, Math.min(1, curls[finger] ?? 0));
    for (let seg = 1; seg <= 3; seg++) {
      const name = BONES.finger(finger, seg, side);
      const o = HAND_OPEN[name], f = HAND_FIST[name];
      pose[name] = (o && f) ? slerpToEulerDeg(o, f, curl) : [0, 0, 0];
    }
  }
  return pose;
}

// ── Hand shape library ───────────────────────────────────────────────────────
// Curl amounts per finger: 0 = straight, 1 = fully closed.
export const HAND_SHAPES = {
  flat:    { thumb: 0.0, index: 0.0, middle: 0.0, ring: 0.0, pinky: 0.0 },
  fist:    { thumb: 0.9, index: 1.0, middle: 1.0, ring: 1.0, pinky: 1.0 },
  point:   { thumb: 0.8, index: 0.0, middle: 1.0, ring: 1.0, pinky: 1.0 },
  peace:   { thumb: 0.9, index: 0.0, middle: 0.0, ring: 1.0, pinky: 1.0 },
  thumbUp: { thumb: 0.0, index: 1.0, middle: 1.0, ring: 1.0, pinky: 1.0 },
  pinch:   { thumb: 0.5, index: 0.5, middle: 1.0, ring: 1.0, pinky: 1.0 },
  cup:     { thumb: 0.4, index: 0.4, middle: 0.4, ring: 0.4, pinky: 0.4 },
  open5:   { thumb: 0.1, index: 0.1, middle: 0.1, ring: 0.1, pinky: 0.1 },
};

/** Convenience: full finger rotation map for a named shape. */
export function handShape(shapeName, side) {
  const curls = HAND_SHAPES[shapeName];
  if (!curls) {
    console.warn(`[signing] unknown hand shape: ${shapeName}`);
    return {};
  }
  return fingerPose(curls, side);
}

// ── Arm positions ───────────────────────────────────────────────────
// أين تقف اليد في الفضاء. لغة الإشارة شديدة الاعتماد على الموضع (الصدر مقابل
// الوجه مقابل الفضاء المحايد)، فالتسمية بالموضع لا بالزاوية.
//
// من أين جاءت هذه الأرقام  —  حُلَّت من جديد على Tarjuman_Signer.glb
// ------------------------------------------------------------------
// الهيكل تبدّل: كان `TarjumanRobot2.glb` (‏57 مفصلاً، ارتفاع ~3.3 وحدة،
// وضعية ارتباط T-pose)، وصار `Tarjuman_Rig` داخل `Tarjuman_Signer.glb`
// (‏54 مفصلاً، ارتفاع 1.0 متر، ووضعية السكون ذراعان مسترخيتان إلى الجنبين).
// الأرقام القديمة كانت تخصّ الهيكل الأوّل، فكانت تُخرج الذراعين إلى الأمام
// في كل موضع وتقلب الكفّ — وهذا ما كان يُرى في الواجهة.
//
// الحلّ الجديد: قِيس الجسم من الشبكة نفسها (الصدر z≈0.74‑0.80، الذقن z≈0.86،
// الجبهة z≈0.94، مقدّمة الوجه y≈‑0.076)، ثمّ لكل موضع هدفٌ للرسغ أمام ذلك
// السطح، وحُلَّت ذراعٌ بعظمتين (IK) تصل إليه ومرفقٌ موجَّه إلى الأسفل والخلف،
// ثمّ حُوّل الحلّ إلى إزاحات أويلر (XYZ، درجات) عن وضعية السكون — وهي نفسها
// ما يضربه المشغِّل فوق `neutral`. خطأ الوصول 0.00 مم في المواضع العشرة.
//
// والرسغ حُلَّ معها: يُبنى إطارٌ متعامد من (اتّجاه الأصابع، عموديّ الكفّ)
// ويُقارَن بإطار السكون، فيخرج الكفّ متّجهاً إلى الأمام والأصابع إلى أعلى في
// كل موضع. وعموديّ الكفّ يُقاس من الهيكل نفسه: cross(index.01, pinky.01)،
// وهو على هذا الهيكل ظهرُ اليد في اليمنى والكفُّ في اليسرى — ولذلك تُقلب
// إشارته لليمنى، لا مرآةٌ حسابية.
//
// اليمين واليسار غير متماثلين عمداً: وضعية السكون نفسها غير متماثلة
// (Shoulder.L و Shoulder.R يحملان دورانين مختلفين)، فالمرآة كانت ستُنتج
// ذراعاً يسرى مزحزحة.
//
// ⚠️  إن استُبدل ملفّ الهيكل مرّة أخرى، أعِد الحلّ — هذه الأرقام تخصّ
//     `Tarjuman_Rig` وحده. ولوحة المعايرة تبقى الطريق لضبطها بالعين.
const ARM_SOLVED = {
  chest:    { R: { u: [  -45.9,   -60.6,   -45.2], l: [   78.0,     0.9,  -145.1], h: [ -108.9,    24.5,   173.5] },
              L: { u: [  -45.4,    68.6,    45.6], l: [ -101.1,     7.2,     1.6], h: [  -83.4,    22.0,    20.6] } },
  face:     { R: { u: [ -106.7,   -43.2,   -43.9], l: [   81.3,     3.2,  -145.2], h: [ -157.2,    12.0,  -129.7] },
              L: { u: [ -100.9,    42.9,    51.9], l: [   77.1,    -7.1,   178.2], h: [ -155.0,   -27.4,   144.7] } },
  forehead: { R: { u: [  109.6,   -83.4,  -107.3], l: [   95.7,    12.8,  -147.2], h: [ -117.6,   -17.9,   -45.7] },
              L: { u: [  179.0,    80.8,    44.6], l: [   91.7,    -7.4,  -179.9], h: [ -128.6,     7.9,    64.3] } },
  outward:  { R: { u: [  -64.7,   -55.5,   -25.4], l: [  -87.4,   -10.8,   -33.4], h: [  -71.7,    -9.7,    26.7] },
              L: { u: [  -64.2,    60.3,    29.1], l: [  -84.4,     7.3,    -0.6], h: [  -78.9,    12.2,   -12.2] } },
  side:     { R: { u: [    3.2,   -48.2,    38.4], l: [ -103.7,     0.3,   -34.9], h: [  -91.1,    -1.7,    33.4] },
              L: { u: [    8.9,    52.4,   -41.9], l: [  -99.2,     7.2,     1.3], h: [  -98.7,    -1.3,   -19.1] } },
};

// ملاحظة على لفّة الرسغ: 65٪ منها نُقلت إلى الساعد (دوران حول محوره الطولي،
// وهو ما يفعله الساعد الحقيقي: كبٌّ وبسط). الرسغ وحده كان يلتوي 78°–102°
// فينفتق الكُمّ عند المعصم وتبدو اليد مخلوعة. الموضع النهائي للكفّ لم يتغيّر.

/** يبني خريطة العظام لموضعٍ محلول. */
function solvedArm(name) {
  return (side) => {
    const a = ARM_SOLVED[name][side];
    return {
      // الكتف يبقى في وضعية السكون: تدويره يزحزح مفصل الكتف عن مكانه في الجسم.
      [BONES.shoulder(side)]: [0, 0, 0],
      [BONES.upperArm(side)]: a.u,
      [BONES.lowerArm(side)]: a.l,
      // الرسغ صار جزءاً من الحلّ. بدونه كانت لفّة الساعد وحدها تقرّر اتّجاه
      // الكفّ، فيخرج مقلوباً في أغلب المواضع.
      [BONES.hand(side)]: a.h,
    };
  };
}

export const ARM_POSITIONS = {
  // صفرٌ في كل شيء = وضعية السكون تماماً: ذراعٌ مسترخية إلى جانب الجسم.
  // والرسغ مذكورٌ هنا أيضاً: بدونه يبقى الكفّ على دوران المفتاح السابق
  // فتعود الذراع إلى جنبها والكفّ ملتوٍ.
  rest: (s) => ({
    [BONES.shoulder(s)]: [0, 0, 0],
    [BONES.upperArm(s)]: [0, 0, 0],
    [BONES.lowerArm(s)]: [0, 0, 0],
    [BONES.hand(s)]:     [0, 0, 0],
  }),
  chest:    solvedArm('chest'),
  face:     solvedArm('face'),
  forehead: solvedArm('forehead'),
  outward:  solvedArm('outward'),
  side:     solvedArm('side'),
};

/**
 * Compose one full keyframe pose.
 *
 * @param {Object} spec
 *   arm   : { R: 'chest', L: 'rest' }      — position name per side
 *   hand  : { R: 'flat',  L: 'fist' }      — hand shape per side
 *   head  : [x, y, z]                      — optional head tilt
 *   extra : { 'BoneName': [x,y,z] }        — manual overrides, applied last
 */
export function composePose({ arm = {}, hand = {}, head, extra = {} } = {}) {
  let pose = {};

  for (const side of SIDES) {
    const position = arm[side];
    if (position && ARM_POSITIONS[position]) {
      pose = { ...pose, ...ARM_POSITIONS[position](side) };
    }
    const shape = hand[side];
    if (shape) pose = { ...pose, ...handShape(shape, side) };
  }

  if (head) pose[BONES.head] = head;

  return { ...pose, ...extra };
}
