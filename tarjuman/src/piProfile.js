/**
 * piProfile.js — مِلَفُّ الرسم على الرازبيري باي.
 *
 * لماذا متغيّرُ بيئةٍ لا استنتاجٌ من وكيل المستخدم (user agent):
 * نفسُ الباي يُستعمل حاسوبَ تطويرٍ عاديّاً أثناء العمل، فالاستنتاجُ من
 * المتصفّح يخفض الجودةَ في الحالتين ولا يُطفأ حين نريده مطفأً. المتغيّر
 * صريحٌ ويُقرأ وقت البناء:
 *
 *     VITE_TARJUMAN_PI=1 npm run build        ← بناءٌ للباي
 *     npm run dev                             ← تطويرٌ عاديّ
 *
 * ويمكن تجاوزُه وقتَ التشغيل للقياس، بلا إعادة بناء:
 *     http://localhost:5173/?pi=1   أو   ?pi=0
 *
 * ما الذي يغيّره (كلُّ رقمٍ مبرَّرٌ لا مُخمَّن):
 *   dpr = 1           شاشةُ الباي غالباً dpr = 1 أصلاً، لكنّ سقف R3F
 *                     الافتراضيّ [1, 2] يضاعف البكسلات أربع مرّاتٍ على أيّ
 *                     شاشةٍ عالية الكثافة تُوصَل بها — وهو أغلى تغييرٍ منفرد.
 *   antialias = false تنعيمُ الحواف يكلّف عيّناتٍ إضافيّةً لكلّ بكسل.
 *   powerPreference   'high-performance' يطلب معالجَ رسمٍ منفصلاً؛ لا وجود
 *     = 'default'     له في VideoCore، فالطلبُ بلا معنىً وقد يُربك السائق.
 *   30 إطاراً/ثانية   الأفاتار يوقّع بيده، ولا تحتاج العينُ ستّين. وكلُّ
 *                     إطارٍ موفَّرٍ نواةٌ تعود إلى MediaPipe — والمعالجُ هو
 *                     العنق لا الذاكرة.
 *
 * ما لا يغيّره: الظلالُ ومعالجةُ ما بعد الرسم وخرائطُ البيئة والظلالُ
 * اللامسة — لأنّ أيّاً منها غيرُ مستعملٍ في المشهد أصلاً (قُرئ في
 * RobotStage.jsx: لا `shadows` على <Canvas>، ولا <Environment>، ولا
 * <ContactShadows>). إطفاءُ ما هو مطفأٌ تعقيدٌ ميّت.
 */

const buildFlag = import.meta.env.VITE_TARJUMAN_PI === '1';

function runtimeOverride() {
  if (typeof window === 'undefined') return null;
  const v = new URLSearchParams(window.location.search).get('pi');
  if (v === '1') return true;
  if (v === '0') return false;
  return null;
}

/** هل نحن في ملفّ الباي؟ */
export const PI = runtimeOverride() ?? buildFlag;

/** سقفُ كثافة البكسل الممرَّر إلى <Canvas dpr=…>. */
export const DPR = PI ? 1 : [1, 2];

/** خصائصُ WebGLRenderer الممرَّرة إلى <Canvas gl=…>. */
export const GL = PI
  ? { antialias: false, powerPreference: 'default' }
  : { antialias: true, powerPreference: 'high-performance' };

/** سقفُ الإطارات، أو 0 لبلا سقف. */
export const MAX_FPS = PI ? 30 : 0;

/** مقاسُ الالتقاط ومعدّلُه اللذان تُبلَّغ بهما البايثون. */
export const CAPTURE = PI
  ? { width: 640, height: 480, fps: 15 }
  : { width: 640, height: 480, fps: 30 };

if (typeof window !== 'undefined') {
  // سطرٌ واحدٌ في الطرفيّة يحسم «أيُّ ملفٍّ يعمل الآن؟» عند القياس على المسرح.
  console.log(
    `[pi] الملفّ: ${PI ? 'رازبيري باي' : 'مكتبيّ'} — dpr=${JSON.stringify(DPR)}`
    + ` antialias=${GL.antialias} سقف=${MAX_FPS || 'بلا'}`
  );
}
