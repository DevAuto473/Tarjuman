import { GraduationCap, Play, ArrowRight, CameraOff } from 'lucide-react';
import Modal from '../Modal';

const QUALITY = {
  great: { text: 'text-good',    box: 'bg-good-wash border-good/25' },
  good:  { text: 'text-good',    box: 'bg-good-wash border-good/25' },
  fair:  { text: 'text-caution', box: 'bg-caution-wash border-caution/25' },
  poor:  { text: 'text-live',    box: 'bg-live-wash border-live/25' },
};

/**
 * LearnModal — «تعلّم مع ترجمان».
 *
 * كانت هذه النافذة فقرةَ نصّ وزرَّ إغلاق، بينما الخادم ينفّذ وضع التدريب
 * كاملاً منذ البداية: `start_practice` يضبط الهدف، ومطابقة DTW تقارن أداءك
 * بالتسجيلة المرجعية وتعيد درجة. هنا وُصِل النصفان أخيراً.
 *
 * حلقة التدريب: اختر كلمة ← شاهد الروبوت يؤدّيها ← أدِّها أمام الكاميرا ←
 * اقرأ درجتك ← أعِد. ولا حاجة إلى تدريب جديد للنموذج: المرجع هو تسجيلاتك.
 */
export default function LearnModal({
  signs, cameraOn, capturing, target, result,
  onStart, onStop, onPlay, onClose,
}) {
  const close = () => { onStop(); onClose(); };
  const current = signs.find((s) => s.id === target);
  const quality = QUALITY[result?.quality];

  return (
    <Modal title="تعلّم مع ترجمان" icon={GraduationCap} onClose={close} size="md">
      {!cameraOn && (
        <div className="flex items-start gap-2.5 p-3 rounded-ctl bg-caution-wash border border-caution/25">
          <CameraOff className="w-4 h-4 text-caution shrink-0 mt-1" />
          <p className="text-sm text-caution text-pretty">
            شغِّل الكاميرا أوّلاً — التدريب يقارن حركتك بالتسجيلة المرجعية،
            ولا يستطيع ذلك وهو لا يراك.
          </p>
        </div>
      )}

      {!target ? (
        /* ── اختيار الكلمة ─────────────────────────────────────────────── */
        signs.length === 0 ? (
          <div className="text-center py-10">
            <p className="text-base text-ink-2">لا توجد إشارات للتدرّب عليها بعد</p>
            <p className="text-sm text-ink-3 mt-2 text-pretty">
              سجّل إشارات بـ <code className="text-brand font-mono text-sm">npm run collect</code>،
              ثمّ صدّرها بـ <code className="text-brand font-mono text-sm">npm run export3d</code>
            </p>
          </div>
        ) : (
          <>
            <p className="text-sm text-ink-2 text-pretty">
              اختر كلمة لتتدرّب عليها. يؤدّيها الروبوت أوّلاً، ثمّ تؤدّيها أنت أمام
              الكاميرا ويعطيك ترجمان درجةَ تطابق حركتك مع المرجع.
            </p>
            <div className="grid gap-2 max-h-[42dvh] overflow-y-auto scroll-thin
                            grid-cols-[repeat(auto-fill,minmax(min(100%,9rem),1fr))]">
              {signs.map((s) => (
                <button
                  key={s.id}
                  onClick={() => onStart(s.id)}
                  disabled={!cameraOn}
                  className="text-right p-3 rounded-ctl border border-hair bg-surface
                             hover:bg-sunken hover:border-hair-2 transition cursor-pointer
                             disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <span className="block text-base font-medium text-ink truncate">{s.label}</span>
                  <span className="block text-2xs text-ink-3 font-mono truncate mt-0.5">{s.id}</span>
                </button>
              ))}
            </div>
          </>
        )
      ) : (
        /* ── التدريب على كلمة ──────────────────────────────────────────── */
        <>
          <div className="flex items-center justify-between gap-3 p-4 rounded-ctl
                          bg-brand-wash border border-brand/25">
            <span className="min-w-0">
              <span className="block text-xs text-brand/70">تتدرّب الآن على</span>
              <span className="block font-display text-h1 text-brand truncate mt-0.5">
                {current?.label || target}
              </span>
            </span>
            {current && (
              <button
                onClick={() => onPlay(current)}
                className="h-10 px-3 shrink-0 rounded-ctl bg-brand text-white text-sm font-medium
                           hover:bg-brand-hi transition cursor-pointer flex items-center gap-2"
              >
                <Play className="w-4 h-4" />
                شاهد الإشارة
              </button>
            )}
          </div>

          {/* حالة الالتقاط — يرى المستخدم أنّ محاولته تُسجَّل الآن */}
          <div
            className={`flex items-center gap-2.5 h-10 px-3 rounded-ctl border text-sm font-medium
              ${capturing ? 'bg-live-wash border-live/25 text-live'
                          : 'bg-sunken border-hair text-ink-3'}`}
            aria-live="polite"
          >
            <span className={`size-2 rounded-full ${capturing ? 'bg-live animate-pulse' : 'bg-ink-3/50'}`} />
            {capturing ? 'أسجّل محاولتك…' : 'أدِّ الإشارة أمام الكاميرا'}
          </div>

          {result && (
            result.error ? (
              <p className="text-sm text-caution text-pretty p-3 rounded-ctl
                            bg-caution-wash border border-caution/25">
                {result.message || 'لا يوجد مرجع مسجَّل لهذه الإشارة.'}
              </p>
            ) : (
              <div className={`flex items-center gap-4 p-4 rounded-ctl border
                               ${quality?.box || 'bg-sunken border-hair'}`}>
                <span className={`tnum font-display text-h1 leading-none shrink-0
                                  ${quality?.text || 'text-ink'}`}>
                  {Math.round(result.score)}٪
                </span>
                <span className={`text-base text-pretty ${quality?.text || 'text-ink'}`}>
                  {result.verdict}
                </span>
              </div>
            )
          )}

          <button
            onClick={onStop}
            className="h-10 rounded-ctl border border-hair bg-surface text-base font-medium text-ink-2
                       hover:bg-sunken hover:text-ink transition cursor-pointer
                       flex items-center justify-center gap-2"
          >
            <ArrowRight className="w-4 h-4" />
            اختر كلمة أخرى
          </button>
        </>
      )}
    </Modal>
  );
}
