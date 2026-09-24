import { Volume2, Mic, MicOff, Delete, Eraser } from 'lucide-react';

function Action({ onClick, disabled, title, active, children }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      className={`size-10 rounded-ctl border transition cursor-pointer
                  flex items-center justify-center
                  disabled:opacity-40 disabled:cursor-not-allowed
        ${active ? 'bg-live-wash border-live/30 text-live'
                 : 'bg-surface border-hair text-ink-2 hover:text-ink hover:border-hair-2'}`}
    >
      {children}
    </button>
  );
}

/**
 * TranscriptBar — النصّ الناتج عن الترجمة، وما يُفعَل به.
 *
 * يجتمع هنا اتجاها الترجمة: ما يشير به الأصمّ يُكتب هنا ليقرأه السامع، وما
 * ينطقه السامع يُفرَّغ هنا أيضاً ثمّ يؤدّيه الروبوت. لذلك زرّ الميكروفون
 * جزءٌ من هذا الصندوق لا من مكان آخر.
 *
 * زرّ «مسافة» ليس زينة: مُخرَج التعرّف كلماتٌ متتابعة، فيحتاج المستخدم أن
 * يفصل الجمل بنفسه.
 */
export default function TranscriptBar({ text, onChange, onSpeak, recorder }) {
  const empty = !text.trim();

  return (
    <section className="bg-surface border border-hair rounded-card shadow-card shrink-0">
      <div className="px-3.5 py-2">
        <p className="text-2xs font-medium text-ink-3 mb-0.5">النصّ المترجَم</p>
        <p
          className={`font-display text-h2 sm:text-hero text-pretty break-words min-h-[1.3em] line-clamp-2
            ${empty ? 'text-ink-3/60' : 'text-ink'}`}
          aria-live="polite"
        >
          {empty ? 'سيظهر هنا ما تترجمه الكاميرا أو الميكروفون…' : text}
        </p>
      </div>

      <div className="flex items-center gap-1.5 px-3 py-1.5 border-t border-hair">
        <Action onClick={onSpeak} disabled={empty} title="انطق النصّ بصوت مسموع">
          <Volume2 className="w-4 h-4" />
        </Action>

        <Action
          onClick={recorder.toggle}
          active={recorder.recording}
          title={recorder.recording ? 'إيقاف التسجيل' : 'تحدَّث لتحويل صوتك إلى نصّ وإشارة'}
        >
          {recorder.recording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
        </Action>

        <span className="w-px h-5 bg-hair mx-0.5" aria-hidden="true" />

        <Action onClick={() => onChange(text + ' ')} title="أضِف مسافة">
          <span className="w-3.5 h-0.5 bg-current rounded-full" />
        </Action>

        <Action onClick={() => onChange(text.slice(0, -1))} disabled={empty} title="احذف آخر حرف">
          <Delete className="w-4 h-4" />
        </Action>

        <Action onClick={() => onChange('')} disabled={empty} title="امسح الكلّ">
          <Eraser className="w-4 h-4" />
        </Action>

        {recorder.recording && (
          <span className="ms-auto flex items-center gap-1.5 text-xs text-live font-medium">
            <span className="size-2 rounded-full bg-live animate-pulse" />
            أستمع…
          </span>
        )}
      </div>
    </section>
  );
}
