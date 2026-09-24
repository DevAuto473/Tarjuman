import { useState } from 'react';
import { Hand } from 'lucide-react';

/**
 * SignComposer — اكتب كلمة فيؤدّيها الروبوت.
 *
 * الاتجاه المعاكس للترجمة، والنصف الذي يسهل إغفاله: بدونه يقرأ السامع فقط
 * ولا يردّ. الاختصارات أسفله ليست زينة — هي توثيقٌ حيٌّ لما يعرفه القاموس،
 * فلا يملك المستخدم طريقة أخرى لمعرفة الكلمات المتاحة.
 */
export default function SignComposer({ onSign, quickWords = [] }) {
  const [value, setValue] = useState('');

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    onSign(text);
    setValue('');
  };

  return (
    <section className="bg-surface border border-hair rounded-card shadow-card p-2.5 sm:p-3 shrink-0">
      <p className="text-2xs font-medium text-ink-3 mb-1.5">اكتب ليؤدّيها الروبوت</p>

      <div className="flex items-center gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
          placeholder="مثال: أهلاً"
          aria-label="نصّ ليؤدّيه الروبوت إشارةً"
          className="flex-1 min-w-0 h-9 bg-sunken border border-hair rounded-ctl px-2.5
                     text-sm text-ink placeholder:text-ink-3/70
                     focus:outline-none focus:border-brand focus:bg-surface transition"
        />
        <button
          onClick={submit}
          disabled={!value.trim()}
          title="أدِّ الإشارة"
          aria-label="أدِّ الإشارة"
          className="size-9 shrink-0 rounded-ctl bg-brand text-white hover:bg-brand-hi
                     disabled:opacity-40 disabled:cursor-not-allowed transition cursor-pointer
                     flex items-center justify-center"
        >
          <Hand className="w-4 h-4" strokeWidth={1.9} />
        </button>
      </div>

      {quickWords.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {quickWords.map((w) => (
            <button
              key={w}
              onClick={() => onSign(w)}
              className="h-7 px-2.5 rounded-chip border border-hair bg-sunken text-xs text-ink-2
                         hover:bg-brand-wash hover:border-brand/30 hover:text-brand
                         transition cursor-pointer"
            >
              {w}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
