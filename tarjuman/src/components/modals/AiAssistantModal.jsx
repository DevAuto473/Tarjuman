import { useEffect, useRef, useState } from 'react';
import { Sparkles, Send, Mic, MicOff } from 'lucide-react';
import Modal from '../Modal';

/**
 * AiAssistantModal — محادثة نصّية أو صوتية مع المساعد.
 *
 * ردّ المساعد يُؤدَّى إشارةً على الروبوت أيضاً (يتولّى ذلك App)، فالمستخدم
 * الأصمّ يقرأ الجواب بطريقتين لا بواحدة.
 */
export default function AiAssistantModal({ history, thinking, onSend, recorder, onClose }) {
  const [input, setInput] = useState('');
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [history, thinking]);

  const submit = () => {
    const text = input.trim();
    if (!text || thinking) return;
    onSend(text);
    setInput('');
  };

  return (
    <Modal title="المساعد الذكي" icon={Sparkles} onClose={onClose} size="md">
      <div ref={scrollRef} className="h-72 overflow-y-auto scroll-thin flex flex-col gap-2">
        {history.length === 0 && !thinking && (
          <div className="m-auto text-center px-4">
            <p className="text-base text-ink-2">اسأل ترجمان عن أيّ شيء</p>
            <p className="text-sm text-ink-3 mt-1 text-pretty">
              نصّاً أو بصوتك — والجواب يصلك مكتوباً ومؤدّى بالإشارة
            </p>
          </div>
        )}

        {history.map((msg, idx) => (
          <div
            key={idx}
            className={`max-w-[85%] px-3 py-2 rounded-ctl text-base text-pretty
              ${msg.role === 'user'
                ? 'self-start bg-brand text-white'
                : 'self-end bg-sunken border border-hair text-ink'}`}
          >
            {msg.content}
          </div>
        ))}

        {thinking && (
          <div className="self-end flex items-center gap-2 px-3 py-2 rounded-ctl
                          bg-sunken border border-hair text-sm text-ink-3">
            <span className="flex gap-1" aria-hidden="true">
              <span className="size-1.5 rounded-full bg-ink-3 animate-pulse" />
              <span className="size-1.5 rounded-full bg-ink-3 animate-pulse [animation-delay:150ms]" />
              <span className="size-1.5 rounded-full bg-ink-3 animate-pulse [animation-delay:300ms]" />
            </span>
            يفكّر…
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 pt-4 border-t border-hair">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
          placeholder="اكتب سؤالك…"
          aria-label="سؤالك للمساعد"
          className="flex-1 min-w-0 h-10 bg-sunken border border-hair rounded-ctl px-3
                     text-base text-ink placeholder:text-ink-3/70
                     focus:outline-none focus:border-brand focus:bg-surface transition"
        />

        <button
          onClick={recorder.toggle}
          disabled={thinking}
          title={recorder.recording ? 'إيقاف التسجيل' : 'اسأل بصوتك'}
          aria-label={recorder.recording ? 'إيقاف التسجيل' : 'اسأل بصوتك'}
          className={`size-10 shrink-0 rounded-ctl border transition cursor-pointer
                      flex items-center justify-center disabled:opacity-40
            ${recorder.recording ? 'bg-live-wash border-live/30 text-live'
                                 : 'bg-surface border-hair text-ink-2 hover:text-ink hover:border-hair-2'}`}
        >
          {recorder.recording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
        </button>

        <button
          onClick={submit}
          disabled={!input.trim() || thinking}
          title="إرسال"
          aria-label="إرسال"
          className="size-10 shrink-0 rounded-ctl bg-brand text-white hover:bg-brand-hi
                     disabled:opacity-40 disabled:cursor-not-allowed transition cursor-pointer
                     flex items-center justify-center"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>

      {recorder.error && <p className="text-sm text-live text-center">{recorder.error}</p>}
    </Modal>
  );
}
