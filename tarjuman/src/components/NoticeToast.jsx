import { AlertTriangle, Info } from 'lucide-react';

/**
 * NoticeToast — رسالة عابرة: إشارة لم تُفهَم، كاميرا مشغولة، خطأ خادم.
 *
 * موضعها أسفل وسط الشاشة لا داخل عمود التحكّم: المستخدم ينظر إلى المسرح وهو
 * يشير، لا إلى الأزرار. ورسالة لا يراها أحد لا وجود لها.
 */
export default function NoticeToast({ notice }) {
  if (!notice) return null;
  const isError = notice.kind === 'error';

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed inset-x-0 bottom-6 z-40 flex justify-center px-4 pointer-events-none"
      style={{ bottom: 'max(1.5rem, env(safe-area-inset-bottom))' }}
    >
      <div
        className={`flex items-center gap-2.5 h-10 px-4 rounded-ctl border shadow-pop
                    text-base font-medium max-w-full
          ${isError ? 'bg-live-wash border-live/30 text-live'
                    : 'bg-caution-wash border-caution/30 text-caution'}`}
      >
        {isError ? <AlertTriangle className="w-4 h-4 shrink-0" />
                 : <Info className="w-4 h-4 shrink-0" />}
        <span className="truncate">{notice.text}</span>
      </div>
    </div>
  );
}
