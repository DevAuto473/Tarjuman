import { Camera, CameraOff } from 'lucide-react';

/**
 * CameraToggle — الفعل الأساسي الوحيد في الواجهة.
 *
 * كانت ستّة أزرار بستّة ألوان مكدّسة بعرض واحد، فلم يبدُ أيٌّ منها أهمّ من
 * الآخر. هذا الزرّ وحده يحمل لون التمييز بملئه، وارتفاعه 56px بينما كل ما
 * سواه 40px. التسلسل يصنعه الامتلاء والارتفاع معاً، لا اللون وحده.
 */
export default function CameraToggle({ on, disabled, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full h-11 sm:h-12 rounded-ctl font-display text-base sm:text-h2 shadow-card shrink-0
                  flex items-center justify-center gap-2 transition cursor-pointer
                  disabled:opacity-40 disabled:cursor-not-allowed
        ${on ? 'bg-live text-white hover:brightness-[1.08]'
             : 'bg-brand text-white hover:bg-brand-hi'}`}
    >
      {on ? <CameraOff className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={1.9} />
          : <Camera className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={1.9} />}
      {on ? 'إيقاف الكاميرا' : 'تشغيل الكاميرا'}
    </button>
  );
}
