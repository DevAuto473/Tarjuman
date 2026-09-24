import { useState } from 'react';
import { Hand, Play, SlidersHorizontal } from 'lucide-react';
import Modal from '../Modal';

/**
 * TrainedSignsModal — كل ما عُلِّم للمتعرِّف، قابلاً للتشغيل على الروبوت.
 *
 * تأتي هذه الإشارات من `npm run export3d`: مشتقّة من تسجيلاتك أنت، فالروبوت
 * يؤدّيها كما أدّيتها لا بزوايا مكتوبة يدوياً.
 */
export default function TrainedSignsModal({ signs, onPlay, onPlayAll, onCalibrate, onClose }) {
  const [playingId, setPlayingId] = useState(null);

  const play = (s) => {
    onPlay(s);
    setPlayingId(s.id);
    setTimeout(() => setPlayingId(null), (s.duration || 1.4) * 1000);
  };

  const empty = signs.length === 0;

  return (
    <Modal
      title={`الإشارات المدرَّبة (${signs.length})`}
      icon={Hand}
      onClose={onClose}
      size="lg"
      footer={empty ? null : (
        <>
          <button
            onClick={onPlayAll}
            className="flex-1 h-10 rounded-ctl bg-brand text-white text-base font-medium
                       hover:bg-brand-hi transition cursor-pointer
                       flex items-center justify-center gap-2"
          >
            <Play className="w-4 h-4" />
            تشغيل الكلّ بالتتابع
          </button>
          <button
            onClick={onCalibrate}
            title="الأوضاع تبدو خاطئة؟"
            className="h-10 px-4 rounded-ctl border border-hair bg-surface text-base text-ink-2
                       hover:bg-sunken hover:text-ink transition cursor-pointer
                       flex items-center justify-center gap-2"
          >
            <SlidersHorizontal className="w-4 h-4" />
            معايرة
          </button>
        </>
      )}
    >
      {empty ? (
        <div className="text-center py-10">
          <p className="text-base text-ink-2">لا توجد إشارات مصدَّرة بعد</p>
          <p className="text-sm text-ink-3 mt-2 text-pretty">
            سجّل إشارات بـ <code className="text-brand font-mono text-sm">npm run collect</code>
            <br />
            ثمّ صدّرها للروبوت بـ <code className="text-brand font-mono text-sm">npm run export3d</code>
          </p>
        </div>
      ) : (
        <>
          <p className="text-sm text-ink-2 text-pretty">
            هذه الإشارات مشتقّة من تسجيلاتك أنت — الروبوت يؤدّيها كما أدّيتها.
            اضغط أيّ إشارة ليؤدّيها.
          </p>

          <div className="grid gap-2 max-h-[46dvh] overflow-y-auto scroll-thin
                          grid-cols-[repeat(auto-fill,minmax(min(100%,9rem),1fr))]">
            {signs.map((s) => (
              <button
                key={s.id}
                onClick={() => play(s)}
                className={`text-right p-3 rounded-ctl border transition cursor-pointer
                  ${playingId === s.id ? 'bg-brand-wash border-brand/35'
                                       : 'bg-surface border-hair hover:bg-sunken hover:border-hair-2'}`}
              >
                <span className={`block text-base font-medium truncate
                  ${playingId === s.id ? 'text-brand' : 'text-ink'}`}>
                  {s.label}
                </span>
                <span className="flex justify-between gap-2 text-2xs text-ink-3 mt-0.5">
                  <span className="truncate font-mono">{s.id}</span>
                  <span className="tnum">{Number(s.duration).toFixed(1)}s</span>
                </span>
              </button>
            ))}
          </div>
        </>
      )}
    </Modal>
  );
}
