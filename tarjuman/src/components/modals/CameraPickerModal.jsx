import { Camera, RefreshCw, Check } from 'lucide-react';
import Modal from '../Modal';

const KIND_LABEL = { integrated: 'مدمجة', usb: 'USB', network: 'شبكة', auto: 'تلقائي' };
const KIND_STYLE = {
  integrated: 'bg-brand-wash text-brand border-brand/20',
  usb: 'bg-caution-wash text-caution border-caution/20',
  network: 'bg-sunken text-ink-3 border-hair',
  auto: 'bg-sunken text-ink-3 border-hair',
};

/**
 * CameraPickerModal — أيّ جهاز يفتحه الخادم.
 *
 * لا يظهر طلب إذن من المتصفِّح للفيديو لأن الكاميرا يفتحها بايثون لا الصفحة.
 * أمّا الميكروفون فإذنٌ متصفِّحٌ حقيقي، ولذلك يُعامَل هنا معاملة مختلفة:
 * صفحةٌ مُنِعت مرّة لا تستطيع رفع المنع عن نفسها، فنقول أين الزرّ بدل
 * التظاهر بأننا نستطيع.
 */
export default function CameraPickerModal({
  cameras, micPermission, onSelect, onRescan, onRequestMic, onClose,
}) {
  const { list, scanning, active } = cameras;

  return (
    <Modal title="اختيار الكاميرا" icon={Camera} onClose={onClose} size="md">
      <p className="text-sm text-ink-2 text-pretty">
        الكاميرا يفتحها الخادم (بايثون) لا المتصفِّح، فلا يظهر طلب إذن للفيديو.
        اختر الجهاز وسيُعاد فتحه فوراً.
      </p>

      {scanning ? (
        <div className="flex items-center justify-center gap-2.5 py-10 text-base text-ink-3">
          <RefreshCw className="w-4 h-4 animate-spin" />
          جارٍ فحص الأجهزة المتّصلة…
        </div>
      ) : list.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-base text-ink-2">لم يستجب أيُّ جهاز كاميرا</p>
          <p className="text-sm text-ink-3 mt-1.5 text-pretty">
            تأكّد أنّ الكاميرا موصولة، وأغلِق أيّ برنامج آخر يستعملها
            (Zoom، Teams، Camera، OBS).
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2 max-h-[40dvh] overflow-y-auto scroll-thin">
          {list.map((c) => {
            const isActive = String(active) === String(c.source);
            return (
              <button
                key={c.source}
                onClick={() => onSelect(c.source)}
                className={`text-right p-3 rounded-ctl border transition cursor-pointer
                            flex items-center justify-between gap-3
                  ${isActive ? 'bg-brand-wash border-brand/35'
                             : 'bg-surface border-hair hover:bg-sunken hover:border-hair-2'}`}
              >
                <span className="min-w-0">
                  <span className="flex items-center gap-2">
                    <span className={`text-base font-medium truncate ${isActive ? 'text-brand' : 'text-ink'}`}>
                      {c.label}
                    </span>
                    <span className={`text-2xs h-5 px-1.5 rounded-full border flex items-center
                                      ${KIND_STYLE[c.kind] || KIND_STYLE.auto}`}>
                      {KIND_LABEL[c.kind] || 'تلقائي'}
                    </span>
                  </span>
                  <span className="block text-2xs text-ink-3 truncate mt-0.5">{c.detail}</span>
                </span>
                {isActive && <Check className="w-4 h-4 shrink-0 text-brand" />}
              </button>
            );
          })}
        </div>
      )}

      <button
        onClick={onRescan}
        disabled={scanning}
        className="h-10 rounded-ctl border border-hair bg-surface text-base font-medium text-ink-2
                   hover:bg-sunken hover:text-ink transition cursor-pointer
                   flex items-center justify-center gap-2 disabled:opacity-40"
      >
        <RefreshCw className={`w-4 h-4 ${scanning ? 'animate-spin' : ''}`} />
        إعادة الفحص
      </button>

      <div className="border-t border-hair pt-4 flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3 min-h-6">
          <span className="text-sm text-ink-2">إذن الميكروفون (المتصفِّح)</span>
          <span
            className={`text-2xs h-6 px-2 rounded-chip border flex items-center font-medium
              ${micPermission === 'granted' ? 'bg-good-wash text-good border-good/25'
                : micPermission === 'denied' ? 'bg-live-wash text-live border-live/25'
                  : 'bg-sunken text-ink-3 border-hair'}`}
          >
            {micPermission === 'granted' ? 'مسموح'
              : micPermission === 'denied' ? 'محظور'
                : micPermission === 'prompt' ? 'سيُسأل' : 'غير معروف'}
          </span>
        </div>

        {micPermission === 'denied' ? (
          <p className="text-sm text-ink-3 text-pretty">
            لإعادة السماح: اضغط أيقونة القفل بجانب العنوان أعلى المتصفِّح ← الأذونات ←
            الميكروفون ← «سؤال» أو «سماح»، ثمّ أعِد تحميل الصفحة. الصفحة لا تستطيع
            رفع الحظر عن نفسها.
          </p>
        ) : (
          <button
            onClick={onRequestMic}
            className="h-10 rounded-ctl border border-hair bg-surface text-base text-ink-2
                       hover:bg-sunken hover:text-ink transition cursor-pointer"
          >
            طلب إذن الميكروفون
          </button>
        )}
      </div>
    </Modal>
  );
}
