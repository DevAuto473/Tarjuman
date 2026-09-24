import { Camera, SlidersHorizontal, Hand } from 'lucide-react';

/**
 * TopBar — الهويّة يميناً، وحالة الاتصال والأدوات العامّة يساراً.
 *
 * كل ما في الصفّ بارتفاع 40px: العلامة، والشارة، والزرّان. صفٌّ واحد بارتفاع
 * واحد يستقرّ في العين؛ والارتفاعات المتقاربة المختلفة هي ما يجعل الشريط
 * يبدو مهتزّاً بلا سبب ظاهر.
 *
 * وكان في أسفل عمود التحكّم ثلاث أيقونات (ضبط، خروج، إعدادات) بلا onClick:
 * تتلوّن عند المرور فتبدو قابلة للنقر ولا تفعل شيئاً. اثنتان هنا موصولتان
 * بوظيفة، و«الخروج» حُذفت — لا حسابات في التطبيق فلا شيء يُخرَج منه.
 */
export default function TopBar({ connected, onOpenCameras, onOpenCalibration }) {
  return (
    <header className="flex items-center justify-between gap-4 shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <span className="size-10 shrink-0 rounded-ctl bg-brand flex items-center justify-center">
          <Hand className="w-5 h-5 text-white" strokeWidth={1.9} />
        </span>
        <span className="min-w-0">
          <h1 className="font-display text-h1 text-ink">ترجمان</h1>
          <p className="text-xs text-ink-3 truncate">مترجم لغة الإشارة العربية</p>
        </span>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {/* على الشاشات الضيّقة تبقى النقطة وحدها: حالة الخادم معلومة لا يصحّ
            إخفاؤها، والعنوان وحده هو ما لا يتّسع له المكان. */}
        <span
          title={connected ? 'الخادم متّصل' : 'بانتظار الخادم'}
          className={`flex items-center gap-2 h-10 px-3 rounded-ctl border text-sm font-medium
            ${connected
              ? 'bg-good-wash border-good/25 text-good'
              : 'bg-caution-wash border-caution/25 text-caution'}`}
        >
          <span className={`size-2 rounded-full ${connected ? 'bg-good' : 'bg-caution'}`} />
          <span className="hidden sm:inline">
            {connected ? 'الخادم متّصل' : 'بانتظار الخادم…'}
          </span>
        </span>

        <button
          onClick={onOpenCameras}
          title="اختيار الكاميرا"
          aria-label="اختيار الكاميرا"
          className="size-10 rounded-ctl border border-hair bg-surface text-ink-2
                     hover:text-ink hover:border-hair-2 transition cursor-pointer
                     flex items-center justify-center"
        >
          <Camera className="w-4 h-4" strokeWidth={1.9} />
        </button>

        <button
          onClick={onOpenCalibration}
          title="معايرة حركة الروبوت"
          aria-label="معايرة حركة الروبوت"
          className="size-10 rounded-ctl border border-hair bg-surface text-ink-2
                     hover:text-ink hover:border-hair-2 transition cursor-pointer
                     flex items-center justify-center"
        >
          <SlidersHorizontal className="w-4 h-4" strokeWidth={1.9} />
        </button>
      </div>
    </header>
  );
}
