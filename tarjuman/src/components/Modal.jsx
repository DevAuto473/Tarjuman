import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

/**
 * Modal — القشرة المشتركة لكل النوافذ.
 *
 * كانت كل نافذة تعيد كتابة الخلفية والترويسة وزرّ الإغلاق بنفسها، فاختلفت
 * الأربع في المسافات والأحجام. هنا تُكتب مرّة واحدة، فتصير النوافذ متّسقة
 * بالضرورة لا بالانضباط: ترويسة 56px، وحشوة 16px، ونفس نصف القطر.
 *
 * وتُضيف ما كان ناقصاً في كلّها: الإغلاق بـEsc، ونقل التركيز عند الفتح،
 * وسِمات ARIA حتّى يعرف قارئ الشاشة أنّ هذه نافذة حوار.
 */
export default function Modal({ title, icon: Icon, onClose, children, size = 'md', footer }) {
  const panelRef = useRef(null);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    panelRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const widths = { sm: 'max-w-sm', md: 'max-w-lg', lg: 'max-w-2xl' };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/25 backdrop-blur-[3px]"
      dir="rtl"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        ref={panelRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`cq w-full ${widths[size]} bg-surface border border-hair rounded-card
                    shadow-pop flex flex-col max-h-[88dvh] outline-none`}
      >
        <header className="h-14 shrink-0 px-4 flex items-center justify-between gap-3 border-b border-hair">
          <span className="flex items-center gap-2.5 min-w-0">
            {Icon && <Icon className="w-4 h-4 text-brand shrink-0" strokeWidth={1.9} />}
            <h2 className="font-display text-h2 text-ink truncate">{title}</h2>
          </span>
          <button
            onClick={onClose}
            aria-label="إغلاق"
            className="size-8 shrink-0 rounded-chip text-ink-3 hover:text-ink hover:bg-sunken
                       transition cursor-pointer flex items-center justify-center"
          >
            <X className="w-4 h-4" />
          </button>
        </header>

        <div className="p-4 overflow-y-auto scroll-thin flex flex-col gap-4">
          {children}
        </div>

        {footer && (
          <footer className="p-4 shrink-0 border-t border-hair flex gap-2">{footer}</footer>
        )}
      </div>
    </div>
  );
}
