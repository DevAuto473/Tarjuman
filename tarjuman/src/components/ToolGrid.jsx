import { Hand, Sparkles, GraduationCap, Repeat } from 'lucide-react';

function Tool({ icon: Icon, label, hint, badge, active, disabled, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full rounded-ctl border p-2 text-right transition cursor-pointer shadow-card
                  flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed
        ${active ? 'bg-brand-wash border-brand/35 text-brand'
                 : 'bg-surface border-hair text-ink hover:border-hair-2 hover:bg-sunken'}`}
    >
      <span className={`size-7 shrink-0 rounded-chip flex items-center justify-center
        ${active ? 'bg-brand text-white' : 'bg-sunken text-ink-2'}`}>
        <Icon className="w-3.5 h-3.5" strokeWidth={1.9} />
      </span>

      <span className="min-w-0 flex-1">
        <span className="block text-xs font-medium truncate">{label}</span>
        <span className={`block text-2xs truncate ${active ? 'text-brand/80' : 'text-ink-3'}`}>
          {hint}
        </span>
      </span>

      {badge != null && (
        <span className="tnum shrink-0 h-5 min-w-5 px-1 rounded-chip bg-sunken border border-hair
                         text-2xs text-ink-2 flex items-center justify-center font-medium">
          {badge}
        </span>
      )}
    </button>
  );
}

export default function ToolGrid({
  mirrorOn, onToggleMirror, mirrorDisabled,
  trainedCount, onOpenTrained,
  onOpenAssistant, onOpenLearn,
}) {
  return (
    <div className="grid grid-cols-2 gap-2 shrink-0">
      <Tool
        icon={Repeat}
        label="المرآة الحيّة"
        hint={mirrorOn ? 'يقلّدك الآن' : 'تقليد لحظي'}
        active={mirrorOn}
        disabled={mirrorDisabled}
        onClick={onToggleMirror}
      />
      <Tool
        icon={Hand}
        label="الإشارات المدرَّبة"
        hint="من تسجيلاتك"
        badge={trainedCount}
        onClick={onOpenTrained}
      />
      <Tool
        icon={Sparkles}
        label="المساعد الذكي"
        hint="نصّاً أو صوتاً"
        onClick={onOpenAssistant}
      />
      <Tool
        icon={GraduationCap}
        label="تعلّم الإشارة"
        hint="تدرَّب واختبر"
        onClick={onOpenLearn}
      />
    </div>
  );
}
