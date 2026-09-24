function Row({ label, ok, okText, offText }) {
  return (
    <div className="flex items-center justify-between gap-2 h-7">
      <span className="text-xs text-ink-2">{label}</span>
      <span
        className={`flex items-center gap-1.5 h-5 px-2 rounded-chip border text-2xs font-medium
          ${ok ? 'bg-good-wash text-good border-good/25' : 'bg-sunken text-ink-3 border-hair'}`}
      >
        <span className={`size-1.5 rounded-full ${ok ? 'bg-current' : 'bg-ink-3/50'}`} />
        {ok ? okText : offText}
      </span>
    </div>
  );
}

export default function StatusCard({ cameraOn, handsVisible, bodyVisible }) {
  return (
    <section className="bg-surface border border-hair rounded-card shadow-card p-2.5 sm:p-3 shrink-0">
      <p className="text-2xs font-medium text-ink-3 mb-1">حالة النظام</p>

      <div className="divide-y divide-hair">
        <Row label="الكاميرا" ok={cameraOn} okText="تعمل" offText="متوقّفة" />
        <Row label="اليدان" ok={cameraOn && handsVisible} okText="مرصودتان" offText="غير مرصودتين" />
        <Row label="الجسم" ok={cameraOn && bodyVisible} okText="مرصود" offText="غير مرصود" />
      </div>

      {cameraOn && !bodyVisible && (
        <p className="mt-2 pt-2 border-t border-hair text-xs text-caution text-pretty">
          ابتعد قليلاً حتى يظهر كتفاك. بدون مرجع الجسم تتشابه الإشارات.
        </p>
      )}
    </section>
  );
}
