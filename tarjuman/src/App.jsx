import { useCallback, useEffect, useRef, useState } from 'react';

import { useTarjumanSocket, BIN_STT_TRANSLATE, BIN_STT_CHAT } from './hooks/useTarjumanSocket';
import { useAudioRecorder } from './hooks/useAudioRecorder';

import { tokenise, AVAILABLE_SIGNS } from './signing/dictionary';
import { loadTrainedSigns, trainedList, findTrainedSign } from './signing/trainedSigns';
import CalibrationPanel from './signing/CalibrationPanel';

import { Camera, SlidersHorizontal } from 'lucide-react';
import RobotStage from './components/RobotStage';
import TranscriptBar from './components/TranscriptBar';
import StatusCard from './components/StatusCard';
import CameraToggle from './components/CameraToggle';
import SignComposer from './components/SignComposer';
import ToolGrid from './components/ToolGrid';
import NoticeToast from './components/NoticeToast';
import AiAssistantModal from './components/modals/AiAssistantModal';
import CameraPickerModal from './components/modals/CameraPickerModal';
import TrainedSignsModal from './components/modals/TrainedSignsModal';
import LearnModal from './components/modals/LearnModal';

import './App.css';

/**
 * App — التركيب والتخطيط، لا أكثر.
 *
 * كان هذا الملف 1070 سطراً يجمع الاتصال والصوت والثلاثيّ الأبعاد وأربع نوافذ
 * وكلّ أنماط العرض. الآن يقرأ كما يبدو على الشاشة: شريط علوي، مسرح، شريط
 * نصّ، عمود تحكّم. وكل قطعة تعيش في ملفّها.
 */
export default function App() {
  // ── حالة التطبيق ─────────────────────────────────────────────────────────
  const [cameraOn, setCameraOn] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [notice, setNotice] = useState(null);

  const [trained, setTrained] = useState([]);
  const [chat, setChat] = useState([]);
  const [aiThinking, setAiThinking] = useState(false);

  const [practiceTarget, setPracticeTarget] = useState(null);
  const [practiceResult, setPracticeResult] = useState(null);

  const [micPermission, setMicPermission] = useState(null);
  const [openModal, setOpenModal] = useState(null); // 'ai' | 'cameras' | 'trained' | 'learn' | 'calibration'

  // ── مراجع ────────────────────────────────────────────────────────────────
  const signerRef = useRef(null);      // مشغّل الإشارات، يملؤه RobotStage
  const cameraOnRef = useRef(false);   // للقراءة داخل ردود نداء ثابتة
  const chatRef = useRef([]);          // تاريخ المحادثة، يُقرأ خارج المحدِّثات
  const trainedRef = useRef([]);       // نسخة يقرأها معالج الرسائل الثابت
  const signTextRef = useRef(null);    // يتيح للمعالج استدعاء أحدث signText
  const askAssistantRef = useRef(null);

  useEffect(() => { trainedRef.current = trained; }, [trained]);
  useEffect(() => { chatRef.current = chat; }, [chat]);
  useEffect(() => { cameraOnRef.current = cameraOn; }, [cameraOn]);

  // ملف مفقود أمرٌ طبيعي: معناه أنّ شيئاً لم يُصدَّر بعد.
  useEffect(() => { loadTrainedSigns().then(() => setTrained(trainedList())); }, []);

  // البانر العابر يختفي وحده فلا يصير فوضى دائمة
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3500);
    return () => clearTimeout(t);
  }, [notice]);

  /**
   * يحوّل معرِّف إشارة إلى نصّها العربي.
   *
   * `LABEL_TO_ARABIC` في الخادم يحوي `salam` وحدها من بين الكلمات المدرَّبة،
   * فيصل «أهلاً» إلى الواجهة مكتوباً `hello` بحروف لاتينية. الإشارات
   * المصدَّرة تحمل النصّ العربي لكل معرِّف، فتسدّ الفجوة من هنا. وإصلاح
   * الخريطة في `websocket_server.py` يبقى مطلوباً — هذا يجعل الواجهة صحيحة
   * قبله وبعده.
   */
  const arabicFor = useCallback((value) => {
    const hit = trainedRef.current.find((s) => s.id === value);
    return hit ? hit.label : value;
  }, []);

  // ── الاتصال بالخادم ──────────────────────────────────────────────────────
  const socket = useTarjumanSocket({
    // النموذج يتعرّف على كلمات كاملة لا حروف، فتُفصَل بمسافة. الوصل بلا
    // فاصل كان ينتج «أهلاًاسم» — كلمة واحدة لا كلمتين.
    onLetter: (value) => setTranscript(
      (prev) => (prev ? prev + ' ' : '') + arabicFor(value)),

    onUnrecognized: () =>
      setNotice({ kind: 'warn', text: 'لم أتعرَّف على الإشارة — أعِد المحاولة' }),

    onSpeechUnclear: () =>
      setNotice({ kind: 'warn', text: 'لم أسمع كلاماً واضحاً — أعِد المحاولة' }),

    onTranscript: (purpose, text) => {
      if (purpose === 'translate') {
        setTranscript((prev) => (prev ? prev + ' ' : '') + text);
        // صوت ← إشارة: السامع يتكلّم فيؤدّي الروبوت كلامه، وهذا هو
        // نصف الترجمة الآخر.
        signTextRef.current?.(text);
      } else if (purpose === 'chat') {
        askAssistantRef.current?.(text);
      }
    },

    onAiReply: (reply) => {
      setChat((prev) => [...prev, { role: 'assistant', content: reply }]);
      setAiThinking(false);
      // المساعد يجيب مستخدماً أصمّ، فجوابه يُؤدَّى إشارةً أيضاً.
      signTextRef.current?.(reply);
    },

    onLivePose: (pose) => signerRef.current?.applyLivePose(pose),
    onMirrorOff: () => signerRef.current?.clearLivePose(),

    onPracticeState: (msg) => {
      setPracticeTarget(msg.active ? msg.target : null);
      if (!msg.active) setPracticeResult(null);
    },
    onPracticeResult: (msg) => setPracticeResult(msg),

    onServerError: (msg) => {
      setNotice({ kind: 'error', text: msg.message || 'حدث خطأ في الخادم' });
      if (msg.code === 'camera_busy' || msg.code === 'camera_failed') setCameraOn(false);
    },

    onDisconnect: () => {
      setCameraOn(false);
      setPracticeTarget(null);
      setPracticeResult(null);
    },
  });

  const { send, sendBinary, clearLiveState } = socket;

  // ── الميكروفون ───────────────────────────────────────────────────────────
  const translateRecorder = useAudioRecorder(
    useCallback((buf) => sendBinary(BIN_STT_TRANSLATE, buf), [sendBinary]),
  );
  const chatRecorder = useAudioRecorder(
    useCallback((buf) => sendBinary(BIN_STT_CHAT, buf), [sendBinary]),
  );

  // ── الأفعال ──────────────────────────────────────────────────────────────
  // المحدِّثات تبقى نقيّة: StrictMode يشغّلها مرّتين، فأيّ إرسال بداخلها
  // يُرسَل مرّتين. الحالة السابقة تُقرأ من ref لا من داخل المحدِّث.
  const toggleCamera = useCallback(() => {
    const next = !cameraOnRef.current;
    setCameraOn(next);
    send({ type: next ? 'start_camera' : 'stop_camera' });
    if (!next) clearLiveState();
  }, [send, clearLiveState]);

  const speakTranscript = useCallback(() => {
    const text = transcript.trim();
    if (text) send({ type: 'speak', value: text });
  }, [transcript, send]);

  /**
   * يجعل الروبوت يؤدّي نصّاً عربياً إشارةً — الاتجاه المعاكس للمنتج كلّه.
   */
  const signText = useCallback((text) => {
    const player = signerRef.current;
    if (!player || !text?.trim()) return;

    // التسجيلة الحقيقية تسبق التقدير المكتوب يدوياً: هي كيف تحرّك إنسانٌ
    // فعلاً، لا كيف خمّن أحدهم الزوايا.
    const resolved = tokenise(text).map((t) => {
      const recorded = findTrainedSign(t.word);
      return recorded ? { ...t, sign: recorded } : t;
    });
    const known = resolved.filter((t) => t.sign);
    const unknown = resolved.filter((t) => !t.sign).map((t) => t.word);

    if (known.length === 0) {
      setNotice({ kind: 'warn', text: 'لا توجد إشارات معروفة في هذا النصّ' });
      return;
    }

    player.playSigns(known);

    // قُل صراحةً ما تعذّرت ترجمته — إسقاط الكلمات بصمت يجعل الروبوت يبدو
    // كأنّه ترجم الجملة خطأً.
    if (unknown.length > 0) {
      setNotice({ kind: 'warn', text: `تعذَّرت ترجمة: ${unknown.slice(0, 3).join('، ')}` });
    }
  }, []);

  useEffect(() => { signTextRef.current = signText; }, [signText]);

  const askAssistant = useCallback((question) => {
    const history = chatRef.current.slice(-10);
    setChat((prev) => [...prev, { role: 'user', content: question }]);
    setAiThinking(true);
    setOpenModal('ai');
    send({ type: 'user_question', data: question, history });
  }, [send]);

  useEffect(() => { askAssistantRef.current = askAssistant; }, [askAssistant]);

  const toggleMirror = useCallback(() => {
    send({ type: socket.mirrorActive ? 'stop_mirror' : 'start_mirror' });
  }, [send, socket.mirrorActive]);

  const openCameraPicker = useCallback(() => {
    setOpenModal('cameras');
    send({ type: 'list_cameras' });
    // الميكروفون إذنُ متصفِّحٍ حقيقي، فأبلِغ عن حالته الفعلية.
    navigator.permissions?.query({ name: 'microphone' })
      .then((r) => setMicPermission(r.state))
      .catch(() => setMicPermission(null));
  }, [send]);

  const startPractice = useCallback((target) => {
    setPracticeResult(null);
    send({ type: 'start_practice', target });
  }, [send]);

  const stopPractice = useCallback(() => {
    setPracticeResult(null);
    send({ type: 'stop_practice' });
  }, [send]);

  const playSign = useCallback((sign) => {
    signerRef.current?.playSigns([{ word: sign.label, sign }]);
  }, []);

  // ── العرض ────────────────────────────────────────────────────────────────
  return (
    <div className="h-dvh max-h-dvh w-full flex flex-col bg-canvas text-ink overflow-hidden" dir="rtl">
      <div
        className="flex-1 w-full mx-auto max-w-[90rem] p-3 sm:p-4 flex flex-col min-h-0 h-full overflow-hidden"
        style={{
          paddingInlineStart: 'max(0.75rem, env(safe-area-inset-right))',
          paddingInlineEnd: 'max(0.75rem, env(safe-area-inset-left))',
          paddingBlockStart: 'max(0.5rem, env(safe-area-inset-top))',
          paddingBlockEnd: 'max(0.5rem, env(safe-area-inset-bottom))',
        }}
      >

        {/* التخطيط الرئيسي: عمودان — المسرح يميناً (RTL) والتحكّم يساراً */}
        <main className="flex-1 min-h-0 grid gap-3 lg:grid-rows-1
                         lg:grid-cols-[minmax(0,1fr)_24rem] xl:grid-cols-[minmax(0,1fr)_26rem] overflow-hidden">

          {/* المسرح والنصّ — العمود العريض (يمين الشاشة في RTL) */}
          <div className="cq flex flex-col gap-2.5 min-w-0 min-h-0 h-full overflow-hidden">
            <RobotStage
              signerRef={signerRef}
              cameraOn={cameraOn}
              capturing={socket.capturing}
              previewUrl={socket.previewUrl}
              handsVisible={socket.tracking.hands}
            />
            <TranscriptBar
              text={transcript}
              onChange={setTranscript}
              onSpeak={speakTranscript}
              recorder={translateRecorder}
            />
          </div>

          {/* لوحة التحكّم — العمود الضيّق (يسار الشاشة في RTL) */}
          <aside className="cq flex flex-col gap-2.5 min-w-0 h-full overflow-y-auto overflow-x-hidden scroll-thin">

            {/* حالة الاتصال وأزرار عامّة — بدلاً من الشريط العلوي */}
            <div className="flex items-center justify-between gap-2 shrink-0">
              <span
                title={socket.connected ? 'الخادم متّصل' : 'بانتظار الخادم'}
                className={`flex items-center gap-2 h-9 px-3 rounded-ctl border text-sm font-medium
                  ${socket.connected
                    ? 'bg-good-wash border-good/25 text-good'
                    : 'bg-caution-wash border-caution/25 text-caution'}`}
              >
                <span className={`size-2 rounded-full ${socket.connected ? 'bg-good' : 'bg-caution'}`} />
                {socket.connected ? 'الخادم متّصل' : 'بانتظار الخادم…'}
              </span>

              <div className="flex items-center gap-1.5">
                <button
                  onClick={openCameraPicker}
                  title="اختيار الكاميرا"
                  aria-label="اختيار الكاميرا"
                  className="size-9 rounded-ctl border border-hair bg-surface text-ink-2
                             hover:text-ink hover:border-hair-2 transition cursor-pointer
                             flex items-center justify-center"
                >
                  <Camera className="w-4 h-4" strokeWidth={1.9} />
                </button>
                <button
                  onClick={() => setOpenModal('calibration')}
                  title="معايرة حركة الروبوت"
                  aria-label="معايرة حركة الروبوت"
                  className="size-9 rounded-ctl border border-hair bg-surface text-ink-2
                             hover:text-ink hover:border-hair-2 transition cursor-pointer
                             flex items-center justify-center"
                >
                  <SlidersHorizontal className="w-4 h-4" strokeWidth={1.9} />
                </button>
              </div>
            </div>

            <CameraToggle
              on={cameraOn}
              disabled={!socket.connected}
              onClick={toggleCamera}
            />
            <StatusCard
              cameraOn={cameraOn}
              handsVisible={socket.tracking.hands}
              bodyVisible={socket.tracking.body}
            />
            <SignComposer
              onSign={signText}
              quickWords={AVAILABLE_SIGNS.slice(0, 4)}
            />
            <ToolGrid
              mirrorOn={socket.mirrorActive}
              mirrorDisabled={!cameraOn}
              onToggleMirror={toggleMirror}
              trainedCount={trained.length}
              onOpenTrained={() => setOpenModal('trained')}
              onOpenAssistant={() => setOpenModal('ai')}
              onOpenLearn={() => setOpenModal('learn')}
            />
          </aside>
        </main>
      </div>

      <NoticeToast notice={notice} />

      {/* ── النوافذ ──────────────────────────────────────────────────────── */}
      {openModal === 'ai' && (
        <AiAssistantModal
          history={chat}
          thinking={aiThinking}
          onSend={askAssistant}
          recorder={chatRecorder}
          onClose={() => setOpenModal(null)}
        />
      )}

      {openModal === 'cameras' && (
        <CameraPickerModal
          cameras={socket.cameras}
          micPermission={micPermission}
          onSelect={(source) => { send({ type: 'set_camera', source }); setOpenModal(null); }}
          onRescan={() => send({ type: 'list_cameras' })}
          onRequestMic={() => navigator.mediaDevices.getUserMedia({ audio: true })
            .then((st) => { st.getTracks().forEach((t) => t.stop()); setMicPermission('granted'); })
            .catch(() => setMicPermission('denied'))}
          onClose={() => setOpenModal(null)}
        />
      )}

      {openModal === 'trained' && (
        <TrainedSignsModal
          signs={trained}
          onPlay={playSign}
          onPlayAll={() => signerRef.current?.playSigns(
            trained.map((s) => ({ word: s.label, sign: s })),
          )}
          onCalibrate={() => setOpenModal('calibration')}
          onClose={() => setOpenModal(null)}
        />
      )}

      {openModal === 'learn' && (
        <LearnModal
          signs={trained}
          cameraOn={cameraOn}
          capturing={socket.capturing}
          target={practiceTarget}
          result={practiceResult}
          onStart={startPractice}
          onStop={stopPractice}
          onPlay={playSign}
          onClose={() => setOpenModal(null)}
        />
      )}

      {openModal === 'calibration' && (
        <CalibrationPanel signerRef={signerRef} onClose={() => setOpenModal(null)} />
      )}
    </div>
  );
}
