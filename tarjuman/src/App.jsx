import React, { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { useGLTF, useAnimations } from '@react-three/drei';
import { useSignPlayer } from './signing/useSignPlayer';
import { tokenise, AVAILABLE_SIGNS } from './signing/dictionary';
import { SlidersHorizontal, LogOut, Settings, X, Send, Sparkles, Delete, Mic, MicOff, Volume2, Hand } from 'lucide-react';
import './App.css';

const styles = `
  @font-face {
    font-family: 'Thmanyah';
    src: url('https://cdn.jsdelivr.net/gh/DevAuto473/myfonts@main/fonts/1785928393204-blahblah.ttf');
    font-display: swap;
  }

  body {
    margin: 0;
    font-family: 'Thmanyah', sans-serif;
  }
`;

const WS_URL = 'ws://localhost:8765';
const RECONNECT_DELAY = 3000;

// Binary frame headers, server → client (must match websocket_server.py)
const BIN_AUDIO = 0x01;    // MP3 speech
const BIN_PREVIEW = 0x02;  // JPEG camera preview

// Binary frame headers, client → server
const BIN_STT_TRANSLATE = 0x01;  // speech to transcribe into the translation box
const BIN_STT_CHAT = 0x02;       // speech to send to the AI assistant

/**
 * Microphone recorder built on MediaRecorder.
 *
 * This closes the OTHER half of the conversation: the Deaf user signs and the
 * hearing person reads, and now the hearing person speaks and the Deaf user
 * reads. The backend already supported both directions — nothing in the UI
 * ever called it.
 */
function useAudioRecorder(onComplete) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);

  const stop = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== 'inactive') rec.stop();
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      // Prefer webm/opus — that's what the backend labels the upload as.
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');

      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      recorderRef.current = rec;

      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      rec.onstop = async () => {
        // Always release the mic — leaving it open keeps the OS indicator on
        // and blocks other apps.
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        setRecording(false);

        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        chunksRef.current = [];
        if (blob.size > 0) onComplete(await blob.arrayBuffer());
      };

      rec.start();
      setRecording(true);
    } catch (err) {
      console.error('[Tarjuman] Microphone unavailable:', err);
      setError('تعذَّر الوصول إلى الميكروفون');
      setRecording(false);
    }
  }, [onComplete]);

  const toggle = useCallback(() => {
    if (recording) stop(); else start();
  }, [recording, start, stop]);

  // Release the mic if the component unmounts mid-recording
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  return { recording, error, toggle };
}

/**
 * Pull the assistant's text out of an `ai_response` payload.
 *
 * The backend sends `{"type": "ai_response", "data": {"reply": "..."}}` — an
 * OBJECT, not a string. Storing `msg.data` directly crashed the chat with
 * "Objects are not valid as a React child" on the very first reply, and also
 * silently broke conversation memory: the server drops any history entry whose
 * `content` is not a string, so the model never saw its own past answers.
 *
 * Always returns a string so neither failure mode can come back.
 */
function extractReply(data) {
  if (typeof data === 'string') return data;
  if (data && typeof data === 'object' && typeof data.reply === 'string') {
    return data.reply;
  }
  console.error('[Tarjuman] Unexpected ai_response payload:', data);
  return 'عذراً، تعذَّرَ قراءةُ الرَّدِّ.';
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Robot Avatar (Strict Scale & Position)
// ─────────────────────────────────────────────────────────────────────────────
function RobotAvatar({ signerRef }) {
  const group = useRef();
  const { scene, animations } = useGLTF('/TarjumanRobot2.glb');
  const { actions } = useAnimations(animations, group);

  const player = useSignPlayer({ scene, actions });

  // Expose the player upward so the UI can trigger signs.
  useEffect(() => { if (signerRef) signerRef.current = player; }, [signerRef, player]);

  useEffect(() => {
    if (!actions || Object.keys(actions).length === 0) return;
    const keys = Object.keys(actions);

    keys.forEach((k) => actions[k]?.stop());

    const idleKey =
      keys.find((k) => k.toLowerCase().includes('idle')) ||
      keys.find((k) => k.toLowerCase().includes('stand')) ||
      keys.find((k) => k.toLowerCase().includes('rest')) ||
      keys[0];

    if (idleKey && actions[idleKey]) {
      actions[idleKey].reset().fadeIn(0.3).play();
    }
    return () => {
      if (idleKey && actions[idleKey]) actions[idleKey].fadeOut(0.3);
    };
  }, [actions]);

  // The sign player must run AFTER the animation mixer each frame, otherwise
  // the idle clip would overwrite the pose we just applied.
  useFrame((_, delta) => player.update(delta));

  return (
    <group ref={group} position={[0, -1.7, 0]} scale={0.75}>
      <primitive object={scene} />
    </group>
  );
}

// Aims the camera at the robot's face level so it always looks straight ahead.
function CameraSetup() {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 0.4, 4);
    camera.lookAt(0, 0.4, 0);
    camera.fov = 60;
    camera.updateProjectionMatrix();
  }, [camera]);
  return null;
}

const RobotCanvas = React.memo(({ signerRef }) => {
  return (
    <Canvas
      className="!absolute inset-0 w-full h-full"
      camera={{ position: [0, 0.4, 4], fov: 60 }}
    >
      <CameraSetup />
      <ambientLight intensity={0.6} />
      <directionalLight position={[0, 4, 5]} intensity={0.8} />
      <directionalLight position={[-5, 2, -2]} intensity={0.4} />
      <directionalLight position={[5, 2, -2]} intensity={0.4} />
      <Suspense fallback={null}>
        <RobotAvatar signerRef={signerRef} />
      </Suspense>
    </Canvas>
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  // State
  const [cameraOn, setCameraOn] = useState(false);
  const [bodyVisible, setBodyVisible] = useState(false);
  const [handsVisible, setHandsVisible] = useState(false);
  // True while the backend is actively recording a sign, so a Deaf user gets
  // visual confirmation that their gesture is being captured.
  const [capturing, setCapturing] = useState(false);
  // Transient banner: unrecognised sign, camera busy, server errors.
  const [notice, setNotice] = useState(null);
  // Object URL of the latest camera preview frame, so the user can see
  // themselves and confirm they are inside the frame.
  const [previewUrl, setPreviewUrl] = useState(null);
  // Text the user wants the robot to perform as sign language.
  const [signInput, setSignInput] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [showAiModal, setShowAiModal] = useState(false);
  const [showLearnModal, setShowLearnModal] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [aiThinking, setAiThinking] = useState(false);
  const [aiInput, setAiInput] = useState('');

  // Refs
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const cameraOnRef = useRef(false);
  const chatScrollRef = useRef(null);
  // Mirrors chatHistory so the WebSocket handler (created once) can read the
  // latest value without being re-created on every message.
  const chatHistoryRef = useRef([]);
  // Handle to the robot's sign player, populated once the GLB has loaded.
  const signerRef = useRef(null);
  // Lets the WebSocket handler (created once) call the latest signText.
  const signTextRef = useRef(null);

  useEffect(() => { cameraOnRef.current = cameraOn; }, [cameraOn]);
  useEffect(() => { chatHistoryRef.current = chatHistory; }, [chatHistory]);

  // Auto-dismiss the notice banner so it never becomes permanent clutter
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3500);
    return () => clearTimeout(t);
  }, [notice]);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatHistory, aiThinking]);

  // ── WebSocket ─────────────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState < 2) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      // Binary frames are tagged by a 1-byte header: 0x01 = MP3 speech,
      // 0x02 = JPEG camera preview. Previously every blob was assumed to be
      // audio, so any other binary payload would have been played as sound.
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then((buf) => {
          const kind = new Uint8Array(buf, 0, 1)[0];
          const body = buf.slice(1);

          if (kind === BIN_AUDIO) {
            const url = URL.createObjectURL(new Blob([body], { type: 'audio/mpeg' }));
            const audio = new Audio(url);
            const cleanup = () => URL.revokeObjectURL(url);
            audio.onended = cleanup;
            audio.onerror = cleanup;
            audio.play().catch(cleanup);
          } else if (kind === BIN_PREVIEW) {
            const url = URL.createObjectURL(new Blob([body], { type: 'image/jpeg' }));
            // Revoke the PREVIOUS url only after swapping, so the <img> never
            // points at a freed object — otherwise the preview flickers.
            setPreviewUrl((old) => {
              if (old) URL.revokeObjectURL(old);
              return url;
            });
          }
        });
        return;
      }
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'tracking_status') {
          const nextBody = msg.body_visible ?? false;
          const nextHands = msg.hands_visible ?? false;
          setBodyVisible((prev) => (prev === nextBody ? prev : nextBody));
          setHandsVisible((prev) => (prev === nextHands ? prev : nextHands));
        } else if (msg.type === 'letter') {
          setTranslatedText((prev) => prev + msg.value);
        } else if (msg.type === 'capture_state') {
          setCapturing(Boolean(msg.capturing));
        } else if (msg.type === 'stt_result') {
          const text = (msg.value || '').trim();
          if (!text) {
            setNotice({ kind: 'warn', text: 'لم أسمع كلاماً واضحاً — أعِد المحاولة' });
          } else if (msg.purpose === 'translate') {
            setTranslatedText((prev) => (prev ? prev + ' ' : '') + text);
            // Voice → sign: a hearing person speaks and the robot signs it,
            // which is the whole point of the two-way translation.
            signTextRef.current?.(text);
          } else if (msg.purpose === 'chat') {
            // Read history from a ref, never from inside a state updater —
            // updaters must stay pure (React StrictMode runs them twice, which
            // would fire the request to the model twice).
            const history = chatHistoryRef.current.slice(-10);
            setChatHistory((prev) => [...prev, { role: 'user', content: text }]);
            setAiThinking(true);
            setShowAiModal(true);
            wsRef.current?.send(JSON.stringify({
              type: 'user_question', data: text, history,
            }));
          }
        } else if (msg.type === 'unrecognized') {
          // A sign was performed but scored below the confidence threshold.
          // Saying so beats silence — silence can't be told apart from
          // "the camera never saw me".
          setNotice({ kind: 'warn', text: 'لم أتعرَّف على الإشارة — أعِد المحاولة' });
        } else if (msg.type === 'error') {
          setNotice({ kind: 'error', text: msg.message || 'حدث خطأ في الخادم' });
          if (msg.code === 'camera_busy' || msg.code === 'camera_failed') {
            setCameraOn(false);
          }
        } else if (msg.type === 'ai_response') {
          const reply = extractReply(msg.data);
          setChatHistory((prev) => [...prev, { role: 'assistant', content: reply }]);
          setAiThinking(false);
          // The assistant answers a Deaf user, so its reply is signed too.
          signTextRef.current?.(reply);
        }
      } catch (err) {
        // Never swallow silently — a malformed frame used to vanish without
        // a trace, making protocol bugs almost impossible to diagnose.
        console.error('[Tarjuman] Failed to handle message:', err, event.data);
      }
    };

    ws.onclose = () => {
      setCameraOn(false);
      setBodyVisible(false);
      setHandsVisible(false);
      setCapturing(false);
      setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };
    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const send = useCallback((obj) => {
    if (wsRef.current?.readyState === 1) wsRef.current.send(JSON.stringify(obj));
  }, []);

  /** Send a binary payload prefixed with its 1-byte type header. */
  const sendBinary = useCallback((header, arrayBuffer) => {
    if (wsRef.current?.readyState !== 1) return;
    const out = new Uint8Array(arrayBuffer.byteLength + 1);
    out[0] = header;
    out.set(new Uint8Array(arrayBuffer), 1);
    wsRef.current.send(out);
  }, []);

  // Mic → transcription appended to the translation box (hearing person speaks)
  const translateRecorder = useAudioRecorder(
    useCallback((buf) => sendBinary(BIN_STT_TRANSLATE, buf), [sendBinary])
  );

  // Mic → transcription sent to the AI assistant as a spoken question
  const chatRecorder = useAudioRecorder(
    useCallback((buf) => sendBinary(BIN_STT_CHAT, buf), [sendBinary])
  );

  /** Read the translated text aloud so a hearing person can hear the sign. */
  const speakTranslation = useCallback(() => {
    const text = translatedText.trim();
    if (text) send({ type: 'speak', value: text });
  }, [translatedText, send]);

  /**
   * Make the robot SIGN a piece of Arabic text.
   *
   * This is the reverse direction of the whole product: typed or spoken words
   * become sign language a Deaf user can read off the avatar.
   */
  const signText = useCallback((text) => {
    const player = signerRef.current;
    if (!player || !text?.trim()) return;

    const tokens = tokenise(text);
    const known = tokens.filter((t) => t.sign);
    const unknown = tokens.filter((t) => !t.sign).map((t) => t.word);

    if (known.length === 0) {
      setNotice({ kind: 'warn', text: 'لا توجد إشارات معروفة في هذا النص' });
      return;
    }

    player.playSigns(known);

    // Be explicit about what could NOT be signed — silently dropping words
    // would make the robot look like it mistranslated the sentence.
    if (unknown.length > 0) {
      setNotice({
        kind: 'warn',
        text: `تعذَّرت ترجمة: ${unknown.slice(0, 3).join('، ')}`,
      });
    }
  }, []);

  useEffect(() => { signTextRef.current = signText; }, [signText]);

  const toggleCamera = useCallback(() => {
    const next = !cameraOnRef.current;
    setCameraOn(next);
    send({ type: next ? 'start_camera' : 'stop_camera' });
    if (!next) {
      setBodyVisible(false);
      setHandsVisible(false);
      setCapturing(false);
      setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
    }
  }, [send]);

  const handleSendChat = useCallback(() => {
    const trimmed = aiInput.trim();
    if (!trimmed || aiThinking) return;
    setChatHistory((prev) => [...prev, { role: 'user', content: trimmed }]);
    setAiThinking(true);
    setAiInput('');
    send({ type: 'user_question', data: trimmed, history: chatHistory.slice(-10) });
  }, [aiInput, aiThinking, send, chatHistory]);

  // ══════════════════════════════════════════════════════════════════════════
  // RENDER — Barebones Wireframe Layout
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <>
      <link rel="preload" href="https://cdn.jsdelivr.net/gh/DevAuto473/myfonts@main/fonts/1785928393204-blahblah.ttf" as="font" type="font/ttf" crossOrigin="anonymous" />
      <style>{styles}</style>

      {/* ── Screen Wrapper ─────────────────────────────────────────────── */}
      <div className="min-h-screen bg-zinc-950 text-white flex items-center justify-center p-6" dir="rtl">

        {/* ── Main Container Card ────────────────────────────────────────── */}
        <div className="w-full max-w-6xl border-2 border-zinc-800 bg-zinc-900 rounded-3xl p-8 shadow-2xl">

          {/* ── Two-Column Grid ──────────────────────────────────────────── */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8">

            {/* ═══ RIGHT COLUMN — 3D & Translation Display ═══════════════ */}
            <div className="md:col-span-7">

              {/* Canvas Wrapper */}
              <div className="w-full h-[550px] bg-black rounded-2xl relative border border-zinc-700 overflow-hidden">
                <RobotCanvas signerRef={signerRef} />

                {/* ── Self-view ────────────────────────────────────────────
                    A Deaf user signs with their hands; without seeing
                    themselves they cannot tell whether they are framed,
                    lit, or tracked at all. This is the mirror. */}
                {cameraOn && (
                  <div
                    className={`absolute bottom-4 left-4 w-[200px] rounded-xl overflow-hidden border-2 shadow-2xl bg-black transition-colors ${capturing ? 'border-red-500' : 'border-zinc-600'
                      }`}
                  >
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="معاينة الكاميرا"
                        className="w-full h-auto block"
                      />
                    ) : (
                      <div className="w-full h-[150px] flex items-center justify-center text-zinc-500 text-sm">
                        جارٍ تشغيل الكاميرا…
                      </div>
                    )}

                    <div className="absolute top-1.5 right-1.5 flex items-center gap-1.5 bg-black/60 px-2 py-0.5 rounded-full">
                      <span className={`w-2 h-2 rounded-full ${handsVisible ? 'bg-emerald-400' : 'bg-zinc-500'}`} />
                      <span className="text-[11px] text-white/80">
                        {handsVisible ? 'تم رصد اليد' : 'لا توجد يد'}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Translation Display Box */}
              <div className="w-full min-h-[100px] bg-zinc-800 rounded-2xl mt-4 flex items-center justify-between p-4 border border-zinc-700">

                <div className="flex-1 overflow-hidden">
                  <span className={`text-3xl font-bold ${translatedText ? 'text-white' : 'text-emerald-400/50'}`}>
                    {translatedText ? translatedText : "النص المترجم يظهر هنا..."}
                  </span>
                </div>

                {/* Text Controls */}
                <div className="flex items-center gap-2">
                  {/* Speak the translation aloud — lets a hearing person HEAR
                      what the Deaf user just signed. */}
                  <button
                    onClick={speakTranslation}
                    disabled={!translatedText.trim()}
                    className="text-zinc-400 hover:text-white hover:bg-zinc-700/50 p-2 rounded-xl transition cursor-pointer h-10 disabled:opacity-30 disabled:cursor-not-allowed"
                    title="انطق النص"
                  >
                    <Volume2 className="w-6 h-6" />
                  </button>

                  {/* Record speech → transcribed into this box, so a hearing
                      person can reply and the Deaf user can READ it. */}
                  <button
                    onClick={translateRecorder.toggle}
                    className={`p-2 rounded-xl transition cursor-pointer h-10 ${translateRecorder.recording
                      ? 'text-red-400 bg-red-500/15 animate-pulse'
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-700/50'
                      }`}
                    title={translateRecorder.recording ? 'إيقاف التسجيل' : 'تحدَّث لتحويل صوتك إلى نص'}
                  >
                    {translateRecorder.recording
                      ? <MicOff className="w-6 h-6" />
                      : <Mic className="w-6 h-6" />}
                  </button>

                  <button
                    onClick={() => setTranslatedText(prev => prev + ' ')}
                    className="text-zinc-400 hover:text-white hover:bg-zinc-700/50 p-2 rounded-xl transition cursor-pointer flex items-center justify-center w-12 h-10"
                    title="مسافة"
                  >
                    <div className="w-6 h-1.5 bg-current rounded-full" />
                  </button>

                  <button
                    onClick={() => setTranslatedText(prev => prev.slice(0, -1))}
                    className="text-zinc-400 hover:text-white hover:bg-zinc-700/50 p-2 rounded-xl transition cursor-pointer h-10"
                    title="مسح"
                  >
                    <Delete className="w-6 h-6" />
                  </button>
                </div>

              </div>

            </div>

            {/* ═══ LEFT COLUMN — Controls & Action Buttons ════════════════ */}
            <div className="md:col-span-5 flex flex-col h-full px-12 py-10 justify-center items-center gap-8">

              {/* 1. Camera Status Box */}
              <div className="flex flex-col items-center gap-2">
                <div className="flex items-center gap-2 bg-zinc-800/80 px-4 py-2 rounded-full border border-zinc-700/60 shadow-sm">
                  <span className={`w-2.5 h-2.5 rounded-full ${cameraOn ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-500'}`} />
                  <span className="text-sm font-medium text-zinc-300">
                    {cameraOn ? 'الكاميرا تعمل' : 'الكاميرا متوقفة'}
                  </span>
                  {cameraOn && (
                    <span className="text-xs text-zinc-400 mr-2 border-r border-zinc-700 pr-2">
                      الأيدي: {handsVisible ? '✓' : '✗'}
                      &nbsp;·&nbsp;
                      {/* Without the body reference, signs that differ only by
                          location (forehead vs. chin) cannot be told apart. */}
                      <span className={bodyVisible ? '' : 'text-amber-400'}>
                        الجسم: {bodyVisible ? '✓' : '✗'}
                      </span>
                    </span>
                  )}
                </div>

                {/* Capture indicator — tells a Deaf user their sign is being
                    recorded right now, instead of leaving them guessing. */}
                {cameraOn && (
                  <div
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-sm font-medium transition-colors ${capturing
                      ? 'bg-red-500/15 border-red-500/50 text-red-300'
                      : 'bg-zinc-800/40 border-zinc-700/40 text-zinc-500'
                      }`}
                  >
                    <span className={`w-2.5 h-2.5 rounded-full ${capturing ? 'bg-red-500 animate-pulse' : 'bg-zinc-600'}`} />
                    {capturing ? 'جارٍ تسجيل الإشارة…' : 'بانتظار إشارتك'}
                  </div>
                )}

                {/* Transient notice: unrecognised sign / camera busy / errors */}
                {notice && (
                  <div
                    className={`px-4 py-1.5 rounded-full border text-sm font-medium text-center max-w-[260px] ${notice.kind === 'error'
                      ? 'bg-red-500/15 border-red-500/50 text-red-300'
                      : 'bg-amber-500/15 border-amber-500/50 text-amber-300'
                      }`}
                  >
                    {notice.text}
                  </div>
                )}
              </div>

              {/* 2. Buttons */}
              <button
                onClick={toggleCamera}
                className={`w-full max-w-[200px] h-[90px] text-xl font-bold rounded-2xl transition cursor-pointer shadow-lg flex items-center justify-center text-center ${cameraOn
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-blue-600 hover:bg-blue-700'
                  }`}
              >
                {cameraOn ? 'إيقاف الكاميرا' : 'تشغيل الكاميرا'}
              </button>

              {/* ── Text / speech → sign language ─────────────────────────
                  The reverse direction: a hearing person writes or speaks,
                  and the robot performs it for the Deaf user to read. */}
              <div className="w-full max-w-[260px] flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={signInput}
                    onChange={(e) => setSignInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && signInput.trim()) {
                        signText(signInput);
                        setSignInput('');
                      }
                    }}
                    placeholder="اكتب كلمة ليؤديها الروبوت…"
                    className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2 text-white placeholder-zinc-500 text-sm focus:outline-none focus:border-blue-500"
                  />
                  <button
                    onClick={() => { signText(signInput); setSignInput(''); }}
                    disabled={!signInput.trim()}
                    className="bg-blue-600 hover:bg-blue-700 disabled:opacity-30 disabled:cursor-not-allowed text-white p-2 rounded-xl transition cursor-pointer"
                    title="أدِّ الإشارة"
                  >
                    <Hand className="w-5 h-5" />
                  </button>
                </div>

                {/* Quick picks — also documents what the dictionary contains */}
                <div className="flex flex-wrap gap-1.5 justify-center">
                  {AVAILABLE_SIGNS.slice(0, 6).map((w) => (
                    <button
                      key={w}
                      onClick={() => signText(w)}
                      className="text-xs bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 px-2.5 py-1 rounded-full transition cursor-pointer"
                    >
                      {w}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={() => setShowLearnModal(true)}
                className="bg-blue-600 hover:bg-blue-700 w-full max-w-[200px] h-[70px] text-xl font-bold rounded-2xl transition cursor-pointer shadow-lg flex items-center justify-center text-center"
              >
                تعلم مع ترجمان
              </button>

              <button
                onClick={() => setShowAiModal(true)}
                className="bg-blue-600 hover:bg-blue-700 w-full max-w-[200px] h-[90px] text-xl font-bold rounded-2xl transition cursor-pointer shadow-lg flex items-center justify-center text-center"
              >
                تفعيل المساعد الذكي
              </button>

              {/* 3. Bottom Icons */}
              <div className="flex items-center justify-center gap-6">
                <SlidersHorizontal className="w-7 h-7 text-zinc-400 hover:text-white cursor-pointer transition-colors" strokeWidth={1.8} />
                <LogOut className="w-7 h-7 text-zinc-400 hover:text-white cursor-pointer transition-colors" strokeWidth={1.8} />
                <Settings className="w-7 h-7 text-zinc-400 hover:text-white cursor-pointer transition-colors" strokeWidth={1.8} />
              </div>

            </div>

          </div>
        </div>
      </div>

      {/* ══ AI Assistant Modal ═════════════════════════════════════════════ */}
      {showAiModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4" dir="rtl">
          <div className="bg-[#111827] border border-white/10 rounded-2xl w-full max-w-lg p-5 flex flex-col gap-4 shadow-2xl">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-amber-400" />
                <h3 className="text-xl font-bold text-white">المساعد الذكي ترجمان</h3>
              </div>
              <button onClick={() => setShowAiModal(false)}
                className="text-white/60 hover:text-white p-1 rounded-lg hover:bg-white/10 cursor-pointer">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div ref={chatScrollRef} className="h-64 overflow-y-auto flex flex-col gap-3 p-1">
              {chatHistory.length === 0 && (
                <div className="text-center text-white/40 my-auto py-8 text-lg">اسأل ترجمان عن أي شيء...</div>
              )}
              {chatHistory.map((msg, idx) => (
                <div key={idx}
                  className={`max-w-[85%] px-4 py-2.5 rounded-2xl text-base leading-relaxed ${msg.role === 'user'
                    ? 'self-start bg-[#1c3f77] text-white'
                    : 'self-end bg-emerald-700/25 border border-emerald-500/30 text-emerald-100'
                    }`}>
                  {msg.content}
                </div>
              ))}
              {aiThinking && (
                <div className="self-end bg-white/5 text-white/40 px-4 py-2 rounded-2xl text-sm animate-pulse">
                  جاري التفكير...
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 pt-2 border-t border-white/10">
              <input type="text" value={aiInput}
                onChange={(e) => setAiInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendChat()}
                placeholder="اكتب سؤالك أو تحدَّث..."
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white placeholder-white/30 text-base focus:outline-none focus:border-blue-500" />

              {/* Ask the assistant by voice instead of typing */}
              <button onClick={chatRecorder.toggle} disabled={aiThinking}
                className={`p-2.5 rounded-xl transition cursor-pointer disabled:opacity-40 ${chatRecorder.recording
                  ? 'bg-red-500/20 text-red-300 animate-pulse'
                  : 'bg-white/5 text-white/70 hover:bg-white/10'
                  }`}
                title={chatRecorder.recording ? 'إيقاف التسجيل' : 'اسأل بصوتك'}>
                {chatRecorder.recording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </button>

              <button onClick={handleSendChat} disabled={!aiInput.trim() || aiThinking}
                className="bg-[#1c3f77] hover:brightness-110 text-white p-2.5 rounded-xl disabled:opacity-40 transition cursor-pointer">
                <Send className="w-5 h-5" />
              </button>
            </div>

            {(translateRecorder.error || chatRecorder.error) && (
              <div className="text-center text-sm text-red-300">
                {translateRecorder.error || chatRecorder.error}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ══ Learn Modal ═══════════════════════════════════════════════════ */}
      {showLearnModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4" dir="rtl">
          <div className="bg-[#111827] border border-white/10 rounded-2xl w-full max-w-md p-6 flex flex-col gap-4 text-center shadow-2xl">
            <h3 className="text-2xl font-bold text-white">تعلم لغة الإشارة مع ترجمان</h3>
            <p className="text-white/70 text-base leading-relaxed">
              قدّم الإشارات أمام الكاميرا وسيقوم ترجمان بالتعرف عليها وترجمتها فوراً إلى نص وصوت.
            </p>
            <button onClick={() => setShowLearnModal(false)}
              className="bg-[#1c3f77] hover:brightness-110 text-white py-3 rounded-xl font-bold transition cursor-pointer">
              إغلاق
            </button>
          </div>
        </div>
      )}
    </>
  );
}
