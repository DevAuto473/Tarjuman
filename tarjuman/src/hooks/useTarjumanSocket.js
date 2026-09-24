import { useCallback, useEffect, useRef, useState } from 'react';

const WS_URL = 'ws://localhost:8765';
const RECONNECT_DELAY = 3000;

// ترويسات الإطارات الثنائية، الخادم ← العميل (يجب أن تطابق websocket_server.py)
const BIN_AUDIO = 0x01;    // كلام MP3
const BIN_PREVIEW = 0x02;  // معاينة JPEG

// ترويسات الإطارات الثنائية، العميل ← الخادم
export const BIN_STT_TRANSLATE = 0x01;  // صوت يُفرَّغ في صندوق الترجمة
export const BIN_STT_CHAT = 0x02;       // صوت يُرسَل سؤالاً للمساعد

/**
 * يستخرج نصّ المساعد من حمولة `ai_response`.
 *
 * أرسل الخادم يوماً `{"reply": "..."}` كائناً لا نصّاً، فانهار المحادثة بـ
 * "Objects are not valid as a React child"، وانكسرت ذاكرة الحوار بصمت لأن
 * الخادم يُسقط أي عنصر تاريخ محتواه ليس نصّاً. هذه الدالة تُعيد نصّاً دائماً.
 */
export function extractReply(data) {
  if (typeof data === 'string') return data;
  if (data && typeof data === 'object' && typeof data.reply === 'string') {
    return data.reply;
  }
  console.error('[Tarjuman] Unexpected ai_response payload:', data);
  return 'عذراً، تعذَّرَ قراءةُ الرَّدِّ.';
}

/**
 * useTarjumanSocket — كامل طبقة الاتصال بخادم بايثون في مكان واحد.
 *
 * يملك الـhook ما هو نقلٌ محض: الاتصال وإعادته، حالة التتبّع، معاينة الكاميرا،
 * قائمة الأجهزة، وتشغيل الصوت الوارد. وما يحتاج قراراً على مستوى التطبيق
 * يُمرَّر إلى `handlers`.
 *
 * تُقرأ `handlers` من ref، فلا يُعاد بناء معالج الرسائل عند كل تغيّر حالة —
 * وهذا ما كان يجعل النسخة السابقة تعتمد على أربعة refs متفرّقة.
 */
export function useTarjumanSocket(handlers) {
  const handlersRef = useRef(handlers);
  useEffect(() => { handlersRef.current = handlers; });

  const [connected, setConnected] = useState(false);
  const [tracking, setTracking] = useState({ hands: false, body: false });
  const [capturing, setCapturing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mirrorActive, setMirrorActive] = useState(false);
  const [cameras, setCameras] = useState({ list: [], scanning: false, active: null });

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  /** يُصفِّر كل ما يصفُ جلسة كاميرا حيّة. */
  const clearLiveState = useCallback(() => {
    setTracking({ hands: false, body: false });
    setCapturing(false);
    setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState < 2) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      // الإطارات الثنائية موسومة ببايت واحد: 0x01 كلام، 0x02 معاينة.
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
            // أبطِل الرابط السابق بعد التبديل لا قبله، وإلّا أشار <img>
            // إلى ذاكرة محرَّرة فترتعش المعاينة.
            setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return url; });
          }
        });
        return;
      }

      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (err) {
        console.error('[Tarjuman] Malformed frame:', err, event.data);
        return;
      }

      const h = handlersRef.current;
      try {
        switch (msg.type) {
          case 'tracking_status':
            setTracking((prev) => {
              const hands = msg.hands_visible ?? false;
              const body = msg.body_visible ?? false;
              return (prev.hands === hands && prev.body === body)
                ? prev : { hands, body };
            });
            break;

          case 'letter':
            h.onLetter?.(msg.value, msg.confidence);
            break;

          case 'unrecognized':
            h.onUnrecognized?.(msg.confidence);
            break;

          case 'capture_state':
            setCapturing(Boolean(msg.capturing));
            break;

          case 'camera_scan_started':
            setCameras((c) => ({ ...c, scanning: true, list: [] }));
            break;

          case 'camera_list':
            setCameras({
              list: msg.cameras || [],
              scanning: false,
              active: msg.active ?? null,
            });
            break;

          case 'camera_source_changed':
            setCameras((c) => ({ ...c, active: msg.source }));
            break;

          case 'live_pose':
            h.onLivePose?.(msg.pose, msg.body);
            break;

          case 'mirror_state':
            setMirrorActive(Boolean(msg.active));
            if (!msg.active) h.onMirrorOff?.();
            break;

          case 'stt_result': {
            const text = (msg.value || '').trim();
            if (!text) h.onSpeechUnclear?.();
            else h.onTranscript?.(msg.purpose, text);
            break;
          }

          case 'ai_response':
            h.onAiReply?.(extractReply(msg.data));
            break;

          // ── وضع التدريب ──────────────────────────────────────────────
          // الخادم ينفّذ هذه بالكامل عبر مطابقة DTW. تُمرَّر هنا جاهزةً
          // لنافذة «تعلّم مع ترجمان» حين تُبنى.
          case 'practice_state':
            h.onPracticeState?.(msg);
            break;

          case 'practice_result':
            h.onPracticeResult?.(msg);
            break;

          case 'sign_list':
            h.onSignList?.(msg.signs || []);
            break;

          case 'error':
            h.onServerError?.(msg);
            break;

          default:
            console.warn('[Tarjuman] Unhandled message type:', msg.type);
        }
      } catch (err) {
        console.error('[Tarjuman] Handler failed for', msg.type, err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setMirrorActive(false);
      clearLiveState();
      handlersRef.current.onDisconnect?.();
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => ws.close();
  }, [clearLiveState]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        // امنع إعادة الاتصال بعد الإزالة
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connect]);

  const send = useCallback((obj) => {
    if (wsRef.current?.readyState === 1) wsRef.current.send(JSON.stringify(obj));
  }, []);

  /** يُرسل حمولة ثنائية مسبوقةً ببايت نوعها. */
  const sendBinary = useCallback((header, arrayBuffer) => {
    if (wsRef.current?.readyState !== 1) return;
    const out = new Uint8Array(arrayBuffer.byteLength + 1);
    out[0] = header;
    out.set(new Uint8Array(arrayBuffer), 1);
    wsRef.current.send(out);
  }, []);

  return {
    connected,
    send,
    sendBinary,
    tracking,
    capturing,
    previewUrl,
    mirrorActive,
    cameras,
    clearLiveState,
  };
}
