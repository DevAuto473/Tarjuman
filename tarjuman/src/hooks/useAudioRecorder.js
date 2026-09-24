import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * useAudioRecorder — تسجيل صوتي عبر MediaRecorder.
 *
 * يُغلق النصف الآخر من المحادثة: الأصمّ يشير والسامع يقرأ، ثمّ يتكلّم السامع
 * والأصمّ يقرأ. الخادم كان يدعم الاتجاهين منذ البداية.
 *
 * @param {(buffer: ArrayBuffer) => void} onComplete
 */
export function useAudioRecorder(onComplete) {
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

      // webm/opus أوّلاً — وهو ما يسمّيه الخادم عند الرفع.
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');

      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      recorderRef.current = rec;

      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      rec.onstop = async () => {
        // حرِّر الميكروفون دائماً — تركُه مفتوحاً يُبقي مؤشّر النظام مضاءً
        // ويمنع البرامج الأخرى من استعماله.
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

  // حرِّر الميكروفون إن أُزيل المكوّن أثناء التسجيل
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  return { recording, error, toggle };
}
