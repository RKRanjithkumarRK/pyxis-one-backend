"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useSession } from "next-auth/react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type VoiceState = "idle" | "listening" | "processing" | "speaking";

export function useVoice({
  onTranscript,
  voiceId = "rachel",
}: {
  onTranscript: (text: string) => void;
  voiceId?: string;
}) {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [state, setState] = useState<VoiceState>("idle");
  const [error, setError] = useState<string | null>(null);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, []);

  const stopTTS = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
  }, []);

  const startListening = useCallback(async () => {
    if (state !== "idle" && state !== "speaking") return;
    stopTTS();
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Set up analyser for VAD (silence detection)
      const ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      chunksRef.current = [];
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      mediaRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (chunksRef.current.length === 0) {
          setState("idle");
          return;
        }
        setState("processing");
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await transcribe(blob);
      };

      mr.start(200);
      setState("listening");

      // VAD: stop after 1.5s of silence
      detectSilence(analyser, () => {
        if (mediaRef.current?.state === "recording") {
          mediaRef.current.stop();
        }
      });
    } catch (e: any) {
      setError(e.message);
      setState("idle");
    }
  }, [state, stopTTS]);

  const stop = useCallback(() => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (mediaRef.current?.state === "recording") mediaRef.current.stop();
    stopTTS();
    setState("idle");
  }, [stopTTS]);

  const transcribe = useCallback(
    async (blob: Blob) => {
      if (!token) { setState("idle"); return; }
      try {
        const form = new FormData();
        form.append("audio", blob, "audio.webm");
        const res = await fetch(`${API_BASE}/api/v1/voice/transcribe`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: form,
        });
        if (!res.ok) throw new Error(`Transcribe ${res.status}`);
        const { text } = await res.json();
        if (text?.trim()) onTranscript(text.trim());
        setState("idle");
      } catch (e: any) {
        setError(e.message);
        setState("idle");
      }
    },
    [token, onTranscript],
  );

  const speak = useCallback(
    async (text: string) => {
      if (!token || !text.trim()) return;
      stopTTS();
      setState("speaking");
      try {
        const res = await fetch(`${API_BASE}/api/v1/voice/tts`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice_id: voiceId }),
        });
        if (!res.ok) throw new Error(`TTS ${res.status}`);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setState("idle");
        };
        audio.play();
      } catch (e: any) {
        setError(e.message);
        setState("idle");
      }
    },
    [token, voiceId, stopTTS],
  );

  return { state, error, startListening, stop, speak };
}

function detectSilence(analyser: AnalyserNode, onSilence: () => void) {
  const data = new Uint8Array(analyser.fftSize);
  let silenceStart: number | null = null;
  const SILENCE_THRESHOLD = 10;
  const SILENCE_DURATION = 1500;

  function check() {
    analyser.getByteTimeDomainData(data);
    const rms = Math.sqrt(data.reduce((s, v) => s + (v - 128) ** 2, 0) / data.length);

    if (rms < SILENCE_THRESHOLD) {
      if (!silenceStart) silenceStart = Date.now();
      else if (Date.now() - silenceStart > SILENCE_DURATION) {
        onSilence();
        return;
      }
    } else {
      silenceStart = null;
    }
    requestAnimationFrame(check);
  }
  requestAnimationFrame(check);
}
