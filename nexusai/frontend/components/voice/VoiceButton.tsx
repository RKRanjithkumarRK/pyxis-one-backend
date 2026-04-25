"use client";

import { useVoice } from "@/hooks/use-voice";
import { cn } from "@/lib/cn";

type Props = {
  onTranscript: (text: string) => void;
  className?: string;
};

export function VoiceButton({ onTranscript, className }: Props) {
  const { state, error, startListening, stop } = useVoice({ onTranscript });

  const isActive = state === "listening" || state === "processing";

  return (
    <button
      type="button"
      onClick={isActive ? stop : startListening}
      title={
        state === "idle" ? "Start voice input" :
        state === "listening" ? "Listening… click to stop" :
        state === "processing" ? "Processing…" :
        "Speaking…"
      }
      className={cn(
        "relative h-8 w-8 flex items-center justify-center rounded-lg transition-colors",
        state === "listening"
          ? "bg-red-500 text-white hover:bg-red-600"
          : state === "processing" || state === "speaking"
          ? "bg-primary/20 text-primary cursor-default"
          : "text-muted-foreground hover:text-foreground hover:bg-accent",
        className,
      )}
    >
      {state === "listening" ? (
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-white" />
        </span>
      ) : state === "processing" ? (
        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      ) : (
        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      )}
    </button>
  );
}
