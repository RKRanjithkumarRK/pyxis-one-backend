"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { ModelPicker } from "./ModelPicker";
import { VoiceButton } from "@/components/voice/VoiceButton";
import { cn } from "@/lib/cn";

type Props = {
  onSend: (text: string, model: string, opts: { webSearch: boolean }) => void;
  onStop: () => void;
  isStreaming: boolean;
  defaultModel?: string;
  disabled?: boolean;
};

export function Composer({ onSend, onStop, isStreaming, defaultModel = "claude-sonnet-4", disabled }: Props) {
  const [text, setText] = useState("");
  const [model, setModel] = useState(defaultModel);
  const [webSearch, setWebSearch] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming || disabled) return;
    onSend(trimmed, model, { webSearch });
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [text, model, webSearch, isStreaming, disabled, onSend]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleInput = useCallback(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, []);

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      <div className="mx-auto max-w-3xl">
        <div className="relative flex flex-col gap-2 rounded-2xl border border-border bg-background shadow-sm focus-within:ring-2 focus-within:ring-ring transition-shadow">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Message NexusAI… (Shift+Enter for newline)"
            rows={1}
            disabled={disabled}
            className="resize-none rounded-t-2xl bg-transparent px-4 pt-3.5 pb-2 text-sm focus:outline-none placeholder:text-muted-foreground disabled:opacity-50 min-h-[52px]"
          />

          <div className="flex items-center justify-between px-3 pb-2.5 gap-2">
            <div className="flex items-center gap-2">
              <ModelPicker value={model} onChange={setModel} compact />
              <button
                onClick={() => setWebSearch(!webSearch)}
                title="Web search"
                className={cn(
                  "flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium border transition-colors",
                  webSearch
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground"
                )}
              >
                🌐 Web
              </button>
              <VoiceButton
                onTranscript={(t) => setText((prev) => prev ? `${prev} ${t}` : t)}
              />
            </div>

            <button
              onClick={isStreaming ? onStop : handleSend}
              disabled={!isStreaming && (!text.trim() || disabled)}
              className={cn(
                "h-8 w-8 rounded-lg flex items-center justify-center transition-all font-medium text-sm",
                isStreaming
                  ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  : text.trim() && !disabled
                  ? "bg-primary text-primary-foreground hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              )}
            >
              {isStreaming ? "■" : "↑"}
            </button>
          </div>
        </div>

        <p className="mt-1.5 text-center text-xs text-muted-foreground/60">
          NexusAI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
}
