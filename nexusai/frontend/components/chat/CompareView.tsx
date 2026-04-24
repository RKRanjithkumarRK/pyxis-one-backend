"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { useCompare } from "@/hooks/use-compare";
import { CompareLane } from "./CompareLane";
import { cn } from "@/lib/cn";

const ALL_MODELS = [
  { id: "claude-sonnet-4",  label: "Claude Sonnet 4", provider: "anthropic" },
  { id: "claude-opus-4",   label: "Claude Opus 4",   provider: "anthropic" },
  { id: "gpt-4o",          label: "GPT-4o",          provider: "openai"    },
  { id: "gpt-4o-mini",     label: "GPT-4o Mini",     provider: "openai"    },
  { id: "gemini-2-pro",    label: "Gemini 2 Pro",    provider: "google"    },
  { id: "gemini-2-flash",  label: "Gemini 2 Flash",  provider: "google"    },
  { id: "groq-llama-70b",  label: "Llama 3.3 70B",   provider: "groq"      },
  { id: "mistral-large",   label: "Mistral Large",   provider: "mistral"   },
  { id: "cerebras-llama",  label: "Cerebras Llama",  provider: "cerebras"  },
  { id: "sambanova-llama", label: "SambaNova Llama", provider: "sambanova" },
] as const;

const PROVIDER_PILL: Record<string, string> = {
  anthropic: "bg-orange-500/10 text-orange-600 dark:text-orange-400",
  openai:    "bg-green-500/10  text-green-600  dark:text-green-400",
  google:    "bg-blue-500/10   text-blue-600   dark:text-blue-400",
  groq:      "bg-purple-500/10 text-purple-600 dark:text-purple-400",
  mistral:   "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400",
  cerebras:  "bg-pink-500/10   text-pink-600   dark:text-pink-400",
  sambanova: "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400",
};

type VoteValue = "best" | "tie" | "worst";

export function CompareView() {
  const { startCompare, stopCompare, isComparing, compareColumns } = useCompare();
  const [selected, setSelected] = useState<string[]>(["claude-sonnet-4", "gpt-4o"]);
  const [message, setMessage] = useState("");
  const [votes, setVotes] = useState<Record<number, VoteValue>>({});
  const [hasResults, setHasResults] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const colCount = Object.keys(compareColumns).length;
  const allDone =
    hasResults && colCount > 0 && Object.values(compareColumns).every((c) => c.done);

  const toggleModel = useCallback((id: string) => {
    setSelected((prev) => {
      if (prev.includes(id)) {
        return prev.length > 2 ? prev.filter((m) => m !== id) : prev;
      }
      return prev.length < 3 ? [...prev, id] : prev;
    });
  }, []);

  const handleSend = useCallback(async () => {
    const trimmed = message.trim();
    if (!trimmed || isComparing || selected.length < 2) return;
    setVotes({});
    setHasResults(true);
    await startCompare(trimmed, selected);
  }, [message, isComparing, selected, startCompare]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleReset = useCallback(() => {
    setHasResults(false);
    setVotes({});
    setMessage("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* ── Page header ── */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="text-lg font-semibold shrink-0">Compare Models</span>
          <span className="text-sm text-muted-foreground truncate hidden sm:block">
            Run the same prompt across {selected.length} model{selected.length !== 1 ? "s" : ""} side-by-side
          </span>
        </div>
        {hasResults && (
          <button
            onClick={handleReset}
            className="ml-auto shrink-0 text-sm text-muted-foreground hover:text-foreground px-3 py-1 rounded-lg border border-border hover:bg-accent transition-colors"
          >
            New comparison
          </button>
        )}
      </div>

      {/* ── Model selector (only before first compare) ── */}
      {!hasResults && (
        <div className="px-6 py-4 border-b border-border shrink-0">
          <p className="text-xs text-muted-foreground mb-3 font-medium uppercase tracking-wide">
            Select 2–3 models ({selected.length} selected)
          </p>
          <div className="flex flex-wrap gap-2">
            {ALL_MODELS.map((m) => {
              const active = selected.includes(m.id);
              const maxed = !active && selected.length >= 3;
              return (
                <button
                  key={m.id}
                  onClick={() => !maxed && toggleModel(m.id)}
                  aria-pressed={active}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium transition-all select-none",
                    active
                      ? "border-primary bg-primary/10 text-primary ring-1 ring-primary/40"
                      : maxed
                      ? "border-border text-muted-foreground/40 cursor-not-allowed opacity-50"
                      : "border-border text-foreground hover:border-foreground hover:bg-accent"
                  )}
                >
                  <span className={cn("text-xs px-1.5 py-0.5 rounded font-normal", PROVIDER_PILL[m.provider] ?? "bg-muted")}>
                    {m.provider}
                  </span>
                  {m.label}
                  {active && (
                    <span className="ml-0.5 w-4 h-4 rounded-full bg-primary text-primary-foreground text-[10px] flex items-center justify-center font-bold">
                      {selected.indexOf(m.id) + 1}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Selected models banner (shown after compare starts) ── */}
      {hasResults && (
        <div className="flex items-center gap-3 px-6 py-2 border-b border-border bg-muted/20 shrink-0">
          <span className="text-xs text-muted-foreground font-medium">Comparing:</span>
          {selected.map((id) => {
            const m = ALL_MODELS.find((x) => x.id === id);
            return m ? (
              <span key={id} className={cn("text-xs px-2 py-0.5 rounded font-medium", PROVIDER_PILL[m.provider] ?? "bg-muted")}>
                {m.label}
              </span>
            ) : null;
          })}
        </div>
      )}

      {/* ── Multi-column lanes ── */}
      {hasResults && (
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {selected.map((modelId, idx) => {
            const col = compareColumns[idx];
            return (
              <CompareLane
                key={`${modelId}-${idx}`}
                model={modelId}
                tokens={col?.tokens ?? ""}
                done={col?.done ?? false}
                allDone={allDone}
                voted={votes[idx]}
                onVote={(v) => setVotes((prev) => ({ ...prev, [idx]: v }))}
              />
            );
          })}
        </div>
      )}

      {/* ── Empty state placeholder (before first run) ── */}
      {!hasResults && (
        <div className="flex-1 flex flex-col items-center justify-center text-center px-6 gap-3 text-muted-foreground">
          <span className="text-5xl">⚡</span>
          <p className="text-base font-medium text-foreground">Side-by-side model comparison</p>
          <p className="text-sm max-w-sm">
            Select 2 or 3 models above, type a prompt below, and watch all responses stream simultaneously.
          </p>
        </div>
      )}

      {/* ── Prompt composer ── */}
      <div className="border-t border-border px-6 py-4 shrink-0">
        <div className="relative flex items-end gap-3 rounded-2xl border border-border bg-background shadow-sm focus-within:ring-2 focus-within:ring-ring px-4 py-3 transition-shadow">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              const ta = e.currentTarget;
              ta.style.height = "auto";
              ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
            }}
            onKeyDown={handleKeyDown}
            placeholder="Enter a prompt to compare across models… (Enter to send, Shift+Enter for newline)"
            rows={2}
            disabled={hasResults && isComparing}
            className="flex-1 resize-none bg-transparent text-sm focus:outline-none placeholder:text-muted-foreground disabled:opacity-50 max-h-40"
          />
          <button
            onClick={isComparing ? stopCompare : handleSend}
            disabled={!isComparing && (!message.trim() || selected.length < 2)}
            className={cn(
              "h-9 px-4 rounded-lg flex items-center gap-1.5 font-medium text-sm transition-all shrink-0 whitespace-nowrap",
              isComparing
                ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                : message.trim() && selected.length >= 2
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            {isComparing ? "■ Stop" : "⚡ Compare"}
          </button>
        </div>
        <p className="mt-1.5 text-center text-xs text-muted-foreground/60">
          All models receive the same prompt simultaneously. Vote to help improve model recommendations.
        </p>
      </div>
    </div>
  );
}
