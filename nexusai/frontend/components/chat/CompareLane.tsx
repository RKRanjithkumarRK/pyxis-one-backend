"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { cn } from "@/lib/cn";

const MODEL_DOT: Record<string, string> = {
  "claude-sonnet-4":  "bg-orange-500",
  "claude-opus-4":    "bg-orange-700",
  "gpt-4o":           "bg-green-500",
  "gpt-4o-mini":      "bg-green-400",
  "gemini-2-pro":     "bg-blue-600",
  "gemini-2-flash":   "bg-blue-400",
  "groq-llama-70b":   "bg-purple-500",
  "mistral-large":    "bg-yellow-500",
  "cerebras-llama":   "bg-pink-500",
  "sambanova-llama":  "bg-indigo-500",
};

const MODEL_LABEL: Record<string, string> = {
  "claude-sonnet-4":  "Claude Sonnet 4",
  "claude-opus-4":    "Claude Opus 4",
  "gpt-4o":           "GPT-4o",
  "gpt-4o-mini":      "GPT-4o Mini",
  "gemini-2-pro":     "Gemini 2 Pro",
  "gemini-2-flash":   "Gemini 2 Flash",
  "groq-llama-70b":   "Llama 3.3 70B",
  "mistral-large":    "Mistral Large",
  "cerebras-llama":   "Cerebras Llama",
  "sambanova-llama":  "SambaNova Llama",
};

type VoteValue = "best" | "tie" | "worst";

type Props = {
  model: string;
  tokens: string;
  done: boolean;
  allDone: boolean;
  voted?: VoteValue;
  onVote: (v: VoteValue) => void;
};

export function CompareLane({ model, tokens, done, allDone, voted, onVote }: Props) {
  const dot = MODEL_DOT[model] ?? "bg-slate-500";
  const label = MODEL_LABEL[model] ?? model;

  return (
    <div className="flex flex-col flex-1 min-w-0 border-r last:border-r-0 border-border min-h-0">
      {/* Lane header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-muted/30 shrink-0">
        <span className={cn("w-2.5 h-2.5 rounded-full shrink-0", dot)} />
        <span className="text-sm font-semibold truncate">{label}</span>
        <span className="ml-auto shrink-0 text-xs font-medium">
          {done ? (
            <span className="text-emerald-500">✓ Done</span>
          ) : (
            <span className="text-muted-foreground animate-pulse">Streaming…</span>
          )}
        </span>
      </div>

      {/* Markdown content */}
      <div className="flex-1 overflow-y-auto px-5 py-4 prose prose-sm dark:prose-invert max-w-none">
        {tokens ? (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {tokens}
            </ReactMarkdown>
            {!done && (
              <span className="inline-block w-0.5 h-[1em] bg-primary align-middle animate-pulse ml-0.5" />
            )}
          </>
        ) : (
          <div className="flex items-center gap-2 text-sm text-muted-foreground mt-4">
            <span className="inline-block animate-spin">⟳</span>
            <span>Waiting for first token…</span>
          </div>
        )}
      </div>

      {/* Vote bar — shown only when all lanes are done */}
      {allDone && (
        <div className="shrink-0 flex items-center justify-center gap-2 px-4 py-2.5 border-t border-border bg-muted/20">
          <span className="text-xs text-muted-foreground font-medium">Rate:</span>
          {(["best", "tie", "worst"] as const).map((v) => (
            <button
              key={v}
              onClick={() => onVote(v)}
              className={cn(
                "px-3 py-0.5 rounded-md text-xs font-medium border transition-all",
                voted === v
                  ? v === "best"
                    ? "border-emerald-500 bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 scale-105"
                    : v === "tie"
                    ? "border-yellow-500 bg-yellow-500/20 text-yellow-600 dark:text-yellow-400 scale-105"
                    : "border-rose-500 bg-rose-500/20 text-rose-600 dark:text-rose-400 scale-105"
                  : "border-border text-muted-foreground hover:border-foreground hover:text-foreground"
              )}
            >
              {v === "best" ? "👍 Best" : v === "tie" ? "🤝 Tie" : "👎 Worst"}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
