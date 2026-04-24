"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useSession } from "next-auth/react";
import { cn } from "@/lib/cn";

type ModelInfo = {
  id: string;
  provider: string;
  vision: boolean;
  tool_use: boolean;
  cost_in_per_1k: number;
  cost_out_per_1k: number;
  latency_p50_ms: number | null;
};

const PROVIDER_COLOR: Record<string, string> = {
  anthropic: "bg-orange-500/10 text-orange-600 dark:text-orange-400",
  openai: "bg-green-500/10 text-green-600 dark:text-green-400",
  google: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  groq: "bg-purple-500/10 text-purple-600 dark:text-purple-400",
  mistral: "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400",
  cerebras: "bg-pink-500/10 text-pink-600 dark:text-pink-400",
  sambanova: "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400",
};

const MODEL_LABEL: Record<string, string> = {
  "claude-sonnet-4": "Claude Sonnet 4",
  "claude-opus-4": "Claude Opus 4",
  "gpt-4o": "GPT-4o",
  "gpt-4o-mini": "GPT-4o Mini",
  "gemini-2-pro": "Gemini 2 Pro",
  "gemini-2-flash": "Gemini 2 Flash",
  "groq-llama-70b": "Llama 3.3 70B",
  "mistral-large": "Mistral Large",
  "cerebras-llama": "Cerebras Llama",
  "sambanova-llama": "SambaNova Llama",
};

type Props = {
  value: string;
  onChange: (model: string) => void;
  compact?: boolean;
};

export function ModelPicker({ value, onChange, compact = false }: Props) {
  const { data: session } = useSession();
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    const token = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
    api.get<{ models: ModelInfo[] }>("/api/v1/chat/models", token)
      .then((d) => setModels(d.models))
      .catch(() => {});
  }, [session]);

  const current = models.find((m) => m.id === value);
  const label = MODEL_LABEL[value] ?? value;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-sm font-medium transition-colors hover:bg-accent",
          compact && "px-2 py-1 text-xs"
        )}
      >
        {current && (
          <span className={cn("text-xs px-1 rounded", PROVIDER_COLOR[current.provider] ?? "bg-muted")}>
            {current.provider}
          </span>
        )}
        <span>{label}</span>
        <svg className={cn("h-3 w-3 text-muted-foreground transition-transform", open && "rotate-180")} viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
        </svg>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute bottom-full left-0 mb-2 z-20 w-72 rounded-xl border border-border bg-background shadow-xl overflow-hidden">
            <div className="p-2 border-b border-border">
              <p className="text-xs text-muted-foreground px-2">Select model</p>
            </div>
            <div className="max-h-64 overflow-y-auto p-1">
              {models.map((m) => (
                <button
                  key={m.id}
                  onClick={() => { onChange(m.id); setOpen(false); }}
                  className={cn(
                    "w-full flex items-start justify-between gap-2 px-3 py-2.5 rounded-lg hover:bg-accent transition-colors text-left",
                    m.id === value && "bg-primary/10"
                  )}
                >
                  <div className="space-y-0.5 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={cn("text-xs px-1.5 rounded shrink-0", PROVIDER_COLOR[m.provider] ?? "bg-muted")}>
                        {m.provider}
                      </span>
                      <span className="text-sm font-medium truncate">{MODEL_LABEL[m.id] ?? m.id}</span>
                    </div>
                    <div className="flex gap-2 text-xs text-muted-foreground">
                      {m.vision && <span>👁 Vision</span>}
                      {m.tool_use && <span>🔧 Tools</span>}
                      {m.latency_p50_ms && <span>⚡ {m.latency_p50_ms}ms</span>}
                    </div>
                  </div>
                  <span className="text-xs text-muted-foreground shrink-0">
                    ${(m.cost_in_per_1k * 1000).toFixed(2)}/M
                  </span>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
