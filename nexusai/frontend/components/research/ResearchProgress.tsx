"use client";

import { cn } from "@/lib/cn";
import type { ResearchProgressEvent } from "@/lib/api-types";

const STAGES = [
  { id: "planning", label: "Planning", icon: "🗺️" },
  { id: "searching", label: "Searching", icon: "🔍" },
  { id: "fetching", label: "Fetching", icon: "📥" },
  { id: "summarizing", label: "Summarizing", icon: "📝" },
  { id: "synthesizing", label: "Synthesizing", icon: "🧠" },
  { id: "verifying", label: "Verifying", icon: "✅" },
  { id: "saving", label: "Saving", icon: "💾" },
  { id: "complete", label: "Complete", icon: "🎉" },
];

const STAGE_ORDER = STAGES.map((s) => s.id);

function stageIndex(stage: string): number {
  const i = STAGE_ORDER.indexOf(stage);
  return i < 0 ? 0 : i;
}

type Props = {
  event: ResearchProgressEvent | null;
  query: string;
};

export function ResearchProgress({ event, query }: Props) {
  const currentIdx = event ? stageIndex(event.stage) : -1;
  const progress = event?.progress ?? 0;
  const isError = event?.stage === "error";

  return (
    <div className="flex flex-col gap-6">
      {/* Query */}
      <div className="rounded-xl border border-border bg-card p-5">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
          Researching
        </p>
        <p className="text-lg font-semibold text-foreground">{query}</p>
      </div>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{isError ? "Error" : event?.message ?? "Starting…"}</span>
          <span>{progress}%</span>
        </div>
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              isError ? "bg-destructive" : "bg-primary",
            )}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Stage pipeline */}
      <div className="grid grid-cols-4 gap-3 sm:grid-cols-8">
        {STAGES.slice(0, 7).map((stage, i) => {
          const done = i < currentIdx;
          const active = i === currentIdx;
          const pending = i > currentIdx;
          return (
            <div
              key={stage.id}
              className={cn(
                "flex flex-col items-center gap-1.5 rounded-lg p-2 transition-all",
                done && "opacity-60",
                active && "bg-primary/10 ring-1 ring-primary/30",
                pending && "opacity-30",
              )}
            >
              <span className={cn("text-xl", active && "animate-pulse")}>
                {stage.icon}
              </span>
              <span className="text-[10px] font-medium text-muted-foreground text-center leading-tight">
                {stage.label}
              </span>
              {done && (
                <span className="text-[10px] text-green-500">✓</span>
              )}
              {active && (
                <span className="h-1 w-1 rounded-full bg-primary animate-ping" />
              )}
            </div>
          );
        })}
      </div>

      {isError && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
          {event?.message ?? "Research failed. Please try again."}
        </div>
      )}
    </div>
  );
}
