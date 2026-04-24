"use client";

import Link from "next/link";
import { cn } from "@/lib/cn";
import type { Agent } from "@/lib/api-types";

type Props = {
  agent: Agent;
  onUse?: (agent: Agent) => void;
  selected?: boolean;
};

const CATEGORY_COLORS: Record<string, string> = {
  code: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  writing: "bg-purple-500/10 text-purple-400 border-purple-500/20",
  productivity: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  education: "bg-green-500/10 text-green-400 border-green-500/20",
  business: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  creative: "bg-pink-500/10 text-pink-400 border-pink-500/20",
  data: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  general: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20",
};

function StarRating({ rating, count }: { rating: number | null; count: number }) {
  if (!rating || count === 0) return null;
  const full = Math.floor(rating);
  const half = rating - full >= 0.5;
  return (
    <div className="flex items-center gap-1 text-xs text-muted-foreground">
      <span className="text-amber-400">{"★".repeat(full)}{"☆".repeat(5 - full - (half ? 0 : 0))}</span>
      <span>{rating.toFixed(1)}</span>
      <span>({count})</span>
    </div>
  );
}

export function AgentCard({ agent, onUse, selected }: Props) {
  const catColor = CATEGORY_COLORS[agent.category] ?? CATEGORY_COLORS.general;

  return (
    <div
      className={cn(
        "group relative flex flex-col gap-3 rounded-xl border bg-card p-4 transition-all hover:border-primary/30 hover:shadow-md hover:shadow-primary/5",
        selected && "border-primary/60 ring-1 ring-primary/30",
      )}
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-xl">
          {agent.icon ?? "🤖"}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate font-semibold text-sm text-foreground">{agent.name}</h3>
            {agent.is_builtin && (
              <span className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium bg-primary/10 text-primary">
                Official
              </span>
            )}
          </div>
          <span
            className={cn(
              "mt-0.5 inline-block rounded-full border px-2 py-px text-[10px] font-medium capitalize",
              catColor,
            )}
          >
            {agent.category}
          </span>
        </div>
      </div>

      {/* Description */}
      {agent.description && (
        <p className="line-clamp-2 text-xs text-muted-foreground leading-relaxed">
          {agent.description}
        </p>
      )}

      {/* Starters preview */}
      {agent.starters && agent.starters.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {agent.starters.slice(0, 2).map((s, i) => (
            <span
              key={i}
              className="truncate rounded-md bg-muted px-2 py-0.5 text-[11px] text-muted-foreground max-w-[140px]"
              title={s}
            >
              {s.length > 30 ? s.slice(0, 30) + "…" : s}
            </span>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between gap-2 pt-1 border-t border-border/50">
        <div className="flex items-center gap-3">
          <StarRating rating={agent.rating} count={agent.rating_count} />
          {agent.usage_count > 0 && (
            <span className="text-xs text-muted-foreground">
              {agent.usage_count.toLocaleString()} uses
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100">
          <Link
            href={`/agents/${agent.id}`}
            className="rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            Details
          </Link>
          {onUse && (
            <button
              onClick={() => onUse(agent)}
              className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Use
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
