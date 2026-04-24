"use client";

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { agentsApi } from "@/lib/api";
import { useAgents } from "@/hooks/use-agents";
import { cn } from "@/lib/cn";
import type { Agent, AgentVersion } from "@/lib/api-types";

type Props = { params: Promise<{ agentId: string }> };

function StarButton({ rating, current, onClick }: { rating: number; current: number; onClick: (r: number) => void }) {
  return (
    <button
      onClick={() => onClick(rating)}
      className={cn(
        "text-2xl transition-colors",
        rating <= current ? "text-amber-400" : "text-muted-foreground hover:text-amber-300",
      )}
    >
      ★
    </button>
  );
}

export default function AgentDetailPage({ params }: Props) {
  const { agentId } = use(params);
  const router = useRouter();
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;
  const { deleteAgent, publishAgent, rateAgent } = useAgents();

  const [agent, setAgent] = useState<Agent | null>(null);
  const [versions, setVersions] = useState<AgentVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [userRating, setUserRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [tab, setTab] = useState<"overview" | "starters" | "versions">("overview");

  const isOwner = agent && session && agent.creator_id === (session as any)?.user?.id;

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const a = await agentsApi.get(agentId, token);
        setAgent(a);
      } catch {
        setError("Agent not found");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [agentId, token]);

  useEffect(() => {
    if (!agent || !token || !isOwner) return;
    agentsApi.versions(agent.id, token).then(setVersions).catch(() => {});
  }, [agent, token, isOwner]);

  const handleDelete = async () => {
    if (!agent) return;
    if (!confirm(`Delete "${agent.name}"? This cannot be undone.`)) return;
    await deleteAgent(agent.id);
    router.push("/agents");
  };

  const handlePublish = async () => {
    if (!agent) return;
    const next = await publishAgent(agent.id, agent.visibility !== "public");
    setAgent(next);
  };

  const handleRate = async (rating: number) => {
    if (!agent || !token) return;
    setUserRating(rating);
    const updated = await rateAgent(agent.id, rating);
    setAgent(updated);
  };

  const handleRestore = async (version: number) => {
    if (!agent || !token) return;
    const updated = await agentsApi.restore(agent.id, version, token);
    setAgent(updated);
    const vs = await agentsApi.versions(agent.id, token);
    setVersions(vs);
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (error || !agent) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4">
        <span className="text-5xl">🤖</span>
        <p className="text-lg font-medium">{error ?? "Agent not found"}</p>
        <Link href="/agents" className="text-primary hover:underline text-sm">
          ← Back to Agent Store
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <Link
            href="/agents"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Agent Store
          </Link>
          <div className="flex items-center gap-2">
            {isOwner && (
              <>
                <button
                  onClick={handlePublish}
                  className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent transition-colors"
                >
                  {agent.visibility === "public" ? "Unpublish" : "Publish"}
                </button>
                <Link
                  href={`/agents/${agent.id}/edit`}
                  className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent transition-colors"
                >
                  Edit
                </Link>
                <button
                  onClick={handleDelete}
                  className="rounded-lg border border-destructive/30 px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10 transition-colors"
                >
                  Delete
                </button>
              </>
            )}
            <Link
              href={`/chat?agent=${agent.id}`}
              className="rounded-lg bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Use Agent →
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-8">
        {/* Agent hero */}
        <div className="mb-8 flex items-start gap-5">
          <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-muted text-4xl">
            {agent.icon ?? "🤖"}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-2xl font-bold text-foreground">{agent.name}</h1>
              {agent.is_builtin && (
                <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                  Official
                </span>
              )}
              <span className="rounded-full border border-border px-2.5 py-0.5 text-xs text-muted-foreground capitalize">
                {agent.category}
              </span>
              <span
                className={cn(
                  "rounded-full px-2.5 py-0.5 text-xs font-medium capitalize",
                  agent.visibility === "public"
                    ? "bg-green-500/10 text-green-400"
                    : "bg-zinc-500/10 text-zinc-400",
                )}
              >
                {agent.visibility}
              </span>
            </div>
            {agent.description && (
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-2xl">
                {agent.description}
              </p>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
              <span>{agent.usage_count.toLocaleString()} uses</span>
              {agent.rating && <span>★ {agent.rating.toFixed(1)} ({agent.rating_count} ratings)</span>}
              <span>v{agent.version}</span>
              <span>Model: {agent.default_model}</span>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6 flex gap-1 border-b border-border">
          {(["overview", "starters", "versions"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "px-4 py-2 text-sm transition-colors capitalize border-b-2 -mb-px",
                tab === t
                  ? "border-primary text-foreground font-medium"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              {t}
              {t === "versions" && versions.length > 0 && (
                <span className="ml-1.5 rounded-full bg-muted px-1.5 py-0.5 text-[10px]">
                  {versions.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "overview" && (
          <div className="space-y-6">
            {agent.instructions && (
              <section className="rounded-xl border border-border bg-card p-5">
                <h2 className="mb-3 text-sm font-semibold text-foreground">System Prompt</h2>
                <pre className="whitespace-pre-wrap font-mono text-xs text-muted-foreground leading-relaxed max-h-80 overflow-y-auto">
                  {agent.instructions}
                </pre>
              </section>
            )}

            {agent.capabilities && (
              <section className="rounded-xl border border-border bg-card p-5">
                <h2 className="mb-3 text-sm font-semibold text-foreground">Capabilities</h2>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(agent.capabilities).map(([k, v]) =>
                    v ? (
                      <span key={k} className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary capitalize">
                        {k.replace("_", " ")}
                      </span>
                    ) : null,
                  )}
                  {!Object.values(agent.capabilities).some(Boolean) && (
                    <span className="text-xs text-muted-foreground">No special capabilities</span>
                  )}
                </div>
              </section>
            )}

            {/* Rating */}
            {agent.visibility === "public" && token && (
              <section className="rounded-xl border border-border bg-card p-5">
                <h2 className="mb-3 text-sm font-semibold text-foreground">Rate this agent</h2>
                <div
                  className="flex gap-1"
                  onMouseLeave={() => setHoverRating(0)}
                >
                  {[1, 2, 3, 4, 5].map((r) => (
                    <button
                      key={r}
                      onClick={() => handleRate(r)}
                      onMouseEnter={() => setHoverRating(r)}
                      className={cn(
                        "text-3xl transition-colors",
                        r <= (hoverRating || userRating)
                          ? "text-amber-400"
                          : "text-muted-foreground hover:text-amber-300",
                      )}
                    >
                      ★
                    </button>
                  ))}
                </div>
                {userRating > 0 && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    You rated this agent {userRating}/5
                  </p>
                )}
              </section>
            )}
          </div>
        )}

        {tab === "starters" && (
          <div className="space-y-3">
            {agent.starters && agent.starters.length > 0 ? (
              agent.starters.map((s, i) => (
                <Link
                  key={i}
                  href={`/chat?agent=${agent.id}&message=${encodeURIComponent(s)}`}
                  className="flex items-center gap-3 rounded-xl border border-border bg-card p-4 text-sm text-foreground hover:border-primary/30 hover:bg-card/80 transition-all group"
                >
                  <span className="text-muted-foreground group-hover:text-primary transition-colors">→</span>
                  {s}
                </Link>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">No conversation starters configured.</p>
            )}
          </div>
        )}

        {tab === "versions" && (
          <div className="space-y-3">
            {isOwner && versions.length > 0 ? (
              versions.map((v) => (
                <div
                  key={v.id}
                  className="flex items-center justify-between rounded-xl border border-border bg-card p-4"
                >
                  <div>
                    <p className="text-sm font-medium text-foreground">Version {v.version}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(v.created_at).toLocaleString()}
                    </p>
                  </div>
                  {v.version !== agent.version && (
                    <button
                      onClick={() => handleRestore(v.version)}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent transition-colors"
                    >
                      Restore
                    </button>
                  )}
                  {v.version === agent.version && (
                    <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs text-primary">
                      Current
                    </span>
                  )}
                </div>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">
                {isOwner ? "No previous versions." : "Version history is only visible to the agent owner."}
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
