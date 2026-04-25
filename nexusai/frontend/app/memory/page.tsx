"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useMemory } from "@/hooks/use-memory";
import { cn } from "@/lib/cn";

export default function MemoryPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { memories, loading, count, deleteMemory, clearAll } = useMemory();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  if (status === "loading") {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!session) {
    router.push("/login");
    return null;
  }

  const handleDelete = async (id: string) => {
    setDeleteId(id);
    try {
      await deleteMemory(id);
    } finally {
      setDeleteId(null);
    }
  };

  const handleClearAll = async () => {
    if (!confirmClear) {
      setConfirmClear(true);
      return;
    }
    setClearing(true);
    setConfirmClear(false);
    try {
      await clearAll();
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Link
              href="/chat"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Memory</h1>
              <p className="text-xs text-muted-foreground">
                {count !== null ? `${count} fact${count !== 1 ? "s" : ""} stored` : "What NexusAI remembers about you"}
              </p>
            </div>
          </div>

          {memories.length > 0 && (
            <button
              onClick={handleClearAll}
              disabled={clearing}
              className={cn(
                "rounded-xl px-4 py-2 text-sm font-medium transition-colors",
                confirmClear
                  ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  : "border border-destructive/30 text-destructive hover:bg-destructive/10",
              )}
            >
              {clearing ? (
                <span className="flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Clearing…
                </span>
              ) : confirmClear ? (
                "Confirm — delete all"
              ) : (
                "Clear All"
              )}
            </button>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-10">
        {/* Explainer */}
        <div className="mb-8 rounded-2xl border border-border bg-card p-6">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-2xl">
              🧠
            </div>
            <div>
              <h2 className="text-sm font-semibold text-foreground">How Memory Works</h2>
              <p className="mt-1 text-sm text-muted-foreground leading-relaxed">
                NexusAI automatically extracts personal facts, preferences, and context from your
                conversations using AI. These memories are embedded and retrieved using vector
                similarity so that future responses are personalised to you.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {[
                  "Auto-extracted after each chat",
                  "pgvector similarity retrieval",
                  "Per-conversation toggle",
                  "Delete any fact anytime",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Memory toggle hint */}
        <div className="mb-6 rounded-xl border border-border bg-card/50 px-4 py-3 text-sm text-muted-foreground">
          💡 You can disable memory for any individual conversation via the conversation settings
          in the chat header.
        </div>

        {/* Memory list */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : memories.length === 0 ? (
          <div className="flex flex-col items-center gap-4 py-20">
            <div className="rounded-2xl border-2 border-dashed border-border p-12 text-center">
              <p className="text-4xl">🧠</p>
              <h3 className="mt-4 text-lg font-semibold text-foreground">No memories yet</h3>
              <p className="mt-1 text-sm text-muted-foreground max-w-xs mx-auto">
                Start chatting and NexusAI will automatically learn your preferences, goals, and
                context over time.
              </p>
              <Link
                href="/chat"
                className="mt-6 inline-block rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Start a conversation →
              </Link>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">
              {memories.length} stored {memories.length === 1 ? "memory" : "memories"}
            </p>
            {memories.map((mem) => (
              <MemoryCard
                key={mem.id}
                memory={mem}
                deleting={deleteId === mem.id}
                onDelete={() => handleDelete(mem.id)}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function MemoryCard({
  memory,
  deleting,
  onDelete,
}: {
  memory: { id: string; fact: string; use_count: number; created_at: string; last_used_at: string | null };
  deleting: boolean;
  onDelete: () => void;
}) {
  return (
    <div className="group flex items-start gap-4 rounded-xl border border-border bg-card p-4 hover:border-primary/30 transition-colors">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
        <span className="text-sm">💭</span>
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm text-foreground leading-relaxed">{memory.fact}</p>
        <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span>
            Added {new Date(memory.created_at).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
              year: new Date(memory.created_at).getFullYear() !== new Date().getFullYear()
                ? "numeric"
                : undefined,
            })}
          </span>
          {memory.use_count > 0 && (
            <>
              <span>·</span>
              <span>Used {memory.use_count}×</span>
            </>
          )}
          {memory.last_used_at && (
            <>
              <span>·</span>
              <span>
                Last used{" "}
                {new Date(memory.last_used_at).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                })}
              </span>
            </>
          )}
        </div>
      </div>

      <button
        onClick={onDelete}
        disabled={deleting}
        className="shrink-0 rounded-lg border border-transparent px-2 py-1 text-xs text-muted-foreground opacity-0 group-hover:opacity-100 hover:border-destructive/30 hover:text-destructive hover:bg-destructive/10 transition-all disabled:opacity-50"
      >
        {deleting ? "…" : "Delete"}
      </button>
    </div>
  );
}
