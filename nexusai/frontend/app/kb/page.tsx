"use client";

import { useState } from "react";
import Link from "next/link";
import { useKnowledgeBases } from "@/hooks/use-knowledge-base";
import type { KnowledgeBase } from "@/lib/api-types";

function KBCard({ kb, onDelete }: { kb: KnowledgeBase; onDelete: (id: string) => void }) {
  const doneFiles = kb.files.filter((f) => f.status === "done").length;
  const processingFiles = kb.files.filter(
    (f) => f.status === "pending" || f.status === "processing",
  ).length;

  return (
    <div className="group relative rounded-xl border border-border bg-card p-4 hover:border-primary/40 transition-colors">
      <Link href={`/kb/${kb.id}`} className="block">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center text-sm">
              🗂️
            </div>
            <div>
              <p className="font-medium text-sm">{kb.name}</p>
              {kb.description && (
                <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{kb.description}</p>
              )}
            </div>
          </div>
        </div>

        <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
          <span>{kb.files.length} file{kb.files.length !== 1 ? "s" : ""}</span>
          {doneFiles > 0 && <span className="text-green-600">{doneFiles} indexed</span>}
          {processingFiles > 0 && (
            <span className="text-amber-600 animate-pulse">{processingFiles} processing…</span>
          )}
        </div>
      </Link>

      <button
        onClick={() => {
          if (confirm(`Delete "${kb.name}"? This cannot be undone.`)) onDelete(kb.id);
        }}
        className="absolute top-3 right-3 h-6 w-6 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all text-xs"
        title="Delete"
      >
        🗑
      </button>
    </div>
  );
}

export default function KBListPage() {
  const { kbs, loading, create, remove } = useKnowledgeBases();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [showForm, setShowForm] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    try {
      await create(name.trim(), description.trim() || undefined);
      setName("");
      setDescription("");
      setShowForm(false);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold">Knowledge Bases</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Upload documents and use them in your chats with RAG retrieval.
            </p>
          </div>
          <button
            onClick={() => setShowForm((v) => !v)}
            className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
          >
            + New KB
          </button>
        </div>

        {/* Create form */}
        {showForm && (
          <form
            onSubmit={handleCreate}
            className="mb-6 rounded-xl border border-border bg-card p-4 space-y-3"
          >
            <p className="font-medium text-sm">New Knowledge Base</p>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Name (e.g. Research Papers)"
              required
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Description (optional)"
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <div className="flex items-center gap-2">
              <button
                type="submit"
                disabled={creating || !name.trim()}
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                {creating ? "Creating…" : "Create"}
              </button>
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-4 py-2 rounded-lg border border-border text-sm hover:bg-accent transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        )}

        {/* KB grid */}
        {loading ? (
          <div className="text-sm text-muted-foreground">Loading…</div>
        ) : kbs.length === 0 ? (
          <div className="text-center py-16 text-muted-foreground">
            <div className="text-4xl mb-3">🗂️</div>
            <p className="font-medium">No knowledge bases yet</p>
            <p className="text-sm mt-1">Create one to start uploading documents</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {kbs.map((kb) => (
              <KBCard key={kb.id} kb={kb} onDelete={remove} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
