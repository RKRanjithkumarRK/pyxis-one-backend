"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useCanvasList } from "@/hooks/use-canvas";
import { cn } from "@/lib/cn";

export default function CanvasListPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { docs, loading, create, remove } = useCanvasList();
  const [creating, setCreating] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);

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

  const handleNew = async () => {
    setCreating(true);
    try {
      const doc = await create("Untitled");
      router.push(`/canvas/${doc.id}`);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    setDeleteId(id);
    try {
      await remove(id);
    } finally {
      setDeleteId(null);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
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
              <h1 className="text-lg font-semibold text-foreground">Canvas</h1>
              <p className="text-xs text-muted-foreground">
                AI-powered collaborative documents
              </p>
            </div>
          </div>

          <button
            onClick={handleNew}
            disabled={creating}
            className="flex items-center gap-2 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {creating ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
            ) : (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
            )}
            New Document
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        {/* Hero */}
        <div className="mb-10 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-4xl">
            📝
          </div>
          <h2 className="text-3xl font-bold text-foreground">Your Documents</h2>
          <p className="mt-2 text-muted-foreground">
            Write and edit with AI assistance. Share with collaborators in real-time.
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : docs.length === 0 ? (
          <div className="flex flex-col items-center gap-4 py-20">
            <div className="rounded-2xl border-2 border-dashed border-border p-12 text-center">
              <p className="text-4xl">📄</p>
              <h3 className="mt-4 text-lg font-semibold text-foreground">No documents yet</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Create your first document to get started.
              </p>
              <button
                onClick={handleNew}
                disabled={creating}
                className="mt-6 rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Create Document →
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {/* New doc card */}
            <button
              onClick={handleNew}
              disabled={creating}
              className="group flex min-h-40 flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-border bg-card hover:border-primary/50 hover:bg-primary/5 transition-colors"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary group-hover:bg-primary/20 transition-colors">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <span className="text-sm font-medium text-muted-foreground group-hover:text-primary transition-colors">
                New Document
              </span>
            </button>

            {docs.map((doc) => (
              <div
                key={doc.id}
                className="group relative flex flex-col rounded-2xl border border-border bg-card p-5 hover:border-primary/40 hover:shadow-md transition-all"
              >
                <Link href={`/canvas/${doc.id}`} className="flex-1">
                  <div className="mb-3 flex items-start gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-xl">
                      📝
                    </div>
                    <div className="min-w-0 flex-1">
                      <h3 className="truncate text-sm font-semibold text-foreground">
                        {doc.title}
                      </h3>
                      <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                        <span>v{doc.version}</span>
                        {doc.is_public && (
                          <>
                            <span>·</span>
                            <span className="text-green-500">Public</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <p className="text-xs text-muted-foreground">
                    Updated {new Date(doc.updated_at).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      year: new Date(doc.updated_at).getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
                    })}
                  </p>
                </Link>

                {/* Actions */}
                <div className="mt-4 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Link
                    href={`/canvas/${doc.id}`}
                    className="flex-1 rounded-lg bg-primary/10 py-1.5 text-center text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                  >
                    Open
                  </Link>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    disabled={deleteId === doc.id}
                    className="rounded-lg border border-destructive/30 px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10 transition-colors disabled:opacity-50"
                  >
                    {deleteId === doc.id ? "…" : "Delete"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Feature callouts */}
        <div className="mt-16 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[
            {
              icon: "✨",
              title: "AI Inline Edits",
              desc: "Select any text and ask AI to rewrite, expand, or improve it.",
            },
            {
              icon: "🔄",
              title: "Version History",
              desc: "Every snapshot is saved. Restore any previous version instantly.",
            },
            {
              icon: "📤",
              title: "Export Anywhere",
              desc: "Download as Markdown, HTML, or Word (.docx) with one click.",
            },
          ].map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-border bg-card p-5"
            >
              <span className="text-2xl">{f.icon}</span>
              <h4 className="mt-3 text-sm font-semibold text-foreground">{f.title}</h4>
              <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
