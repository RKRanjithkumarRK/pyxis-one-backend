"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useProjects } from "@/hooks/use-projects";
import { cn } from "@/lib/cn";

const ROLE_COLOR: Record<string, string> = {
  owner: "text-primary bg-primary/10",
  editor: "text-blue-500 bg-blue-500/10",
  viewer: "text-muted-foreground bg-muted",
};

export default function ProjectsPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { projects, loading, create, remove } = useProjects();

  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) { setError("Name is required"); return; }
    setError(null);
    setCreating(true);
    try {
      const project = await create({
        name: name.trim(),
        description: description.trim() || undefined,
        system_prompt: systemPrompt.trim() || undefined,
      });
      setShowCreate(false);
      setName("");
      setDescription("");
      setSystemPrompt("");
      router.push(`/projects/${project.id}`);
    } catch (err: any) {
      setError(err?.message ?? "Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this project and all its data?")) return;
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
            <Link href="/chat" className="text-muted-foreground hover:text-foreground transition-colors">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Projects</h1>
              <p className="text-xs text-muted-foreground">Scoped chats, custom instructions, team members</p>
            </div>
          </div>

          <button
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-2 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Project
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        {/* Create form */}
        {showCreate && (
          <div className="mb-8 rounded-2xl border border-border bg-card p-6">
            <h2 className="mb-4 text-base font-semibold text-foreground">New Project</h2>
            <form onSubmit={handleCreate} className="space-y-4">
              {error && (
                <p className="rounded-lg bg-destructive/10 border border-destructive/30 px-4 py-2 text-sm text-destructive">
                  {error}
                </p>
              )}
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  Name <span className="text-destructive">*</span>
                </label>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Product Research"
                  maxLength={256}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  Description
                </label>
                <input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What is this project for?"
                  maxLength={2000}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  System Prompt
                </label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="e.g. You are a product research assistant. Always cite sources."
                  rows={3}
                  maxLength={8000}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>
              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={creating}
                  className="rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  {creating ? "Creating…" : "Create Project"}
                </button>
                <button
                  type="button"
                  onClick={() => { setShowCreate(false); setError(null); }}
                  className="rounded-xl border border-border px-5 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Project grid */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : projects.length === 0 && !showCreate ? (
          <div className="flex flex-col items-center gap-4 py-20">
            <div className="rounded-2xl border-2 border-dashed border-border p-12 text-center">
              <p className="text-4xl">📁</p>
              <h3 className="mt-4 text-lg font-semibold text-foreground">No projects yet</h3>
              <p className="mt-1 text-sm text-muted-foreground max-w-xs mx-auto">
                Create a project to organise conversations, set a system prompt, and invite collaborators.
              </p>
              <button
                onClick={() => setShowCreate(true)}
                className="mt-6 rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Create Project →
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((project) => (
              <div
                key={project.id}
                className="group relative flex flex-col rounded-2xl border border-border bg-card p-5 hover:border-primary/40 hover:shadow-md transition-all"
              >
                <Link href={`/projects/${project.id}`} className="flex-1">
                  <div className="mb-3 flex items-start gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-2xl">
                      📁
                    </div>
                    <div className="min-w-0 flex-1">
                      <h3 className="truncate font-semibold text-foreground text-sm">{project.name}</h3>
                      {project.role && (
                        <span className={cn(
                          "mt-0.5 inline-block rounded-full px-2 py-0.5 text-xs font-medium",
                          ROLE_COLOR[project.role] ?? "text-muted-foreground bg-muted",
                        )}>
                          {project.role}
                        </span>
                      )}
                    </div>
                  </div>
                  {project.description && (
                    <p className="mb-3 text-xs text-muted-foreground line-clamp-2">
                      {project.description}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Updated {new Date(project.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                  </p>
                </Link>

                <div className="mt-4 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Link
                    href={`/projects/${project.id}`}
                    className="flex-1 rounded-lg bg-primary/10 py-1.5 text-center text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                  >
                    Open
                  </Link>
                  {project.role === "owner" && (
                    <button
                      onClick={() => handleDelete(project.id)}
                      disabled={deleteId === project.id}
                      className="rounded-lg border border-destructive/30 px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10 transition-colors disabled:opacity-50"
                    >
                      {deleteId === project.id ? "…" : "Delete"}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
