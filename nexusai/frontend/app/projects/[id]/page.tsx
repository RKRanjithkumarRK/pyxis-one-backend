"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter, useParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { useProject } from "@/hooks/use-projects";
import { cn } from "@/lib/cn";
import type { ProjectRole } from "@/lib/api-types";

const ROLE_OPTIONS: ProjectRole[] = ["owner", "editor", "viewer"];

const ROLE_COLOR: Record<string, string> = {
  owner: "text-primary bg-primary/10 border-primary/30",
  editor: "text-blue-500 bg-blue-500/10 border-blue-500/30",
  viewer: "text-muted-foreground bg-muted border-border",
};

type Tab = "chats" | "settings" | "members";

export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { data: session } = useSession();
  const { project, members, conversations, loading, update, invite, removeMember, updateMemberRole } =
    useProject(id);

  const [tab, setTab] = useState<Tab>("chats");

  // Settings form state
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [saving, setSaving] = useState(false);
  const [settingsMsg, setSettingsMsg] = useState<string | null>(null);

  // Members form state
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<ProjectRole>("viewer");
  const [inviting, setInviting] = useState(false);
  const [inviteError, setInviteError] = useState<string | null>(null);

  const token = (session as any)?.accessToken as string | undefined;
  const myUserId = (session?.user as any)?.id as string | undefined;

  const isOwner = project?.role === "owner";
  const canEdit = project?.role === "owner" || project?.role === "editor";

  const handleOpenSettings = () => {
    if (project) {
      setEditName(project.name);
      setEditDesc(project.description ?? "");
      setEditPrompt(project.system_prompt ?? "");
    }
    setTab("settings");
  };

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSettingsMsg(null);
    try {
      await update({
        name: editName.trim(),
        description: editDesc.trim() || undefined,
        system_prompt: editPrompt.trim() || undefined,
      });
      setSettingsMsg("Saved!");
      setTimeout(() => setSettingsMsg(null), 2000);
    } catch {
      setSettingsMsg("Failed to save");
    } finally {
      setSaving(false);
    }
  };

  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) { setInviteError("Email required"); return; }
    setInviteError(null);
    setInviting(true);
    try {
      await invite(inviteEmail.trim(), inviteRole);
      setInviteEmail("");
    } catch (err: any) {
      const detail = err?.detail ?? err?.message ?? "Failed to invite";
      setInviteError(typeof detail === "string" ? detail : JSON.stringify(detail));
    } finally {
      setInviting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-background text-foreground">
        <p className="text-lg font-medium">Project not found</p>
        <Link href="/projects" className="text-primary underline underline-offset-4">← Projects</Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl items-center gap-4 px-6 py-4">
          <Link href="/projects" className="text-muted-foreground hover:text-foreground transition-colors">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-lg font-semibold text-foreground">{project.name}</h1>
            {project.description && (
              <p className="truncate text-xs text-muted-foreground">{project.description}</p>
            )}
          </div>
          {project.role && (
            <span className={cn(
              "rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize",
              ROLE_COLOR[project.role],
            )}>
              {project.role}
            </span>
          )}
          <Link
            href={`/chat?project_id=${project.id}`}
            className="flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            + New Chat
          </Link>
        </div>

        {/* Tabs */}
        <div className="mx-auto flex max-w-5xl gap-1 px-6 pb-1">
          {(["chats", "members", "settings"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => t === "settings" ? handleOpenSettings() : setTab(t)}
              className={cn(
                "rounded-lg px-3 py-1.5 text-sm font-medium capitalize transition-colors",
                tab === t ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t}
            </button>
          ))}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {/* Chats tab */}
        {tab === "chats" && (
          <div>
            {conversations.length === 0 ? (
              <div className="flex flex-col items-center gap-4 py-20 text-center">
                <p className="text-4xl">💬</p>
                <h3 className="text-lg font-semibold text-foreground">No chats yet</h3>
                <p className="text-sm text-muted-foreground">Start a conversation scoped to this project.</p>
                <Link
                  href={`/chat?project_id=${project.id}`}
                  className="rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Start Chat →
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">
                  {conversations.length} conversation{conversations.length !== 1 ? "s" : ""}
                </p>
                {conversations.map((conv) => (
                  <Link
                    key={conv.id}
                    href={`/chat/${conv.id}`}
                    className="flex items-center gap-4 rounded-xl border border-border bg-card p-4 hover:border-primary/30 hover:shadow-sm transition-all"
                  >
                    <span className="text-xl">💬</span>
                    <div className="flex-1 min-w-0">
                      <p className="truncate text-sm font-medium text-foreground">{conv.title}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(conv.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                        {" · "}
                        {conv.model_id}
                      </p>
                    </div>
                    <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Members tab */}
        {tab === "members" && (
          <div className="space-y-6">
            {/* Invite form */}
            {canEdit && (
              <div className="rounded-2xl border border-border bg-card p-6">
                <h2 className="mb-4 text-sm font-semibold text-foreground">Invite Member</h2>
                <form onSubmit={handleInvite} className="flex flex-col gap-3 sm:flex-row sm:items-end">
                  <div className="flex-1">
                    <label className="mb-1 block text-xs text-muted-foreground">Email</label>
                    <input
                      type="email"
                      value={inviteEmail}
                      onChange={(e) => setInviteEmail(e.target.value)}
                      placeholder="colleague@example.com"
                      className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                  </div>
                  <div className="w-32">
                    <label className="mb-1 block text-xs text-muted-foreground">Role</label>
                    <select
                      value={inviteRole}
                      onChange={(e) => setInviteRole(e.target.value as ProjectRole)}
                      className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none"
                    >
                      <option value="viewer">Viewer</option>
                      <option value="editor">Editor</option>
                      {isOwner && <option value="owner">Owner</option>}
                    </select>
                  </div>
                  <button
                    type="submit"
                    disabled={inviting}
                    className="rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                  >
                    {inviting ? "Inviting…" : "Invite"}
                  </button>
                </form>
                {inviteError && (
                  <p className="mt-2 text-xs text-destructive">{inviteError}</p>
                )}
              </div>
            )}

            {/* Member list */}
            <div className="rounded-2xl border border-border bg-card overflow-hidden">
              <div className="border-b border-border px-5 py-3">
                <p className="text-sm font-semibold text-foreground">{members.length} member{members.length !== 1 ? "s" : ""}</p>
              </div>
              <div className="divide-y divide-border">
                {members.map((m) => (
                  <div key={m.user_id} className="flex items-center gap-4 px-5 py-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
                      {(m.email ?? "?")[0].toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-foreground truncate">{m.email ?? m.user_id}</p>
                    </div>
                    {isOwner && m.user_id !== myUserId ? (
                      <select
                        value={m.role}
                        onChange={(e) => updateMemberRole(m.user_id, e.target.value)}
                        className="rounded-lg border border-border bg-background px-2 py-1 text-xs focus:outline-none"
                      >
                        {ROLE_OPTIONS.map((r) => (
                          <option key={r} value={r}>{r}</option>
                        ))}
                      </select>
                    ) : (
                      <span className={cn(
                        "rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize",
                        ROLE_COLOR[m.role],
                      )}>
                        {m.role}
                      </span>
                    )}
                    {canEdit && m.role !== "owner" && m.user_id !== myUserId && (
                      <button
                        onClick={() => removeMember(m.user_id)}
                        className="text-xs text-muted-foreground hover:text-destructive transition-colors"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Settings tab */}
        {tab === "settings" && (
          <div className="max-w-2xl space-y-6">
            <form onSubmit={handleSaveSettings} className="rounded-2xl border border-border bg-card p-6 space-y-4">
              <h2 className="text-sm font-semibold text-foreground">Project Settings</h2>
              {settingsMsg && (
                <p className={cn(
                  "text-sm",
                  settingsMsg === "Saved!" ? "text-green-500" : "text-destructive",
                )}>
                  {settingsMsg}
                </p>
              )}
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">Name</label>
                <input
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  disabled={!canEdit}
                  maxLength={256}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-60"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">Description</label>
                <input
                  value={editDesc}
                  onChange={(e) => setEditDesc(e.target.value)}
                  disabled={!canEdit}
                  maxLength={2000}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-60"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  System Prompt
                  <span className="ml-2 font-normal text-muted-foreground/60">
                    Injected into every conversation in this project
                  </span>
                </label>
                <textarea
                  value={editPrompt}
                  onChange={(e) => setEditPrompt(e.target.value)}
                  disabled={!canEdit}
                  rows={6}
                  maxLength={8000}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm font-mono resize-none focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-60"
                />
              </div>
              {canEdit && (
                <button
                  type="submit"
                  disabled={saving}
                  className="rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  {saving ? "Saving…" : "Save Changes"}
                </button>
              )}
            </form>

            {/* Danger zone */}
            {isOwner && (
              <div className="rounded-2xl border border-destructive/30 bg-destructive/5 p-6">
                <h3 className="text-sm font-semibold text-destructive">Danger Zone</h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Deleting this project will remove all conversations and data permanently.
                </p>
                <button
                  onClick={async () => {
                    if (!confirm(`Delete "${project.name}"? This cannot be undone.`)) return;
                    try {
                      const { projectsApi } = await import("@/lib/api");
                      if (token) await projectsApi.delete(project.id, token);
                      router.push("/projects");
                    } catch {}
                  }}
                  className="mt-4 rounded-xl border border-destructive/50 bg-destructive/10 px-4 py-2 text-sm font-medium text-destructive hover:bg-destructive/20 transition-colors"
                >
                  Delete Project
                </button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
