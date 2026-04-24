"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useAgents } from "@/hooks/use-agents";
import { AgentCard } from "@/components/agents/AgentCard";
import type { Agent } from "@/lib/api-types";

export default function MyAgentsPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { myAgents, fetchMyAgents, deleteAgent, publishAgent } = useAgents();

  useEffect(() => {
    if (session) fetchMyAgents();
  }, [session]);

  if (status === "loading") {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!session) {
    router.replace("/login");
    return null;
  }

  const handleUse = (agent: Agent) => router.push(`/chat?agent=${agent.id}`);

  const handleDelete = async (agent: Agent) => {
    if (!confirm(`Delete "${agent.name}"?`)) return;
    await deleteAgent(agent.id);
  };

  const handleTogglePublish = async (agent: Agent) => {
    await publishAgent(agent.id, agent.visibility !== "public");
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Link href="/agents" className="text-muted-foreground hover:text-foreground transition-colors">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <h1 className="text-lg font-semibold text-foreground">My Agents</h1>
          </div>
          <Link
            href="/agents/create"
            className="rounded-lg bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            + Create Agent
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {myAgents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
            <span className="text-5xl">🤖</span>
            <p className="text-lg font-medium text-foreground">No agents yet</p>
            <p className="text-sm text-muted-foreground">
              Create your first custom agent and share it with the world
            </p>
            <Link
              href="/agents/create"
              className="mt-2 rounded-lg bg-primary px-5 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Create Agent
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {myAgents.map((agent) => (
              <div key={agent.id} className="group relative">
                <AgentCard agent={agent} onUse={handleUse} />
                {/* Overlay actions */}
                <div className="absolute right-3 top-3 hidden gap-1.5 group-hover:flex">
                  <button
                    onClick={() => handleTogglePublish(agent)}
                    className="rounded-md bg-background/90 border border-border px-2 py-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {agent.visibility === "public" ? "Unpublish" : "Publish"}
                  </button>
                  <Link
                    href={`/agents/${agent.id}/edit`}
                    className="rounded-md bg-background/90 border border-border px-2 py-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Edit
                  </Link>
                  <button
                    onClick={() => handleDelete(agent)}
                    className="rounded-md bg-background/90 border border-destructive/30 px-2 py-1 text-[11px] text-destructive hover:bg-destructive/10 transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
