"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { AgentEditor } from "@/components/agents/AgentEditor";
import { agentsApi } from "@/lib/api";
import { useAgents } from "@/hooks/use-agents";
import type { Agent, UpdateAgentPayload } from "@/lib/api-types";

type Props = { params: Promise<{ agentId: string }> };

export default function EditAgentPage({ params }: Props) {
  const { agentId } = use(params);
  const router = useRouter();
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;
  const { updateAgent } = useAgents();

  const [agent, setAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const a = await agentsApi.get(agentId, token);
        if (a.is_builtin) {
          setError("Built-in agents cannot be edited");
          return;
        }
        const userId = (session as any)?.user?.id;
        if (a.creator_id !== userId) {
          setError("You don't own this agent");
          return;
        }
        setAgent(a);
      } catch {
        setError("Agent not found");
      } finally {
        setLoading(false);
      }
    };
    if (session !== undefined) load();
  }, [agentId, token, session]);

  const handleSave = async (payload: UpdateAgentPayload) => {
    if (!agent) return;
    setSaving(true);
    try {
      await updateAgent(agent.id, payload);
      router.push(`/agents/${agent.id}`);
    } finally {
      setSaving(false);
    }
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
        <p className="text-lg font-medium text-foreground">{error ?? "Agent not found"}</p>
        <Link href="/agents" className="text-primary hover:underline text-sm">
          ← Back to Agent Store
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-3xl items-center gap-4 px-6 py-4">
          <Link
            href={`/agents/${agent.id}`}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <div>
            <h1 className="text-lg font-semibold text-foreground">Edit Agent</h1>
            <p className="text-xs text-muted-foreground">{agent.name} · v{agent.version}</p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-8">
        <AgentEditor
          initial={agent}
          onSave={handleSave}
          onCancel={() => router.push(`/agents/${agent.id}`)}
          saving={saving}
        />
      </main>
    </div>
  );
}
