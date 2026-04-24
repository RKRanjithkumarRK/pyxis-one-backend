"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { AgentEditor } from "@/components/agents/AgentEditor";
import { useAgents } from "@/hooks/use-agents";
import type { CreateAgentPayload } from "@/lib/api-types";

export default function CreateAgentPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { createAgent } = useAgents();
  const [saving, setSaving] = useState(false);

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

  const handleSave = async (payload: CreateAgentPayload) => {
    setSaving(true);
    try {
      const agent = await createAgent(payload);
      router.push(`/agents/${agent.id}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-3xl items-center gap-4 px-6 py-4">
          <Link
            href="/agents"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <h1 className="text-lg font-semibold text-foreground">Create Agent</h1>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-8">
        <AgentEditor
          onSave={handleSave}
          onCancel={() => router.push("/agents")}
          saving={saving}
        />
      </main>
    </div>
  );
}
