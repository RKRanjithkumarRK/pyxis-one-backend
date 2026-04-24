"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useConversations } from "@/hooks/use-conversations";
import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";
import { useChatStore } from "@/lib/store/chat";
import { useChat } from "@/hooks/use-chat";
import { agentsApi } from "@/lib/api";
import { useSession } from "next-auth/react";
import type { Agent } from "@/lib/api-types";

export default function ChatHomePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;
  const { createConversation } = useConversations();
  const { setActive } = useChatStore();
  const { sendMessage, stopStream, isStreaming, streamBuffer, currentMessages } = useChat(null);
  const [model, setModel] = useState("claude-sonnet-4");
  const [agent, setAgent] = useState<Agent | null>(null);

  const agentId = searchParams.get("agent");
  const preMessage = searchParams.get("message");

  useEffect(() => {
    if (!agentId) return;
    agentsApi.get(agentId, token)
      .then((a) => {
        setAgent(a);
        setModel(a.default_model);
      })
      .catch(() => {});
  }, [agentId, token]);

  const handleSend = async (text: string, mdl: string, opts: { webSearch: boolean }) => {
    const conv = await createConversation(mdl);
    if (!conv) return;
    setActive(conv.id);
    await sendMessage(text, { model: mdl, webSearch: opts.webSearch, agentId: agentId ?? undefined });
    router.push(`/chat/${conv.id}`);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Agent banner */}
      {agent && (
        <div className="flex items-center gap-3 border-b border-border bg-card/50 px-6 py-3">
          <span className="text-2xl">{agent.icon ?? "🤖"}</span>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-foreground">{agent.name}</p>
            {agent.description && (
              <p className="text-xs text-muted-foreground truncate">{agent.description}</p>
            )}
          </div>
          <button
            onClick={() => { setAgent(null); router.replace("/chat"); }}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            ✕ Clear agent
          </button>
        </div>
      )}

      <MessageList
        messages={currentMessages}
        conversationId=""
        isStreaming={isStreaming}
        streamBuffer={streamBuffer}
        onRegenerate={() => {}}
      />

      {/* Starters */}
      {agent?.starters && currentMessages.length === 0 && (
        <div className="px-4 pb-2">
          <p className="mb-2 text-xs text-muted-foreground text-center">Try asking:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {agent.starters.map((s, i) => (
              <button
                key={i}
                onClick={() => handleSend(s, model, { webSearch: false })}
                className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground hover:border-primary/30 hover:text-foreground transition-colors max-w-xs truncate"
                title={s}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      <Composer
        onSend={handleSend}
        onStop={stopStream}
        isStreaming={isStreaming}
        defaultModel={model}
      />
    </div>
  );
}
