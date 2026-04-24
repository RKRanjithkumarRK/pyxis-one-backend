"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { useChatStore } from "@/lib/store/chat";
import { useChat } from "@/hooks/use-chat";
import { MessageList } from "@/components/chat/MessageList";
import { Composer } from "@/components/chat/Composer";
import { api } from "@/lib/api";
import type { Message } from "@/lib/types";

export default function ConversationPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const { data: session } = useSession();
  const { setMessages, setActive } = useChatStore();
  const { sendMessage, stopStream, isStreaming, streamBuffer, currentMessages } =
    useChat(conversationId);
  const [loadError, setLoadError] = useState(false);

  const token = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;

  // Load messages when conversation changes
  useEffect(() => {
    if (!conversationId || !token) return;
    setActive(conversationId);

    api
      .get<{ messages: Message[] }>(`/api/v1/conversations/${conversationId}/messages`, token)
      .then((data) => setMessages(conversationId, data.messages))
      .catch(() => setLoadError(true));
  }, [conversationId, token, setActive, setMessages]);

  const handleSend = async (text: string, model: string, opts: { webSearch: boolean }) => {
    await sendMessage(text, { model, webSearch: opts.webSearch });
  };

  const handleRegenerate = async (messageId: string) => {
    // Find the user message before this assistant message and resend
    const msgs = currentMessages;
    const idx = msgs.findIndex((m) => m.id === messageId);
    if (idx <= 0) return;
    const userMsg = msgs.slice(0, idx).reverse().find((m) => m.role === "user");
    if (!userMsg) return;
    await sendMessage(userMsg.content, { model: "claude-sonnet-4", webSearch: false });
  };

  if (loadError) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground text-sm">
        Failed to load conversation. <a href="/chat" className="ml-1 text-primary underline">Go home</a>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <MessageList
        messages={currentMessages}
        conversationId={conversationId}
        isStreaming={isStreaming}
        streamBuffer={streamBuffer}
        onRegenerate={handleRegenerate}
      />
      <Composer
        onSend={handleSend}
        onStop={stopStream}
        isStreaming={isStreaming}
      />
    </div>
  );
}
