"use client";

import { useRouter } from "next/navigation";
import { useConversations } from "@/hooks/use-conversations";
import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";
import { useChatStore } from "@/lib/store/chat";
import { useChat } from "@/hooks/use-chat";
import { useState } from "react";

export default function ChatHomePage() {
  const router = useRouter();
  const { createConversation } = useConversations();
  const { setActive } = useChatStore();
  const { sendMessage, stopStream, isStreaming, streamBuffer, currentMessages } = useChat(null);
  const [model, setModel] = useState("claude-sonnet-4");

  const handleSend = async (text: string, model: string, opts: { webSearch: boolean }) => {
    const conv = await createConversation(model);
    if (!conv) return;
    setActive(conv.id);
    await sendMessage(text, { model, webSearch: opts.webSearch });
    router.push(`/chat/${conv.id}`);
  };

  return (
    <div className="flex flex-col h-full">
      <MessageList
        messages={currentMessages}
        conversationId=""
        isStreaming={isStreaming}
        streamBuffer={streamBuffer}
        onRegenerate={() => {}}
      />
      <Composer
        onSend={handleSend}
        onStop={stopStream}
        isStreaming={isStreaming}
        defaultModel={model}
      />
    </div>
  );
}
