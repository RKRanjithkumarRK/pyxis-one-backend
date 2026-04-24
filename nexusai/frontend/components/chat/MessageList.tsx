"use client";

import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { useChatStore } from "@/lib/store/chat";
import type { Message } from "@/lib/types";
import { api } from "@/lib/api";
import { useSession } from "next-auth/react";

type Props = {
  messages: Message[];
  conversationId: string;
  isStreaming: boolean;
  streamBuffer: string;
  onRegenerate: (messageId: string) => void;
};

export function MessageList({
  messages,
  conversationId,
  isStreaming,
  streamBuffer,
  onRegenerate,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const { data: session } = useSession();
  const { appendMessage } = useChatStore();

  // Auto-scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streamBuffer]);

  const token = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;

  const handleFeedback = async (messageId: string, type: "good" | "bad") => {
    await api.post(`/api/v1/chat/${messageId}/feedback`, { feedback: type }, token);
  };

  const handleEdit = async (messageId: string, newContent: string) => {
    if (!token) return;
    await api.post(
      `/api/v1/conversations/${conversationId}/messages/${messageId}/edit`,
      { new_content: newContent },
      token
    );
    // After edit, UI will re-stream from the new branch (handled by parent)
    onRegenerate(messageId);
  };

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-6 p-8">
        <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center">
          <span className="text-3xl">✨</span>
        </div>
        <div className="text-center space-y-1">
          <h2 className="text-lg font-semibold">How can I help you today?</h2>
          <p className="text-sm text-muted-foreground">
            Ask anything — chat, code, research, and more.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 max-w-md w-full">
          {STARTERS.map((s) => (
            <button
              key={s}
              className="px-3 py-2.5 text-sm text-left border border-border rounded-xl hover:bg-accent transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl py-4">
        {messages.map((msg, idx) => {
          const isLastAssistant =
            msg.role === "assistant" && idx === messages.length - 1 && isStreaming;
          return (
            <MessageBubble
              key={msg.id}
              message={msg}
              isStreaming={isLastAssistant}
              streamContent={isLastAssistant ? streamBuffer : undefined}
              onRegenerate={() => onRegenerate(msg.id)}
              onFeedback={(type) => handleFeedback(msg.id, type)}
              onEdit={(newContent) => handleEdit(msg.id, newContent)}
            />
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

const STARTERS = [
  "Explain quantum computing simply",
  "Write a Python web scraper",
  "Summarize this document",
  "Help me debug my code",
];
