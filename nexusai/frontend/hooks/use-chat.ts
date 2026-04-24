"use client";

import { useCallback, useRef } from "react";
import { useSession } from "next-auth/react";
import { useChatStore } from "@/lib/store/chat";
import { streamSSE } from "@/lib/api";
import type { Message, SSEEvent } from "@/lib/types";

function getToken(session: ReturnType<typeof useSession>["data"]): string | undefined {
  return (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
}

export function useChat(conversationId: string | null) {
  const { data: session } = useSession();
  const {
    streamState,
    streamBuffer,
    messages,
    startStream,
    appendToken,
    finalizeStream,
    setStreamError,
    resetStream,
    appendMessage,
    upsertConversation,
    setActive,
  } = useChatStore();

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (
      text: string,
      opts: {
        model: string;
        attachments?: unknown[];
        webSearch?: boolean;
        useMemory?: boolean;
        agentId?: string;
      }
    ) => {
      const token = getToken(session);
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      startStream(ctrl);

      // Optimistically add user message
      const tempUserMsg: Message = {
        id: crypto.randomUUID(),
        conversation_id: conversationId ?? "",
        branch_id: "",
        parent_branch_id: null,
        sequence: (messages[conversationId ?? ""]?.length ?? 0),
        role: "user",
        content: text,
        model_id: null,
        usage: null,
        citations: null,
        attachments: (opts.attachments as Message["attachments"]) ?? null,
        feedback: null,
        created_at: new Date().toISOString(),
      };
      if (conversationId) appendMessage(conversationId, tempUserMsg);

      // Optimistically add empty assistant message
      const tempAssistantId = crypto.randomUUID();
      const tempAssistant: Message = {
        id: tempAssistantId,
        conversation_id: conversationId ?? "",
        branch_id: "",
        parent_branch_id: null,
        sequence: tempUserMsg.sequence + 1,
        role: "assistant",
        content: "",
        model_id: opts.model,
        usage: null,
        citations: null,
        attachments: null,
        feedback: null,
        created_at: new Date().toISOString(),
      };
      if (conversationId) appendMessage(conversationId, tempAssistant);

      let activeConvId = conversationId;

      try {
        const gen = streamSSE(
          "/api/v1/chat/stream",
          {
            conversation_id: conversationId,
            model: opts.model,
            message: text,
            attachments: opts.attachments ?? [],
            web_search: opts.webSearch ?? false,
            use_memory: opts.useMemory ?? true,
            agent_id: opts.agentId,
          },
          token
        );

        let accumulated = "";

        for await (const ev of gen) {
          const event = ev as SSEEvent & { type: string };

          if (event.type === "conversation_id") {
            activeConvId = (ev as Record<string, string>).conversation_id;
            setActive(activeConvId);
          } else if (event.type === "token") {
            accumulated += (event as { content: string }).content;
            appendToken((event as { content: string }).content);
            // Update optimistic assistant message
            if (activeConvId) {
              const msgs = useChatStore.getState().messages[activeConvId] ?? [];
              const last = msgs[msgs.length - 1];
              if (last?.role === "assistant") {
                useChatStore.getState().updateLastAssistant(activeConvId, accumulated);
              }
            }
          } else if (event.type === "done") {
            const done = event as { model: string; usage: Record<string, number> };
            finalizeStream(done.model, done.usage);
          } else if (event.type === "error") {
            setStreamError((event as { message: string }).message);
          }
        }
      } catch (err: unknown) {
        if ((err as Error)?.name !== "AbortError") {
          setStreamError(String(err));
        }
      } finally {
        abortRef.current = null;
      }
    },
    [session, conversationId, messages, startStream, appendToken, finalizeStream, setStreamError, appendMessage, setActive]
  );

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    resetStream();
  }, [resetStream]);

  return {
    sendMessage,
    stopStream,
    streamState,
    streamBuffer,
    isStreaming: streamState === "streaming",
    currentMessages: messages[conversationId ?? ""] ?? [],
  };
}
