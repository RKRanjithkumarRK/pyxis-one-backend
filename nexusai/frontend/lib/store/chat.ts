import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import type { Conversation, Message, SSEEvent } from "@/lib/types";

type StreamState = "idle" | "streaming" | "done" | "error";

type ChatState = {
  conversations: Conversation[];
  activeConversationId: string | null;
  messages: Record<string, Message[]>;   // conversationId → messages
  streamState: StreamState;
  streamBuffer: string;                  // accumulating tokens for active stream
  streamModel: string | null;
  abortController: AbortController | null;
  compareMode: boolean;
  compareColumns: Record<number, { model: string; tokens: string; done: boolean }>;
};

type ChatActions = {
  setConversations: (convs: Conversation[]) => void;
  upsertConversation: (conv: Conversation) => void;
  removeConversation: (id: string) => void;
  setActive: (id: string | null) => void;
  setMessages: (conversationId: string, msgs: Message[]) => void;
  appendMessage: (conversationId: string, msg: Message) => void;
  updateLastAssistant: (conversationId: string, content: string) => void;
  startStream: (controller: AbortController) => void;
  appendToken: (token: string) => void;
  finalizeStream: (model: string, usage: Record<string, number>) => void;
  setStreamError: (msg: string) => void;
  resetStream: () => void;
  setCompareMode: (on: boolean) => void;
  appendCompareToken: (column: number, model: string, token: string) => void;
  markCompareDone: (column: number) => void;
  resetCompare: () => void;
};

export const useChatStore = create<ChatState & ChatActions>()(
  immer((set) => ({
    conversations: [],
    activeConversationId: null,
    messages: {},
    streamState: "idle",
    streamBuffer: "",
    streamModel: null,
    abortController: null,
    compareMode: false,
    compareColumns: {},

    setConversations: (convs) =>
      set((s) => { s.conversations = convs; }),

    upsertConversation: (conv) =>
      set((s) => {
        const idx = s.conversations.findIndex((c) => c.id === conv.id);
        if (idx >= 0) s.conversations[idx] = conv;
        else s.conversations.unshift(conv);
      }),

    removeConversation: (id) =>
      set((s) => { s.conversations = s.conversations.filter((c) => c.id !== id); }),

    setActive: (id) =>
      set((s) => { s.activeConversationId = id; }),

    setMessages: (cid, msgs) =>
      set((s) => { s.messages[cid] = msgs; }),

    appendMessage: (cid, msg) =>
      set((s) => {
        if (!s.messages[cid]) s.messages[cid] = [];
        s.messages[cid].push(msg);
      }),

    updateLastAssistant: (cid, content) =>
      set((s) => {
        const msgs = s.messages[cid];
        if (!msgs) return;
        const last = msgs[msgs.length - 1];
        if (last && last.role === "assistant") last.content = content;
      }),

    startStream: (controller) =>
      set((s) => {
        s.streamState = "streaming";
        s.streamBuffer = "";
        s.streamModel = null;
        s.abortController = controller;
      }),

    appendToken: (token) =>
      set((s) => { s.streamBuffer += token; }),

    finalizeStream: (model, _usage) =>
      set((s) => {
        s.streamState = "done";
        s.streamModel = model;
        s.abortController = null;
      }),

    setStreamError: (_msg) =>
      set((s) => {
        s.streamState = "error";
        s.abortController = null;
      }),

    resetStream: () =>
      set((s) => {
        s.streamState = "idle";
        s.streamBuffer = "";
        s.streamModel = null;
        s.abortController = null;
      }),

    setCompareMode: (on) =>
      set((s) => { s.compareMode = on; }),

    appendCompareToken: (col, model, token) =>
      set((s) => {
        if (!s.compareColumns[col]) s.compareColumns[col] = { model, tokens: "", done: false };
        s.compareColumns[col].tokens += token;
      }),

    markCompareDone: (col) =>
      set((s) => { if (s.compareColumns[col]) s.compareColumns[col].done = true; }),

    resetCompare: () =>
      set((s) => { s.compareColumns = {}; }),
  }))
);
