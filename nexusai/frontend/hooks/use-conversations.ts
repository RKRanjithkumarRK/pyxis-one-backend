"use client";

import { useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { useChatStore } from "@/lib/store/chat";
import { api } from "@/lib/api";
import type { Conversation } from "@/lib/types";
import { startOfDay, subDays, isAfter } from "date-fns";

function getToken(session: ReturnType<typeof useSession>["data"]): string | undefined {
  return (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
}

export type ConversationGroup = {
  label: string;
  conversations: Conversation[];
};

export function groupConversations(convs: Conversation[]): ConversationGroup[] {
  const now = new Date();
  const todayStart = startOfDay(now);
  const yesterdayStart = subDays(todayStart, 1);
  const day7Start = subDays(todayStart, 7);
  const day30Start = subDays(todayStart, 30);

  const pinned = convs.filter((c) => c.pinned_at);
  const unpinned = convs.filter((c) => !c.pinned_at);

  const today: Conversation[] = [];
  const yesterday: Conversation[] = [];
  const week: Conversation[] = [];
  const month: Conversation[] = [];
  const older: Conversation[] = [];

  for (const c of unpinned) {
    const d = new Date(c.updated_at);
    if (isAfter(d, todayStart)) today.push(c);
    else if (isAfter(d, yesterdayStart)) yesterday.push(c);
    else if (isAfter(d, day7Start)) week.push(c);
    else if (isAfter(d, day30Start)) month.push(c);
    else older.push(c);
  }

  const groups: ConversationGroup[] = [];
  if (pinned.length) groups.push({ label: "Pinned", conversations: pinned });
  if (today.length) groups.push({ label: "Today", conversations: today });
  if (yesterday.length) groups.push({ label: "Yesterday", conversations: yesterday });
  if (week.length) groups.push({ label: "Previous 7 Days", conversations: week });
  if (month.length) groups.push({ label: "Previous 30 Days", conversations: month });
  if (older.length) groups.push({ label: "Older", conversations: older });
  return groups;
}

export function useConversations() {
  const { data: session } = useSession();
  const { conversations, setConversations, upsertConversation, removeConversation } =
    useChatStore();

  const fetchAll = useCallback(async () => {
    const token = getToken(session);
    if (!token) return;
    try {
      const data = (await api.get<{ conversations: Conversation[] }>(
        "/api/v1/conversations",
        token
      ));
      setConversations(data.conversations);
    } catch {
      // silently fail; user may be offline
    }
  }, [session, setConversations]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const createConversation = useCallback(
    async (model: string = "claude-sonnet-4") => {
      const token = getToken(session);
      if (!token) return null;
      const conv = await api.post<Conversation>("/api/v1/conversations", { model_id: model }, token);
      upsertConversation(conv);
      return conv;
    },
    [session, upsertConversation]
  );

  const deleteConversation = useCallback(
    async (id: string) => {
      const token = getToken(session);
      if (!token) return;
      await api.delete(`/api/v1/conversations/${id}`, token);
      removeConversation(id);
    },
    [session, removeConversation]
  );

  const updateConversation = useCallback(
    async (id: string, updates: Record<string, unknown>) => {
      const token = getToken(session);
      if (!token) return;
      const conv = await api.patch<Conversation>(`/api/v1/conversations/${id}`, updates, token);
      upsertConversation(conv);
      return conv;
    },
    [session, upsertConversation]
  );

  return {
    conversations,
    groups: groupConversations(conversations),
    fetchAll,
    createConversation,
    deleteConversation,
    updateConversation,
  };
}
