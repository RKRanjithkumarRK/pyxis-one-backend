"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { memoryApi } from "@/lib/api";
import type { UserMemory } from "@/lib/api-types";

export function useMemory() {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const [mems, stats] = await Promise.all([
        memoryApi.list(token),
        memoryApi.stats(token),
      ]);
      setMemories(mems);
      setCount(stats.count);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const deleteMemory = useCallback(
    async (id: string) => {
      if (!token) return;
      await memoryApi.delete(id, token);
      setMemories((prev) => prev.filter((m) => m.id !== id));
      setCount((c) => (c !== null ? c - 1 : null));
    },
    [token],
  );

  const clearAll = useCallback(async () => {
    if (!token) return;
    const result = await memoryApi.clearAll(token);
    setMemories([]);
    setCount(0);
    return result.deleted;
  }, [token]);

  return { memories, loading, count, refresh, deleteMemory, clearAll };
}
