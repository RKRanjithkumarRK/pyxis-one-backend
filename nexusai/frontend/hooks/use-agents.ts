"use client";

import { useCallback, useEffect } from "react";
import { useSession } from "next-auth/react";
import { agentsApi } from "@/lib/api";
import { useAgentsStore } from "@/lib/store/agents";
import type { CreateAgentPayload, UpdateAgentPayload } from "@/lib/api-types";

export function useAgents() {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;
  const store = useAgentsStore();

  const fetchAgents = useCallback(async () => {
    store.setLoading(true);
    try {
      const res = await agentsApi.list(
        {
          category: store.category ?? undefined,
          search: store.search || undefined,
          sort: store.sort,
          page: store.page,
          page_size: 24,
        },
        token,
      );
      store.setAgents(res.agents, res.total, res.pages);
    } catch (err) {
      console.error("Failed to fetch agents:", err);
    } finally {
      store.setLoading(false);
    }
  }, [store.category, store.search, store.sort, store.page, token]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const fetchMyAgents = useCallback(async () => {
    if (!token) return;
    try {
      const agents = await agentsApi.mine(token);
      store.setMyAgents(agents);
    } catch (err) {
      console.error("Failed to fetch my agents:", err);
    }
  }, [token]);

  const createAgent = useCallback(
    async (payload: CreateAgentPayload) => {
      if (!token) throw new Error("Not authenticated");
      const agent = await agentsApi.create(payload, token);
      store.upsertAgent(agent);
      return agent;
    },
    [token],
  );

  const updateAgent = useCallback(
    async (id: string, payload: UpdateAgentPayload) => {
      if (!token) throw new Error("Not authenticated");
      const agent = await agentsApi.update(id, payload, token);
      store.upsertAgent(agent);
      return agent;
    },
    [token],
  );

  const deleteAgent = useCallback(
    async (id: string) => {
      if (!token) throw new Error("Not authenticated");
      await agentsApi.delete(id, token);
      store.removeAgent(id);
    },
    [token],
  );

  const publishAgent = useCallback(
    async (id: string, isPublic: boolean) => {
      if (!token) throw new Error("Not authenticated");
      const agent = await agentsApi.publish(id, isPublic, token);
      store.upsertAgent(agent);
      return agent;
    },
    [token],
  );

  const rateAgent = useCallback(
    async (id: string, rating: number) => {
      if (!token) throw new Error("Not authenticated");
      const agent = await agentsApi.rate(id, rating, token);
      store.upsertAgent(agent);
      return agent;
    },
    [token],
  );

  return {
    agents: store.agents,
    myAgents: store.myAgents,
    total: store.total,
    pages: store.pages,
    loading: store.loading,
    category: store.category,
    search: store.search,
    sort: store.sort,
    page: store.page,
    setCategory: store.setCategory,
    setSearch: store.setSearch,
    setSort: store.setSort,
    setPage: store.setPage,
    fetchAgents,
    fetchMyAgents,
    createAgent,
    updateAgent,
    deleteAgent,
    publishAgent,
    rateAgent,
  };
}
