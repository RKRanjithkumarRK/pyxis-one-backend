"use client";

import { create } from "zustand";
import type { Agent } from "@/lib/api-types";

type AgentsState = {
  agents: Agent[];
  myAgents: Agent[];
  selectedAgent: Agent | null;
  category: string | null;
  search: string;
  sort: string;
  page: number;
  total: number;
  pages: number;
  loading: boolean;

  setAgents: (agents: Agent[], total: number, pages: number) => void;
  setMyAgents: (agents: Agent[]) => void;
  setSelected: (agent: Agent | null) => void;
  setCategory: (category: string | null) => void;
  setSearch: (search: string) => void;
  setSort: (sort: string) => void;
  setPage: (page: number) => void;
  setLoading: (loading: boolean) => void;
  upsertAgent: (agent: Agent) => void;
  removeAgent: (id: string) => void;
};

export const useAgentsStore = create<AgentsState>((set) => ({
  agents: [],
  myAgents: [],
  selectedAgent: null,
  category: null,
  search: "",
  sort: "popular",
  page: 1,
  total: 0,
  pages: 0,
  loading: false,

  setAgents: (agents, total, pages) => set({ agents, total, pages }),
  setMyAgents: (myAgents) => set({ myAgents }),
  setSelected: (selectedAgent) => set({ selectedAgent }),
  setCategory: (category) => set({ category, page: 1 }),
  setSearch: (search) => set({ search, page: 1 }),
  setSort: (sort) => set({ sort, page: 1 }),
  setPage: (page) => set({ page }),
  setLoading: (loading) => set({ loading }),

  upsertAgent: (agent) =>
    set((s) => {
      const update = (list: Agent[]) => {
        const idx = list.findIndex((a) => a.id === agent.id);
        if (idx >= 0) {
          const next = [...list];
          next[idx] = agent;
          return next;
        }
        return [agent, ...list];
      };
      return { agents: update(s.agents), myAgents: update(s.myAgents) };
    }),

  removeAgent: (id) =>
    set((s) => ({
      agents: s.agents.filter((a) => a.id !== id),
      myAgents: s.myAgents.filter((a) => a.id !== id),
    })),
}));
