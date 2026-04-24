"use client";

import { useCallback, useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useAgents } from "@/hooks/use-agents";
import { AgentCard } from "@/components/agents/AgentCard";
import { AgentFilters } from "@/components/agents/AgentFilters";
import type { Agent } from "@/lib/api-types";

function SearchIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
    </svg>
  );
}

function Skeleton() {
  return (
    <div className="animate-pulse rounded-xl border border-border bg-card p-4 h-44">
      <div className="flex gap-3 mb-3">
        <div className="h-10 w-10 rounded-lg bg-muted" />
        <div className="flex-1 space-y-2">
          <div className="h-3 w-3/4 rounded bg-muted" />
          <div className="h-2 w-1/3 rounded bg-muted" />
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-2 rounded bg-muted" />
        <div className="h-2 w-5/6 rounded bg-muted" />
      </div>
    </div>
  );
}

export default function AgentStorePage() {
  const router = useRouter();
  const { data: session } = useSession();
  const {
    agents, total, pages, loading,
    category, search, sort, page,
    setCategory, setSearch, setSort, setPage,
  } = useAgents();

  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleSearch = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      if (searchTimeout.current) clearTimeout(searchTimeout.current);
      searchTimeout.current = setTimeout(() => setSearch(val), 300);
    },
    [setSearch],
  );

  const handleUse = (agent: Agent) => {
    router.push(`/chat?agent=${agent.id}`);
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      {/* Top bar */}
      <header className="flex items-center justify-between border-b border-border px-6 py-4">
        <div className="flex items-center gap-3">
          <Link href="/chat" className="text-muted-foreground hover:text-foreground transition-colors">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <div>
            <h1 className="text-lg font-semibold text-foreground">Agent Store</h1>
            <p className="text-xs text-muted-foreground">{total.toLocaleString()} agents available</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {session && (
            <>
              <Link
                href="/agents/mine"
                className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                My Agents
              </Link>
              <Link
                href="/agents/create"
                className="rounded-lg bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                + Create Agent
              </Link>
            </>
          )}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Filters sidebar */}
        <div className="hidden md:flex flex-col border-r border-border bg-card/40 px-4 py-6 overflow-y-auto">
          <AgentFilters
            category={category}
            sort={sort}
            onCategoryChange={setCategory}
            onSortChange={setSort}
          />
        </div>

        {/* Main content */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {/* Search bar */}
          <div className="border-b border-border px-6 py-3">
            <div className="relative max-w-xl">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                <SearchIcon />
              </span>
              <input
                type="search"
                placeholder="Search agents…"
                defaultValue={search}
                onChange={handleSearch}
                className="h-9 w-full rounded-lg border border-border bg-muted pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
          </div>

          {/* Agent grid */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            {loading ? (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {Array.from({ length: 12 }).map((_, i) => (
                  <Skeleton key={i} />
                ))}
              </div>
            ) : agents.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 text-center gap-3">
                <span className="text-5xl">🤖</span>
                <p className="text-base font-medium text-foreground">No agents found</p>
                <p className="text-sm text-muted-foreground">
                  {search ? "Try a different search term" : "No agents in this category yet"}
                </p>
                {session && (
                  <Link
                    href="/agents/create"
                    className="mt-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                  >
                    Create the first one
                  </Link>
                )}
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {agents.map((agent) => (
                    <AgentCard key={agent.id} agent={agent} onUse={handleUse} />
                  ))}
                </div>

                {/* Pagination */}
                {pages > 1 && (
                  <div className="mt-8 flex items-center justify-center gap-2">
                    <button
                      onClick={() => setPage(page - 1)}
                      disabled={page <= 1}
                      className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-40"
                    >
                      Previous
                    </button>
                    <span className="text-sm text-muted-foreground">
                      Page {page} of {pages}
                    </span>
                    <button
                      onClick={() => setPage(page + 1)}
                      disabled={page >= pages}
                      className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-40"
                    >
                      Next
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
