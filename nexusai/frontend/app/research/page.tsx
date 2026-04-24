"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useResearch } from "@/hooks/use-research";
import { cn } from "@/lib/cn";
import type { ResearchDepth } from "@/lib/api-types";

const DEPTH_OPTIONS: { id: ResearchDepth; label: string; desc: string; time: string }[] = [
  { id: "quick", label: "Quick", desc: "3–6 sources, ~1 min", time: "~1 min" },
  { id: "standard", label: "Standard", desc: "8–12 sources, ~3 min", time: "~3 min" },
  { id: "deep", label: "Deep", desc: "14–18 sources, ~7 min", time: "~7 min" },
];

const EXAMPLE_QUERIES = [
  "What are the most effective treatments for long COVID?",
  "How does Anthropic's constitutional AI approach work?",
  "What is the current state of nuclear fusion energy?",
  "Compare the economic impacts of UBI pilots worldwide",
  "How are large language models being used in drug discovery?",
];

export default function ResearchPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { reports, loading, fetchReports, startResearch, deleteReport } = useResearch();
  const [query, setQuery] = useState("");
  const [depth, setDepth] = useState<ResearchDepth>("standard");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (session) fetchReports();
  }, [session]);

  if (status === "loading") {
    return <div className="flex h-screen items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" /></div>;
  }

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || query.trim().length < 5) {
      setError("Query must be at least 5 characters");
      return;
    }
    if (!session) { router.push("/login"); return; }
    setError(null);
    setStarting(true);
    try {
      const report = await startResearch(query.trim(), depth);
      router.push(`/research/${report.id}`);
    } catch (err: any) {
      setError(err?.message ?? "Failed to start research");
      setStarting(false);
    }
  };

  const handleExampleClick = (q: string) => setQuery(q);

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Link href="/chat" className="text-muted-foreground hover:text-foreground transition-colors">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Deep Research</h1>
              <p className="text-xs text-muted-foreground">AI-powered multi-source research reports</p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-10">
        {/* Hero */}
        <div className="mb-10 text-center">
          <span className="text-5xl">🔬</span>
          <h2 className="mt-4 text-3xl font-bold text-foreground">What do you want to research?</h2>
          <p className="mt-2 text-muted-foreground">
            NexusAI searches the web, reads sources, and writes a comprehensive report with inline citations.
          </p>
        </div>

        {/* Search form */}
        <form onSubmit={handleStart} className="flex flex-col gap-4">
          {error && (
            <div className="rounded-lg bg-destructive/10 border border-destructive/30 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          )}
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. What is the current state of quantum computing and its near-term applications?"
            rows={3}
            maxLength={1000}
            className="w-full rounded-xl border border-border bg-card px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/50 shadow-sm"
          />

          {/* Depth selector */}
          <div className="grid grid-cols-3 gap-3">
            {DEPTH_OPTIONS.map((opt) => (
              <label
                key={opt.id}
                className={cn(
                  "flex cursor-pointer flex-col gap-1 rounded-xl border p-3 transition-colors",
                  depth === opt.id
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/30",
                )}
              >
                <input
                  type="radio"
                  name="depth"
                  value={opt.id}
                  checked={depth === opt.id}
                  onChange={() => setDepth(opt.id)}
                  className="sr-only"
                />
                <span className="text-sm font-semibold text-foreground">{opt.label}</span>
                <span className="text-xs text-muted-foreground">{opt.desc}</span>
              </label>
            ))}
          </div>

          <button
            type="submit"
            disabled={starting || !query.trim()}
            className="w-full rounded-xl bg-primary py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {starting ? (
              <span className="flex items-center justify-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Starting research…
              </span>
            ) : (
              "Start Research →"
            )}
          </button>
        </form>

        {/* Examples */}
        <div className="mt-8">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Example queries
          </p>
          <div className="flex flex-col gap-2">
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                onClick={() => handleExampleClick(q)}
                className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2.5 text-left text-sm text-muted-foreground hover:border-primary/30 hover:text-foreground transition-colors"
              >
                <span className="text-primary">→</span>
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Past reports */}
        {reports.length > 0 && (
          <div className="mt-12">
            <p className="mb-4 text-sm font-semibold text-foreground">Past Research</p>
            <div className="flex flex-col gap-3">
              {reports.map((r) => (
                <div
                  key={r.id}
                  className="flex items-center gap-4 rounded-xl border border-border bg-card p-4 group"
                >
                  <div className="text-2xl">
                    {r.status === "complete" ? "📄" : r.status === "error" ? "❌" : "⏳"}
                  </div>
                  <div className="flex-1 min-w-0">
                    <Link
                      href={`/research/${r.id}`}
                      className="block text-sm font-medium text-foreground hover:text-primary transition-colors truncate"
                    >
                      {r.title ?? r.query}
                    </Link>
                    <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                      <span className="capitalize">{r.status}</span>
                      {r.sources_count > 0 && <><span>·</span><span>{r.sources_count} sources</span></>}
                      <span>·</span>
                      <span>{new Date(r.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Link
                      href={`/research/${r.id}`}
                      className="rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-accent transition-colors"
                    >
                      View
                    </Link>
                    <button
                      onClick={() => deleteReport(r.id)}
                      className="rounded-md border border-destructive/30 px-2.5 py-1 text-xs text-destructive hover:bg-destructive/10 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
