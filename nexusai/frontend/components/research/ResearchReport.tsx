"use client";

import { useState } from "react";
import { cn } from "@/lib/cn";
import type { ResearchReportData } from "@/lib/api-types";

type Props = {
  report: ResearchReportData;
  query: string;
};

function CitationRef({ id, citations }: { id: number; citations: ResearchReportData["citations"] }) {
  const [open, setOpen] = useState(false);
  const cit = citations.find((c) => c.id === id);
  if (!cit) return <span className="text-muted-foreground">[{id}]</span>;

  return (
    <span className="relative inline-block">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex h-4 w-4 items-center justify-center rounded text-[10px] font-bold bg-primary/10 text-primary hover:bg-primary/20 transition-colors cursor-pointer"
      >
        {id}
      </button>
      {open && (
        <span className="absolute bottom-full left-0 z-20 mb-1 w-64 rounded-lg border border-border bg-card p-3 shadow-lg text-xs">
          <span className="font-semibold text-foreground block mb-1 truncate">{cit.title}</span>
          {cit.snippet && <span className="text-muted-foreground block mb-2 line-clamp-3">{cit.snippet}</span>}
          <a
            href={cit.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline truncate block"
            onClick={(e) => e.stopPropagation()}
          >
            {cit.url}
          </a>
        </span>
      )}
    </span>
  );
}

function renderWithCitations(text: string, citations: ResearchReportData["citations"]) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      return <CitationRef key={i} id={parseInt(match[1])} citations={citations} />;
    }
    return <span key={i}>{part}</span>;
  });
}

export function ResearchReport({ report, query }: Props) {
  const [activeSection, setActiveSection] = useState<number | null>(null);
  const [copiedReport, setCopiedReport] = useState(false);

  const handleCopyMarkdown = () => {
    const md = [
      `# ${report.title}`,
      "",
      `## Executive Summary`,
      report.executive_summary,
      "",
      ...report.sections.flatMap((s) => [`## ${s.heading}`, "", s.content, ""]),
      "## References",
      ...report.citations.map((c) => `[${c.id}] [${c.title}](${c.url})`),
    ].join("\n");
    navigator.clipboard.writeText(md);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2000);
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">{report.title}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span>{report.citations.length} sources</span>
            <span>·</span>
            <span>{report.sections.length} sections</span>
            <span>·</span>
            <span className="capitalize">{report.depth} research</span>
            {report.generated_at && (
              <>
                <span>·</span>
                <span>{new Date(report.generated_at).toLocaleDateString()}</span>
              </>
            )}
          </div>
        </div>
        <button
          onClick={handleCopyMarkdown}
          className="shrink-0 rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
        >
          {copiedReport ? "Copied!" : "Copy MD"}
        </button>
      </div>

      {/* Key findings */}
      {report.key_findings && report.key_findings.length > 0 && (
        <section className="rounded-xl border border-primary/20 bg-primary/5 p-5">
          <h2 className="mb-3 text-sm font-semibold text-primary">Key Findings</h2>
          <ul className="space-y-2">
            {report.key_findings.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                <span className="mt-0.5 shrink-0 text-primary">→</span>
                <span>{renderWithCitations(f, report.citations)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Executive summary */}
      <section className="rounded-xl border border-border bg-card p-5">
        <h2 className="mb-3 text-sm font-semibold text-foreground">Executive Summary</h2>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {renderWithCitations(report.executive_summary, report.citations)}
        </p>
      </section>

      {/* Table of contents */}
      {report.sections.length > 2 && (
        <nav className="rounded-xl border border-border bg-card/50 p-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Contents
          </p>
          <ol className="space-y-1">
            {report.sections.map((s, i) => (
              <li key={i}>
                <button
                  onClick={() => {
                    setActiveSection(i);
                    document.getElementById(`section-${i}`)?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className="text-sm text-primary hover:underline text-left"
                >
                  {i + 1}. {s.heading}
                </button>
              </li>
            ))}
          </ol>
        </nav>
      )}

      {/* Report sections */}
      <div className="space-y-6">
        {report.sections.map((section, i) => (
          <section
            key={i}
            id={`section-${i}`}
            className={cn(
              "rounded-xl border border-border bg-card p-5 transition-all",
              activeSection === i && "border-primary/40 ring-1 ring-primary/20",
            )}
          >
            <h2 className="mb-3 text-base font-semibold text-foreground">{section.heading}</h2>
            <div className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
              {renderWithCitations(section.content, report.citations)}
            </div>
          </section>
        ))}
      </div>

      {/* Citations */}
      {report.citations.length > 0 && (
        <section className="rounded-xl border border-border bg-card p-5">
          <h2 className="mb-4 text-sm font-semibold text-foreground">
            Sources ({report.citations.length})
          </h2>
          <div className="space-y-3">
            {report.citations.map((cit) => (
              <div key={cit.id} className="flex items-start gap-3">
                <span className="shrink-0 flex h-5 w-5 items-center justify-center rounded bg-muted text-[10px] font-bold text-muted-foreground">
                  {cit.id}
                </span>
                <div className="min-w-0 flex-1">
                  <a
                    href={cit.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-foreground hover:text-primary transition-colors truncate block"
                  >
                    {cit.title}
                  </a>
                  {cit.snippet && (
                    <p className="mt-0.5 text-xs text-muted-foreground line-clamp-2">
                      {cit.snippet}
                    </p>
                  )}
                  <a
                    href={cit.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-0.5 text-[11px] text-primary hover:underline truncate block"
                  >
                    {cit.url}
                  </a>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
