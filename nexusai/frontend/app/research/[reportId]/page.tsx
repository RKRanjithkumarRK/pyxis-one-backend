"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { researchApi } from "@/lib/api";
import { useResearchStream } from "@/hooks/use-research";
import { ResearchProgress } from "@/components/research/ResearchProgress";
import { ResearchReport } from "@/components/research/ResearchReport";
import type { ResearchProgressEvent, ResearchReport as ResearchReportType } from "@/lib/api-types";

type Props = { params: Promise<{ reportId: string }> };

export default function ResearchReportPage({ params }: Props) {
  const { reportId } = use(params);
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [report, setReport] = useState<ResearchReportType | null>(null);
  const [loading, setLoading] = useState(true);
  const [latestEvent, setLatestEvent] = useState<ResearchProgressEvent | null>(null);
  const [streamDone, setStreamDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isRunning = report && (report.status === "pending" || report.status === "running");

  const reloadReport = useCallback(async () => {
    if (!token) return;
    try {
      const r = await researchApi.get(reportId, token);
      setReport(r);
    } catch {
      setError("Report not found");
    }
  }, [reportId, token]);

  useEffect(() => {
    const load = async () => {
      if (!token) return;
      setLoading(true);
      try {
        const r = await researchApi.get(reportId, token);
        setReport(r);
      } catch {
        setError("Report not found");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [reportId, token]);

  const handleEvent = useCallback((ev: ResearchProgressEvent) => {
    setLatestEvent(ev);
  }, []);

  const handleComplete = useCallback(() => {
    setStreamDone(true);
    reloadReport();
  }, [reloadReport]);

  const { startStream } = useResearchStream(reportId, token, handleEvent, handleComplete);

  useEffect(() => {
    if (isRunning && !streamDone) {
      startStream();
    }
  }, [isRunning, streamDone, startStream]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4">
        <span className="text-5xl">🔬</span>
        <p className="text-lg font-medium text-foreground">{error ?? "Report not found"}</p>
        <Link href="/research" className="text-primary hover:underline text-sm">
          ← Back to Research
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Link href="/research" className="text-muted-foreground hover:text-foreground transition-colors">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <div>
              <h1 className="max-w-lg truncate text-base font-semibold text-foreground">
                {report.title ?? report.query}
              </h1>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span
                  className={
                    report.status === "complete"
                      ? "text-green-500"
                      : report.status === "error"
                      ? "text-destructive"
                      : "text-amber-500"
                  }
                >
                  ●
                </span>
                <span className="capitalize">{report.status}</span>
                {report.sources_count > 0 && (
                  <><span>·</span><span>{report.sources_count} sources</span></>
                )}
              </div>
            </div>
          </div>
          <Link
            href="/research"
            className="rounded-lg bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            New Research
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-8">
        {(report.status === "pending" || report.status === "running") && (
          <ResearchProgress event={latestEvent} query={report.query} />
        )}

        {report.status === "error" && (
          <div className="flex flex-col items-center gap-4 py-16 text-center">
            <span className="text-5xl">❌</span>
            <p className="text-lg font-medium text-foreground">Research failed</p>
            <p className="text-sm text-destructive">{report.error}</p>
            <Link
              href="/research"
              className="mt-2 rounded-lg bg-primary px-5 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              Try Again
            </Link>
          </div>
        )}

        {report.status === "complete" && report.report && (
          <ResearchReport report={report.report} query={report.query} />
        )}
      </main>
    </div>
  );
}
