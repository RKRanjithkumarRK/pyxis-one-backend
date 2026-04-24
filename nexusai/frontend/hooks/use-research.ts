"use client";

import { useCallback, useState } from "react";
import { useSession } from "next-auth/react";
import { researchApi, streamResearchProgress } from "@/lib/api";
import type { ResearchDepth, ResearchProgressEvent, ResearchReport } from "@/lib/api-types";

export function useResearch() {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [reports, setReports] = useState<ResearchReport[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchReports = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const data = await researchApi.list(token);
      setReports(data);
    } catch (err) {
      console.error("Failed to fetch research reports:", err);
    } finally {
      setLoading(false);
    }
  }, [token]);

  const startResearch = useCallback(
    async (query: string, depth: ResearchDepth): Promise<ResearchReport> => {
      if (!token) throw new Error("Not authenticated");
      const report = await researchApi.start(query, depth, token);
      setReports((prev) => [report, ...prev]);
      return report;
    },
    [token],
  );

  const deleteReport = useCallback(
    async (id: string) => {
      if (!token) throw new Error("Not authenticated");
      await researchApi.delete(id, token);
      setReports((prev) => prev.filter((r) => r.id !== id));
    },
    [token],
  );

  const updateReport = useCallback((id: string, updates: Partial<ResearchReport>) => {
    setReports((prev) =>
      prev.map((r) => (r.id === id ? { ...r, ...updates } : r)),
    );
  }, []);

  return {
    reports,
    loading,
    token,
    fetchReports,
    startResearch,
    deleteReport,
    updateReport,
  };
}

export function useResearchStream(
  reportId: string,
  token: string | undefined,
  onEvent: (event: ResearchProgressEvent) => void,
  onComplete: () => void,
) {
  const start = useCallback(async () => {
    if (!token) return;
    try {
      for await (const event of streamResearchProgress(reportId, token)) {
        onEvent(event);
        if (event.stage === "complete" || event.stage === "error") {
          onComplete();
          break;
        }
      }
    } catch (err) {
      console.error("Research stream error:", err);
      onComplete();
    }
  }, [reportId, token, onEvent, onComplete]);

  return { startStream: start };
}
