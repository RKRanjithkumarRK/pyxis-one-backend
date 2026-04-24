"use client";

import { useCallback, useRef } from "react";
import { useSession } from "next-auth/react";
import { useChatStore } from "@/lib/store/chat";

export function useCompare() {
  const { data: session } = useSession();
  const { appendCompareToken, markCompareDone, resetCompare } = useChatStore();
  const compareColumns = useChatStore((s) => s.compareColumns);
  const abortRef = useRef<AbortController | null>(null);

  const token = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
  const isComparing =
    Object.keys(compareColumns).length > 0 &&
    Object.values(compareColumns).some((c) => !c.done);

  const startCompare = useCallback(
    async (message: string, models: string[]) => {
      if (!token || models.length < 2) return;
      resetCompare();

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      let resp: Response;
      try {
        resp = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL ?? ""}/api/v1/chat/compare`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ message, models }),
            signal: ctrl.signal,
          }
        );
      } catch {
        return;
      }

      if (!resp.ok || !resp.body) return;

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parts = buf.split("\n\n");
          buf = parts.pop() ?? "";
          for (const part of parts) {
            if (!part.startsWith("data: ")) continue;
            const raw = part.slice(6).trim();
            if (raw === "[DONE]") continue;
            try {
              const ev = JSON.parse(raw) as Record<string, unknown>;
              if (typeof ev.column !== "number") continue;
              if (ev.type === "token") {
                appendCompareToken(ev.column as number, ev.model as string, ev.content as string);
              } else if (ev.type === "done" || ev.type === "error") {
                markCompareDone(ev.column as number);
              }
            } catch {
              // malformed SSE line — skip
            }
          }
        }
      } catch (err: unknown) {
        if ((err as Error)?.name !== "AbortError") console.error("[compare]", err);
      } finally {
        // Ensure all columns are marked done on disconnect / stop
        for (let i = 0; i < models.length; i++) markCompareDone(i);
        abortRef.current = null;
      }
    },
    [token, appendCompareToken, markCompareDone, resetCompare]
  );

  const stopCompare = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { startCompare, stopCompare, isComparing, compareColumns };
}
