"use client";

import { useState, useCallback, useEffect } from "react";
import { useSession } from "next-auth/react";
import { imageApi } from "@/lib/api";
import type { ImageRequest } from "@/lib/api-types";

function useToken() {
  const { data: session } = useSession();
  return (session as any)?.accessToken as string | undefined;
}

export function useImageStudio() {
  const token = useToken();
  const [history, setHistory] = useState<ImageRequest[]>([]);
  const [generating, setGenerating] = useState(false);
  const [current, setCurrent] = useState<ImageRequest | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    if (!token) return;
    try {
      const data = await imageApi.history(token);
      setHistory(data);
    } catch {}
  }, [token]);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // Poll the current request until done
  useEffect(() => {
    if (!current || !token) return;
    if (current.status === "done" || current.status === "error") return;

    const id = setInterval(async () => {
      try {
        const updated = await imageApi.getRequest(current.id, token);
        setCurrent(updated);
        if (updated.status === "done" || updated.status === "error") {
          setHistory((prev) => {
            const idx = prev.findIndex((r) => r.id === updated.id);
            if (idx === -1) return [updated, ...prev];
            const next = [...prev];
            next[idx] = updated;
            return next;
          });
          setGenerating(false);
        }
      } catch {}
    }, 2000);

    return () => clearInterval(id);
  }, [current, token]);

  const generate = useCallback(
    async (params: {
      prompt: string;
      negative_prompt?: string;
      model?: string;
      width?: number;
      height?: number;
      num_images?: number;
    }) => {
      if (!token) return;
      setGenerating(true);
      setError(null);
      try {
        const req = await imageApi.generate(params, token);
        setCurrent(req);
        setHistory((prev) => [req, ...prev]);
        return req;
      } catch (e: any) {
        setError(e.message);
        setGenerating(false);
        return null;
      }
    },
    [token],
  );

  const upscale = useCallback(
    async (imageUrl: string) => {
      if (!token) return null;
      const { url } = await imageApi.upscale(imageUrl, 4, token);
      return url;
    },
    [token],
  );

  const removeBg = useCallback(
    async (imageUrl: string) => {
      if (!token) return null;
      const { url } = await imageApi.removeBg(imageUrl, token);
      return url;
    },
    [token],
  );

  return { history, generating, current, error, generate, upscale, removeBg, reload: loadHistory };
}
