"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useSession } from "next-auth/react";
import { canvasApi } from "@/lib/api";
import type { CanvasDoc, CanvasDocListItem, CanvasDocVersion } from "@/lib/api-types";

const WS_BASE =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/^http/, "ws");

// ─── Document list ────────────────────────────────────────

export function useCanvasList() {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;
  const [docs, setDocs] = useState<CanvasDocListItem[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const data = await canvasApi.list(token);
      setDocs(data);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = useCallback(
    async (title = "Untitled") => {
      if (!token) throw new Error("Not authenticated");
      const doc = await canvasApi.create(title, token);
      setDocs((prev) => [
        {
          id: doc.id,
          title: doc.title,
          version: doc.version,
          is_public: doc.is_public,
          created_at: doc.created_at,
          updated_at: doc.updated_at,
        },
        ...prev,
      ]);
      return doc;
    },
    [token],
  );

  const remove = useCallback(
    async (id: string) => {
      if (!token) throw new Error("Not authenticated");
      await canvasApi.delete(id, token);
      setDocs((prev) => prev.filter((d) => d.id !== id));
    },
    [token],
  );

  return { docs, loading, refresh, create, remove };
}

// ─── Single document editor ───────────────────────────────

export function useCanvasDoc(docId: string) {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [doc, setDoc] = useState<CanvasDoc | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [peers, setPeers] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingTextRef = useRef<string | null>(null);

  // Initial load
  useEffect(() => {
    if (!docId || !token) return;
    setLoading(true);
    canvasApi
      .get(docId, token)
      .then(setDoc)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [docId, token]);

  // WebSocket
  useEffect(() => {
    if (!docId || !token) return;
    let ws: WebSocket;
    let alive = true;

    const connect = () => {
      ws = new WebSocket(`${WS_BASE}/ws/canvas/${docId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "auth", token }));
      };

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data as string);
          if (msg.type === "init") {
            setWsConnected(true);
            setPeers(msg.peers ?? 0);
            // Apply remote content only if we have no pending local changes
            if (msg.content && pendingTextRef.current === null) {
              setDoc((prev) =>
                prev ? { ...prev, content: msg.content } : prev,
              );
            }
          } else if (msg.type === "update") {
            if (pendingTextRef.current === null) {
              setDoc((prev) =>
                prev ? { ...prev, content: msg.content } : prev,
              );
            }
          } else if (msg.type === "peer_joined") {
            setPeers(msg.peers ?? 0);
          } else if (msg.type === "peer_left") {
            setPeers(msg.peers ?? 0);
          } else if (msg.type === "title") {
            setDoc((prev) => (prev ? { ...prev, title: msg.title } : prev));
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (alive) setTimeout(connect, 3000);
      };

      ws.onerror = () => {};
    };

    connect();

    return () => {
      alive = false;
      ws?.close();
      wsRef.current = null;
    };
  }, [docId, token]);

  const broadcast = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  // Debounced save + broadcast on content change
  const updateContent = useCallback(
    (text: string) => {
      const content: Record<string, unknown> = { type: "markdown", text };
      pendingTextRef.current = text;
      setDoc((prev) => (prev ? { ...prev, content } : prev));
      broadcast({ type: "update", content });

      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(async () => {
        if (!token) return;
        setSaving(true);
        try {
          await canvasApi.update(
            docId,
            { content, content_text: text },
            token,
          );
        } finally {
          setSaving(false);
          pendingTextRef.current = null;
        }
      }, 2000);
    },
    [docId, token, broadcast],
  );

  const updateTitle = useCallback(
    async (title: string) => {
      if (!token) return;
      setDoc((prev) => (prev ? { ...prev, title } : prev));
      broadcast({ type: "title", title });
      try {
        await canvasApi.update(docId, { title }, token);
      } catch {
        // ignore
      }
    },
    [docId, token, broadcast],
  );

  const saveVersion = useCallback(async () => {
    if (!token) return;
    setSaving(true);
    try {
      const updated = await canvasApi.update(docId, { save_version: true }, token);
      setDoc(updated);
    } finally {
      setSaving(false);
    }
  }, [docId, token]);

  const togglePublic = useCallback(async () => {
    if (!doc || !token) return;
    const updated = await canvasApi.update(docId, { is_public: !doc.is_public }, token);
    setDoc(updated);
  }, [doc, docId, token]);

  // Extract markdown text from content JSONB
  const getMarkdownText = useCallback((d: CanvasDoc | null): string => {
    if (!d?.content) return "";
    const c = d.content as any;
    if (typeof c.text === "string") return c.text;
    return "";
  }, []);

  return {
    doc,
    loading,
    saving,
    peers,
    wsConnected,
    getMarkdownText,
    updateContent,
    updateTitle,
    saveVersion,
    togglePublic,
  };
}

// ─── Version history ──────────────────────────────────────

export function useCanvasVersions(docId: string) {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;
  const [versions, setVersions] = useState<CanvasDocVersion[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchVersions = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const data = await canvasApi.versions(docId, token);
      setVersions(data);
    } finally {
      setLoading(false);
    }
  }, [docId, token]);

  const restore = useCallback(
    async (version: number) => {
      if (!token) return null;
      return canvasApi.restoreVersion(docId, version, token);
    },
    [docId, token],
  );

  return { versions, loading, fetchVersions, restore };
}
