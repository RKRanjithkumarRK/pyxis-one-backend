"use client";

import { useState, useCallback, useEffect } from "react";
import { useSession } from "next-auth/react";
import { kbApi } from "@/lib/api";
import type { KnowledgeBase, KBFile } from "@/lib/api-types";

function useToken() {
  const { data: session } = useSession();
  return (session as any)?.accessToken as string | undefined;
}

export function useKnowledgeBases() {
  const token = useToken();
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setLoading(true);
      const data = await kbApi.list(token);
      setKbs(data);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  const create = useCallback(
    async (name: string, description?: string) => {
      if (!token) return null;
      const kb = await kbApi.create({ name, description }, token);
      setKbs((prev) => [kb, ...prev]);
      return kb;
    },
    [token],
  );

  const remove = useCallback(
    async (id: string) => {
      if (!token) return;
      await kbApi.delete(id, token);
      setKbs((prev) => prev.filter((k) => k.id !== id));
    },
    [token],
  );

  return { kbs, loading, error, refresh: load, create, remove };
}

export function useKnowledgeBase(kbId: string) {
  const token = useToken();
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setLoading(true);
      const data = await kbApi.get(kbId, token);
      setKb(data);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [kbId, token]);

  useEffect(() => { load(); }, [load]);

  // Poll files that are pending/processing every 3s
  useEffect(() => {
    if (!kb) return;
    const active = kb.files.filter((f) => f.status === "pending" || f.status === "processing");
    if (active.length === 0) return;
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, [kb, load]);

  const uploadFile = useCallback(
    async (file: File) => {
      if (!token) return null;
      setUploading(true);
      try {
        const kbFile = await kbApi.uploadFile(kbId, file, token);
        setKb((prev) =>
          prev ? { ...prev, files: [kbFile, ...prev.files] } : prev,
        );
        return kbFile;
      } catch (e: any) {
        setError(e.message);
        return null;
      } finally {
        setUploading(false);
      }
    },
    [kbId, token],
  );

  const deleteFile = useCallback(
    async (fileId: string) => {
      if (!token) return;
      await kbApi.deleteFile(kbId, fileId, token);
      setKb((prev) =>
        prev ? { ...prev, files: prev.files.filter((f) => f.id !== fileId) } : prev,
      );
    },
    [kbId, token],
  );

  const updateName = useCallback(
    async (name: string) => {
      if (!token || !kb) return;
      const updated = await kbApi.update(kbId, { name }, token);
      setKb(updated);
    },
    [kbId, token, kb],
  );

  return { kb, loading, uploading, error, refresh: load, uploadFile, deleteFile, updateName };
}
