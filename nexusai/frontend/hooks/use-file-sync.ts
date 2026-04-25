"use client";
import { useEffect, useRef, useCallback, useState } from "react";
import { useSession } from "next-auth/react";

export interface FileEntry {
  path: string;
  content: string;
}

export interface FileSyncHook {
  files: Record<string, string>;
  sendChange: (path: string, content: string) => void;
  saveFile: (path: string, content: string) => void;
  refreshFileList: () => void;
  openFile: (path: string) => void;
  fileList: Array<{ path: string; size?: number }>;
  connected: boolean;
  saving: boolean;
}

export function useFileSync(projectId: string): FileSyncHook {
  const { data: session } = useSession();
  const wsRef = useRef<WebSocket | null>(null);
  const [files, setFiles] = useState<Record<string, string>>({});
  const [fileList, setFileList] = useState<Array<{ path: string; size?: number }>>([]);
  const [connected, setConnected] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const accessToken = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
    if (!accessToken || !projectId) return;

    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000"}/ws/filesync/${projectId}?token=${accessToken}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ type: "file_list" }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "file_list") {
          setFileList(msg.files || []);
        } else if (msg.type === "file_content") {
          setFiles((prev) => ({ ...prev, [msg.path]: msg.content }));
        } else if (msg.type === "file_change") {
          setFiles((prev) => ({ ...prev, [msg.path]: msg.content }));
        } else if (msg.type === "saved") {
          setSaving(false);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => setConnected(false);

    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 20_000);

    return () => {
      clearInterval(ping);
      ws.close();
    };
  }, [projectId, session?.user]);

  const sendChange = useCallback((path: string, content: string) => {
    setFiles((prev) => ({ ...prev, [path]: content }));
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "file_change", path, content }));
    }
  }, []);

  const saveFile = useCallback((path: string, content: string) => {
    setSaving(true);
    setFiles((prev) => ({ ...prev, [path]: content }));
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "file_save", path, content }));
    }
  }, []);

  const refreshFileList = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "file_list" }));
    }
  }, []);

  const openFile = useCallback((path: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "file_read", path }));
    }
  }, []);

  return { files, sendChange, saveFile, refreshFileList, openFile, fileList, connected, saving };
}
