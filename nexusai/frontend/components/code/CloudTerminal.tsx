"use client";
import { useEffect, useRef, useCallback } from "react";
import { useSession } from "next-auth/react";

interface CloudTerminalProps {
  projectId: string;
  className?: string;
}

export function CloudTerminal({ projectId, className = "" }: CloudTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<import("@xterm/xterm").Terminal | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const fitRef = useRef<import("@xterm/addon-fit").FitAddon | null>(null);
  const { data: session } = useSession();

  const connect = useCallback(async () => {
    const accessToken = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
    if (!containerRef.current || !accessToken) return;

    const { Terminal } = await import("@xterm/xterm");
    const { FitAddon } = await import("@xterm/addon-fit");
    const { WebLinksAddon } = await import("@xterm/addon-web-links");

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: '"JetBrains Mono", "Fira Code", Consolas, monospace',
      theme: {
        background: "#0d0d0f",
        foreground: "#e2e8f0",
        cursor: "#7c3aed",
        selectionBackground: "#7c3aed44",
        black: "#1e1e2e",
        red: "#f38ba8",
        green: "#a6e3a1",
        yellow: "#f9e2af",
        blue: "#89b4fa",
        magenta: "#cba6f7",
        cyan: "#89dceb",
        white: "#cdd6f4",
      },
      scrollback: 5000,
      allowProposedApi: true,
    });

    const fit = new FitAddon();
    term.loadAddon(fit);
    term.loadAddon(new WebLinksAddon());
    term.open(containerRef.current);
    fit.fit();
    fitRef.current = fit;
    termRef.current = term;

    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000"}/ws/terminal/${projectId}?token=${accessToken}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      term.writeln("\x1b[32mConnected to NexusCode sandbox\x1b[0m");
      term.writeln("\x1b[90mType commands below. Your environment is ready.\x1b[0m\r\n");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "output") {
          term.write(msg.data);
        } else if (msg.type === "error") {
          term.writeln(`\x1b[31mError: ${msg.message}\x1b[0m`);
        }
      } catch {
        term.write(event.data);
      }
    };

    ws.onclose = () => {
      term.writeln("\r\n\x1b[33mDisconnected from sandbox\x1b[0m");
    };

    ws.onerror = () => {
      term.writeln("\r\n\x1b[31mWebSocket error — check backend connection\x1b[0m");
    };

    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "input", data }));
      }
    });

    // Handle terminal resize
    const observer = new ResizeObserver(() => {
      fit.fit();
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
      }
    });
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
      ws.close();
      term.dispose();
    };
  }, [projectId, session?.user]);

  useEffect(() => {
    let cleanup: (() => void) | undefined;
    connect().then((fn) => { cleanup = fn; });
    return () => { cleanup?.(); };
  }, [connect]);

  return (
    <div
      ref={containerRef}
      className={`h-full w-full bg-[#0d0d0f] rounded-b-lg overflow-hidden ${className}`}
      style={{ minHeight: 200 }}
    />
  );
}
