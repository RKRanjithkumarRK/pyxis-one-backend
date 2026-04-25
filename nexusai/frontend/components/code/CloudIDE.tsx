"use client";
import { useState, useCallback, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { useSession } from "next-auth/react";
import { useFileSync } from "@/hooks/use-file-sync";
import { PreviewPanel } from "./PreviewPanel";
import {
  FolderTree,
  Terminal as TermIcon,
  Eye,
  Code2,
  ChevronRight,
  Plus,
  Save,
  Play,
  Loader2,
  Wifi,
  WifiOff,
  Bot,
  Send,
  X,
} from "lucide-react";

const MonacoEditor = dynamic(() => import("@monaco-editor/react").then((m) => m.Editor), { ssr: false });
const CloudTerminal = dynamic(() => import("./CloudTerminal").then((m) => m.CloudTerminal), { ssr: false });

interface CloudIDEProps {
  projectId: string;
  projectName?: string;
}

type Pane = "editor" | "terminal" | "preview" | "split";

interface AIMessage {
  role: "user" | "assistant";
  content: string;
}

export function CloudIDE({ projectId, projectName = "Project" }: CloudIDEProps) {
  const { data: session } = useSession();
  const { files, sendChange, saveFile, refreshFileList, openFile, fileList, connected, saving } = useFileSync(projectId);

  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [pane, setPane] = useState<Pane>("split");
  const [sandboxId, setSandboxId] = useState<string | null>(null);
  const [sandboxLoading, setSandboxLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // AI shell sidebar
  const [aiOpen, setAiOpen] = useState(false);
  const [aiMessages, setAiMessages] = useState<AIMessage[]>([]);
  const [aiInput, setAiInput] = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const getToken = () => (session?.user as Record<string, unknown>)?.accessToken as string | undefined;

  // Start sandbox on mount
  useEffect(() => {
    const token = getToken();
    if (!token || sandboxId) return;
    setSandboxLoading(true);
    fetch(`${apiBase}/api/v1/sandbox/projects/${projectId}/start`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => r.json())
      .then((data) => {
        setSandboxId(data.sandbox_id);
        refreshFileList();
      })
      .catch(console.error)
      .finally(() => setSandboxLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.user, projectId, apiBase, sandboxId, refreshFileList]);

  const handleExpose = useCallback(
    async (port: number) => {
      const token = getToken();
      if (!token) return;
      setPreviewLoading(true);
      try {
        const r = await fetch(`${apiBase}/api/v1/sandbox/projects/${projectId}/expose/${port}`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await r.json();
        setPreviewUrl(data.url);
      } finally {
        setPreviewLoading(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [session?.user, projectId, apiBase],
  );

  const handleAiShell = useCallback(async () => {
    const token = getToken();
    if (!aiInput.trim() || !token) return;
    const prompt = aiInput.trim();
    setAiInput("");
    setAiMessages((prev) => [...prev, { role: "user", content: prompt }]);
    setAiLoading(true);
    try {
      const r = await fetch(`${apiBase}/api/v1/sandbox/projects/${projectId}/ai-shell`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ prompt }),
      });
      const data = await r.json();
      const reply = `$ ${data.command}\n${data.stdout || ""}${data.stderr ? `\n[stderr] ${data.stderr}` : ""}`;
      setAiMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (e) {
      setAiMessages((prev) => [...prev, { role: "assistant", content: `Error: ${e}` }]);
    } finally {
      setAiLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiInput, session?.user, projectId, apiBase]);

  const editorContent = activeFile ? files[activeFile] ?? "" : "";

  return (
    <div className="flex h-screen bg-[#0d0d0f] text-sm overflow-hidden">
      {/* Sidebar: file tree */}
      <FileTree
        fileList={fileList}
        activeFile={activeFile}
        onSelect={(path) => {
          setActiveFile(path);
          openFile(path);
        }}
        onRefresh={refreshFileList}
      />

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="flex items-center gap-1 px-3 py-1.5 border-b border-white/5 bg-[#13131a]">
          <span className="text-xs font-medium text-muted-foreground mr-2 truncate">{projectName}</span>

          <div className="flex items-center gap-0.5 bg-white/5 rounded p-0.5 ml-auto">
            {(["editor", "split", "preview"] as Pane[]).map((p) => (
              <button
                key={p}
                onClick={() => setPane(p)}
                className={`px-2 py-0.5 rounded text-xs capitalize transition-colors ${pane === p ? "bg-violet-600 text-white" : "text-muted-foreground hover:text-white"}`}
              >
                {p === "split" ? "Split" : p === "editor" ? "Code" : "Preview"}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 ml-2">
            {connected ? (
              <Wifi className="w-3.5 h-3.5 text-green-500" title="File sync connected" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-red-500" title="File sync disconnected" />
            )}
            {saving && <Loader2 className="w-3.5 h-3.5 animate-spin text-violet-400" title="Saving..." />}
          </div>

          {activeFile && (
            <button
              onClick={() => saveFile(activeFile, editorContent)}
              className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/5 hover:bg-white/10 text-xs text-muted-foreground hover:text-white ml-1"
            >
              <Save className="w-3 h-3" /> Save
            </button>
          )}

          <button
            onClick={() => handleExpose(3000)}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-violet-600/20 hover:bg-violet-600/40 text-xs text-violet-300 ml-1"
            title="Preview port 3000"
          >
            <Play className="w-3 h-3" /> Preview
          </button>

          <button
            onClick={() => setAiOpen((o) => !o)}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/5 hover:bg-white/10 text-xs text-muted-foreground hover:text-white ml-1"
          >
            <Bot className="w-3 h-3" /> AI Shell
          </button>

          {sandboxLoading && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground ml-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Starting sandbox...
            </div>
          )}
        </div>

        {/* Editor / Preview split */}
        <div className="flex-1 flex min-h-0">
          {/* Code editor */}
          {(pane === "editor" || pane === "split") && (
            <div className={`flex flex-col ${pane === "split" ? "w-1/2" : "flex-1"} border-r border-white/5`}>
              {/* Active file tab */}
              {activeFile && (
                <div className="flex items-center gap-2 px-3 py-1 border-b border-white/5 bg-[#13131a] text-xs text-muted-foreground">
                  <Code2 className="w-3 h-3" />
                  <span>{activeFile}</span>
                </div>
              )}
              <div className="flex-1">
                {activeFile ? (
                  <MonacoEditor
                    value={editorContent}
                    language={detectLanguage(activeFile)}
                    theme="vs-dark"
                    onChange={(v) => { if (activeFile) sendChange(activeFile, v ?? ""); }}
                    options={{
                      fontSize: 13,
                      fontFamily: '"JetBrains Mono", "Fira Code", Consolas, monospace',
                      minimap: { enabled: false },
                      scrollBeyondLastLine: false,
                      tabSize: 2,
                      wordWrap: "on",
                      automaticLayout: true,
                      quickSuggestions: true,
                    }}
                  />
                ) : (
                  <div className="flex-1 flex items-center justify-center text-muted-foreground h-full">
                    <div className="text-center">
                      <Code2 className="w-10 h-10 mx-auto opacity-20 mb-2" />
                      <p className="text-sm">Select a file to start editing</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Terminal panel (bottom half) */}
              <div className="h-48 border-t border-white/5 flex flex-col">
                <div className="flex items-center gap-1 px-3 py-1 bg-[#13131a] border-b border-white/5 text-xs text-muted-foreground">
                  <TermIcon className="w-3 h-3" /> Terminal
                </div>
                <div className="flex-1 overflow-hidden">
                  {sandboxId ? <CloudTerminal projectId={projectId} /> : (
                    <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
                      {sandboxLoading ? "Starting sandbox..." : "Sandbox not running"}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Preview panel */}
          {(pane === "preview" || pane === "split") && (
            <div className={pane === "split" ? "w-1/2" : "flex-1"}>
              <PreviewPanel previewUrl={previewUrl} isLoading={previewLoading} className="h-full" />
            </div>
          )}
        </div>
      </div>

      {/* AI Shell sidebar */}
      {aiOpen && (
        <div className="w-80 border-l border-white/5 flex flex-col bg-[#13131a]">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-white/5">
            <Bot className="w-4 h-4 text-violet-400" />
            <span className="text-sm font-medium">AI Shell</span>
            <button onClick={() => setAiOpen(false)} className="ml-auto text-muted-foreground hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {aiMessages.length === 0 && (
              <p className="text-xs text-muted-foreground">Ask AI to run commands in your sandbox.</p>
            )}
            {aiMessages.map((m, i) => (
              <div key={i} className={`text-xs rounded p-2 ${m.role === "user" ? "bg-violet-600/20 text-violet-200" : "bg-white/5 text-muted-foreground font-mono whitespace-pre-wrap"}`}>
                {m.content}
              </div>
            ))}
            {aiLoading && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Running...
              </div>
            )}
          </div>
          <div className="flex gap-2 p-3 border-t border-white/5">
            <input
              value={aiInput}
              onChange={(e) => setAiInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAiShell(); } }}
              placeholder="Create a Next.js app..."
              className="flex-1 bg-white/5 rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-violet-500"
            />
            <button
              onClick={handleAiShell}
              disabled={aiLoading || !aiInput.trim()}
              className="p-1.5 rounded bg-violet-600 hover:bg-violet-500 disabled:opacity-40"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── File tree ────────────────────────────────────────────

interface FileTreeProps {
  fileList: Array<{ path: string; size?: number }>;
  activeFile: string | null;
  onSelect: (path: string) => void;
  onRefresh: () => void;
}

function FileTree({ fileList, activeFile, onSelect, onRefresh }: FileTreeProps) {
  return (
    <div className="w-52 border-r border-white/5 flex flex-col bg-[#0f0f17]">
      <div className="flex items-center gap-1 px-3 py-2 border-b border-white/5">
        <FolderTree className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Files</span>
        <button onClick={onRefresh} className="ml-auto text-muted-foreground hover:text-white" title="Refresh">
          <Plus className="w-3 h-3" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto py-1">
        {fileList.length === 0 ? (
          <p className="px-3 py-2 text-xs text-muted-foreground">No files yet</p>
        ) : (
          fileList.map((f) => (
            <button
              key={f.path}
              onClick={() => onSelect(f.path)}
              className={`w-full flex items-center gap-1.5 px-3 py-1 text-left text-xs hover:bg-white/5 ${activeFile === f.path ? "bg-violet-600/20 text-violet-300" : "text-muted-foreground"}`}
            >
              <ChevronRight className="w-3 h-3 opacity-40 shrink-0" />
              <span className="truncate">{f.path}</span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

// ─── Language detection ──────────────────────────────────

function detectLanguage(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript",
    py: "python", rs: "rust", go: "go", java: "java", cpp: "cpp", c: "c",
    css: "css", scss: "scss", html: "html", json: "json", md: "markdown",
    yaml: "yaml", yml: "yaml", toml: "toml", sh: "shell", bash: "shell",
    sql: "sql", graphql: "graphql",
  };
  return map[ext] ?? "plaintext";
}
