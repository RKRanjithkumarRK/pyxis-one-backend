"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import { Code2, Terminal as TermIcon, Eye, Bot } from "lucide-react";
import { useFileSync } from "@/hooks/use-file-sync";

const CloudTerminal = dynamic(() => import("./CloudTerminal").then((m) => m.CloudTerminal), { ssr: false });
const MonacoEditor = dynamic(() => import("@monaco-editor/react").then((m) => m.Editor), { ssr: false });

interface MobileIDEProps {
  projectId: string;
}

type MobileTab = "files" | "editor" | "terminal" | "preview";

export function MobileIDE({ projectId }: MobileIDEProps) {
  const [tab, setTab] = useState<MobileTab>("files");
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const { files, fileList, sendChange, openFile } = useFileSync(projectId);

  const tabs: Array<{ id: MobileTab; icon: typeof Code2; label: string }> = [
    { id: "files", icon: Bot, label: "Files" },
    { id: "editor", icon: Code2, label: "Code" },
    { id: "terminal", icon: TermIcon, label: "Terminal" },
    { id: "preview", icon: Eye, label: "Preview" },
  ];

  return (
    <div className="flex flex-col h-screen bg-[#0d0d0f]">
      {/* Content area */}
      <div className="flex-1 overflow-hidden">
        {tab === "files" && (
          <div className="h-full overflow-y-auto py-2">
            {fileList.map((f) => (
              <button
                key={f.path}
                onClick={() => { setActiveFile(f.path); openFile(f.path); setTab("editor"); }}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-left text-sm hover:bg-white/5 border-b border-white/5 text-muted-foreground"
              >
                <Code2 className="w-4 h-4 shrink-0" />
                <span className="truncate">{f.path}</span>
              </button>
            ))}
            {fileList.length === 0 && (
              <p className="text-center text-muted-foreground text-sm py-10">No files. Create one in the terminal.</p>
            )}
          </div>
        )}

        {tab === "editor" && (
          <div className="h-full">
            {activeFile ? (
              <MonacoEditor
                value={files[activeFile] ?? ""}
                language={detectLang(activeFile)}
                theme="vs-dark"
                onChange={(v) => activeFile && sendChange(activeFile, v ?? "")}
                options={{
                  fontSize: 12,
                  minimap: { enabled: false },
                  wordWrap: "on",
                  automaticLayout: true,
                  scrollBeyondLastLine: false,
                }}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                Select a file from the Files tab
              </div>
            )}
          </div>
        )}

        {tab === "terminal" && (
          <div className="h-full">
            <CloudTerminal projectId={projectId} />
          </div>
        )}

        {tab === "preview" && (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            <p>Run your app in the terminal, then tap Expose Port 3000 here.</p>
          </div>
        )}
      </div>

      {/* Bottom tab bar */}
      <nav className="flex border-t border-white/5 bg-[#13131a] safe-area-inset-bottom">
        {tabs.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex-1 flex flex-col items-center gap-0.5 py-2 text-xs ${tab === id ? "text-violet-400" : "text-muted-foreground"}`}
          >
            <Icon className="w-5 h-5" />
            {label}
          </button>
        ))}
      </nav>
    </div>
  );
}

function detectLang(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript",
    py: "python", rs: "rust", go: "go", css: "css", html: "html", json: "json",
  };
  return map[ext] ?? "plaintext";
}
