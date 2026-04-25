"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Link from "next/link";
import dynamic from "next/dynamic";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useCanvasDoc, useCanvasVersions } from "@/hooks/use-canvas";
import { canvasApi } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { CanvasDoc } from "@/lib/api-types";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

type ViewMode = "split" | "edit" | "preview";
type Panel = "none" | "ai" | "history";

function getMarkdown(doc: CanvasDoc | null): string {
  if (!doc?.content) return "";
  const c = doc.content as any;
  return typeof c.text === "string" ? c.text : "";
}

export default function CanvasEditor({ docId }: { docId: string }) {
  const router = useRouter();
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const { doc, loading, saving, peers, wsConnected, updateContent, updateTitle, saveVersion, togglePublic } =
    useCanvasDoc(docId);
  const { versions, loading: versionsLoading, fetchVersions, restore } = useCanvasVersions(docId);

  const [text, setText] = useState("");
  const [titleEdit, setTitleEdit] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("split");
  const [panel, setPanel] = useState<Panel>("none");

  // AI edit state
  const [selectedText, setSelectedText] = useState("");
  const [aiInstruction, setAIInstruction] = useState("");
  const [aiResult, setAIResult] = useState<{ original: string; suggested: string } | null>(null);
  const [aiLoading, setAILoading] = useState(false);
  const [aiError, setAIError] = useState<string | null>(null);

  // Version restore state
  const [restoring, setRestoring] = useState(false);

  const editorRef = useRef<any>(null);

  // Initialize text from loaded doc (once per doc)
  useEffect(() => {
    if (doc) {
      const md = getMarkdown(doc);
      setText(md);
      setTitleDraft(doc.title);
    }
  }, [doc?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync title draft when doc title changes (e.g. from WS)
  useEffect(() => {
    if (doc && !titleEdit) setTitleDraft(doc.title);
  }, [doc?.title, titleEdit]);

  const handleEditorMount = (editor: any) => {
    editorRef.current = editor;
    editor.onDidChangeCursorSelection((e: any) => {
      const model = editor.getModel();
      const sel = model?.getValueInRange(e.selection) ?? "";
      setSelectedText(sel);
    });
  };

  const handleEditorChange = (value: string | undefined) => {
    const v = value ?? "";
    setText(v);
    updateContent(v);
  };

  const handleTitleCommit = async () => {
    setTitleEdit(false);
    if (titleDraft.trim() && titleDraft !== doc?.title) {
      await updateTitle(titleDraft.trim());
    }
  };

  const handleOpenPanel = (p: Panel) => {
    if (p === panel) {
      setPanel("none");
      return;
    }
    setPanel(p);
    if (p === "history") fetchVersions();
  };

  // ── AI Edit ────────────────────────────────────────────
  const handleAIEdit = async () => {
    if (!selectedText.trim()) {
      setAIError("Select some text in the editor first.");
      return;
    }
    if (!aiInstruction.trim()) {
      setAIError("Enter an instruction.");
      return;
    }
    if (!token) return;
    setAIError(null);
    setAILoading(true);
    try {
      const result = await canvasApi.aiEdit(
        docId,
        selectedText,
        aiInstruction,
        text.slice(0, 500),
        token,
      );
      setAIResult(result);
    } catch (err: any) {
      setAIError(err?.message ?? "AI edit failed");
    } finally {
      setAILoading(false);
    }
  };

  const handleAcceptAIEdit = () => {
    if (!aiResult || !editorRef.current) return;
    const editor = editorRef.current;
    const selection = editor.getSelection();
    editor.executeEdits("ai-edit", [
      { range: selection, text: aiResult.suggested, forceMoveMarkers: true },
    ]);
    const newText = editor.getValue() as string;
    setText(newText);
    updateContent(newText);
    setAIResult(null);
    setAIInstruction("");
    setSelectedText("");
  };

  const handleRejectAIEdit = () => {
    setAIResult(null);
    setAIInstruction("");
  };

  // ── Version restore ────────────────────────────────────
  const handleRestore = async (version: number) => {
    setRestoring(true);
    try {
      const updated = await restore(version);
      if (updated) {
        const md = getMarkdown(updated);
        setText(md);
        if (editorRef.current) editorRef.current.setValue(md);
        updateContent(md);
        setPanel("none");
      }
    } finally {
      setRestoring(false);
    }
  };

  // ── Export ─────────────────────────────────────────────
  const handleExport = async (format: "md" | "html" | "docx") => {
    if (!token) return;
    const url = canvasApi.exportUrl(docId, format);
    const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) return;
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${doc?.title ?? "canvas"}.${format}`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!doc) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-background text-foreground">
        <p className="text-lg font-medium">Document not found</p>
        <Link href="/canvas" className="text-primary underline underline-offset-4">
          ← Back to Canvas
        </Link>
      </div>
    );
  }

  const showEditor = viewMode !== "preview";
  const showPreview = viewMode !== "edit";
  const panelOpen = panel !== "none";

  return (
    <div className="flex h-screen flex-col bg-background text-foreground overflow-hidden">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border px-4">
        <Link
          href="/canvas"
          className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          <span className="text-sm">Canvas</span>
        </Link>

        <div className="mx-2 h-5 w-px bg-border" />

        {/* Title */}
        {titleEdit ? (
          <input
            autoFocus
            value={titleDraft}
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={handleTitleCommit}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleTitleCommit();
              if (e.key === "Escape") { setTitleEdit(false); setTitleDraft(doc.title); }
            }}
            className="flex-1 bg-transparent text-sm font-semibold text-foreground outline-none border-b border-primary min-w-0 max-w-xs"
          />
        ) : (
          <button
            onClick={() => { setTitleEdit(true); setTitleDraft(doc.title); }}
            className="flex-1 text-left text-sm font-semibold text-foreground truncate max-w-xs hover:text-primary transition-colors"
          >
            {doc.title}
          </button>
        )}

        {/* Save status */}
        <span className={cn(
          "text-xs transition-colors",
          saving ? "text-primary animate-pulse" : "text-muted-foreground",
        )}>
          {saving ? "Saving…" : "Saved"}
        </span>

        {/* Peers */}
        {peers > 0 && (
          <span className="flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-xs text-green-500">
            <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
            {peers} online
          </span>
        )}

        <div className="flex-1" />

        {/* View mode */}
        <div className="flex items-center rounded-lg border border-border bg-muted/30 p-0.5 text-xs">
          {(["split", "edit", "preview"] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className={cn(
                "rounded-md px-2.5 py-1 capitalize transition-colors",
                viewMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {m}
            </button>
          ))}
        </div>

        <div className="mx-1 h-5 w-px bg-border" />

        {/* Toolbar actions */}
        <button
          onClick={() => handleOpenPanel("ai")}
          className={cn(
            "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
            panel === "ai" ? "bg-primary text-primary-foreground" : "border border-border hover:bg-accent",
          )}
        >
          ✨ AI Edit
        </button>

        <button
          onClick={() => handleOpenPanel("history")}
          className={cn(
            "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
            panel === "history" ? "bg-primary text-primary-foreground" : "border border-border hover:bg-accent",
          )}
        >
          History
        </button>

        <button
          onClick={saveVersion}
          className="rounded-lg border border-border px-2.5 py-1.5 text-xs font-medium hover:bg-accent transition-colors"
          title="Save a version snapshot"
        >
          Snapshot
        </button>

        {/* Export dropdown */}
        <ExportMenu onExport={handleExport} />

        {/* Share toggle */}
        <button
          onClick={togglePublic}
          className={cn(
            "rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
            doc.is_public
              ? "bg-green-500/10 text-green-600 border border-green-500/30 hover:bg-green-500/20"
              : "border border-border hover:bg-accent",
          )}
        >
          {doc.is_public ? "Public" : "Share"}
        </button>
      </header>

      {/* ── Body ─────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Editor pane */}
        {showEditor && (
          <div className={cn("flex flex-col", showPreview && !panelOpen ? "w-1/2 border-r border-border" : panelOpen ? "flex-1" : "flex-1")}>
            <MonacoEditor
              language="markdown"
              value={text}
              onChange={handleEditorChange}
              onMount={handleEditorMount}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                wordWrap: "on",
                lineNumbers: "off",
                scrollBeyondLastLine: false,
                fontSize: 15,
                lineHeight: 1.7,
                padding: { top: 20, bottom: 20 },
                fontFamily: "var(--font-geist-mono), 'Fira Code', 'Cascadia Code', monospace",
                renderLineHighlight: "none",
                scrollbar: { vertical: "auto", horizontal: "hidden" },
                overviewRulerBorder: false,
                hideCursorInOverviewRuler: true,
                contextmenu: false,
                renderWhitespace: "none",
                bracketPairColorization: { enabled: false },
                guides: { indentation: false },
                folding: false,
                glyphMargin: false,
              }}
              className="flex-1"
            />
          </div>
        )}

        {/* Preview pane */}
        {showPreview && !panelOpen && (
          <div className={cn(
            "overflow-y-auto bg-background",
            showEditor ? "w-1/2" : "flex-1",
          )}>
            <div className="mx-auto max-w-3xl px-10 py-8">
              <h1 className="mb-6 text-3xl font-bold text-foreground">{doc.title}</h1>
              <div className="prose prose-neutral dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {text || "*Start writing…*"}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}

        {/* AI Edit panel */}
        {panel === "ai" && (
          <div className="w-96 shrink-0 border-l border-border flex flex-col bg-card">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <h2 className="text-sm font-semibold text-foreground">✨ AI Edit</h2>
              <button
                onClick={() => setPanel("none")}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Selected text indicator */}
              <div>
                <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Selected Text
                </p>
                {selectedText ? (
                  <div className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-xs text-foreground font-mono leading-relaxed max-h-32 overflow-y-auto">
                    {selectedText}
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground italic">
                    Select text in the editor to edit it with AI
                  </p>
                )}
              </div>

              {/* Instruction input */}
              <div>
                <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Instruction
                </p>
                <textarea
                  value={aiInstruction}
                  onChange={(e) => setAIInstruction(e.target.value)}
                  placeholder="e.g. Make this more concise, fix grammar, expand with examples…"
                  rows={3}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {aiError && (
                <p className="rounded-lg bg-destructive/10 border border-destructive/30 px-3 py-2 text-xs text-destructive">
                  {aiError}
                </p>
              )}

              <button
                onClick={handleAIEdit}
                disabled={aiLoading || !selectedText.trim() || !aiInstruction.trim()}
                className="w-full rounded-lg bg-primary py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {aiLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                    Rewriting…
                  </span>
                ) : (
                  "Rewrite →"
                )}
              </button>

              {/* Diff view */}
              {aiResult && (
                <div className="space-y-3">
                  <div>
                    <p className="mb-1.5 text-xs font-medium text-red-500 uppercase tracking-wide">Original</p>
                    <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2 text-xs font-mono leading-relaxed whitespace-pre-wrap text-foreground">
                      {aiResult.original}
                    </div>
                  </div>
                  <div>
                    <p className="mb-1.5 text-xs font-medium text-green-500 uppercase tracking-wide">Suggested</p>
                    <div className="rounded-lg border border-green-500/30 bg-green-500/5 px-3 py-2 text-xs font-mono leading-relaxed whitespace-pre-wrap text-foreground">
                      {aiResult.suggested}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={handleAcceptAIEdit}
                      className="flex-1 rounded-lg bg-green-600 py-2 text-sm font-semibold text-white hover:bg-green-700 transition-colors"
                    >
                      Accept
                    </button>
                    <button
                      onClick={handleRejectAIEdit}
                      className="flex-1 rounded-lg border border-border py-2 text-sm font-semibold hover:bg-accent transition-colors"
                    >
                      Reject
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Version history panel */}
        {panel === "history" && (
          <div className="w-80 shrink-0 border-l border-border flex flex-col bg-card">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <h2 className="text-sm font-semibold text-foreground">Version History</h2>
              <button
                onClick={() => setPanel("none")}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              {versionsLoading ? (
                <div className="flex items-center justify-center py-10">
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                </div>
              ) : versions.length === 0 ? (
                <div className="px-4 py-10 text-center">
                  <p className="text-sm text-muted-foreground">No saved versions yet.</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Click Snapshot to save the current state.
                  </p>
                </div>
              ) : (
                <div className="divide-y divide-border">
                  {versions.map((v) => (
                    <div key={v.id} className="flex items-center gap-3 px-4 py-3 hover:bg-accent/40 transition-colors">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold">
                        v{v.version}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-foreground truncate">
                          {v.title ?? "Untitled"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(v.created_at).toLocaleString()}
                        </p>
                      </div>
                      <button
                        onClick={() => handleRestore(v.version)}
                        disabled={restoring}
                        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors disabled:opacity-50"
                      >
                        Restore
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Current version indicator */}
            <div className="border-t border-border px-4 py-3">
              <p className="text-xs text-muted-foreground">
                Current version: <span className="font-semibold text-foreground">v{doc.version}</span>
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Export dropdown ────────────────────────────────────────

function ExportMenu({ onExport }: { onExport: (fmt: "md" | "html" | "docx") => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 rounded-lg border border-border px-2.5 py-1.5 text-xs font-medium hover:bg-accent transition-colors"
      >
        Export
        <svg className="h-3 w-3 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 w-36 rounded-xl border border-border bg-card shadow-xl">
          {(["md", "html", "docx"] as const).map((fmt) => (
            <button
              key={fmt}
              onClick={() => { onExport(fmt); setOpen(false); }}
              className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm hover:bg-accent transition-colors first:rounded-t-xl last:rounded-b-xl"
            >
              <span className="text-base">{fmt === "md" ? "📝" : fmt === "html" ? "🌐" : "📄"}</span>
              <span className="uppercase font-mono text-xs text-foreground">.{fmt}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
