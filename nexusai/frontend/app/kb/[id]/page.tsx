"use client";

import { useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { useKnowledgeBase } from "@/hooks/use-knowledge-base";
import type { KBFile } from "@/lib/api-types";

function StatusBadge({ status }: { status: KBFile["status"] }) {
  const map: Record<KBFile["status"], { label: string; cls: string }> = {
    pending: { label: "Pending", cls: "bg-muted text-muted-foreground" },
    processing: { label: "Processing…", cls: "bg-amber-100 text-amber-700 animate-pulse" },
    done: { label: "Indexed", cls: "bg-green-100 text-green-700" },
    error: { label: "Error", cls: "bg-red-100 text-red-700" },
  };
  const { label, cls } = map[status] ?? map.error;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

function FileRow({ file, onDelete }: { file: KBFile; onDelete: (id: string) => void }) {
  const kb = Math.round(file.file_size / 1024);
  return (
    <div className="flex items-center gap-3 px-4 py-3 border-b border-border last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.filename}</p>
        <p className="text-xs text-muted-foreground mt-0.5">
          {kb} KB · {file.file_type.toUpperCase()}
          {file.status === "done" && ` · ${file.chunk_count} chunks`}
          {file.error_msg && ` · ${file.error_msg}`}
        </p>
      </div>
      <StatusBadge status={file.status} />
      <button
        onClick={() => onDelete(file.id)}
        className="h-7 w-7 flex items-center justify-center rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors text-xs"
        title="Delete file"
      >
        🗑
      </button>
    </div>
  );
}

export default function KBDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { kb, loading, uploading, error, uploadFile, deleteFile } = useKnowledgeBase(id);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      const files = Array.from(e.dataTransfer.files);
      for (const f of files) await uploadFile(f);
    },
    [uploadFile],
  );

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      for (const f of files) await uploadFile(f);
      e.target.value = "";
    },
    [uploadFile],
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground text-sm">Loading…</p>
      </div>
    );
  }

  if (!kb) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground text-sm">Knowledge base not found.</p>
      </div>
    );
  }

  const totalChunks = kb.files.reduce((s, f) => s + f.chunk_count, 0);

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-3xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={() => router.push("/kb")}
            className="text-muted-foreground hover:text-foreground transition-colors text-sm"
          >
            ← Back
          </button>
          <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
            🗂️
          </div>
          <div>
            <h1 className="text-xl font-bold">{kb.name}</h1>
            {kb.description && (
              <p className="text-sm text-muted-foreground">{kb.description}</p>
            )}
          </div>
        </div>

        {/* Stats bar */}
        <div className="flex items-center gap-6 mb-6 text-sm text-muted-foreground">
          <span>{kb.files.length} file{kb.files.length !== 1 ? "s" : ""}</span>
          <span>{totalChunks} chunks indexed</span>
        </div>

        {/* Upload zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className="mb-6 rounded-xl border-2 border-dashed border-border hover:border-primary/50 bg-card cursor-pointer transition-colors p-8 text-center"
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.xlsx,.pptx,.txt,.md,.html,.csv"
            className="hidden"
            onChange={handleFileChange}
          />
          {uploading ? (
            <p className="text-sm text-primary animate-pulse">Uploading…</p>
          ) : (
            <>
              <div className="text-3xl mb-2">📄</div>
              <p className="text-sm font-medium">Drop files here or click to upload</p>
              <p className="text-xs text-muted-foreground mt-1">
                PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, CSV — max 50 MB each
              </p>
            </>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded-lg bg-destructive/10 text-destructive text-sm px-4 py-2">
            {error}
          </div>
        )}

        {/* Files list */}
        {kb.files.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="font-medium">No files yet</p>
            <p className="text-sm mt-1">Upload documents above to start building your knowledge base</p>
          </div>
        ) : (
          <div className="rounded-xl border border-border bg-card overflow-hidden">
            {kb.files.map((file) => (
              <FileRow key={file.id} file={file} onDelete={deleteFile} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
