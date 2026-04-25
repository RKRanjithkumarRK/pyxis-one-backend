"use client";
import { useState, useEffect, useRef } from "react";
import { ExternalLink, RefreshCw, Monitor, Smartphone } from "lucide-react";

interface PreviewPanelProps {
  previewUrl: string | null;
  isLoading?: boolean;
  className?: string;
}

export function PreviewPanel({ previewUrl, isLoading, className = "" }: PreviewPanelProps) {
  const [viewport, setViewport] = useState<"desktop" | "mobile">("desktop");
  const [refreshKey, setRefreshKey] = useState(0);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    setRefreshKey((k) => k + 1);
  }, [previewUrl]);

  if (isLoading) {
    return (
      <div className={`flex flex-col h-full bg-[#0d0d0f] ${className}`}>
        <PreviewToolbar previewUrl={null} viewport={viewport} onViewportChange={setViewport} onRefresh={() => {}} />
        <div className="flex-1 flex items-center justify-center text-muted-foreground">
          <div className="text-center space-y-3">
            <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-sm">Starting preview server...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!previewUrl) {
    return (
      <div className={`flex flex-col h-full bg-[#0d0d0f] ${className}`}>
        <PreviewToolbar previewUrl={null} viewport={viewport} onViewportChange={setViewport} onRefresh={() => {}} />
        <div className="flex-1 flex items-center justify-center text-muted-foreground">
          <div className="text-center space-y-2">
            <Monitor className="w-10 h-10 mx-auto opacity-30" />
            <p className="text-sm">Run your app to see the preview</p>
            <p className="text-xs opacity-50">e.g. <code className="bg-white/5 px-1 rounded">npm run dev</code></p>
          </div>
        </div>
      </div>
    );
  }

  const iframeWidth = viewport === "mobile" ? "375px" : "100%";

  return (
    <div className={`flex flex-col h-full bg-[#0d0d0f] ${className}`}>
      <PreviewToolbar
        previewUrl={previewUrl}
        viewport={viewport}
        onViewportChange={setViewport}
        onRefresh={() => setRefreshKey((k) => k + 1)}
      />
      <div className="flex-1 overflow-hidden flex items-start justify-center bg-[#111] p-2">
        <iframe
          key={refreshKey}
          ref={iframeRef}
          src={previewUrl}
          className="h-full bg-white rounded shadow-lg"
          style={{ width: iframeWidth, border: "none", transition: "width 0.3s ease" }}
          sandbox="allow-scripts allow-same-origin allow-forms allow-modals allow-popups"
          title="Preview"
        />
      </div>
    </div>
  );
}

interface ToolbarProps {
  previewUrl: string | null;
  viewport: "desktop" | "mobile";
  onViewportChange: (v: "desktop" | "mobile") => void;
  onRefresh: () => void;
}

function PreviewToolbar({ previewUrl, viewport, onViewportChange, onRefresh }: ToolbarProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-white/5 bg-[#13131a]">
      <div className="flex-1 flex items-center gap-1.5 bg-white/5 rounded px-2 py-1 text-xs text-muted-foreground font-mono truncate">
        {previewUrl || "No preview URL"}
      </div>
      <button
        onClick={onRefresh}
        className="p-1 rounded hover:bg-white/10 text-muted-foreground"
        title="Refresh"
      >
        <RefreshCw className="w-3.5 h-3.5" />
      </button>
      <button
        onClick={() => onViewportChange("desktop")}
        className={`p-1 rounded ${viewport === "desktop" ? "bg-violet-600 text-white" : "hover:bg-white/10 text-muted-foreground"}`}
        title="Desktop viewport"
      >
        <Monitor className="w-3.5 h-3.5" />
      </button>
      <button
        onClick={() => onViewportChange("mobile")}
        className={`p-1 rounded ${viewport === "mobile" ? "bg-violet-600 text-white" : "hover:bg-white/10 text-muted-foreground"}`}
        title="Mobile viewport"
      >
        <Smartphone className="w-3.5 h-3.5" />
      </button>
      {previewUrl && (
        <a
          href={previewUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="p-1 rounded hover:bg-white/10 text-muted-foreground"
          title="Open in new tab"
        >
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      )}
    </div>
  );
}
