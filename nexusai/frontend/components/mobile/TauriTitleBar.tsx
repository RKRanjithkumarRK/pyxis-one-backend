"use client";
import { useIsTauri } from "@/hooks/use-platform";

export function TauriTitleBar() {
  const isTauri = useIsTauri();
  if (!isTauri) return null;

  return (
    <div
      data-tauri-drag-region
      className="fixed top-0 inset-x-0 z-[9999] h-8 hidden mac:flex items-center justify-center"
    >
      <span
        data-tauri-drag-region
        className="text-xs font-medium text-muted-foreground select-none pointer-events-none"
      >
        NexusAI
      </span>
    </div>
  );
}
