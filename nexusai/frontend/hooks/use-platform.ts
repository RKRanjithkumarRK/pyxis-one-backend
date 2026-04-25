"use client";
import { useEffect, useState } from "react";

type Platform = "web" | "tauri" | "capacitor-ios" | "capacitor-android";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export function usePlatform(): Platform {
  const [platform, setPlatform] = useState<Platform>("web");

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.__TAURI_INTERNALS__) {
      setPlatform("tauri");
    } else if (window.Capacitor?.isNative) {
      const p = (window.Capacitor as { getPlatform?: () => string }).getPlatform?.();
      setPlatform(p === "ios" ? "capacitor-ios" : "capacitor-android");
    }
  }, []);

  return platform;
}

export function useIsNative(): boolean {
  const p = usePlatform();
  return p !== "web";
}

export function useIsTauri(): boolean {
  return usePlatform() === "tauri";
}
