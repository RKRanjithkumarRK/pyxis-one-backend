"use client";
import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { BottomNav } from "./BottomNav";
import { useIsMobile } from "@/hooks/use-is-mobile";

const HIDE_NAV_PREFIXES = ["/code/", "/login", "/signup", "/shared/"];

export function MobileShell({ children }: { children: React.ReactNode }) {
  const isMobile = useIsMobile();
  const pathname = usePathname();
  const touchStartX = useRef(0);

  const showNav =
    isMobile &&
    !HIDE_NAV_PREFIXES.some((p) => pathname.startsWith(p));

  // Swipe-to-go-back for PWA/Capacitor
  useEffect(() => {
    if (!isMobile) return;

    const onTouchStart = (e: TouchEvent) => {
      touchStartX.current = e.touches[0].clientX;
    };
    const onTouchEnd = (e: TouchEvent) => {
      const dx = e.changedTouches[0].clientX - touchStartX.current;
      if (dx > 60 && touchStartX.current < 24) {
        window.history.back();
      }
    };

    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });
    return () => {
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchend", onTouchEnd);
    };
  }, [isMobile]);

  return (
    <>
      <div className={showNav ? "pb-safe" : undefined}>{children}</div>
      {showNav && <BottomNav />}
    </>
  );
}
