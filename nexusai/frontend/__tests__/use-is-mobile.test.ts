import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useIsMobile } from "@/hooks/use-is-mobile";

function mockMediaQuery(matches: boolean) {
  const listeners: ((e: { matches: boolean }) => void)[] = [];
  return {
    matches,
    addEventListener: vi.fn((_: string, cb: (e: { matches: boolean }) => void) => listeners.push(cb)),
    removeEventListener: vi.fn(),
  };
}

describe("useIsMobile", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns true on narrow viewport", () => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn(() => mockMediaQuery(true)),
    });
    const { result } = renderHook(() => useIsMobile(768));
    expect(result.current).toBe(true);
  });

  it("returns false on wide viewport", () => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn(() => mockMediaQuery(false)),
    });
    const { result } = renderHook(() => useIsMobile(768));
    expect(result.current).toBe(false);
  });
});
