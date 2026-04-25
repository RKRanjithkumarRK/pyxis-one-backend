import { describe, it, expect, vi, beforeEach } from "vitest";

describe("haptics", () => {
  beforeEach(() => {
    vi.resetModules();
    Object.defineProperty(window, "Capacitor", { value: undefined, configurable: true, writable: true });
  });

  it("does not throw when vibrate is unavailable", async () => {
    Object.defineProperty(navigator, "vibrate", { value: undefined, configurable: true });
    const { haptic } = await import("@/lib/haptics");
    await expect(haptic("light")).resolves.toBeUndefined();
  });

  it("calls navigator.vibrate with correct pattern for medium", async () => {
    const vibrateMock = vi.fn();
    Object.defineProperty(navigator, "vibrate", { value: vibrateMock, configurable: true });
    const { haptic } = await import("@/lib/haptics");
    await haptic("medium");
    expect(vibrateMock).toHaveBeenCalledWith(20);
  });

  it("calls navigator.vibrate with array for selection", async () => {
    const vibrateMock = vi.fn();
    Object.defineProperty(navigator, "vibrate", { value: vibrateMock, configurable: true });
    const { haptic } = await import("@/lib/haptics");
    await haptic("selection");
    expect(vibrateMock).toHaveBeenCalledWith([5, 10, 5]);
  });
});
