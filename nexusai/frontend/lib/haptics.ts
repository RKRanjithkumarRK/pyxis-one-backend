// Native haptic feedback — Capacitor-aware, graceful fallback
declare global {
  interface Window {
    Capacitor?: { isNative?: boolean; Plugins?: Record<string, unknown> };
  }
}

type HapticStyle = "light" | "medium" | "heavy" | "selection";

const VIBRATION_MAP: Record<HapticStyle, number | number[]> = {
  light: 10,
  medium: 20,
  heavy: 40,
  selection: [5, 10, 5],
};

export async function haptic(style: HapticStyle = "light"): Promise<void> {
  try {
    // Capacitor native path (iOS/Android)
    if (window.Capacitor?.isNative && window.Capacitor.Plugins) {
      const { Haptics } = window.Capacitor.Plugins as {
        Haptics?: { impact: (opts: { style: string }) => Promise<void> };
      };
      if (Haptics) {
        await Haptics.impact({ style: style.toUpperCase() });
        return;
      }
    }
    // Web Vibration API fallback
    if ("vibrate" in navigator) {
      navigator.vibrate(VIBRATION_MAP[style]);
    }
  } catch {
    // silently ignore — haptics are purely additive UX
  }
}
