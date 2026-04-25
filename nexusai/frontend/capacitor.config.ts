import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "ai.nexus.app",
  appName: "NexusAI",
  webDir: "out",
  server: {
    // In dev, point to Next.js dev server; in prod, use bundled assets
    url: process.env.CAPACITOR_DEV === "true" ? "http://localhost:3000" : undefined,
    cleartext: process.env.CAPACITOR_DEV === "true",
  },
  ios: {
    contentInset: "always",
    preferredContentMode: "mobile",
    scrollEnabled: true,
  },
  android: {
    buildOptions: {
      keystorePath: "nexusai.keystore",
      keystoreAlias: "nexusai",
    },
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 1500,
      backgroundColor: "#0d0d0f",
      showSpinner: false,
      androidScaleType: "CENTER_CROP",
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
    Keyboard: {
      resize: "body",
      style: "dark",
      resizeOnFullScreen: true,
    },
    StatusBar: {
      style: "dark",
      backgroundColor: "#0d0d0f",
    },
  },
};

export default config;
