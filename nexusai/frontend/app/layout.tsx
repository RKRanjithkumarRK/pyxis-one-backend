import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import Script from "next/script";
import "./globals.css";
import { Providers } from "@/components/providers";
import { MobileShell } from "@/components/mobile/MobileShell";
import { TauriTitleBar } from "@/components/mobile/TauriTitleBar";

export const metadata: Metadata = {
  title: {
    default: "NexusAI — All-in-One AI Platform",
    template: "%s | NexusAI",
  },
  description:
    "NexusAI combines ChatGPT, Cursor, Perplexity, Midjourney and ElevenLabs into one unified AI platform.",
  keywords: ["AI", "chat", "code", "image generation", "voice", "research"],
  authors: [{ name: "NexusAI" }],
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "NexusAI",
  },
  openGraph: {
    type: "website",
    siteName: "NexusAI",
    title: "NexusAI — All-in-One AI Platform",
    description: "Chat · Code · Research · Image · Voice — unified.",
  },
  twitter: {
    card: "summary_large_image",
    title: "NexusAI",
    description: "All-in-One AI Platform",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0f" },
  ],
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        <Providers>
          <TauriTitleBar />
          <MobileShell>{children}</MobileShell>
        </Providers>
        <Script id="sw-register" strategy="afterInteractive">
          {`if ('serviceWorker' in navigator) { navigator.serviceWorker.register('/sw.js'); }`}
        </Script>
      </body>
    </html>
  );
}
