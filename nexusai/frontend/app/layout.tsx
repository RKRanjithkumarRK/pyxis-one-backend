import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
  title: {
    default: "NexusAI — All-in-One AI Platform",
    template: "%s | NexusAI",
  },
  description:
    "NexusAI combines ChatGPT, Cursor, Perplexity, Midjourney and ElevenLabs into one unified AI platform.",
  keywords: ["AI", "chat", "code", "image generation", "voice", "research"],
  authors: [{ name: "NexusAI" }],
  manifest: "/manifest.json",
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
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
