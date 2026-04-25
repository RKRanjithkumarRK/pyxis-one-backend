import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "NexusAI",
    short_name: "NexusAI",
    description: "All-in-one AI platform — Chat, Code, Research, Image, Voice",
    start_url: "/",
    display: "standalone",
    background_color: "#0d0d0f",
    theme_color: "#7c3aed",
    orientation: "any",
    categories: ["productivity", "utilities"],
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
    shortcuts: [
      { name: "Chat", url: "/chat", icons: [{ src: "/icons/icon-192.png", sizes: "192x192" }] },
      { name: "Code", url: "/code", icons: [{ src: "/icons/icon-192.png", sizes: "192x192" }] },
    ],
  };
}
