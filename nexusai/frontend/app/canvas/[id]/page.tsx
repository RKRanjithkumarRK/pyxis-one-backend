"use client";

import dynamic from "next/dynamic";
import type { Metadata } from "next";

const CanvasEditor = dynamic(
  () => import("@/components/canvas/CanvasEditor"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    ),
  },
);

export default function CanvasDocPage({
  params,
}: {
  params: { id: string };
}) {
  return <CanvasEditor docId={params.id} />;
}
