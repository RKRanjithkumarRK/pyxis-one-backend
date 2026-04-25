"use client";
import dynamic from "next/dynamic";
import { useParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { useIsMobile } from "@/hooks/use-is-mobile";

const Loader = () => (
  <div className="flex h-screen items-center justify-center bg-[#0d0d0f] text-muted-foreground">
    <div className="text-center space-y-3">
      <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto" />
      <p className="text-sm">Loading NexusCode...</p>
    </div>
  </div>
);

const CloudIDE = dynamic(() => import("@/components/code/CloudIDE").then((m) => m.CloudIDE), {
  ssr: false,
  loading: Loader,
});

const MobileIDE = dynamic(() => import("@/components/code/MobileIDE").then((m) => m.MobileIDE), {
  ssr: false,
  loading: Loader,
});

export default function ProjectIDEPage() {
  const params = useParams<{ projectId: string }>();
  const { data: session, status } = useSession();
  const isMobile = useIsMobile();

  if (status === "loading") {
    return <Loader />;
  }

  if (!session) {
    return null; // middleware handles redirect
  }

  if (isMobile) {
    return <MobileIDE projectId={params.projectId} />;
  }

  return <CloudIDE projectId={params.projectId} />;
}
