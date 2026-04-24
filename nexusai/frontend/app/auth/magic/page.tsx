"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { useAuthStore } from "@/lib/store/auth";
import type { AuthResponse } from "@/lib/api-types";

export default function MagicLinkPage() {
  const router = useRouter();
  const params = useSearchParams();
  const { fetchMe } = useAuthStore();
  const [status, setStatus] = useState<"verifying" | "success" | "error">("verifying");
  const [message, setMessage] = useState("");

  useEffect(() => {
    const token = params.get("token");
    if (!token) {
      setStatus("error");
      setMessage("No token found in URL.");
      return;
    }

    api
      .post<AuthResponse>("/api/v1/auth/magic-link/verify", { token })
      .then((data) => {
        fetchMe(data.access_token);
        setStatus("success");
        setTimeout(() => router.push("/chat"), 1200);
      })
      .catch((err) => {
        setStatus("error");
        setMessage(err?.message ?? "Invalid or expired link.");
      });
  }, [params, fetchMe, router]);

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <div className="text-center space-y-3 max-w-sm">
        {status === "verifying" && (
          <>
            <div className="h-8 w-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="font-medium">Verifying your link…</p>
          </>
        )}
        {status === "success" && (
          <>
            <p className="text-4xl">✅</p>
            <p className="font-medium text-lg">Signed in! Redirecting…</p>
          </>
        )}
        {status === "error" && (
          <>
            <p className="text-4xl">❌</p>
            <p className="font-medium text-lg">Link expired or invalid</p>
            <p className="text-sm text-muted-foreground">{message}</p>
            <a
              href="/login"
              className="inline-block px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium"
            >
              Back to sign in
            </a>
          </>
        )}
      </div>
    </main>
  );
}
