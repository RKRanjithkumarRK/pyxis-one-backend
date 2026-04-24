"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { useAuthStore } from "@/lib/store/auth";
import type { AuthResponse } from "@/lib/api-types";
import Link from "next/link";

export default function VerifyEmailPage() {
  const params = useSearchParams();
  const { fetchMe } = useAuthStore();
  const [status, setStatus] = useState<"verifying" | "success" | "error">("verifying");
  const [message, setMessage] = useState("");

  useEffect(() => {
    const token = params.get("token");
    if (!token) {
      setStatus("error");
      setMessage("No token in URL.");
      return;
    }

    api
      .post<AuthResponse>("/api/v1/auth/verify-email", { token })
      .then((data) => {
        fetchMe(data.access_token);
        setStatus("success");
      })
      .catch((err) => {
        setStatus("error");
        setMessage(err?.message ?? "Invalid or expired link.");
      });
  }, [params, fetchMe]);

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <div className="text-center space-y-3 max-w-sm">
        {status === "verifying" && (
          <>
            <div className="h-8 w-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="font-medium">Verifying your email…</p>
          </>
        )}
        {status === "success" && (
          <>
            <p className="text-4xl">🎉</p>
            <p className="font-medium text-lg">Email verified!</p>
            <p className="text-sm text-muted-foreground">Your account is now fully activated.</p>
            <Link
              href="/chat"
              className="inline-block px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium"
            >
              Go to NexusAI
            </Link>
          </>
        )}
        {status === "error" && (
          <>
            <p className="text-4xl">❌</p>
            <p className="font-medium text-lg">Verification failed</p>
            <p className="text-sm text-muted-foreground">{message}</p>
            <Link href="/login" className="text-primary hover:underline text-sm">
              Back to sign in
            </Link>
          </>
        )}
      </div>
    </main>
  );
}
