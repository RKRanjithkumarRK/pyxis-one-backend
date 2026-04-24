"use client";

import { useSession, signIn, signOut } from "next-auth/react";
import { useEffect } from "react";
import { useAuthStore } from "@/lib/store/auth";
import { api } from "@/lib/api";

export function useAuth() {
  const { data: session, status } = useSession();
  const { user, isLoading, fetchMe, clear, setUser } = useAuthStore();

  useEffect(() => {
    const token = (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
    if (token && !user) {
      fetchMe(token);
    }
    if (status === "unauthenticated") {
      clear();
    }
  }, [session, status, user, fetchMe, clear]);

  const loginWithEmail = async (email: string, password: string) => {
    const result = await signIn("credentials", {
      email,
      password,
      redirect: false,
    });
    return result;
  };

  const loginWithGoogle = () => signIn("google", { callbackUrl: "/chat" });
  const loginWithGitHub = () => signIn("github", { callbackUrl: "/chat" });

  const sendMagicLink = async (email: string) => {
    const resp = await api.post("/api/v1/auth/magic-link", { email });
    return resp;
  };

  const logout = async () => {
    clear();
    await signOut({ callbackUrl: "/" });
  };

  const startGuestSession = async () => {
    const data = (await api.post("/api/v1/auth/guest", {})) as {
      guest_id: string;
      messages_remaining: number;
      token: string;
    };
    fetchMe(data.token);
    return data;
  };

  return {
    user,
    session,
    isLoading: status === "loading" || isLoading,
    isAuthenticated: status === "authenticated",
    isGuest: !session && !!user,
    loginWithEmail,
    loginWithGoogle,
    loginWithGitHub,
    sendMagicLink,
    logout,
    startGuestSession,
  };
}
