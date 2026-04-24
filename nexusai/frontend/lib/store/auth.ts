import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import type { User } from "@/lib/types";
import { api } from "@/lib/api";

type AuthState = {
  user: User | null;
  accessToken: string | null;
  isLoading: boolean;
  error: string | null;
};

type AuthActions = {
  setUser: (user: User | null) => void;
  setAccessToken: (token: string | null) => void;
  fetchMe: (token: string) => Promise<void>;
  clear: () => void;
};

export const useAuthStore = create<AuthState & AuthActions>()(
  immer((set) => ({
    user: null,
    accessToken: null,
    isLoading: false,
    error: null,

    setUser: (user) =>
      set((s) => {
        s.user = user;
      }),

    setAccessToken: (token) =>
      set((s) => {
        s.accessToken = token;
      }),

    fetchMe: async (token: string) => {
      set((s) => {
        s.isLoading = true;
        s.error = null;
      });
      try {
        const user = await api.get<User>("/api/v1/auth/me", token);
        set((s) => {
          s.user = user;
          s.accessToken = token;
          s.isLoading = false;
        });
      } catch (err) {
        set((s) => {
          s.error = err instanceof Error ? err.message : "Failed to fetch user";
          s.isLoading = false;
        });
      }
    },

    clear: () =>
      set((s) => {
        s.user = null;
        s.accessToken = null;
        s.error = null;
      }),
  }))
);
