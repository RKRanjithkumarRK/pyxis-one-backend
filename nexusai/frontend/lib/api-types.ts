import type { User } from "./types";

export type AuthResponse = {
  user: User;
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
};

export type GuestSessionResponse = {
  guest_id: string;
  messages_remaining: number;
  token: string;
};
