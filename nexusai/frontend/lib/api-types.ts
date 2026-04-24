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

export type AgentCapabilities = {
  vision: boolean;
  tool_use: boolean;
  web_search: boolean;
};

export type Agent = {
  id: string;
  slug: string;
  name: string;
  description: string | null;
  icon: string | null;
  icon_url: string | null;
  category: string;
  instructions: string | null;
  starters: string[] | null;
  capabilities: AgentCapabilities | null;
  default_model: string;
  visibility: "public" | "private" | "unlisted";
  version: number;
  is_builtin: boolean;
  usage_count: number;
  rating: number | null;
  rating_count: number;
  creator_id: string | null;
  created_at: string;
  updated_at: string;
};

export type AgentListResponse = {
  agents: Agent[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
};

export type AgentVersion = {
  id: string;
  agent_id: string;
  version: number;
  snapshot: Record<string, unknown>;
  created_at: string;
};

export type CreateAgentPayload = {
  name: string;
  slug?: string;
  description?: string;
  icon?: string;
  category?: string;
  instructions?: string;
  starters?: string[];
  capabilities?: Partial<AgentCapabilities>;
  default_model?: string;
  visibility?: "public" | "private" | "unlisted";
};

export type UpdateAgentPayload = Partial<CreateAgentPayload>;
