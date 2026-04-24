export type AuthProvider =
  | "email"
  | "google"
  | "github"
  | "apple"
  | "microsoft"
  | "magic_link"
  | "guest";

export type SubscriptionPlan = "free" | "plus" | "team" | "enterprise";

export type User = {
  id: string;
  email: string | null;
  name: string | null;
  avatar_url: string | null;
  provider: AuthProvider;
  plan: SubscriptionPlan;
  is_admin: boolean;
  memory_enabled: boolean;
  custom_instructions: string | null;
  created_at: string;
};

export type MessageRole = "user" | "assistant" | "system" | "tool";

export type Attachment = {
  id: string;
  type: "image" | "document" | "audio" | "code";
  name: string;
  url: string;
  size: number;
  extracted_text?: string;
  mime_type: string;
};

export type Citation = {
  type: "kb" | "web";
  title: string;
  url?: string;
  text?: string;
};

export type Message = {
  id: string;
  conversation_id: string;
  branch_id: string;
  parent_branch_id: string | null;
  sequence: number;
  role: MessageRole;
  content: string;
  model_id: string | null;
  usage: { prompt_tokens: number; completion_tokens: number } | null;
  citations: Citation[] | null;
  attachments: Attachment[] | null;
  feedback: "good" | "bad" | null;
  created_at: string;
};

export type Conversation = {
  id: string;
  user_id: string;
  title: string;
  model_id: string;
  active_branch_id: string | null;
  project_id: string | null;
  agent_id: string | null;
  pinned_at: string | null;
  archived_at: string | null;
  is_shared: boolean;
  share_id: string | null;
  memory_enabled: boolean;
  web_search_enabled: boolean;
  created_at: string;
  updated_at: string;
};

export type ModelRoute = {
  id: string;
  provider: string;
  name: string;
  vision: boolean;
  tool_use: boolean;
  max_input: number;
  cost_in_per_1k: number;
  cost_out_per_1k: number;
  latency_p50_ms?: number;
};

export type Agent = {
  id: string;
  slug: string;
  name: string;
  description: string | null;
  icon_url: string | null;
  category: string;
  instructions: string | null;
  starters: string[] | null;
  capabilities: Record<string, boolean> | null;
  default_model: string;
  visibility: "private" | "unlisted" | "public";
  version: number;
  is_builtin: boolean;
  usage_count: number;
  rating: number | null;
};

export type Project = {
  id: string;
  owner_id: string;
  name: string;
  description: string | null;
  system_prompt: string | null;
  icon_url: string | null;
  created_at: string;
};

export type SSEEvent =
  | { type: "token"; content: string }
  | { type: "done"; usage: { prompt_tokens: number; completion_tokens: number }; model: string }
  | { type: "error"; message: string; model?: string }
  | { type: "tool_call"; tool_calls: unknown[] };
