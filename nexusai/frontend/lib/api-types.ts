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

export type ResearchDepth = "quick" | "standard" | "deep";
export type ResearchStatus = "pending" | "running" | "complete" | "error";

export type ResearchSection = {
  heading: string;
  content: string;
};

export type ResearchCitation = {
  id: number;
  title: string;
  url: string;
  snippet: string;
};

export type ResearchReportData = {
  title: string;
  executive_summary: string;
  sections: ResearchSection[];
  key_findings: string[];
  citations: ResearchCitation[];
  sub_questions: string[];
  depth: ResearchDepth;
  generated_at: string;
};

export type ResearchReport = {
  id: string;
  query: string;
  depth: ResearchDepth;
  status: ResearchStatus;
  title: string | null;
  report: ResearchReportData | null;
  error: string | null;
  sources_count: number;
  task_id: string | null;
  created_at: string;
  completed_at: string | null;
};

export type ResearchProgressEvent = {
  stage: string;
  progress: number;
  message: string;
  report_id?: string;
  title?: string;
  sources_count?: number;
};

// ─── Projects ────────────────────────────────────────────
export type ProjectRole = "owner" | "editor" | "viewer";

export type Project = {
  id: string;
  owner_id: string;
  name: string;
  description: string | null;
  system_prompt: string | null;
  icon_url: string | null;
  role: ProjectRole | null;
  created_at: string;
  updated_at: string;
};

export type ProjectMember = {
  user_id: string;
  project_id: string;
  role: ProjectRole;
  email: string | null;
};

export type ProjectConversation = {
  id: string;
  title: string;
  model_id: string;
  created_at: string;
  updated_at: string;
};

// ─── Memory ──────────────────────────────────────────────
export type UserMemory = {
  id: string;
  fact: string;
  use_count: number;
  created_at: string;
  last_used_at: string | null;
};

export type MemoryStats = {
  count: number;
};

// ─── Canvas ──────────────────────────────────────────────
export type CanvasDoc = {
  id: string;
  title: string;
  content: Record<string, unknown> | null;
  version: number;
  is_public: boolean;
  created_at: string;
  updated_at: string;
};

export type CanvasDocListItem = {
  id: string;
  title: string;
  version: number;
  is_public: boolean;
  created_at: string;
  updated_at: string;
};

export type CanvasDocVersion = {
  id: string;
  document_id: string;
  version: number;
  title: string | null;
  created_at: string;
};

export type AIEditResponse = {
  original: string;
  suggested: string;
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

// ─── Knowledge Base ──────────────────────────────────────
export type KBFileStatus = "pending" | "processing" | "done" | "error";

export type KBFile = {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: KBFileStatus;
  chunk_count: number;
  error_msg: string | null;
};

export type KnowledgeBase = {
  id: string;
  name: string;
  description: string | null;
  project_id: string | null;
  files: KBFile[];
};
