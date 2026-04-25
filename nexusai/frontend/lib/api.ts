const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { token?: string },
): Promise<T> {
  const { token, ...fetchInit } = init ?? {};
  const headers = new Headers(fetchInit.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_BASE}${path}`, { ...fetchInit, headers });

  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    throw new ApiError(res.status, `API ${res.status}: ${path}`, detail);
  }

  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return res.json() as Promise<T>;
  return res.text() as unknown as T;
}

export const api = {
  get: <T>(path: string, token?: string) =>
    request<T>(path, { method: "GET", token }),
  post: <T>(path: string, body: unknown, token?: string) =>
    request<T>(path, {
      method: "POST",
      body: JSON.stringify(body),
      token,
    }),
  patch: <T>(path: string, body: unknown, token?: string) =>
    request<T>(path, {
      method: "PATCH",
      body: JSON.stringify(body),
      token,
    }),
  delete: <T>(path: string, token?: string) =>
    request<T>(path, { method: "DELETE", token }),
};

export async function* streamSSE(
  path: string,
  body: unknown,
  token?: string,
): AsyncGenerator<Record<string, unknown>> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    throw new ApiError(res.status, `SSE failed: ${path}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (!raw || raw === "[DONE]") continue;
      try {
        yield JSON.parse(raw);
      } catch {
        // skip malformed
      }
    }
  }
}

export type HealthResponse = {
  status: string;
  service: string;
  environment: string;
  version: string;
};

export const healthApi = {
  check: () => api.get<HealthResponse>("/api/v1/health"),
  ready: () => api.get("/api/v1/health/ready"),
};

import type {
  Agent,
  AgentListResponse,
  AgentVersion,
  CreateAgentPayload,
  UpdateAgentPayload,
} from "./api-types";

import type { ResearchReport, ResearchDepth } from "./api-types";

export const researchApi = {
  start: (query: string, depth: ResearchDepth, token: string) =>
    api.post<ResearchReport>("/api/v1/research", { query, depth }, token),

  list: (token: string) => api.get<ResearchReport[]>("/api/v1/research", token),

  get: (id: string, token: string) => api.get<ResearchReport>(`/api/v1/research/${id}`, token),

  delete: (id: string, token: string) => api.delete<void>(`/api/v1/research/${id}`, token),

  streamProgress: (id: string, token: string): EventSource => {
    const url = `${API_BASE}/api/v1/research/${id}/stream`;
    const es = new EventSource(`${url}?token=${encodeURIComponent(token)}`);
    return es;
  },
};

export async function* streamResearchProgress(
  reportId: string,
  token: string,
): AsyncGenerator<import("./api-types").ResearchProgressEvent> {
  const url = `${API_BASE}/api/v1/research/${reportId}/stream`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, "Stream failed");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (!raw || raw === "[DONE]") continue;
      try {
        yield JSON.parse(raw);
      } catch {
        // skip malformed
      }
    }
  }
}

import type { Project, ProjectMember, ProjectConversation } from "./api-types";

export const projectsApi = {
  list: (token: string) => api.get<Project[]>("/api/v1/projects", token),
  create: (
    payload: { name: string; description?: string; system_prompt?: string; icon_url?: string },
    token: string,
  ) => api.post<Project>("/api/v1/projects", payload, token),
  get: (id: string, token: string) => api.get<Project>(`/api/v1/projects/${id}`, token),
  update: (
    id: string,
    payload: { name?: string; description?: string; system_prompt?: string; icon_url?: string },
    token: string,
  ) => api.patch<Project>(`/api/v1/projects/${id}`, payload, token),
  delete: (id: string, token: string) => api.delete<void>(`/api/v1/projects/${id}`, token),
  members: (id: string, token: string) =>
    api.get<ProjectMember[]>(`/api/v1/projects/${id}/members`, token),
  invite: (id: string, email: string, role: string, token: string) =>
    api.post<ProjectMember>(`/api/v1/projects/${id}/members`, { email, role }, token),
  updateMemberRole: (id: string, userId: string, role: string, token: string) =>
    api.patch<ProjectMember>(`/api/v1/projects/${id}/members/${userId}`, { role }, token),
  removeMember: (id: string, userId: string, token: string) =>
    api.delete<void>(`/api/v1/projects/${id}/members/${userId}`, token),
  conversations: (id: string, token: string) =>
    api.get<ProjectConversation[]>(`/api/v1/projects/${id}/conversations`, token),
};

import type { UserMemory, MemoryStats } from "./api-types";

export const memoryApi = {
  list: (token: string) => api.get<UserMemory[]>("/api/v1/memory", token),
  stats: (token: string) => api.get<MemoryStats>("/api/v1/memory/stats", token),
  delete: (id: string, token: string) => api.delete<void>(`/api/v1/memory/${id}`, token),
  clearAll: (token: string) =>
    request<{ deleted: number }>("/api/v1/memory", { method: "DELETE", token }),
};

import type { CanvasDoc, CanvasDocListItem, CanvasDocVersion, AIEditResponse } from "./api-types";

export const canvasApi = {
  list: (token: string) => api.get<CanvasDocListItem[]>("/api/v1/canvas", token),
  create: (title: string, token: string) =>
    api.post<CanvasDoc>("/api/v1/canvas", { title }, token),
  get: (id: string, token?: string) =>
    api.get<CanvasDoc>(`/api/v1/canvas/${id}`, token),
  update: (id: string, payload: Record<string, unknown>, token: string) =>
    api.patch<CanvasDoc>(`/api/v1/canvas/${id}`, payload, token),
  delete: (id: string, token: string) =>
    api.delete<void>(`/api/v1/canvas/${id}`, token),
  versions: (id: string, token: string) =>
    api.get<CanvasDocVersion[]>(`/api/v1/canvas/${id}/versions`, token),
  restoreVersion: (id: string, version: number, token: string) =>
    api.post<CanvasDoc>(`/api/v1/canvas/${id}/versions/${version}/restore`, {}, token),
  aiEdit: (
    id: string,
    selectedText: string,
    instruction: string,
    context: string,
    token: string,
  ) =>
    api.post<AIEditResponse>(
      `/api/v1/canvas/${id}/ai-edit`,
      { selected_text: selectedText, instruction, context },
      token,
    ),
  exportUrl: (id: string, format: "md" | "html" | "docx") =>
    `${API_BASE}/api/v1/canvas/${id}/export?format=${format}`,
};

import type { KnowledgeBase, KBFile } from "./api-types";

export const kbApi = {
  list: (token: string) => api.get<KnowledgeBase[]>("/api/v1/kb", token),

  create: (payload: { name: string; description?: string; project_id?: string }, token: string) =>
    api.post<KnowledgeBase>("/api/v1/kb", payload, token),

  get: (id: string, token: string) => api.get<KnowledgeBase>(`/api/v1/kb/${id}`, token),

  update: (id: string, payload: { name?: string; description?: string }, token: string) =>
    api.patch<KnowledgeBase>(`/api/v1/kb/${id}`, payload, token),

  delete: (id: string, token: string) => api.delete<void>(`/api/v1/kb/${id}`, token),

  uploadFile: async (kbId: string, file: File, token: string): Promise<KBFile> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/api/v1/kb/${kbId}/files`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => res.text());
      throw new ApiError(res.status, `Upload failed`, detail);
    }
    return res.json();
  },

  deleteFile: (kbId: string, fileId: string, token: string) =>
    api.delete<void>(`/api/v1/kb/${kbId}/files/${fileId}`, token),

  getFile: (kbId: string, fileId: string, token: string) =>
    api.get<KBFile>(`/api/v1/kb/${kbId}/files/${fileId}`, token),
};

export const agentsApi = {
  list: (
    params: {
      category?: string;
      search?: string;
      sort?: string;
      page?: number;
      page_size?: number;
    },
    token?: string,
  ) => {
    const q = new URLSearchParams();
    if (params.category) q.set("category", params.category);
    if (params.search) q.set("search", params.search);
    if (params.sort) q.set("sort", params.sort);
    if (params.page) q.set("page", String(params.page));
    if (params.page_size) q.set("page_size", String(params.page_size));
    const qs = q.toString();
    return api.get<AgentListResponse>(`/api/v1/agents${qs ? `?${qs}` : ""}`, token);
  },

  mine: (token: string) => api.get<Agent[]>("/api/v1/agents/mine", token),

  get: (ref: string, token?: string) =>
    api.get<Agent>(`/api/v1/agents/${ref}`, token),

  create: (payload: CreateAgentPayload, token: string) =>
    api.post<Agent>("/api/v1/agents", payload, token),

  update: (id: string, payload: UpdateAgentPayload, token: string) =>
    api.patch<Agent>(`/api/v1/agents/${id}`, payload, token),

  delete: (id: string, token: string) =>
    api.delete<void>(`/api/v1/agents/${id}`, token),

  publish: (id: string, isPublic: boolean, token: string) =>
    api.post<Agent>(`/api/v1/agents/${id}/publish?public=${isPublic}`, {}, token),

  versions: (id: string, token: string) =>
    api.get<AgentVersion[]>(`/api/v1/agents/${id}/versions`, token),

  restore: (id: string, version: number, token: string) =>
    api.post<Agent>(`/api/v1/agents/${id}/restore/${version}`, {}, token),

  rate: (id: string, rating: number, token: string) =>
    api.post<Agent>(`/api/v1/agents/${id}/rate`, { rating }, token),
};
