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
