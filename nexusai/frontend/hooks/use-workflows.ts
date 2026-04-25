"use client";
import { useState, useCallback } from "react";
import { useSession } from "next-auth/react";

export interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, unknown>;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  trigger_type: "manual" | "schedule" | "webhook";
  trigger_config: Record<string, unknown>;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  is_active: boolean;
  last_run_at?: string;
  created_at: string;
  updated_at: string;
}

export interface WorkflowRun {
  id: string;
  workflow_id: string;
  status: "pending" | "running" | "completed" | "failed";
  trigger: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  error?: string;
  node_results: Record<string, unknown>;
  duration_ms?: number;
  created_at: string;
}

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function useToken() {
  const { data: session } = useSession();
  return (session?.user as Record<string, unknown>)?.accessToken as string | undefined;
}

export function useWorkflows() {
  const token = useToken();
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(false);

  const headers = () => ({ "Content-Type": "application/json", Authorization: `Bearer ${token}` });

  const fetchWorkflows = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    const r = await fetch(`${API}/api/v1/workflows`, { headers: headers() });
    if (r.ok) setWorkflows(await r.json());
    setLoading(false);
  }, [token]);

  const createWorkflow = useCallback(async (data: Partial<Workflow>) => {
    const r = await fetch(`${API}/api/v1/workflows`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(data),
    });
    const wf = await r.json();
    setWorkflows((prev) => [wf, ...prev]);
    return wf as Workflow;
  }, [token]);

  const updateWorkflow = useCallback(async (id: string, data: Partial<Workflow>) => {
    const r = await fetch(`${API}/api/v1/workflows/${id}`, {
      method: "PATCH",
      headers: headers(),
      body: JSON.stringify(data),
    });
    const wf = await r.json();
    setWorkflows((prev) => prev.map((w) => (w.id === id ? wf : w)));
    return wf as Workflow;
  }, [token]);

  const deleteWorkflow = useCallback(async (id: string) => {
    await fetch(`${API}/api/v1/workflows/${id}`, { method: "DELETE", headers: headers() });
    setWorkflows((prev) => prev.filter((w) => w.id !== id));
  }, [token]);

  const triggerWorkflow = useCallback(async (id: string, inputs: Record<string, unknown> = {}) => {
    const r = await fetch(`${API}/api/v1/workflows/${id}/trigger`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ inputs }),
    });
    return r.json();
  }, [token]);

  const getRuns = useCallback(async (id: string): Promise<WorkflowRun[]> => {
    const r = await fetch(`${API}/api/v1/workflows/${id}/runs`, { headers: headers() });
    return r.ok ? r.json() : [];
  }, [token]);

  return { workflows, loading, fetchWorkflows, createWorkflow, updateWorkflow, deleteWorkflow, triggerWorkflow, getRuns };
}
