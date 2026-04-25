"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { projectsApi } from "@/lib/api";
import type { Project, ProjectMember, ProjectConversation } from "@/lib/api-types";

export function useProjects() {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const data = await projectsApi.list(token);
      setProjects(data);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = useCallback(
    async (payload: { name: string; description?: string; system_prompt?: string }) => {
      if (!token) throw new Error("Not authenticated");
      const project = await projectsApi.create(payload, token);
      setProjects((prev) => [project, ...prev]);
      return project;
    },
    [token],
  );

  const remove = useCallback(
    async (id: string) => {
      if (!token) return;
      await projectsApi.delete(id, token);
      setProjects((prev) => prev.filter((p) => p.id !== id));
    },
    [token],
  );

  return { projects, loading, refresh, create, remove };
}

export function useProject(projectId: string) {
  const { data: session } = useSession();
  const token = (session as any)?.accessToken as string | undefined;

  const [project, setProject] = useState<Project | null>(null);
  const [members, setMembers] = useState<ProjectMember[]>([]);
  const [conversations, setConversations] = useState<ProjectConversation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token || !projectId) return;
    setLoading(true);
    Promise.all([
      projectsApi.get(projectId, token),
      projectsApi.members(projectId, token),
      projectsApi.conversations(projectId, token),
    ])
      .then(([proj, mems, convs]) => {
        setProject(proj);
        setMembers(mems);
        setConversations(convs);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [projectId, token]);

  const update = useCallback(
    async (payload: { name?: string; description?: string; system_prompt?: string }) => {
      if (!token || !project) return;
      const updated = await projectsApi.update(projectId, payload, token);
      setProject(updated);
      return updated;
    },
    [project, projectId, token],
  );

  const invite = useCallback(
    async (email: string, role: string) => {
      if (!token) return;
      const member = await projectsApi.invite(projectId, email, role, token);
      setMembers((prev) => {
        const idx = prev.findIndex((m) => m.user_id === member.user_id);
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = member;
          return next;
        }
        return [...prev, member];
      });
      return member;
    },
    [projectId, token],
  );

  const removeMember = useCallback(
    async (userId: string) => {
      if (!token) return;
      await projectsApi.removeMember(projectId, userId, token);
      setMembers((prev) => prev.filter((m) => m.user_id !== userId));
    },
    [projectId, token],
  );

  const updateMemberRole = useCallback(
    async (userId: string, role: string) => {
      if (!token) return;
      const updated = await projectsApi.updateMemberRole(projectId, userId, role, token);
      setMembers((prev) => prev.map((m) => (m.user_id === userId ? updated : m)));
    },
    [projectId, token],
  );

  return {
    project,
    members,
    conversations,
    loading,
    update,
    invite,
    removeMember,
    updateMemberRole,
  };
}
