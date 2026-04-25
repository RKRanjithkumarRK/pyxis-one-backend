"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { useWorkflows, type Workflow, type WorkflowRun } from "@/hooks/use-workflows";
import {
  Plus,
  Play,
  Trash2,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronRight,
  Webhook,
  Calendar,
} from "lucide-react";

const WorkflowBuilder = dynamic(
  () => import("@/components/workflows/WorkflowBuilder").then((m) => m.WorkflowBuilder),
  { ssr: false }
);

type View = "list" | "builder" | "runs";

export default function WorkflowsPage() {
  const { workflows, loading, fetchWorkflows, createWorkflow, updateWorkflow, deleteWorkflow, triggerWorkflow, getRuns } =
    useWorkflows();
  const [view, setView] = useState<View>("list");
  const [activeWf, setActiveWf] = useState<Workflow | null>(null);
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [triggering, setTriggering] = useState<string | null>(null);

  useEffect(() => { fetchWorkflows(); }, [fetchWorkflows]);

  const handleNew = async () => {
    const wf = await createWorkflow({
      name: "Untitled Workflow",
      trigger_type: "manual",
      nodes: [{ id: "trigger_1", type: "trigger", position: { x: 250, y: 50 }, data: { label: "Start" } }],
      edges: [],
    });
    setActiveWf(wf);
    setView("builder");
  };

  const handleOpenBuilder = (wf: Workflow) => {
    setActiveWf(wf);
    setView("builder");
  };

  const handleViewRuns = async (wf: Workflow) => {
    setActiveWf(wf);
    const r = await getRuns(wf.id);
    setRuns(r);
    setView("runs");
  };

  const handleSaveBuilder = async (nodes: unknown[], edges: unknown[]) => {
    if (!activeWf) return;
    await updateWorkflow(activeWf.id, { nodes: nodes as Workflow["nodes"], edges: edges as Workflow["edges"] });
  };

  const handleTrigger = async (wf: Workflow) => {
    setTriggering(wf.id);
    await triggerWorkflow(wf.id);
    setTimeout(() => setTriggering(null), 2000);
  };

  if (view === "builder" && activeWf) {
    return (
      <div className="flex flex-col h-screen bg-[#0d0d0f]">
        <div className="flex items-center gap-3 px-4 py-2 border-b border-white/5 bg-[#13131a]">
          <button onClick={() => setView("list")} className="text-muted-foreground hover:text-white text-sm">
            ← Workflows
          </button>
          <span className="text-sm font-medium truncate">{activeWf.name}</span>
          <button
            onClick={() => handleTrigger(activeWf)}
            className="ml-auto flex items-center gap-1.5 px-3 py-1 rounded bg-violet-600 hover:bg-violet-500 text-white text-sm"
          >
            <Play className="w-3.5 h-3.5" /> Run
          </button>
          <button
            onClick={() => handleViewRuns(activeWf)}
            className="flex items-center gap-1.5 px-3 py-1 rounded bg-white/5 hover:bg-white/10 text-muted-foreground text-sm"
          >
            <Clock className="w-3.5 h-3.5" /> History
          </button>
        </div>
        <div className="flex-1 overflow-hidden">
          <WorkflowBuilder
            nodes={activeWf.nodes}
            edges={activeWf.edges}
            onChange={(nodes, edges) => handleSaveBuilder(nodes, edges)}
          />
        </div>
      </div>
    );
  }

  if (view === "runs" && activeWf) {
    return (
      <div className="flex flex-col h-screen bg-[#0d0d0f]">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-white/5 bg-[#13131a]">
          <button onClick={() => setView("list")} className="text-muted-foreground hover:text-white text-sm">
            ← Workflows
          </button>
          <span className="text-sm font-medium">{activeWf.name} — Run History</span>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {runs.length === 0 && (
            <p className="text-center text-muted-foreground text-sm py-10">No runs yet. Trigger the workflow to start.</p>
          )}
          {runs.map((run) => (
            <div key={run.id} className="bg-white/5 rounded-lg p-4 space-y-2">
              <div className="flex items-center gap-2">
                <RunStatusIcon status={run.status} />
                <span className="text-sm font-medium capitalize">{run.status}</span>
                <span className="text-xs text-muted-foreground ml-auto">
                  {run.duration_ms ? `${run.duration_ms}ms` : "—"}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">{new Date(run.created_at).toLocaleString()}</p>
              {run.error && (
                <p className="text-xs text-red-400 bg-red-500/10 rounded p-2 font-mono">{run.error}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-[#0d0d0f]">
      <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5">
        <h1 className="text-lg font-semibold">Workflows</h1>
        <button
          onClick={handleNew}
          className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm"
        >
          <Plus className="w-4 h-4" /> New Workflow
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : workflows.length === 0 ? (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-sm">No workflows yet. Create one to automate tasks.</p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {workflows.map((wf) => (
              <div key={wf.id} className="bg-white/5 rounded-xl p-5 space-y-3 hover:bg-white/8 transition-colors">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-medium text-sm">{wf.name}</h3>
                    {wf.description && (
                      <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">{wf.description}</p>
                    )}
                  </div>
                  <TriggerBadge type={wf.trigger_type} />
                </div>

                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span>{wf.nodes.length} nodes</span>
                  <span>·</span>
                  <span>{wf.edges.length} connections</span>
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <button
                    onClick={() => handleOpenBuilder(wf)}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 rounded bg-white/5 hover:bg-white/10 text-xs text-muted-foreground hover:text-white"
                  >
                    <ChevronRight className="w-3.5 h-3.5" /> Edit
                  </button>
                  <button
                    onClick={() => handleTrigger(wf)}
                    disabled={triggering === wf.id}
                    className="flex items-center gap-1 px-2 py-1.5 rounded bg-violet-600/20 hover:bg-violet-600/40 text-xs text-violet-300 disabled:opacity-50"
                  >
                    {triggering === wf.id ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    Run
                  </button>
                  <button
                    onClick={() => handleViewRuns(wf)}
                    className="p-1.5 rounded hover:bg-white/10 text-muted-foreground"
                    title="Run history"
                  >
                    <Clock className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => deleteWorkflow(wf.id)}
                    className="p-1.5 rounded hover:bg-red-500/20 text-muted-foreground hover:text-red-400"
                    title="Delete"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RunStatusIcon({ status }: { status: string }) {
  if (status === "completed") return <CheckCircle2 className="w-4 h-4 text-green-500" />;
  if (status === "failed") return <XCircle className="w-4 h-4 text-red-500" />;
  if (status === "running") return <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />;
  return <Clock className="w-4 h-4 text-muted-foreground" />;
}

function TriggerBadge({ type }: { type: string }) {
  const map: Record<string, { icon: typeof Play; label: string; color: string }> = {
    manual: { icon: Play, label: "Manual", color: "violet" },
    schedule: { icon: Calendar, label: "Schedule", color: "blue" },
    webhook: { icon: Webhook, label: "Webhook", color: "green" },
  };
  const cfg = map[type] || map.manual;
  const Icon = cfg.icon;
  return (
    <span className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded bg-${cfg.color}-500/20 text-${cfg.color}-400`}>
      <Icon className="w-2.5 h-2.5" />
      {cfg.label}
    </span>
  );
}
