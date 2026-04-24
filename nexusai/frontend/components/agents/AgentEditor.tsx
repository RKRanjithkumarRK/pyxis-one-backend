"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/cn";
import type { Agent, CreateAgentPayload } from "@/lib/api-types";

const CATEGORIES = [
  "code", "writing", "productivity", "education",
  "business", "creative", "data", "general",
];

const MODELS = [
  { id: "claude-sonnet-4", label: "Claude Sonnet 4" },
  { id: "claude-opus-4", label: "Claude Opus 4" },
  { id: "gpt-4o", label: "GPT-4o" },
  { id: "gpt-4o-mini", label: "GPT-4o mini" },
  { id: "gemini-2-pro", label: "Gemini 2.0 Pro" },
  { id: "gemini-2-flash", label: "Gemini 2.0 Flash" },
  { id: "llama-3-3-70b", label: "Llama 3.3 70B (Groq)" },
];

type Props = {
  initial?: Agent;
  onSave: (payload: CreateAgentPayload) => Promise<void>;
  onCancel: () => void;
  saving?: boolean;
};

export function AgentEditor({ initial, onSave, onCancel, saving = false }: Props) {
  const router = useRouter();
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [icon, setIcon] = useState(initial?.icon ?? "🤖");
  const [category, setCategory] = useState(initial?.category ?? "general");
  const [instructions, setInstructions] = useState(initial?.instructions ?? "");
  const [defaultModel, setDefaultModel] = useState(initial?.default_model ?? "claude-sonnet-4");
  const [visibility, setVisibility] = useState<"public" | "private">(
    (initial?.visibility as "public" | "private") ?? "private",
  );
  const [starters, setStarters] = useState<string[]>(initial?.starters ?? ["", "", "", ""]);
  const [capVision, setCapVision] = useState(initial?.capabilities?.vision ?? false);
  const [capTools, setCapTools] = useState(initial?.capabilities?.tool_use ?? false);
  const [capSearch, setCapSearch] = useState(initial?.capabilities?.web_search ?? false);
  const [error, setError] = useState<string | null>(null);

  const updateStarter = (i: number, val: string) => {
    const next = [...starters];
    next[i] = val;
    setStarters(next);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    setError(null);
    const filteredStarters = starters.filter((s) => s.trim());
    try {
      await onSave({
        name: name.trim(),
        description: description.trim() || undefined,
        icon: icon || undefined,
        category,
        instructions: instructions.trim() || undefined,
        starters: filteredStarters.length ? filteredStarters : undefined,
        capabilities: { vision: capVision, tool_use: capTools, web_search: capSearch },
        default_model: defaultModel,
        visibility,
      });
    } catch (err: any) {
      setError(err?.message ?? "Failed to save agent");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6 max-w-2xl">
      {error && (
        <div className="rounded-lg bg-destructive/10 border border-destructive/30 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Identity */}
      <section className="flex flex-col gap-4 rounded-xl border border-border bg-card p-5">
        <h2 className="text-sm font-semibold text-foreground">Identity</h2>
        <div className="flex items-center gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Icon</label>
            <input
              type="text"
              value={icon}
              onChange={(e) => setIcon(e.target.value)}
              maxLength={4}
              className="h-12 w-16 rounded-lg border border-border bg-muted text-center text-2xl focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div className="flex flex-1 flex-col gap-1">
            <label className="text-xs text-muted-foreground">Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Python Expert"
              required
              maxLength={256}
              className="h-10 rounded-lg border border-border bg-muted px-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="What does this agent do?"
            maxLength={2000}
            rows={2}
            className="rounded-lg border border-border bg-muted px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="h-10 rounded-lg border border-border bg-muted px-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {c.charAt(0).toUpperCase() + c.slice(1)}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Default Model</label>
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              className="h-10 rounded-lg border border-border bg-muted px-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      {/* System Prompt */}
      <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 className="text-sm font-semibold text-foreground">System Prompt</h2>
        <textarea
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
          placeholder="You are an expert in... Your job is to..."
          maxLength={16000}
          rows={8}
          className="rounded-lg border border-border bg-muted px-3 py-2 font-mono text-sm text-foreground placeholder:text-muted-foreground resize-y focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <p className="text-xs text-muted-foreground">{instructions.length}/16,000 characters</p>
      </section>

      {/* Conversation Starters */}
      <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 className="text-sm font-semibold text-foreground">Conversation Starters</h2>
        <p className="text-xs text-muted-foreground">Up to 4 starter prompts shown to users</p>
        <div className="flex flex-col gap-2">
          {starters.map((s, i) => (
            <input
              key={i}
              type="text"
              value={s}
              onChange={(e) => updateStarter(i, e.target.value)}
              placeholder={`Starter ${i + 1}`}
              maxLength={200}
              className="h-9 rounded-lg border border-border bg-muted px-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          ))}
        </div>
      </section>

      {/* Capabilities */}
      <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 className="text-sm font-semibold text-foreground">Capabilities</h2>
        <div className="flex flex-col gap-2">
          {(
            [
              { key: "vision", label: "Vision (image understanding)", value: capVision, set: setCapVision },
              { key: "tool_use", label: "Tool use", value: capTools, set: setCapTools },
              { key: "web_search", label: "Web search", value: capSearch, set: setCapSearch },
            ] as const
          ).map(({ key, label, value, set }) => (
            <label key={key} className="flex cursor-pointer items-center gap-3">
              <button
                type="button"
                role="switch"
                aria-checked={value}
                onClick={() => set(!value)}
                className={cn(
                  "relative h-5 w-9 rounded-full transition-colors focus:outline-none",
                  value ? "bg-primary" : "bg-muted-foreground/30",
                )}
              >
                <span
                  className={cn(
                    "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                    value ? "translate-x-4" : "translate-x-0.5",
                  )}
                />
              </button>
              <span className="text-sm text-foreground">{label}</span>
            </label>
          ))}
        </div>
      </section>

      {/* Visibility */}
      <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 className="text-sm font-semibold text-foreground">Visibility</h2>
        <div className="flex gap-3">
          {(["private", "public"] as const).map((v) => (
            <label
              key={v}
              className={cn(
                "flex flex-1 cursor-pointer flex-col gap-1 rounded-lg border p-3 transition-colors",
                visibility === v
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/30",
              )}
            >
              <input
                type="radio"
                name="visibility"
                value={v}
                checked={visibility === v}
                onChange={() => setVisibility(v)}
                className="sr-only"
              />
              <span className="text-sm font-medium text-foreground capitalize">{v}</span>
              <span className="text-xs text-muted-foreground">
                {v === "private"
                  ? "Only you can see and use this agent"
                  : "Listed in the Agent Store for everyone"}
              </span>
            </label>
          ))}
        </div>
      </section>

      {/* Actions */}
      <div className="flex items-center justify-end gap-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={saving}
          className="rounded-lg border border-border px-4 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={saving || !name.trim()}
          className="rounded-lg bg-primary px-5 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {saving ? "Saving…" : initial ? "Update Agent" : "Create Agent"}
        </button>
      </div>
    </form>
  );
}
