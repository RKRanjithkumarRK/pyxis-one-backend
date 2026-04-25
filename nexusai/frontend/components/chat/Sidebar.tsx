"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useConversations, groupConversations } from "@/hooks/use-conversations";
import { useChatStore } from "@/lib/store/chat";
import { cn } from "@/lib/cn";
import type { Conversation } from "@/lib/types";

type Props = {
  onNewChat: () => void;
};

export function Sidebar({ onNewChat }: Props) {
  const router = useRouter();
  const { conversations, createConversation, deleteConversation, updateConversation } =
    useConversations();
  const { activeConversationId, setActive } = useChatStore();
  const [search, setSearch] = useState("");
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const filtered = search
    ? conversations.filter((c) => c.title.toLowerCase().includes(search.toLowerCase()))
    : conversations;

  const groups = groupConversations(filtered);

  const handleNew = async () => {
    const conv = await createConversation();
    if (conv) {
      setActive(conv.id);
      router.push(`/chat/${conv.id}`);
    }
    onNewChat();
  };

  const handlePin = async (e: React.MouseEvent, conv: Conversation) => {
    e.preventDefault();
    e.stopPropagation();
    await updateConversation(conv.id, { pinned: !conv.pinned_at });
  };

  const handleArchive = async (e: React.MouseEvent, conv: Conversation) => {
    e.preventDefault();
    e.stopPropagation();
    await updateConversation(conv.id, { archived: true });
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm("Delete this conversation?")) {
      await deleteConversation(id);
      if (activeConversationId === id) {
        setActive(null);
        router.push("/chat");
      }
    }
  };

  return (
    <aside className="flex h-full w-64 flex-col border-r border-border bg-sidebar">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-sidebar-border">
        <Link href="/" className="flex items-center gap-2 font-semibold text-sm">
          <div className="h-6 w-6 rounded-lg bg-primary flex items-center justify-center">
            <span className="text-xs font-bold text-white">N</span>
          </div>
          NexusAI
        </Link>
        <button
          onClick={handleNew}
          className="h-7 w-7 flex items-center justify-center rounded-lg hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="New chat"
        >
          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
          </svg>
        </button>
      </div>

      {/* Search */}
      <div className="px-3 py-2">
        <div className="relative">
          <svg className="absolute left-2.5 top-2 h-3.5 w-3.5 text-muted-foreground" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd" />
          </svg>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search chats…"
            className="w-full rounded-lg border border-border bg-background pl-8 pr-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
      </div>

      {/* Nav links */}
      <div className="px-2 space-y-0.5 pb-2">
        {[
          { label: "Compare",      href: "/chat/compare", icon: "⚡" },
          { label: "Agents",       href: "/agents",       icon: "🤖" },
          { label: "Projects",     href: "/projects",     icon: "📁" },
          { label: "Research",     href: "/research",     icon: "🔬" },
          { label: "Canvas",       href: "/canvas",       icon: "📝" },
          { label: "Knowledge",    href: "/kb",            icon: "🗂️" },
          { label: "Memory",       href: "/memory",       icon: "🧠" },
          { label: "Image Studio", href: "/image",         icon: "🎨" },
        ].map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          >
            <span>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </div>

      <div className="mx-3 border-t border-sidebar-border" />

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-4">
        {groups.length === 0 && (
          <p className="text-xs text-muted-foreground px-2 py-4 text-center">No conversations yet</p>
        )}
        {groups.map((group) => (
          <div key={group.label}>
            <p className="px-2 mb-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {group.label}
            </p>
            {group.conversations.map((conv) => (
              <div
                key={conv.id}
                className="relative"
                onMouseEnter={() => setHoveredId(conv.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <Link
                  href={`/chat/${conv.id}`}
                  onClick={() => setActive(conv.id)}
                  className={cn(
                    "flex items-center gap-2 px-2 py-2 rounded-lg text-sm transition-colors group",
                    activeConversationId === conv.id
                      ? "bg-accent text-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                  )}
                >
                  {conv.pinned_at && <span className="text-xs shrink-0">📌</span>}
                  <span className="truncate flex-1">{conv.title}</span>
                </Link>

                {/* Hover actions */}
                {hoveredId === conv.id && (
                  <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5 bg-sidebar">
                    <button
                      onClick={(e) => handlePin(e, conv)}
                      className="h-6 w-6 flex items-center justify-center rounded hover:bg-accent text-muted-foreground hover:text-foreground"
                      title={conv.pinned_at ? "Unpin" : "Pin"}
                    >
                      {conv.pinned_at ? "📌" : "📍"}
                    </button>
                    <button
                      onClick={(e) => handleArchive(e, conv)}
                      className="h-6 w-6 flex items-center justify-center rounded hover:bg-accent text-muted-foreground hover:text-foreground"
                      title="Archive"
                    >
                      📦
                    </button>
                    <button
                      onClick={(e) => handleDelete(e, conv.id)}
                      className="h-6 w-6 flex items-center justify-center rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive"
                      title="Delete"
                    >
                      🗑
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="border-t border-sidebar-border p-3">
        <Link
          href="/settings"
          className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
        >
          ⚙️ Settings
        </Link>
      </div>
    </aside>
  );
}
