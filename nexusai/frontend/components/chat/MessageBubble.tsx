"use client";

import { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import { cn } from "@/lib/cn";
import type { Message } from "@/lib/types";

type Props = {
  message: Message;
  isStreaming?: boolean;
  streamContent?: string;
  onRegenerate?: () => void;
  onCopy?: () => void;
  onFeedback?: (type: "good" | "bad") => void;
  onEdit?: (newContent: string) => void;
};

export function MessageBubble({
  message,
  isStreaming,
  streamContent,
  onRegenerate,
  onCopy,
  onFeedback,
  onEdit,
}: Props) {
  const [hovering, setHovering] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(message.content);
  const [copied, setCopied] = useState(false);

  const content = isStreaming ? streamContent ?? "" : message.content;
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    onCopy?.();
  }, [content, onCopy]);

  const handleEditSubmit = useCallback(() => {
    if (editValue.trim() && editValue !== message.content) {
      onEdit?.(editValue.trim());
    }
    setEditing(false);
  }, [editValue, message.content, onEdit]);

  return (
    <div
      className={cn(
        "group relative flex gap-3 px-4 py-3 transition-colors",
        isUser ? "flex-row-reverse" : "flex-row",
        hovering && "bg-muted/30"
      )}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      {/* Avatar */}
      <div className={cn(
        "mt-1 h-7 w-7 shrink-0 rounded-full flex items-center justify-center text-xs font-bold",
        isUser ? "bg-primary text-primary-foreground" : "bg-gradient-to-br from-purple-500 to-blue-500 text-white"
      )}>
        {isUser ? "U" : "N"}
      </div>

      {/* Content */}
      <div className={cn("min-w-0 max-w-[calc(100%-4rem)]", isUser && "items-end flex flex-col")}>
        {editing ? (
          <div className="w-full space-y-2">
            <textarea
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              className="w-full min-h-[80px] px-3 py-2 rounded-lg border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring resize-none"
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={handleEditSubmit}
                className="px-3 py-1 bg-primary text-primary-foreground rounded-md text-xs font-medium"
              >
                Submit & regenerate
              </button>
              <button
                onClick={() => { setEditing(false); setEditValue(message.content); }}
                className="px-3 py-1 border border-border rounded-md text-xs"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            {isUser ? (
              <div className="bg-primary/10 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm whitespace-pre-wrap break-words">
                {content}
              </div>
            ) : (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeHighlight, rehypeKatex]}
                >
                  {content || " "}
                </ReactMarkdown>
                {isStreaming && (
                  <span className="inline-block w-0.5 h-4 bg-primary animate-pulse ml-0.5 align-middle" />
                )}
              </div>
            )}

            {/* Citations */}
            {message.citations && message.citations.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {message.citations.map((c, i) => (
                  <a
                    key={i}
                    href={c.url ?? "#"}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary/10 text-xs text-primary hover:bg-primary/20 transition-colors"
                  >
                    [{i + 1}] {c.title}
                  </a>
                ))}
              </div>
            )}
          </>
        )}

        {/* Model tag for assistant */}
        {isAssistant && message.model_id && !isStreaming && (
          <p className="mt-1 text-xs text-muted-foreground/60">{message.model_id}</p>
        )}
      </div>

      {/* Action menu */}
      {hovering && !editing && !isStreaming && (
        <div className={cn(
          "absolute top-2 flex items-center gap-1 rounded-lg border border-border bg-background shadow-sm p-0.5",
          isUser ? "left-12" : "right-4"
        )}>
          <ActionBtn onClick={handleCopy} title="Copy">
            {copied ? "✓" : "⎘"}
          </ActionBtn>
          {isAssistant && (
            <>
              <ActionBtn onClick={onRegenerate} title="Regenerate">↺</ActionBtn>
              <ActionBtn onClick={() => onFeedback?.("good")} title="Good response"
                className={cn(message.feedback === "good" && "text-green-500")}>👍</ActionBtn>
              <ActionBtn onClick={() => onFeedback?.("bad")} title="Bad response"
                className={cn(message.feedback === "bad" && "text-red-500")}>👎</ActionBtn>
            </>
          )}
          {isUser && (
            <ActionBtn onClick={() => setEditing(true)} title="Edit message">✎</ActionBtn>
          )}
        </div>
      )}
    </div>
  );
}

function ActionBtn({
  onClick,
  children,
  title,
  className,
}: {
  onClick?: () => void;
  children: React.ReactNode;
  title: string;
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cn(
        "h-7 w-7 flex items-center justify-center rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors",
        className
      )}
    >
      {children}
    </button>
  );
}
