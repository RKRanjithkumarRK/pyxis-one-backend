import type { Metadata } from "next";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ShareSnapshot {
  title: string;
  owner: string;
  model: string;
  created_at: string;
  messages: Array<{ role: string; content: string; created_at?: string }>;
}

export async function generateMetadata({ params }: { params: { token: string } }): Promise<Metadata> {
  try {
    const data = await fetchSnapshot(params.token);
    return { title: `${data.title} | NexusAI`, description: `Shared by ${data.owner}` };
  } catch {
    return { title: "Shared Conversation | NexusAI" };
  }
}

async function fetchSnapshot(token: string): Promise<ShareSnapshot> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const r = await fetch(`${apiUrl}/api/v1/share/${token}`, { next: { revalidate: 3600 } });
  if (!r.ok) throw new Error("Not found");
  return r.json();
}

export default async function SharedPage({ params }: { params: { token: string } }) {
  let snapshot: ShareSnapshot;
  try {
    snapshot = await fetchSnapshot(params.token);
  } catch {
    return (
      <div className="flex h-screen items-center justify-center bg-[#0d0d0f] text-muted-foreground">
        <p>This share link has expired or does not exist.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0d0d0f] py-10 px-4">
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-white">{snapshot.title}</h1>
          <p className="text-sm text-muted-foreground">
            Shared by <span className="text-white">{snapshot.owner}</span> · {snapshot.model} ·{" "}
            {new Date(snapshot.created_at).toLocaleDateString()}
          </p>
        </div>

        <div className="space-y-4">
          {snapshot.messages.map((msg, i) => (
            <div key={i} className={`flex gap-3 ${msg.role === "assistant" ? "" : "justify-end"}`}>
              {msg.role === "assistant" && (
                <div className="w-7 h-7 rounded-full bg-violet-600 flex items-center justify-center text-xs font-bold shrink-0 mt-1">
                  N
                </div>
              )}
              <div
                className={`rounded-xl px-4 py-3 max-w-[80%] text-sm ${
                  msg.role === "user"
                    ? "bg-violet-600/20 text-violet-100 ml-auto"
                    : "bg-white/5 text-foreground"
                }`}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              </div>
            </div>
          ))}
        </div>

        <div className="text-center pt-8">
          <a
            href="/"
            className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm"
          >
            Try NexusAI
          </a>
        </div>
      </div>
    </div>
  );
}
