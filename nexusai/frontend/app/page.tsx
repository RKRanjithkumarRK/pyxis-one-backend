import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-background via-background to-primary/5 p-8">
      <div className="flex flex-col items-center gap-6 text-center max-w-2xl">
        {/* Logo */}
        <div className="relative">
          <div className="h-16 w-16 rounded-2xl bg-primary flex items-center justify-center shadow-lg shadow-primary/30">
            <span className="text-2xl font-bold text-white">N</span>
          </div>
          <div className="absolute -bottom-1 -right-1 h-5 w-5 rounded-full bg-green-500 border-2 border-background animate-pulse" />
        </div>

        <div className="space-y-2">
          <h1 className="text-5xl font-bold tracking-tight">
            Nexus<span className="text-primary">AI</span>
          </h1>
          <p className="text-xl text-muted-foreground">
            All-in-One AI Platform
          </p>
        </div>

        <p className="text-muted-foreground max-w-md leading-relaxed">
          Chat · Code · Research · Canvas · Image · Voice — seven AI providers,
          one unified experience. Built for developers, researchers, and
          creators.
        </p>

        <div className="flex gap-3 flex-wrap justify-center">
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors shadow-lg shadow-primary/25"
          >
            Start Chatting
          </Link>
          <Link
            href="/code"
            className="inline-flex items-center gap-2 px-6 py-3 bg-secondary text-secondary-foreground rounded-xl font-medium hover:bg-secondary/80 transition-colors"
          >
            Open NexusCode
          </Link>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full mt-4">
          {features.map((f) => (
            <div
              key={f.label}
              className="flex flex-col items-center gap-1.5 p-3 rounded-xl bg-card border border-border hover:border-primary/40 transition-colors"
            >
              <span className="text-2xl">{f.icon}</span>
              <span className="text-xs font-medium text-muted-foreground">{f.label}</span>
            </div>
          ))}
        </div>

        <p className="text-xs text-muted-foreground/60">
          Phase 1 — Foundation ready · All 7 providers wired · PostgreSQL + Redis + Qdrant
        </p>
      </div>
    </main>
  );
}

const features = [
  { icon: "💬", label: "NexusChat" },
  { icon: "💻", label: "NexusCode" },
  { icon: "🔬", label: "Deep Research" },
  { icon: "🎨", label: "Image Studio" },
  { icon: "🎙️", label: "Voice Mode" },
  { icon: "🧠", label: "Memory" },
  { icon: "📚", label: "Knowledge Base" },
  { icon: "⚡", label: "Workflows" },
];
