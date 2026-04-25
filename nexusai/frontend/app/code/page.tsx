import Link from "next/link";

export default function CodePage() {
  return (
    <div className="flex h-screen items-center justify-center bg-[#0d0d0f]">
      <div className="text-center space-y-4">
        <h1 className="text-2xl font-semibold text-white">NexusCode</h1>
        <p className="text-muted-foreground text-sm">Select or create a project to open the Cloud IDE</p>
        <div className="flex gap-3 justify-center">
          <Link
            href="/projects"
            className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium"
          >
            My Projects
          </Link>
        </div>
      </div>
    </div>
  );
}
