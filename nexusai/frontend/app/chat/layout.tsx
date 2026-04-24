import { Sidebar } from "@/components/chat/Sidebar";

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar onNewChat={() => {}} />
      <main className="flex flex-1 flex-col overflow-hidden min-w-0">
        {children}
      </main>
    </div>
  );
}
