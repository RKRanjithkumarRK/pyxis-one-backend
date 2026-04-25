"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  Code2,
  FlaskConical,
  Image,
  Mic,
  LayoutGrid,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/research", label: "Research", icon: FlaskConical },
  { href: "/code", label: "Code", icon: Code2 },
  { href: "/image", label: "Image", icon: Image },
  { href: "/voice", label: "Voice", icon: Mic },
  { href: "/projects", label: "More", icon: LayoutGrid },
];

export function BottomNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 bg-background/95 backdrop-blur border-t border-border bottom-nav md:hidden">
      <div className="flex items-center justify-around h-16">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex flex-col items-center gap-0.5 px-3 py-2 rounded-xl transition-colors",
                active
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon
                size={22}
                strokeWidth={active ? 2.5 : 1.75}
                className={active ? "drop-shadow-[0_0_6px_rgba(124,58,237,0.6)]" : ""}
              />
              <span className="text-[10px] font-medium leading-none">{label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
