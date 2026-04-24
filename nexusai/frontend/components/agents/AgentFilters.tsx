"use client";

import { cn } from "@/lib/cn";

const CATEGORIES = [
  { id: null, label: "All", icon: "✦" },
  { id: "code", label: "Code", icon: "💻" },
  { id: "writing", label: "Writing", icon: "✍️" },
  { id: "productivity", label: "Productivity", icon: "⚡" },
  { id: "education", label: "Education", icon: "🎓" },
  { id: "business", label: "Business", icon: "📊" },
  { id: "creative", label: "Creative", icon: "🎨" },
  { id: "data", label: "Data & SEO", icon: "📈" },
  { id: "general", label: "General", icon: "🌐" },
];

const SORT_OPTIONS = [
  { id: "popular", label: "Most Popular" },
  { id: "newest", label: "Newest" },
  { id: "rating", label: "Top Rated" },
  { id: "name", label: "A–Z" },
];

type Props = {
  category: string | null;
  sort: string;
  onCategoryChange: (cat: string | null) => void;
  onSortChange: (sort: string) => void;
};

export function AgentFilters({ category, sort, onCategoryChange, onSortChange }: Props) {
  return (
    <aside className="flex w-52 shrink-0 flex-col gap-6">
      {/* Categories */}
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Category
        </p>
        <nav className="flex flex-col gap-0.5">
          {CATEGORIES.map((cat) => (
            <button
              key={String(cat.id)}
              onClick={() => onCategoryChange(cat.id)}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors text-left",
                category === cat.id
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <span className="text-base leading-none">{cat.icon}</span>
              {cat.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Sort */}
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Sort by
        </p>
        <div className="flex flex-col gap-0.5">
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              onClick={() => onSortChange(opt.id)}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors text-left",
                sort === opt.id
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {sort === opt.id && (
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
              )}
              {sort !== opt.id && <span className="h-1.5 w-1.5 rounded-full opacity-0" />}
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}
