"use client";

import { createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "light" | "system";

type ThemeProviderProps = {
  children: React.ReactNode;
  defaultTheme?: Theme;
  attribute?: string;
  enableSystem?: boolean;
  disableTransitionOnChange?: boolean;
};

type ThemeProviderState = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  resolvedTheme: "dark" | "light";
};

const ThemeProviderContext = createContext<ThemeProviderState>({
  theme: "system",
  setTheme: () => null,
  resolvedTheme: "light",
});

export function ThemeProvider({
  children,
  defaultTheme = "system",
  attribute = "class",
  enableSystem = true,
  disableTransitionOnChange = false,
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(defaultTheme);
  const [resolvedTheme, setResolvedTheme] = useState<"dark" | "light">("light");

  useEffect(() => {
    const stored = (localStorage.getItem("nexusai-theme") as Theme) || defaultTheme;
    setThemeState(stored);
  }, [defaultTheme]);

  useEffect(() => {
    const root = document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const apply = (t: Theme) => {
      const resolved = t === "system" ? (mediaQuery.matches ? "dark" : "light") : t;
      setResolvedTheme(resolved);
      if (disableTransitionOnChange) {
        root.classList.add("[&_*]:!transition-none");
        setTimeout(() => root.classList.remove("[&_*]:!transition-none"), 0);
      }
      if (attribute === "class") {
        root.classList.remove("dark", "light");
        root.classList.add(resolved);
      } else {
        root.setAttribute(attribute, resolved);
      }
    };

    apply(theme);

    if (theme === "system" && enableSystem) {
      const listener = () => apply("system");
      mediaQuery.addEventListener("change", listener);
      return () => mediaQuery.removeEventListener("change", listener);
    }
  }, [theme, attribute, enableSystem, disableTransitionOnChange]);

  const setTheme = (t: Theme) => {
    localStorage.setItem("nexusai-theme", t);
    setThemeState(t);
  };

  return (
    <ThemeProviderContext.Provider value={{ theme, setTheme, resolvedTheme }}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeProviderContext);
