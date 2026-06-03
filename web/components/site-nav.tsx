"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Leaf } from "lucide-react";

import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const ROUTES = [
  { href: "/", label: "Diagnose" },
  { href: "/analytics", label: "Analytics" },
  { href: "/about", label: "About" },
] as const;

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 max-w-6xl items-center px-4">
        <Link href="/" className="mr-8 flex items-center gap-2">
          <Leaf className="h-6 w-6 text-emerald-500" />
          <span className="text-lg font-semibold tracking-tight">
            GreenVision
          </span>
        </Link>

        <nav className="flex flex-1 items-center gap-1">
          {ROUTES.map((route) => {
            const active = pathname === route.href;
            return (
              <Link
                key={route.href}
                href={route.href}
                className={cn(
                  "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}
              >
                {route.label}
              </Link>
            );
          })}
        </nav>

        <ThemeToggle />
      </div>
    </header>
  );
}
