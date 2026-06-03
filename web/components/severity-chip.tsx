import { AlertCircle, AlertTriangle, Info } from "lucide-react";

import { cn } from "@/lib/utils";

interface SeverityChipProps {
  severity?: "low" | "medium" | "high" | null;
  className?: string;
}

const CONFIG = {
  low: {
    label: "Low severity",
    Icon: Info,
    classes:
      "border-slate-500/20 bg-slate-500/10 text-slate-700 dark:text-slate-300",
  },
  medium: {
    label: "Medium severity",
    Icon: AlertCircle,
    classes:
      "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  },
  high: {
    label: "High severity — act soon",
    Icon: AlertTriangle,
    classes:
      "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300",
  },
} as const;

export function SeverityChip({ severity, className }: SeverityChipProps) {
  if (!severity) return null;
  const { label, Icon, classes } = CONFIG[severity];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        classes,
        className,
      )}
    >
      <Icon className="h-3 w-3" />
      {label}
    </span>
  );
}
