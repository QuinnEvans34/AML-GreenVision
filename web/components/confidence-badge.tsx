import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { getConfidenceBand } from "@/lib/types";

interface ConfidenceBadgeProps {
  confidence: number;
  className?: string;
}

const COLORS = {
  high: "border-emerald-500/30 bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  moderate:
    "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  medium:
    "border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-400",
  low: "border-rose-500/30 bg-rose-500/15 text-rose-700 dark:text-rose-400",
} as const;

const LABELS = {
  high: "High confidence",
  moderate: "Moderate-high",
  medium: "Medium confidence",
  low: "Low confidence",
} as const;

export function ConfidenceBadge({ confidence, className }: ConfidenceBadgeProps) {
  const band = getConfidenceBand(confidence);
  return (
    <Badge
      variant="outline"
      className={cn("border tabular-nums", COLORS[band], className)}
    >
      {LABELS[band]} · {(confidence * 100).toFixed(1)}%
    </Badge>
  );
}
