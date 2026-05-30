import { Progress } from "@/components/ui/progress";

import type { TopKPrediction } from "@/lib/types";

interface TopKAlternativesProps {
  topK: TopKPrediction[];
}

export function TopKAlternatives({ topK }: TopKAlternativesProps) {
  const alternatives = topK.slice(1);
  if (alternatives.length === 0) return null;
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-muted-foreground">
        Alternative predictions
      </h4>
      <div className="space-y-2">
        {alternatives.map((p) => (
          <div key={p.class_name} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="truncate pr-2">{p.display_name}</span>
              <span className="tabular-nums text-muted-foreground">
                {(p.probability * 100).toFixed(1)}%
              </span>
            </div>
            <Progress value={p.probability * 100} className="h-1.5" />
          </div>
        ))}
      </div>
    </div>
  );
}
