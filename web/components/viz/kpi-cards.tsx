import { Card, CardContent } from "@/components/ui/card";

import type { TrainingData } from "@/lib/types";

interface KpiCardsProps {
  data: TrainingData;
}

export function KpiCards({ data }: KpiCardsProps) {
  const { best_run } = data;
  const valAcc = best_run.best_val_acc;
  const testAcc = best_run.test_acc;
  const gap =
    valAcc !== null && testAcc !== null ? Math.abs(valAcc - testAcc) : null;
  const minutes =
    best_run.training_time_seconds !== null
      ? best_run.training_time_seconds / 60
      : null;

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Stat
        label="Best val accuracy"
        value={valAcc !== null ? `${(valAcc * 100).toFixed(2)}%` : "—"}
        sub={`epoch ${best_run.best_epoch_global ?? "—"} / ${best_run.total_epochs}`}
        accent="emerald"
      />
      <Stat
        label="Held-out test accuracy"
        value={testAcc !== null ? `${(testAcc * 100).toFixed(2)}%` : "—"}
        sub={
          gap !== null
            ? `${(gap * 100).toFixed(3)}% val/test gap`
            : "test eval skipped"
        }
        accent="emerald"
      />
      <Stat
        label="Random baseline"
        value="2.56%"
        sub={`1 of ${best_run.num_classes ?? 39} classes`}
        accent="muted"
      />
      <Stat
        label="Training time"
        value={minutes !== null ? `${minutes.toFixed(1)} min` : "—"}
        sub={`${best_run.total_epochs} epochs · attempt ${best_run.attempt_id}`}
        accent="muted"
      />
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub: string;
  accent: "emerald" | "muted";
}) {
  return (
    <Card>
      <CardContent className="space-y-1 p-5">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <p
          className={
            accent === "emerald"
              ? "text-2xl font-semibold tabular-nums text-emerald-600 dark:text-emerald-400"
              : "text-2xl font-semibold tabular-nums"
          }
        >
          {value}
        </p>
        <p className="text-xs text-muted-foreground">{sub}</p>
      </CardContent>
    </Card>
  );
}
