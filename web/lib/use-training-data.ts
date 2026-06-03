"use client";

import * as React from "react";

import type { TrainingData } from "@/lib/types";

interface UseTrainingDataResult {
  data: TrainingData | null;
  error: string | null;
  loading: boolean;
}

/**
 * Fetch the static training_data.json baked by scripts/export_mlflow_for_dashboard.py.
 *
 * Runs once on mount. The file is in /public so it's served at the root.
 * If the file isn't there, the dashboard shows a "run the export script"
 * error so it's discoverable.
 */
export function useTrainingData(): UseTrainingDataResult {
  const [data, setData] = React.useState<TrainingData | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    fetch("/training_data.json", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) {
          throw new Error(
            `Could not load training_data.json (HTTP ${r.status}). ` +
              "Run scripts/export_mlflow_for_dashboard.py to regenerate it.",
          );
        }
        return r.json();
      })
      .then((json: TrainingData) => {
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Unknown error");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, error, loading };
}
