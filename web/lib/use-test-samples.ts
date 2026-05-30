"use client";

import * as React from "react";

export interface TestSample {
  class_name: string;
  display_name: string;
  crop: string;
  is_healthy: boolean;
  is_background: boolean;
  images: string[];
}

interface UseTestSamplesResult {
  samples: TestSample[] | null;
  error: string | null;
  loading: boolean;
}

/**
 * Load the test-set manifest baked by scripts/copy_test_samples.py.
 *
 * Returns the list grouped by crop. UI components can re-group or filter
 * however they want.
 */
export function useTestSamples(): UseTestSamplesResult {
  const [samples, setSamples] = React.useState<TestSample[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    fetch("/test-samples.json", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) {
          throw new Error(
            `Could not load test-samples.json (HTTP ${r.status}). ` +
              "Run scripts/copy_test_samples.py to generate it.",
          );
        }
        return r.json();
      })
      .then((data: TestSample[]) => {
        if (!cancelled) {
          setSamples(data);
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

  return { samples, error, loading };
}

/**
 * Fetch a sample image URL and return a File suitable for FormData upload.
 */
export async function sampleToFile(
  url: string,
  className: string,
): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not fetch sample (${res.status})`);
  const blob = await res.blob();
  // Extension from URL
  const ext = url.split(".").pop() ?? "jpg";
  return new File([blob], `${className}.${ext}`, {
    type: blob.type || "image/jpeg",
  });
}
