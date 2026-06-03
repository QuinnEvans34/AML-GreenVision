"use client";

import * as React from "react";
import { AlertCircle, CheckCircle2, ServerOff } from "lucide-react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { PredictionResult } from "@/components/prediction-result";
import { UploadCard } from "@/components/upload-card";

import { APIError, health, predict } from "@/lib/api-client";
import type { HealthResponse, PredictionResponse } from "@/lib/types";

export default function HomePage() {
  const [healthStatus, setHealthStatus] = React.useState<HealthResponse | null>(
    null,
  );
  const [healthError, setHealthError] = React.useState<string | null>(null);
  const [prediction, setPrediction] = React.useState<PredictionResponse | null>(
    null,
  );
  const [isPredicting, setIsPredicting] = React.useState(false);
  const [predictionError, setPredictionError] = React.useState<string | null>(
    null,
  );
  // Decision 14 — rembg toggle state.
  // Auto-on for arbitrary uploads (likely field photos), auto-off for
  // test-set thumbnails (already in-distribution). The upload card
  // notifies us of mode changes so we can apply the heuristic.
  const [removeBg, setRemoveBg] = React.useState(true);

  // Mount-time health check.
  React.useEffect(() => {
    health()
      .then((h) => {
        setHealthStatus(h);
        setHealthError(null);
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setHealthError(msg);
        setHealthStatus(null);
      });
  }, []);

  async function handlePredict(
    file: File,
    opts: { removeBg: boolean } = { removeBg },
  ) {
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const result = await predict(file, { removeBg: opts.removeBg });
      setPrediction(result);
      if (result.is_background) {
        toast.warning("No leaf detected — please retake the photo");
      } else if (result.is_healthy) {
        toast.success(`${result.display_name} — no disease detected`);
      } else {
        toast.info(`Diagnosis: ${result.display_name}`);
      }
    } catch (e: unknown) {
      const msg =
        e instanceof APIError ? e.message : e instanceof Error ? e.message : "Unexpected error";
      setPredictionError(msg);
      toast.error(msg);
    } finally {
      setIsPredicting(false);
    }
  }

  // Auto-on/off heuristic per the W10 resilience plan:
  //  • Switching to Upload tab → assume an arbitrary photo, enable rembg
  //  • Switching to Test set → already in-distribution, disable rembg
  function handleModeChange(mode: "test" | "upload") {
    setRemoveBg(mode === "upload");
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Diagnose a leaf</h1>
        <p className="mt-2 text-muted-foreground">
          This is the user interface for the GreenVision FastAPI inference
          endpoint. Pick a leaf from the held-out test set, or upload your own
          photo — the dashboard <code className="font-mono">POST</code>s to{" "}
          <code className="font-mono">/api/predict</code>, which loads{" "}
          <code className="font-mono">models:/GreenVision/Production</code> from
          the MLflow Registry, returns disease name + confidence + cited
          treatment recommendation, and the result renders below.
        </p>
      </div>

      {healthError && (
        <Alert variant="destructive">
          <ServerOff className="h-4 w-4" />
          <AlertTitle>Backend not reachable</AlertTitle>
          <AlertDescription>
            {healthError} Make sure the FastAPI server is running:
            <code className="mt-2 block rounded bg-muted p-2 font-mono text-xs">
              PYTHONPATH=src .venv/bin/uvicorn api.main:app --port 8000
            </code>
          </AlertDescription>
        </Alert>
      )}

      {healthStatus?.model_loaded && (
        <Alert className="border-emerald-500/40 bg-emerald-500/5">
          <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          <AlertTitle className="text-sm">Backend ready</AlertTitle>
          <AlertDescription className="text-xs text-muted-foreground">
            {healthStatus.num_classes} classes loaded · KB v
            {healthStatus.kb_version} ({healthStatus.kb_entry_count} entries) ·{" "}
            <code className="font-mono">{healthStatus.model_version}</code>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <UploadCard
          onPredict={handlePredict}
          isPredicting={isPredicting}
          disabled={!healthStatus?.model_loaded}
          removeBg={removeBg}
          onRemoveBgChange={setRemoveBg}
          onModeChange={handleModeChange}
          onInvalidFile={(reason) => toast.error(reason)}
        />

        {prediction && <PredictionResult prediction={prediction} />}

        {!prediction && !predictionError && (
          <div className="flex min-h-[280px] items-center justify-center rounded-lg border border-dashed p-12 text-center">
            <p className="text-sm text-muted-foreground">
              Upload a leaf photo to see the diagnosis and treatment here.
            </p>
          </div>
        )}

        {predictionError && !prediction && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Prediction failed</AlertTitle>
            <AlertDescription>{predictionError}</AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
}
