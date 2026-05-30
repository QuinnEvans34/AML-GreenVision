"use client";

import * as React from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  ExternalLink,
  Eye,
  EyeOff,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

import { AIDisclaimer } from "@/components/ai-disclaimer";
import { ConfidenceBadge } from "@/components/confidence-badge";
import { SeverityChip } from "@/components/severity-chip";
import { TopKAlternatives } from "@/components/top-k-alternatives";

import { getConfidenceBand, type PredictionResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

interface PredictionResultProps {
  prediction: PredictionResponse;
}

type Tone = "emerald" | "rose" | "amber" | "slate";

function toneClasses(tone: Tone) {
  switch (tone) {
    case "emerald":
      return {
        accent: "bg-emerald-500",
        ring: "border-emerald-500/40",
        soft: "bg-emerald-500/5",
        textStrong: "text-emerald-700 dark:text-emerald-400",
      };
    case "rose":
      return {
        accent: "bg-rose-500",
        ring: "border-rose-500/40",
        soft: "bg-rose-500/5",
        textStrong: "text-rose-700 dark:text-rose-400",
      };
    case "amber":
      return {
        accent: "bg-amber-500",
        ring: "border-amber-500/40",
        soft: "bg-amber-500/5",
        textStrong: "text-amber-700 dark:text-amber-400",
      };
    default:
      return {
        accent: "bg-slate-400",
        ring: "border-slate-300",
        soft: "bg-slate-100 dark:bg-slate-900",
        textStrong: "text-slate-700 dark:text-slate-300",
      };
  }
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const [forceShow, setForceShow] = React.useState(false);
  const band = getConfidenceBand(prediction.confidence);
  const isLowConfidence = band === "low";

  // Pick a tone driven by prediction kind + severity
  let tone: Tone = "slate";
  if (prediction.is_background) tone = "amber";
  else if (prediction.is_healthy) tone = "emerald";
  else if (prediction.treatment.severity === "high") tone = "rose";
  else if (prediction.treatment.severity === "medium") tone = "amber";
  else tone = "slate";
  const toneCls = toneClasses(tone);

  // ── No-leaf state ──────────────────────────────────────────────
  if (prediction.is_background) {
    return (
      <Card className={cn("overflow-hidden", toneCls.ring)}>
        <div className={cn("h-1 w-full", toneCls.accent)} />
        <CardContent className="space-y-4 p-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className={cn("h-5 w-5", toneCls.textStrong)} />
              <span
                className={cn(
                  "text-xs font-medium uppercase tracking-wide",
                  toneCls.textStrong,
                )}
              >
                No leaf detected
              </span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight">
              Please retake the photo
            </h2>
            <p className="text-sm text-muted-foreground">
              {prediction.treatment.summary}
            </p>
          </div>
          {prediction.treatment.retake_guidance && (
            <div>
              <h3 className="mb-2 text-sm font-medium">
                How to retake the photo
              </h3>
              <ul className="space-y-2 text-sm">
                {prediction.treatment.retake_guidance.map((step, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="tabular-nums text-muted-foreground">
                      {i + 1}.
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <AIDisclaimer />
          <InferenceFooter prediction={prediction} />
        </CardContent>
      </Card>
    );
  }

  // ── Healthy state ──────────────────────────────────────────────
  if (prediction.is_healthy) {
    return (
      <Card className={cn("overflow-hidden", toneCls.ring)}>
        <div className={cn("h-1 w-full", toneCls.accent)} />
        <CardContent className="space-y-4 p-6">
          <ResultHero
            prediction={prediction}
            tone={tone}
            icon={<CheckCircle2 className={cn("h-5 w-5", toneCls.textStrong)} />}
            kicker="No disease detected"
          />
          {prediction.treatment.maintenance_tips && (
            <div>
              <h3 className="mb-2 text-sm font-medium">Maintenance tips</h3>
              <ul className="space-y-2 text-sm">
                {prediction.treatment.maintenance_tips.map((tip, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-0.5 text-emerald-500">✓</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {prediction.warnings.length > 0 && (
            <WarningsBlock warnings={prediction.warnings} />
          )}

          <Separator />
          <TopKAlternatives topK={prediction.top_k} />

          {prediction.treatment.sources.length > 0 && (
            <>
              <Separator />
              <SourcesList sources={prediction.treatment.sources} />
            </>
          )}

          <AIDisclaimer />
          <InferenceFooter prediction={prediction} />
        </CardContent>
      </Card>
    );
  }

  // ── Disease state ──────────────────────────────────────────────
  const showTreatment = !isLowConfidence || forceShow;

  return (
    <Card className={cn("overflow-hidden", toneCls.ring)}>
      <div className={cn("h-1 w-full", toneCls.accent)} />
      <CardContent className="space-y-4 p-6">
        <ResultHero
          prediction={prediction}
          tone={tone}
          kicker={
            prediction.treatment.severity === "high"
              ? "High-severity diagnosis"
              : "Diagnosis"
          }
        />

        {prediction.warnings.length > 0 && (
          <WarningsBlock warnings={prediction.warnings} />
        )}

        {!showTreatment ? (
          <div className="space-y-3 rounded-md border border-dashed p-6 text-center">
            <EyeOff className="mx-auto h-8 w-8 text-muted-foreground" />
            <div className="space-y-1">
              <p className="text-sm font-medium">
                Treatment hidden — confidence too low
              </p>
              <p className="text-xs text-muted-foreground">
                Applying treatment for the wrong disease can damage the plant
                and waste money. Retake the photo if possible.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setForceShow(true)}
            >
              <Eye className="mr-1 h-3 w-3" />
              Show anyway
            </Button>
          </div>
        ) : (
          <>
            {prediction.treatment.time_sensitivity && (
              <div
                className={cn(
                  "flex items-start gap-2 rounded-md p-3 text-sm",
                  toneCls.soft,
                )}
              >
                <Clock
                  className={cn("mt-0.5 h-4 w-4 shrink-0", toneCls.textStrong)}
                />
                <div>
                  <p className="font-medium">When to act</p>
                  <p className="text-xs text-muted-foreground">
                    {prediction.treatment.time_sensitivity}
                  </p>
                </div>
              </div>
            )}

            {prediction.treatment.action_steps && (
              <div>
                <h3 className="mb-2 text-sm font-medium">Treatment steps</h3>
                <ol className="space-y-2 text-sm">
                  {prediction.treatment.action_steps.map((step, i) => (
                    <li key={i} className="flex gap-3">
                      <span
                        className={cn(
                          "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                          toneCls.soft,
                          toneCls.textStrong,
                        )}
                      >
                        {i + 1}
                      </span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </>
        )}

        <Separator />
        <TopKAlternatives topK={prediction.top_k} />

        {prediction.treatment.sources.length > 0 && (
          <>
            <Separator />
            <SourcesList sources={prediction.treatment.sources} />
          </>
        )}

        <AIDisclaimer strengthened={isLowConfidence} />
        <InferenceFooter prediction={prediction} />
      </CardContent>
    </Card>
  );
}

// ── Reusable hero block ────────────────────────────────────────────

function ResultHero({
  prediction,
  tone,
  kicker,
  icon,
}: {
  prediction: PredictionResponse;
  tone: Tone;
  kicker: string;
  icon?: React.ReactNode;
}) {
  const toneCls = toneClasses(tone);
  const conf = prediction.confidence * 100;
  return (
    <div className={cn("rounded-md p-4", toneCls.soft)}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        {icon}
        <span
          className={cn(
            "text-xs font-medium uppercase tracking-wide",
            toneCls.textStrong,
          )}
        >
          {kicker}
        </span>
        <span className="ml-auto flex items-center gap-2">
          <ConfidenceBadge confidence={prediction.confidence} />
          <SeverityChip severity={prediction.treatment.severity} />
        </span>
      </div>
      <div className="flex items-end justify-between gap-4">
        <h2 className="text-2xl font-bold leading-tight tracking-tight">
          {prediction.display_name}
        </h2>
        <div className="text-right shrink-0">
          <div
            className={cn(
              "text-3xl font-bold leading-none tabular-nums",
              toneCls.textStrong,
            )}
          >
            {conf.toFixed(1)}%
          </div>
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            confidence
          </div>
        </div>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        {prediction.treatment.summary}
      </p>
    </div>
  );
}

// ── Small subcomponents ────────────────────────────────────────────

function WarningsBlock({ warnings }: { warnings: string[] }) {
  return (
    <div className="space-y-1 rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
      {warnings.map((w, i) => (
        <p key={i} className="text-xs text-amber-700 dark:text-amber-300">
          {w}
        </p>
      ))}
    </div>
  );
}

function SourcesList({
  sources,
}: {
  sources: { name: string; url: string }[];
}) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-medium">Sources</h3>
      <ul className="space-y-1 text-xs">
        {sources.map((s) => (
          <li key={s.url}>
            <a
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              {s.name}
              <ExternalLink className="h-3 w-3" />
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

function InferenceFooter({ prediction }: { prediction: PredictionResponse }) {
  return (
    <p className="text-center text-[10px] text-muted-foreground">
      Inference: {prediction.inference_time_ms.toFixed(1)} ms · model:{" "}
      <code className="font-mono">{prediction.model_version}</code>
    </p>
  );
}
