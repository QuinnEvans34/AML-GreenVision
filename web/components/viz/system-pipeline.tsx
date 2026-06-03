"use client";

import * as React from "react";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  Box,
  Brain,
  Database,
  FileCode,
  Layout,
  Server,
  Sliders,
  Upload,
} from "lucide-react";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/**
 * Full-project architecture diagram for the "Architecture" segment of the
 * W10P1 presentation. "Architecture" in the rubric refers to the entire
 * pipeline + codebase, not just the neural net — this component is the
 * answer.
 *
 * Three sections stacked vertically:
 *   1. Training pipeline (offline build)
 *   2. Model registry (the contract between training and serving)
 *   3. Serving layer (live, per-request)
 *
 * Each stage box is pointable: it has a title, an icon, key implementation
 * details, the file path, and a "Rubric:" tag tying it to a specific
 * assignment requirement.
 */
export function SystemPipeline() {
  return (
    <div className="space-y-6">
      <SectionHeader
        kicker="Phase 1 — offline, one-time"
        title="Training pipeline"
        subtitle="From raw images to a trained, registered model"
        color="blue"
      />
      <PipelineRow>
        <StageBox
          icon={<Database className="h-5 w-5" />}
          title="Dataset"
          subtitle="PlantVillage"
          details={[
            "54,306 leaf images",
            "39 classes (38 disease/healthy + 1 background)",
            "Stratified 80 / 10 / 10 split (seed=42)",
          ]}
          codePath="data/raw/PlantVillage/"
          color="blue"
        />
        <ArrowRightIcon />
        <StageBox
          icon={<Sliders className="h-5 w-5" />}
          title="Preprocessing"
          subtitle="Validation transforms"
          details={[
            "Resize(256) → CenterCrop(224)",
            "ImageNet mean/std normalization",
            "Deterministic — no augmentation at eval",
          ]}
          codePath="src/greenvision/data/transforms.py"
          rubricTieIn="Part 1 · Part 3"
          rubricQuote="“same transforms used for validation”"
          color="blue"
        />
        <ArrowRightIcon />
        <StageBox
          icon={<Brain className="h-5 w-5" />}
          title="Model"
          subtitle="EfficientNet-B0"
          details={[
            "Pretrained on ImageNet",
            "Custom head: Dropout(0.3) → Linear(1280, 39)",
            "Two-phase fine-tuning + gradual unfreezing",
          ]}
          codePath="src/greenvision/models/efficientnet_head.py"
          color="blue"
        />
        <ArrowRightIcon />
        <StageBox
          icon={<Activity className="h-5 w-5" />}
          title="Training + MLflow"
          subtitle="AdamW · nested runs"
          details={[
            "AdamW, decoupled weight decay (W9D2)",
            "ReduceLROnPlateau, early stopping",
            "24 epochs · best val 99.73% at epoch 18",
          ]}
          codePath="scripts/train.py + src/greenvision/training/"
          color="blue"
        />
      </PipelineRow>

      <CenteredArrow />

      <SectionHeader
        kicker="Phase 2 — the contract"
        title="MLflow Model Registry"
        subtitle="The single bridge between training and serving"
        color="violet"
      />
      <PipelineRow center>
        <StageBox
          icon={<Box className="h-5 w-5" />}
          title="GreenVision v3 @ Production"
          subtitle="file:./mlruns/"
          details={[
            "mlflow.pytorch.log_model(registered_model_name=\"GreenVision\")",
            "scripts/promote.py transitions stage to Production",
            "Class-names artifact attached to the same run",
          ]}
          codePath="scripts/promote.py · file:./mlruns/"
          rubricTieIn="Part 1 · Deliverables"
          rubricQuote="“Load model from MLflow Registry: models:/GreenVision/Production”"
          color="violet"
          wide
        />
      </PipelineRow>

      <CenteredArrow />

      <SectionHeader
        kicker="Phase 3 — live, per-request"
        title="Serving layer"
        subtitle="From a farmer's photo to a treatment recommendation"
        color="emerald"
      />
      <PipelineRow>
        <StageBox
          icon={<Layout className="h-5 w-5" />}
          title="Dashboard"
          subtitle="Next.js 16 + React 19"
          details={[
            "Tailwind 4 + shadcn/ui (this UI)",
            "React Three Fiber for 3D analytics",
            "Mount-time /health check + state",
          ]}
          codePath="web/app/ + web/components/"
          rubricTieIn="Part 2"
          rubricQuote="“user interface that calls your FastAPI endpoint”"
          color="emerald"
        />
        <ArrowRightIcon />
        <StageBox
          icon={<Upload className="h-5 w-5" />}
          title="User upload"
          subtitle="POST /api/predict"
          details={[
            "multipart/form-data (JPG/PNG/WebP, max 10 MB)",
            "Same-origin via next.config rewrite",
            "Test-set picker → auto-predict on click",
          ]}
          codePath="web/lib/api-client.ts + web/components/upload-card.tsx"
          color="emerald"
        />
        <ArrowRightIcon />
        <StageBox
          icon={<Server className="h-5 w-5" />}
          title="FastAPI backend"
          subtitle="Inference + treatment merge"
          details={[
            "GET /health · POST /predict",
            "Lifespan loads Production model + class names + KB",
            "Pydantic v2 response (typed contract)",
          ]}
          codePath="api/main.py + api/routes/"
          rubricTieIn="Part 1"
          rubricQuote="“GET /health · POST /predict”"
          color="emerald"
        />
        <ArrowRightIcon />
        <StageBox
          icon={<FileCode className="h-5 w-5" />}
          title="Treatment KB"
          subtitle="Static JSON"
          details={[
            "39 entries · severity · action steps",
            "Cited US extension publications",
            "Loaded once at API startup",
          ]}
          codePath="data/treatments.json"
          rubricTieIn="Part 1 · design decision"
          rubricQuote="“Where does treatment recommendation text come from?”"
          color="emerald"
        />
      </PipelineRow>

      {/* Footer caption */}
      <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
        <strong>How to read this diagram:</strong> Each box is a real file in the
        repo. The path under each title points to the actual implementation.
        Boxes tagged with a <span className="font-medium">Rubric:</span> badge
        satisfy a specific assignment requirement — hover or click to see the
        rubric quote.
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Layout primitives
// ──────────────────────────────────────────────────────────────────

type StageColor = "blue" | "emerald" | "violet";

function SectionHeader({
  kicker,
  title,
  subtitle,
  color,
}: {
  kicker: string;
  title: string;
  subtitle: string;
  color: StageColor;
}) {
  const colors = {
    blue: "text-blue-700 dark:text-blue-400",
    emerald: "text-emerald-700 dark:text-emerald-400",
    violet: "text-violet-700 dark:text-violet-400",
  } as const;
  return (
    <div className="space-y-0.5">
      <p
        className={cn(
          "text-[10px] font-semibold uppercase tracking-widest",
          colors[color],
        )}
      >
        {kicker}
      </p>
      <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
      <p className="text-sm text-muted-foreground">{subtitle}</p>
    </div>
  );
}

function PipelineRow({
  children,
  center = false,
}: {
  children: React.ReactNode;
  center?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-stretch gap-3 lg:flex-row",
        center && "lg:justify-center",
      )}
    >
      {children}
    </div>
  );
}

function ArrowRightIcon() {
  return (
    <div className="flex items-center justify-center">
      <ArrowRight className="h-5 w-5 rotate-90 text-muted-foreground lg:rotate-0" />
    </div>
  );
}

function CenteredArrow() {
  return (
    <div className="flex justify-center">
      <ArrowDown className="h-5 w-5 text-muted-foreground" />
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Stage box
// ──────────────────────────────────────────────────────────────────

interface StageBoxProps {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  details: string[];
  codePath: string;
  rubricTieIn?: string;
  rubricQuote?: string;
  color: StageColor;
  wide?: boolean;
}

function StageBox({
  icon,
  title,
  subtitle,
  details,
  codePath,
  rubricTieIn,
  rubricQuote,
  color,
  wide = false,
}: StageBoxProps) {
  const colors = {
    blue: {
      iconBg: "bg-blue-500/10",
      iconText: "text-blue-700 dark:text-blue-400",
      accent: "border-l-blue-500",
      rubricBg: "bg-blue-500/10 text-blue-700 dark:text-blue-400",
    },
    emerald: {
      iconBg: "bg-emerald-500/10",
      iconText: "text-emerald-700 dark:text-emerald-400",
      accent: "border-l-emerald-500",
      rubricBg: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
    },
    violet: {
      iconBg: "bg-violet-500/10",
      iconText: "text-violet-700 dark:text-violet-400",
      accent: "border-l-violet-500",
      rubricBg: "bg-violet-500/10 text-violet-700 dark:text-violet-400",
    },
  } as const;
  const c = colors[color];

  return (
    <Card
      className={cn(
        "flex flex-col gap-2 border-l-4 p-4",
        c.accent,
        wide && "max-w-3xl",
        "flex-1",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-md",
            c.iconBg,
            c.iconText,
          )}
        >
          {icon}
        </span>
        <div className="flex-1">
          <h4 className="font-semibold leading-tight">{title}</h4>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </div>
      </div>

      <ul className="space-y-1 text-xs text-foreground">
        {details.map((d, i) => (
          <li key={i} className="flex gap-1.5">
            <span className="mt-1 inline-block h-1 w-1 rounded-full bg-muted-foreground/60" />
            <span>{d}</span>
          </li>
        ))}
      </ul>

      <code className="mt-auto block break-all rounded bg-muted/50 px-2 py-1 font-mono text-[10px] text-muted-foreground">
        {codePath}
      </code>

      {rubricTieIn && (
        <div
          className={cn(
            "rounded-md px-2 py-1 text-[10px]",
            c.rubricBg,
          )}
          title={rubricQuote}
        >
          <span className="font-semibold">Rubric · {rubricTieIn}</span>
          {rubricQuote && (
            <span className="block italic opacity-80">{rubricQuote}</span>
          )}
        </div>
      )}
    </Card>
  );
}
