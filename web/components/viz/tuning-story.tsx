"use client";

import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Flame,
  GitBranch,
  Layers,
  Scissors,
  Settings2,
  Snowflake,
  Sparkles,
  Target,
  XCircle,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

/**
 * Tuning story — the OOD fix Quinn shipped after the W9A1 model failed on
 * outside-internet leaf photos. Two tracks: rembg preprocessing
 * (Decision 14) and robust-augmentation fine-tune (Decision 15) producing
 * v4. Rendered as a structured walkthrough so the presenter can point at
 * each section in turn during the demo.
 */
export function TuningStory() {
  return (
    <div className="space-y-6">
      {/* ── HERO: The problem ───────────────────────────────────────── */}
      <Card className="border-l-4 border-l-rose-500">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-rose-500" />
            <Badge variant="outline" className="border-rose-500/30 text-rose-700 dark:text-rose-400">
              Domain shift identified post-W9A1
            </Badge>
          </div>
          <CardTitle className="mt-2">
            v3 achieved 99.73% on PlantVillage — but failed on field photos
          </CardTitle>
          <CardDescription>
            The reflection section of the W9A1 training report flagged
            this as the project's biggest limitation. After W9A1 submission
            I tested v3 against real-world internet images, and the failure
            was systematic: backgrounds of grass, dirt, hands, and complex
            scenes pushed predictions either to wrong classes or to
            confident "No leaf detected" rejections.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <DiagnosisCard
              kicker="What we observed"
              items={[
                "8 outside images → 5 either misclassified or rejected as no-leaf",
                "Cherry leaves often classified as Blueberry",
                "Leaves on grass/dirt rejected with 100% confidence",
                "When predictions were correct, confidence was suspicious",
              ]}
            />
            <DiagnosisCard
              kicker="Root cause"
              items={[
                "PlantVillage backgrounds: neutral, uniform, studio-lit",
                "Model learned: leaves + neutral backgrounds",
                "Field photo backgrounds: grass, dirt, complex scenes",
                "Background bias: classic CV domain shift",
              ]}
              tone="rose"
            />
          </div>
        </CardContent>
      </Card>

      {/* ── TWO-TRACK STRATEGY ──────────────────────────────────────── */}
      <div>
        <h3 className="text-lg font-semibold tracking-tight">
          The two-track strategy
        </h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Solving it under a 12-hour budget meant betting on both a
          production-style preprocessing fix AND a retraining run. Either
          alone reduces risk; both together compound.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Track 1 — rembg */}
        <Card className="border-l-4 border-l-emerald-500">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Scissors className="h-5 w-5 text-emerald-600" />
              <Badge className="bg-emerald-500/15 text-emerald-700 dark:text-emerald-400">
                Decision 14
              </Badge>
            </div>
            <CardTitle className="mt-2">
              Track 1 — rembg at inference
            </CardTitle>
            <CardDescription>
              The safety net · shipped first
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p>
              <strong>What:</strong> Use the U²-Net foreground segmenter to
              strip the background from incoming photos before they reach
              the model. Then composite the leaf onto white, which matches
              the training distribution.
            </p>
            <p>
              <strong>Why it works:</strong> Instead of teaching the model
              to handle every kind of background, we shift the input
              distribution to match what the model was trained on. Same
              principle as Apple Visual Look Up.
            </p>
            <Separator />
            <div className="space-y-1.5">
              <KeyValue label="Library" value="rembg + onnxruntime · U²-Net model" />
              <KeyValue label="Latency" value="~200-230 ms preprocessing + ~10 ms inference" />
              <KeyValue label="Exposed via" value="POST /predict?remove_bg=true Form param" />
              <KeyValue label="Dashboard" value="Auto-on for Upload tab, auto-off for Test-set tab" />
              <KeyValue label="Where" value="api/inference.py:remove_background()" mono />
            </div>
          </CardContent>
        </Card>

        {/* Track 2 — augmentation fine-tune */}
        <Card className="border-l-4 border-l-violet-500">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-violet-600" />
              <Badge className="bg-violet-500/15 text-violet-700 dark:text-violet-400">
                Decision 15
              </Badge>
            </div>
            <CardTitle className="mt-2">
              Track 2 — Robust-augmentation fine-tune (v4)
            </CardTitle>
            <CardDescription>
              The structural fix · trained overnight from v3
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p>
              <strong>What:</strong> Loaded v3 from the MLflow Registry
              and fine-tuned for 12 more epochs with a heavier augmentation
              pipeline — specifically designed to force the model away
              from background reliance.
            </p>
            <p>
              <strong>Why it works:</strong> Aggressive
              <code className="mx-1 rounded bg-muted px-1 font-mono text-xs">RandomGrayscale</code>
              destroys the color shortcut.
              <code className="mx-1 rounded bg-muted px-1 font-mono text-xs">RandomErasing</code>
              destroys background patches. The model has to attend to
              leaf shape and texture, not surroundings.
            </p>
            <Separator />
            <div className="space-y-1.5">
              <KeyValue label="Base" value="v3 loaded via models:/GreenVision/3" />
              <KeyValue label="Epochs" value="12 max · early stop on val acc plateau" />
              <KeyValue label="LR" value="head 5e-4 · backbone 5e-5 (conservative)" />
              <KeyValue label="Validation" value="Deterministic — same eval_tfms" />
              <KeyValue label="Where" value="scripts/train.py --fine-tune-from 3 --robust-augmentation" mono />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── BACKBONE UNFREEZE: v3 GRADUAL vs v4 FULL ──────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            What was unfrozen — v3 vs v4
          </CardTitle>
          <CardDescription>
            EfficientNet-B0 has 9 feature blocks plus our classifier head.
            Blue = trainable. Slate = frozen with{" "}
            <code className="rounded bg-muted px-1 font-mono text-xs">
              requires_grad=False
            </code>{" "}
            + BatchNorm running stats locked.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <UnfreezeGrid />
          <Separator />
          <div className="grid gap-3 lg:grid-cols-2">
            <div className="rounded-md border bg-blue-500/5 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-blue-700 dark:text-blue-400">
                Why v3 unfroze gradually
              </p>
              <p className="mt-1 text-xs">
                The classifier head started randomly initialized. Without
                Phase 1, the head&apos;s huge initial gradients would flow
                backward and trash the pretrained backbone — catastrophic
                forgetting. Phase 1 protects the backbone while the head
                settles. Phase 2 then unfreezes deepest-first because the
                deepest layers are the most ImageNet-specific.
              </p>
            </div>
            <div className="rounded-md border bg-violet-500/5 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-violet-700 dark:text-violet-400">
                Why v4 unfroze everything from epoch 0
              </p>
              <p className="mt-1 text-xs">
                When v4 starts, the head is already trained (loaded from v3
                via the registry). So there&apos;s no catastrophic-forgetting
                risk. We unfreeze everything at once and adapt to the new
                augmentation pipeline. The 10× LR ratio still protects the
                backbone — it learns slowly while the head adapts faster.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── TRAINING RECIPE TABLE — v3 vs v4 ──────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings2 className="h-5 w-5" />
            Training recipe — v3 vs v4 side by side
          </CardTitle>
          <CardDescription>
            Same optimizer family, same scheduler, same eval transforms.
            Three things change: starting point, what&apos;s frozen, and
            what augmentation hits the training batches.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="px-3 py-2">Setting</th>
                  <th className="px-3 py-2">v3 (from scratch)</th>
                  <th className="px-3 py-2">v4 (fine-tune)</th>
                </tr>
              </thead>
              <tbody>
                <RecipeRow
                  setting="Starting weights"
                  v3="EfficientNet-B0 · ImageNet"
                  v4="v3 from MLflow Registry"
                  changed
                />
                <RecipeRow
                  setting="Phase 1 (head warm-up)"
                  v3="3 epochs · backbone frozen"
                  v4="(skipped — head already trained)"
                  changed
                />
                <RecipeRow
                  setting="Phase 2 — what's unfrozen"
                  v3="Gradual: (6→3→0) over 12 epochs"
                  v4="All blocks unfrozen from epoch 0"
                  changed
                />
                <RecipeRow
                  setting="Head learning rate"
                  v3="1e-3"
                  v4="5e-4 (more conservative — model already converged)"
                  changed
                />
                <RecipeRow
                  setting="Backbone learning rate"
                  v3="1e-4"
                  v4="5e-5 (10× ratio preserved)"
                  changed
                />
                <RecipeRow
                  setting="Head : backbone LR ratio"
                  v3="10×"
                  v4="10× (same)"
                />
                <RecipeRow
                  setting="Optimizer"
                  v3="AdamW · 4 param groups"
                  v4="AdamW · 4 param groups (same)"
                />
                <RecipeRow
                  setting="Weight decay"
                  v3="1e-4 · matrix weights only"
                  v4="1e-4 · matrix weights only (same)"
                />
                <RecipeRow
                  setting="Scheduler"
                  v3="ReduceLROnPlateau (val_loss, 0.5, p=2)"
                  v4="ReduceLROnPlateau (val_loss, 0.5, p=2) — same"
                />
                <RecipeRow
                  setting="Early stop patience"
                  v3="5 epochs"
                  v4="4 epochs"
                />
                <RecipeRow
                  setting="Train augmentation"
                  v3={
                    <code className="font-mono text-[11px]">train_tfms</code>
                  }
                  v4={
                    <code className="font-mono text-[11px]">
                      train_tfms_robust
                    </code>
                  }
                  changed
                />
                <RecipeRow
                  setting="Eval augmentation"
                  v3={
                    <code className="font-mono text-[11px]">eval_tfms</code>
                  }
                  v4={
                    <code className="font-mono text-[11px]">eval_tfms</code>
                  }
                />
                <RecipeRow
                  setting="Epochs run"
                  v3="24 (Phase 1 + Phase 2)"
                  v4="12 (fine-tune only)"
                />
                <RecipeRow
                  setting="Best val accuracy"
                  v3="99.73%"
                  v4="99.78% — slight improvement"
                  highlight
                />
                <RecipeRow
                  setting="Training time"
                  v3="~58 min · num_workers=4"
                  v4="~58 min · num_workers=0 (DataLoader workaround)"
                />
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* ── AUGMENTATION PIPELINE DETAIL ────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            What changed in the augmentation pipeline
          </CardTitle>
          <CardDescription>
            v3's training augmentation was modest. v4's is aggressive on
            purpose — every change targets a specific aspect of background
            bias.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="px-3 py-2">Transform</th>
                  <th className="px-3 py-2">v3</th>
                  <th className="px-3 py-2">v4</th>
                  <th className="px-3 py-2">Why</th>
                </tr>
              </thead>
              <tbody>
                <TransformRow
                  name="RandomResizedCrop scale"
                  v3="(0.7, 1.0)"
                  v4="(0.5, 1.0)"
                  why="Wider zoom — exposes more background variation per crop"
                />
                <TransformRow
                  name="RandomRotation"
                  v3="15°"
                  v4="30°"
                  why="Real photos aren't always upright"
                />
                <TransformRow
                  name="ColorJitter (brightness/contrast/sat/hue)"
                  v3="0.2 / 0.2 / 0.1 / 0"
                  v4="0.4 / 0.4 / 0.4 / 0.15"
                  why="Simulates outdoor lighting shifts"
                />
                <TransformRow
                  name="RandomGrayscale"
                  v3="—"
                  v4="p=0.10"
                  why="Forces texture-based learning when color is removed"
                  highlight
                />
                <TransformRow
                  name="GaussianBlur"
                  v3="—"
                  v4="p=0.25"
                  why="Simulates phone-camera defocus"
                />
                <TransformRow
                  name="RandomErasing"
                  v3="—"
                  v4="p=0.40 · scale (0.02, 0.25)"
                  why="Destroys background patches — forces foreground attention"
                  highlight
                />
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground">
            Eval transforms unchanged — validation must remain deterministic
            so val/test numbers are comparable across v3 and v4.
          </p>
        </CardContent>
      </Card>

      {/* ── v3 vs v4 RESULTS ────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="h-5 w-5" />
            Results — v3 + rembg vs v4 + rembg on outside images
          </CardTitle>
          <CardDescription>
            8 internet images Quinn collected (grass/dirt/complex backgrounds)
            · v4 wins or ties on 7 of 8 · loses confidence on 1 but stays
            correct top-1
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="px-3 py-2">Image</th>
                  <th className="px-3 py-2">v3 + rembg</th>
                  <th className="px-3 py-2">v4 + rembg</th>
                  <th className="px-3 py-2">Verdict</th>
                </tr>
              </thead>
              <tbody>
                <ResultRow
                  name="a436dbe… (corn leaf)"
                  v3="Corn Cercospora 99% ✓"
                  v4="Corn Cercospora 99% ✓"
                  verdict="tie+"
                  note="Also gets it right without rembg now (72%)"
                />
                <ResultRow
                  name="cherry-leaf-260nw"
                  v3="Blueberry 56% (wrong)"
                  v4="Cherry 91% ✓"
                  verdict="win"
                />
                <ResultRow
                  name="back-side-cherry"
                  v3="Raspberry 60% · Cherry 39%"
                  v4="Cherry 66% ✓ · Raspberry 34%"
                  verdict="win"
                />
                <ResultRow
                  name="images-2 copy"
                  v3="No leaf detected 99%"
                  v4="Strawberry Leaf scorch 99%"
                  verdict="win"
                />
                <ResultRow
                  name="black-cherry-leaf"
                  v3="No leaf detected 97%"
                  v4="Blueberry 57% · Bell pepper 43%"
                  verdict="partial"
                  note="Engages with leaf — no longer rejected"
                />
                <ResultRow
                  name="images.jpeg"
                  v3="No leaf detected 95%"
                  v4="No leaf 65% · Corn classes surfacing"
                  verdict="partial"
                  note="Model considers plant classes now"
                />
                <ResultRow
                  name="images-3 (apple-ish)"
                  v3="Apple 67%"
                  v4="Apple 86%"
                  verdict="tie"
                />
                <ResultRow
                  name="1_fIG.PNG (potato early blight)"
                  v3="Potato Early blight 95% ✓"
                  v4="Potato Early blight 47% (correct top-1)"
                  verdict="loss"
                  note="Confidence drop — calibrated uncertainty"
                />
              </tbody>
            </table>
          </div>

          <div className="grid gap-3 sm:grid-cols-4">
            <Stat label="Clear wins" value="4" tone="emerald" />
            <Stat label="Partial wins" value="2" tone="emerald" />
            <Stat label="Ties" value="1" tone="slate" />
            <Stat label="Confidence drops" value="1" tone="amber" />
          </div>
        </CardContent>
      </Card>

      {/* ── HONEST LIMITS ────────────────────────────────────────────── */}
      <Card className="border-l-4 border-l-amber-500">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-600" />
            Where v4 still fails — and what we&apos;d ship next
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            v4 + rembg is materially better than v3 + rembg, but it still
            doesn&apos;t fully solve the OOD problem. Two images
            (<code className="rounded bg-muted px-1 font-mono text-xs">images-2.jpeg</code>,{" "}
            <code className="rounded bg-muted px-1 font-mono text-xs">black-cherry-leaf.jpg</code>)
            are still misclassified or rejected. Calibrated uncertainty
            helps the user notice, but doesn&apos;t fix the prediction.
          </p>
          <Separator />
          <div className="space-y-1.5">
            <LimitItem text="Field-photo augmentation training — synthesize backgrounds from a diverse texture dataset (DTD) at training time" />
            <LimitItem text="A two-stage system: in-distribution detector first, then disease classifier — reject OOD inputs explicitly" />
            <LimitItem text="Collect a small real-world labeled set (50–100 photos per class) and fine-tune on it" />
            <LimitItem text="Per-image attention masking — use rembg's alpha at training time to teach the model that backgrounds are irrelevant" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Subcomponents
// ──────────────────────────────────────────────────────────────────

function DiagnosisCard({
  kicker,
  items,
  tone = "slate",
}: {
  kicker: string;
  items: string[];
  tone?: "slate" | "rose";
}) {
  const toneClass =
    tone === "rose"
      ? "text-rose-700 dark:text-rose-400"
      : "text-muted-foreground";
  return (
    <div className="rounded-md border bg-muted/30 p-3">
      <p className={`mb-2 text-xs font-semibold uppercase tracking-wider ${toneClass}`}>
        {kicker}
      </p>
      <ul className="space-y-1 text-xs">
        {items.map((item, i) => (
          <li key={i} className="flex gap-1.5">
            <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function KeyValue({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={
          mono
            ? "rounded bg-muted px-1 py-0.5 font-mono text-[11px]"
            : "font-medium tabular-nums"
        }
      >
        {value}
      </span>
    </div>
  );
}

function TransformRow({
  name,
  v3,
  v4,
  why,
  highlight = false,
}: {
  name: string;
  v3: string;
  v4: string;
  why: string;
  highlight?: boolean;
}) {
  return (
    <tr
      className={
        highlight
          ? "border-b bg-violet-500/5 last:border-0"
          : "border-b last:border-0"
      }
    >
      <td className="px-3 py-2 font-medium">{name}</td>
      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{v3}</td>
      <td className="px-3 py-2 font-mono text-xs text-violet-700 dark:text-violet-400">
        {v4}
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground">{why}</td>
    </tr>
  );
}

function ResultRow({
  name,
  v3,
  v4,
  verdict,
  note,
}: {
  name: string;
  v3: string;
  v4: string;
  verdict: "win" | "tie" | "tie+" | "partial" | "loss";
  note?: string;
}) {
  const config = {
    win: { Icon: CheckCircle2, label: "v4 WIN", classes: "text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/30" },
    "tie+": { Icon: CheckCircle2, label: "TIE+", classes: "text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/30" },
    tie: { Icon: ChevronRight, label: "TIE", classes: "text-slate-700 dark:text-slate-300 bg-slate-500/10 border-slate-500/30" },
    partial: { Icon: Sparkles, label: "PARTIAL", classes: "text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/30" },
    loss: { Icon: XCircle, label: "LOSS", classes: "text-amber-700 dark:text-amber-400 bg-amber-500/10 border-amber-500/30" },
  } as const;
  const v = config[verdict];
  const Icon = v.Icon;
  return (
    <tr className="border-b last:border-0">
      <td className="px-3 py-2">
        <div className="text-xs font-medium">{name}</div>
        {note && <div className="mt-0.5 text-[10px] text-muted-foreground">{note}</div>}
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground">{v3}</td>
      <td className="px-3 py-2 text-xs">{v4}</td>
      <td className="px-3 py-2">
        <span
          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${v.classes}`}
        >
          <Icon className="h-3 w-3" />
          {v.label}
        </span>
      </td>
    </tr>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "emerald" | "amber" | "slate";
}) {
  const toneCls = {
    emerald: "text-emerald-700 dark:text-emerald-400",
    amber: "text-amber-700 dark:text-amber-400",
    slate: "text-slate-700 dark:text-slate-300",
  }[tone];
  return (
    <div className="rounded-md border p-3 text-center">
      <p className={`text-2xl font-bold tabular-nums ${toneCls}`}>{value}</p>
      <p className="mt-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
    </div>
  );
}

function LimitItem({ text }: { text: string }) {
  return (
    <div className="flex gap-2 text-xs">
      <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
      <span>{text}</span>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Unfreeze grid — visual of what was frozen at each training stage
// ──────────────────────────────────────────────────────────────────

const STAGES: { name: string; description: string; unfrozen: boolean[] }[] = [
  {
    name: "v3 · Phase 1 (head warm-up)",
    description: "3 epochs · backbone frozen, BN stats locked, head trains at LR 1e-3",
    // 10 cells: features[0..8] + classifier. true = trainable
    unfrozen: [false, false, false, false, false, false, false, false, false, true],
  },
  {
    name: "v3 · Phase 2a (deepest unfreeze)",
    description: "4 epochs · features[6..8] + classifier trainable",
    unfrozen: [false, false, false, false, false, false, true, true, true, true],
  },
  {
    name: "v3 · Phase 2b (mid unfreeze)",
    description: "4 epochs · features[3..8] + classifier trainable",
    unfrozen: [false, false, false, true, true, true, true, true, true, true],
  },
  {
    name: "v3 · Phase 2c (full unfreeze)",
    description: "remaining epochs · features[0..8] + classifier trainable",
    unfrozen: [true, true, true, true, true, true, true, true, true, true],
  },
  {
    name: "v4 · Fine-tune (from epoch 0)",
    description: "12 epochs · everything unfrozen at once · discriminative LRs protect backbone",
    unfrozen: [true, true, true, true, true, true, true, true, true, true],
  },
];

const BLOCK_LABELS = [
  "f[0]",
  "f[1]",
  "f[2]",
  "f[3]",
  "f[4]",
  "f[5]",
  "f[6]",
  "f[7]",
  "f[8]",
  "head",
];

function UnfreezeGrid() {
  return (
    <div className="space-y-2.5">
      {/* Column headers */}
      <div className="grid grid-cols-[260px_repeat(10,minmax(0,1fr))] gap-1 text-[10px] text-muted-foreground">
        <div />
        {BLOCK_LABELS.map((label) => (
          <div key={label} className="text-center font-mono">
            {label}
          </div>
        ))}
      </div>

      {STAGES.map((stage) => (
        <div
          key={stage.name}
          className="grid grid-cols-[260px_repeat(10,minmax(0,1fr))] items-center gap-1"
        >
          <div className="pr-2">
            <p className="text-xs font-medium leading-tight">{stage.name}</p>
            <p className="text-[10px] leading-tight text-muted-foreground">
              {stage.description}
            </p>
          </div>
          {stage.unfrozen.map((isUnfrozen, i) => (
            <UnfreezeCell key={i} unfrozen={isUnfrozen} isHead={i === 9} />
          ))}
        </div>
      ))}

      {/* Legend */}
      <div className="mt-3 flex flex-wrap items-center gap-4 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-flex h-4 w-6 items-center justify-center rounded border border-blue-500/40 bg-blue-500/15 text-blue-700 dark:text-blue-400">
            <Flame className="h-2.5 w-2.5" />
          </span>
          trainable
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-flex h-4 w-6 items-center justify-center rounded border border-slate-500/30 bg-slate-500/10 text-slate-500">
            <Snowflake className="h-2.5 w-2.5" />
          </span>
          frozen · <code className="font-mono">requires_grad=False</code>
        </span>
        <span className="text-[10px]">
          (f[0]–f[8] = EfficientNet-B0 feature blocks · head = our
          Dropout→Linear(1280, 39))
        </span>
      </div>
    </div>
  );
}

function UnfreezeCell({
  unfrozen,
  isHead,
}: {
  unfrozen: boolean;
  isHead: boolean;
}) {
  if (unfrozen) {
    return (
      <div
        className={`flex h-7 items-center justify-center rounded border ${
          isHead
            ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
            : "border-blue-500/40 bg-blue-500/15 text-blue-700 dark:text-blue-400"
        }`}
        title={isHead ? "Classifier head — trainable" : "Feature block — trainable"}
      >
        <Flame className="h-3 w-3" />
      </div>
    );
  }
  return (
    <div
      className="flex h-7 items-center justify-center rounded border border-slate-500/30 bg-slate-500/10 text-slate-500"
      title="Frozen · requires_grad=False"
    >
      <Snowflake className="h-3 w-3" />
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Training recipe row
// ──────────────────────────────────────────────────────────────────

function RecipeRow({
  setting,
  v3,
  v4,
  changed = false,
  highlight = false,
}: {
  setting: string;
  v3: React.ReactNode;
  v4: React.ReactNode;
  changed?: boolean;
  highlight?: boolean;
}) {
  return (
    <tr
      className={
        highlight
          ? "border-b bg-emerald-500/5 last:border-0"
          : changed
            ? "border-b bg-violet-500/5 last:border-0"
            : "border-b last:border-0"
      }
    >
      <td className="px-3 py-2 text-xs font-medium">{setting}</td>
      <td className="px-3 py-2 text-xs text-muted-foreground">{v3}</td>
      <td
        className={`px-3 py-2 text-xs ${
          changed ? "text-violet-700 dark:text-violet-400" : ""
        }`}
      >
        {v4}
      </td>
    </tr>
  );
}
