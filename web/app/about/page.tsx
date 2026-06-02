import { ExternalLink } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default function AboutPage() {
  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">About GreenVision</h1>
        <p className="mt-2 text-muted-foreground">
          End-to-end machine learning system for diagnosing plant leaf diseases
          — built for Applied Machine Learning at Neumont College (W8A1
          design, W9A1 training, W10P1 serving and dashboard).
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Architecture summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-[140px_1fr] gap-y-2 text-sm">
            <span className="text-muted-foreground">Base model</span>
            <span>EfficientNet-B0, pretrained on ImageNet</span>

            <span className="text-muted-foreground">Dataset</span>
            <span>
              PlantVillage — 54,306 leaf images, 39 classes (38 PlantVillage
              disease/healthy combos + 1 background-without-leaves)
            </span>

            <span className="text-muted-foreground">Fine-tuning</span>
            <span>
              Two-phase — head warm-up (3 epochs) + gradual unfreezing
              (21 epochs)
            </span>

            <span className="text-muted-foreground">Optimizer</span>
            <span>
              AdamW with decoupled weight decay, 10× learning rate ratio
              (head : backbone)
            </span>

            <span className="text-muted-foreground">Tracking</span>
            <span>MLflow — nested runs + Model Registry</span>

            <span className="text-muted-foreground">Serving</span>
            <span>
              FastAPI loading from{" "}
              <code className="rounded bg-muted px-1 font-mono text-xs">
                models:/GreenVision/Production
              </code>
            </span>

            <span className="text-muted-foreground">Dashboard</span>
            <span>
              Next.js 16 + Tailwind 4 + shadcn/ui + React Three Fiber
            </span>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            For the full pipeline diagram with each file referenced, see{" "}
            <a href="/analytics" className="underline hover:text-foreground">
              Analytics → Architecture → System pipeline
            </a>.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Stat label="Validation accuracy" value="99.73%" />
          <Stat label="Test accuracy (held-out)" value="99.73%" />
          <Stat label="Random baseline (1/39)" value="2.56%" />
          <Stat
            label="Improvement over baseline"
            value="+97.17 percentage points"
          />
          <Separator className="my-3" />
          <p className="text-xs text-muted-foreground">
            <strong>Important caveat:</strong> PlantVillage is captured under
            uniform studio conditions against neutral backgrounds. Field photos
            (variable lighting, complex backgrounds, partial occlusion) would
            likely drop accuracy significantly. The dashboard guides users to
            photograph single leaves on plain backgrounds to keep them on the
            trained distribution.
          </p>
        </CardContent>
      </Card>

      {/* ── Design decisions section — mapped to the W10P1 rubric ───────── */}

      <div className="pt-2">
        <h2 className="text-2xl font-semibold tracking-tight">
          Design decisions
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Each card below answers a specific design question raised by the W10P1
          assignment. Hover the rubric tag at the bottom for the exact quote.
        </p>
      </div>

      <DecisionSection
        kicker="Part 1 — FastAPI inference endpoint"
        accent="blue"
      >
        <DecisionCard
          question="How do you handle non-image uploads? Corrupted files?"
          answer="The upload body is size-checked first (413 if > 10 MB, 400 if empty), then PIL.Image.open() attempts to decode. UnidentifiedImageError or any other decode failure returns HTTP 400 with a clear 'Could not decode image — please upload a JPG, PNG, or WebP file' message. The dashboard surfaces the error in the result panel without crashing."
          where="api/routes/predict.py"
          rubricQuote="Your design decisions: How do you handle non-image uploads? Corrupted files?"
          accent="blue"
        />
        <DecisionCard
          question="What do you return when confidence is very low (<40%)?"
          answer="The API always returns a normal PredictionResponse with all fields populated, including a 'warnings' array containing a low-confidence message. The dashboard responds by hiding the treatment recommendation behind a 'Show anyway' disclosure and strengthening the AI disclaimer text. Rationale: a low-confidence prediction shouldn't drive a fungicide decision — gating the treatment behind an explicit opt-in protects the farmer."
          where="api/inference.py (warnings) · web/components/prediction-result.tsx (disclosure UI)"
          rubricQuote="Your design decisions: What do you return when confidence is very low (<40%)?"
          accent="blue"
        />
        <DecisionCard
          question="Where does treatment recommendation text come from?"
          answer="A static knowledge base at data/treatments.json with 39 entries — one per class. Each disease entry includes a one-sentence summary, severity tier, time-sensitivity callout, 3–5 numbered action steps, and at least 2 citations to US university extension publications (Cornell PMEP, Penn State Extension, UC IPM, UMN Extension). Loaded once at API startup."
          where="data/treatments.json · api/treatments.py"
          rubricQuote="Your design decisions: Where does treatment recommendation text come from?"
          accent="blue"
        />
      </DecisionSection>

      <DecisionSection
        kicker="Part 2 — Dashboard"
        accent="emerald"
      >
        <DecisionCard
          question='How do you format class names for display? (Tomato___Late_blight → "Tomato — Late blight")'
          answer="Two layers cooperate. The static treatments.json provides a hand-tuned display_name per class (so 'Pepper,_bell' becomes 'Bell pepper'). A frontend helper lib/format-class-name.ts handles raw-string fallback if a class is missing from the KB. The result: 'Tomato___Late_blight' → 'Tomato — Late blight', 'Apple___healthy' → 'Apple (healthy)', 'Background_without_leaves' → 'No leaf detected'."
          where="data/treatments.json (display_name) · web/lib/format-class-name.ts"
          rubricQuote='Your design decisions: How do you format class names for display? (e.g., "Tomato___Late_blight" → "Tomato — Late blight")'
          accent="emerald"
        />
        <DecisionCard
          question="How do you visually communicate low confidence to a farmer?"
          answer="Four-tier confidence band with distinct color, language, and behavior. High (≥85%) is emerald, treatment shown normally. Medium (40–85%) is amber with a 'review alternatives' warning. Low (<40%) is rose, treatment hidden behind a 'Show anyway' disclosure, primary CTA becomes 'Retake photo', AI disclaimer text strengthens. The card tone (accent strip + tinted hero) reinforces the band at a glance."
          where="web/components/confidence-badge.tsx · web/components/prediction-result.tsx (band logic) · web/lib/types.ts (getConfidenceBand)"
          rubricQuote="Your design decisions: How do you visually communicate low confidence to a farmer?"
          accent="emerald"
        />
        <DecisionCard
          question="What context makes a prediction actionable for your stakeholder?"
          answer="Four ingredients: (1) a one-sentence summary in plain language, (2) a severity tier so the farmer knows urgency, (3) a 'When to act' time-sensitivity callout with a window (e.g., 'apply within 7 days'), (4) numbered action steps starting with the highest-leverage one, (5) citations to extension publications the farmer can read directly. The Top-3 alternatives surface even at high confidence so a near-tie can be inspected."
          where="data/treatments.json (severity, time_sensitivity, action_steps, sources) · web/components/prediction-result.tsx"
          rubricQuote="Your design decisions: What context makes a prediction actionable for your stakeholder?"
          accent="emerald"
        />
      </DecisionSection>

      <DecisionSection
        kicker="Part 3 — End-to-end integration"
        accent="violet"
      >
        <DecisionCard
          question="Preprocessing in serve matches validation transforms exactly"
          answer="The API imports the same eval_tfms object that training imports — a single torchvision.transforms.Compose defined once in src/greenvision/data/transforms.py. Re-implementing transforms inside the API would risk silent drift; importing the same Python object makes drift structurally impossible."
          where="src/greenvision/data/transforms.py (definition) · api/inference.py (import)"
          rubricQuote="Integration checklist: Preprocessing in serve.py matches validation transforms exactly"
          accent="violet"
        />
        <DecisionCard
          question="Class names loaded from MLflow artifact in correct order"
          answer="At training time, mlflow.log_artifact('artifacts/checkpoints/class_names.json') attaches the ImageFolder.classes list to the phase2 child run. At serving time, the API reads the same file as a startup load. The order is the contract between model output indices and human labels — and the source of truth is the file in the run, never re-derived."
          where="scripts/train.py (logs the artifact) · api/inference.py:load_class_names()"
          rubricQuote="Integration checklist: Class names in serve.py loaded from MLflow artifact in correct order"
          accent="violet"
        />
        <DecisionCard
          question="Model in eval mode before inference"
          answer="load_production_model() in api/inference.py calls .to(device) and then .eval() before returning. The lifespan stores the eval-mode model on app.state.model. Every /predict call wraps the forward pass in torch.no_grad(). Together these guarantee deterministic, memory-efficient inference."
          where="api/inference.py:load_production_model() + api/inference.py:predict_image()"
          rubricQuote="Integration checklist: Model in eval() mode before inference"
          accent="violet"
        />
        <DecisionCard
          question="FastAPI and dashboard run simultaneously on different ports"
          answer="FastAPI on :8000 (uvicorn) and Next.js on :3000 (next dev). The Next.js dev server uses a next.config rewrite to proxy /api/* → http://localhost:8000/* so the browser sees a same-origin request and CORS never enters the conversation. scripts/demo.sh starts both under `concurrently` with pre-flight port checks."
          where="api/main.py + web/next.config.ts (rewrite) + scripts/demo.sh"
          rubricQuote="Integration checklist: FastAPI and Streamlit can run simultaneously on different ports"
          accent="violet"
        />
      </DecisionSection>

      <Card>
        <CardHeader>
          <CardTitle>Project links</CardTitle>
        </CardHeader>
        <CardContent>
          <a
            href="https://github.com/QuinnEvans34/AML-GreenVision"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm hover:underline"
          >
            github.com/QuinnEvans34/AML-GreenVision
            <ExternalLink className="h-3 w-3" />
          </a>
        </CardContent>
      </Card>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Subcomponents
// ──────────────────────────────────────────────────────────────────

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{value}</span>
    </div>
  );
}

type Accent = "blue" | "emerald" | "violet";

function DecisionSection({
  kicker,
  accent,
  children,
}: {
  kicker: string;
  accent: Accent;
  children: React.ReactNode;
}) {
  const colors = {
    blue: "text-blue-700 dark:text-blue-400",
    emerald: "text-emerald-700 dark:text-emerald-400",
    violet: "text-violet-700 dark:text-violet-400",
  } as const;
  return (
    <div className="space-y-3">
      <p
        className={`text-[10px] font-semibold uppercase tracking-widest ${colors[accent]}`}
      >
        {kicker}
      </p>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">{children}</div>
    </div>
  );
}

function DecisionCard({
  question,
  answer,
  where,
  rubricQuote,
  accent,
}: {
  question: string;
  answer: string;
  where: string;
  rubricQuote: string;
  accent: Accent;
}) {
  const colors = {
    blue: "border-l-blue-500",
    emerald: "border-l-emerald-500",
    violet: "border-l-violet-500",
  } as const;
  return (
    <Card className={`border-l-4 ${colors[accent]}`}>
      <CardContent className="space-y-2 p-4">
        <p className="text-sm font-semibold leading-tight">Q. {question}</p>
        <p className="text-xs text-muted-foreground">{answer}</p>
        <p className="text-[10px] text-muted-foreground">
          <span className="font-medium">Where:</span>{" "}
          <code className="font-mono">{where}</code>
        </p>
        <p
          className="text-[10px] italic text-muted-foreground"
          title={rubricQuote}
        >
          <span className="font-medium not-italic">Rubric:</span> {rubricQuote}
        </p>
      </CardContent>
    </Card>
  );
}
