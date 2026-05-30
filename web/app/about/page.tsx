import { ExternalLink } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// Inline GitHub mark — lucide-react 1.17 removed brand icons.
function GithubMark({ className }: { className?: string }) {
  return (
    <svg
      role="img"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      fill="currentColor"
      className={className}
      aria-label="GitHub"
    >
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56v-2.16c-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.23-1.28-5.23-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.62 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.43-2.69 5.4-5.25 5.68.41.35.78 1.04.78 2.1v3.11c0 .31.21.66.79.55C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z" />
    </svg>
  );
}

export default function AboutPage() {
  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">About GreenVision</h1>
        <p className="mt-2 text-muted-foreground">
          End-to-end machine learning system for diagnosing plant leaf diseases
          — built for Applied Machine Learning at Neumont College.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Architecture</CardTitle>
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
              Two-phase — head warm-up (3 epochs) + gradual unfreezing (21
              epochs)
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

      <Card>
        <CardHeader>
          <CardTitle>Project links</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <a
            href="https://github.com/QuinnEvans34/AML-GreenVision"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm hover:underline"
          >
            <GithubMark className="h-4 w-4" />
            github.com/QuinnEvans34/AML-GreenVision
            <ExternalLink className="h-3 w-3" />
          </a>
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{value}</span>
    </div>
  );
}
