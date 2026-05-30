"use client";

import { AlertCircle, Loader2 } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { Architecture3D } from "@/components/viz/architecture-3d";
import { ConfusionMatrix3D } from "@/components/viz/confusion-matrix-3d";
import { KpiCards } from "@/components/viz/kpi-cards";
import { OptimizerLandscape3D } from "@/components/viz/optimizer-landscape-3d";
import { PerClassBars3D } from "@/components/viz/per-class-bars-3d";
import { PerClassTable } from "@/components/viz/per-class-table";
import { TrainingCurves2D } from "@/components/viz/training-curves-2d";

import { useTrainingData } from "@/lib/use-training-data";

export default function AnalyticsPage() {
  const { data, error, loading } = useTrainingData();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Training analytics
        </h1>
        <p className="mt-2 text-muted-foreground">
          Inside the model — convergence, per-class accuracy, confusion patterns,
          and a walk-through of inference. All data is rebuilt from MLflow at
          export time; this page never queries the registry live.
        </p>
      </div>

      {loading && (
        <div className="flex h-64 items-center justify-center rounded-lg border bg-card">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading training data…
          </div>
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Could not load training data</AlertTitle>
          <AlertDescription>
            {error}
            <code className="mt-2 block rounded bg-muted p-2 font-mono text-xs">
              PYTHONPATH=src .venv/bin/python scripts/export_mlflow_for_dashboard.py
            </code>
          </AlertDescription>
        </Alert>
      )}

      {data && (
        <>
          <KpiCards data={data} />

          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-2 lg:grid-cols-5">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="per-class">Per-class</TabsTrigger>
              <TabsTrigger value="confusion">Confusion</TabsTrigger>
              <TabsTrigger value="optimizer">Optimizer</TabsTrigger>
              <TabsTrigger value="architecture">Architecture</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <TrainingCurves2D epochMetrics={data.epoch_metrics} />
            </TabsContent>

            <TabsContent value="per-class">
              <PerClassBars3D metrics={data.per_class_metrics} />
            </TabsContent>

            <TabsContent value="confusion">
              <ConfusionMatrix3D
                matrix={data.confusion_matrix}
                classNames={data.class_names}
                perClassMetrics={data.per_class_metrics}
              />
            </TabsContent>

            <TabsContent value="optimizer">
              <OptimizerLandscape3D />
            </TabsContent>

            <TabsContent value="architecture">
              <Architecture3D />
            </TabsContent>
          </Tabs>

          <PerClassTable metrics={data.per_class_metrics} />
        </>
      )}
    </div>
  );
}
