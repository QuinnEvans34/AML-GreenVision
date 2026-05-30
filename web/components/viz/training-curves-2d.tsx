"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import type { EpochMetric } from "@/lib/types";

interface TrainingCurves2DProps {
  epochMetrics: EpochMetric[];
}

// Explicit colors so chart text reads cleanly in both light and dark modes.
const AXIS = "#64748b"; // slate-500
const GRID = "#cbd5e1"; // slate-300, low opacity
const TRAIN = "#3b82f6"; // blue-500
const VAL_ACC = "#10b981"; // emerald-500
const VAL_LOSS = "#f59e0b"; // amber-500
const PHASE1_TINT = "#10b981";

export function TrainingCurves2D({ epochMetrics }: TrainingCurves2DProps) {
  const data = epochMetrics.map((m) => ({
    epoch: m.epoch,
    phase: m.phase,
    train_acc: m.train_acc,
    val_acc: m.val_acc,
    train_loss: m.train_loss,
    val_loss: m.val_loss,
  }));

  const phase1End =
    epochMetrics.filter((m) => m.phase === "phase1").length - 0.5;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Accuracy per epoch</CardTitle>
          <p className="text-xs text-muted-foreground">
            Phase 1 (head warm-up) shaded · Phase 2 unfreezes the backbone
          </p>
        </CardHeader>
        <CardContent className="h-72 pl-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 16, right: 24, bottom: 24, left: 8 }}
            >
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" opacity={0.6} />
              <XAxis
                dataKey="epoch"
                tick={{ fontSize: 12, fill: AXIS }}
                stroke={AXIS}
                label={{
                  value: "Epoch",
                  position: "insideBottom",
                  offset: -8,
                  fontSize: 12,
                  fill: AXIS,
                }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 12, fill: AXIS }}
                stroke={AXIS}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: "white",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  fontSize: 12,
                  color: "#0f172a",
                }}
                labelStyle={{ color: "#0f172a", fontWeight: 500 }}
                formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                labelFormatter={(epoch) => `Epoch ${epoch}`}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <ReferenceArea
                x1={0}
                x2={phase1End}
                fill={PHASE1_TINT}
                fillOpacity={0.08}
                label={{
                  value: "Phase 1",
                  position: "insideTopLeft",
                  fontSize: 11,
                  fill: AXIS,
                  offset: 6,
                }}
              />
              <Line
                type="monotone"
                dataKey="train_acc"
                stroke={TRAIN}
                strokeWidth={2.5}
                dot={false}
                name="Train accuracy"
              />
              <Line
                type="monotone"
                dataKey="val_acc"
                stroke={VAL_ACC}
                strokeWidth={2.5}
                dot={false}
                name="Val accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Loss per epoch</CardTitle>
          <p className="text-xs text-muted-foreground">
            Cross-entropy loss · log-shaped descent characteristic of fine-tuning
          </p>
        </CardHeader>
        <CardContent className="h-72 pl-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 16, right: 24, bottom: 24, left: 8 }}
            >
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" opacity={0.6} />
              <XAxis
                dataKey="epoch"
                tick={{ fontSize: 12, fill: AXIS }}
                stroke={AXIS}
                label={{
                  value: "Epoch",
                  position: "insideBottom",
                  offset: -8,
                  fontSize: 12,
                  fill: AXIS,
                }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: AXIS }}
                stroke={AXIS}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  background: "white",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  fontSize: 12,
                  color: "#0f172a",
                }}
                labelStyle={{ color: "#0f172a", fontWeight: 500 }}
                formatter={(v: number) => v.toFixed(4)}
                labelFormatter={(epoch) => `Epoch ${epoch}`}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <ReferenceArea
                x1={0}
                x2={phase1End}
                fill={PHASE1_TINT}
                fillOpacity={0.08}
              />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke={TRAIN}
                strokeWidth={2.5}
                dot={false}
                name="Train loss"
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke={VAL_LOSS}
                strokeWidth={2.5}
                dot={false}
                name="Val loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
