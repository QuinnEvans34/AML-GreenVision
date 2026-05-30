"use client";

import * as React from "react";
import { Line, Text } from "@react-three/drei";

import { SceneCanvas } from "@/components/viz/scene-canvas";
import type { EpochMetric } from "@/lib/types";

interface TrainingCurves3DProps {
  epochMetrics: EpochMetric[];
}

const X_SCALE = 0.4;
const Y_SCALE = 5;
const Z_TRAIN = 1.2;
const Z_VAL = -1.2;

function metricToPoints(
  metrics: EpochMetric[],
  key: "train_acc" | "val_acc" | "train_loss" | "val_loss",
  z: number,
  baseY: number = 0,
  yScale: number = Y_SCALE,
): [number, number, number][] {
  return metrics
    .map((m) => {
      const v = m[key];
      if (typeof v !== "number") return null;
      return [m.epoch * X_SCALE, baseY + v * yScale, z] as [number, number, number];
    })
    .filter((p): p is [number, number, number] => p !== null);
}

export function TrainingCurves3D({ epochMetrics }: TrainingCurves3DProps) {
  const phase1End =
    epochMetrics.filter((m) => m.phase === "phase1").length - 0.5;

  // Find loss max for scaling
  const maxLoss = Math.max(
    1,
    ...epochMetrics
      .flatMap((m) => [m.train_loss, m.val_loss])
      .filter((v): v is number => typeof v === "number"),
  );

  const trainAcc = metricToPoints(epochMetrics, "train_acc", Z_TRAIN, 0);
  const valAcc = metricToPoints(epochMetrics, "val_acc", Z_VAL, 0);
  const trainLoss = metricToPoints(
    epochMetrics,
    "train_loss",
    Z_TRAIN,
    -6,
    -3 / maxLoss,
  );
  const valLoss = metricToPoints(
    epochMetrics,
    "val_loss",
    Z_VAL,
    -6,
    -3 / maxLoss,
  );

  const totalEpochs = epochMetrics.length;
  const xMax = (totalEpochs - 1) * X_SCALE;

  return (
    <SceneCanvas
      cameraPosition={[10, 5, 12]}
      caption="X = epoch · Y = metric · Z = train (front) vs val (back) · green plane = Phase 1→2 boundary"
    >
      {/* Axis lines */}
      <Line
        points={[
          [-1, 0, 0],
          [xMax + 1, 0, 0],
        ]}
        color="#888"
        lineWidth={1}
      />
      <Line
        points={[
          [0, 0, 0],
          [0, Y_SCALE + 0.5, 0],
        ]}
        color="#888"
        lineWidth={1}
      />
      <Line
        points={[
          [0, 0, Z_TRAIN + 0.5],
          [0, 0, Z_VAL - 0.5],
        ]}
        color="#888"
        lineWidth={1}
      />

      {/* Phase 1 → Phase 2 boundary plane */}
      <mesh position={[phase1End * X_SCALE, Y_SCALE / 2, 0]}>
        <planeGeometry args={[0.05, Y_SCALE + 1, 3.5]} />
        <meshBasicMaterial color="#10b981" transparent opacity={0.18} />
      </mesh>
      <Text
        position={[phase1End * X_SCALE, Y_SCALE + 0.8, 0]}
        fontSize={0.32}
        color="#10b981"
        anchorX="center"
        anchorY="middle"
      >
        Phase 1 → 2
      </Text>

      {/* Accuracy curves (upper) */}
      <Line points={trainAcc} color="#3b82f6" lineWidth={3} />
      <Line points={valAcc} color="#10b981" lineWidth={3} />

      {/* Loss curves (lower, mirrored downward) */}
      <Line points={trainLoss} color="#60a5fa" lineWidth={2} />
      <Line points={valLoss} color="#f59e0b" lineWidth={2} />

      {/* Axis labels */}
      <Text
        position={[xMax / 2, -1, 0]}
        fontSize={0.4}
        color="#888"
        anchorX="center"
        anchorY="middle"
      >
        epoch
      </Text>
      <Text
        position={[-1.2, Y_SCALE / 2 + 0.5, 0]}
        fontSize={0.32}
        color="#888"
        anchorX="right"
        anchorY="middle"
        rotation={[0, 0, Math.PI / 2]}
      >
        accuracy
      </Text>
      <Text
        position={[-1.2, -4, 0]}
        fontSize={0.32}
        color="#888"
        anchorX="right"
        anchorY="middle"
        rotation={[0, 0, Math.PI / 2]}
      >
        loss
      </Text>

      {/* Train/Val depth labels */}
      <Text
        position={[xMax + 1, 0.5, Z_TRAIN]}
        fontSize={0.3}
        color="#3b82f6"
        anchorX="left"
        anchorY="middle"
      >
        train
      </Text>
      <Text
        position={[xMax + 1, 0.5, Z_VAL]}
        fontSize={0.3}
        color="#10b981"
        anchorX="left"
        anchorY="middle"
      >
        val
      </Text>

      {/* Y-axis ticks for accuracy */}
      {[0.25, 0.5, 0.75, 1.0].map((tick) => (
        <Text
          key={tick}
          position={[-0.4, tick * Y_SCALE, 0]}
          fontSize={0.22}
          color="#888"
          anchorX="right"
          anchorY="middle"
        >
          {(tick * 100).toFixed(0)}%
        </Text>
      ))}
    </SceneCanvas>
  );
}
