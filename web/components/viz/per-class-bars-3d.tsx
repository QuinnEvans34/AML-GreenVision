"use client";

import * as React from "react";
import { Html, Text } from "@react-three/drei";

import { SceneCanvas } from "@/components/viz/scene-canvas";
import type { PerClassMetric } from "@/lib/types";

interface PerClassBars3DProps {
  metrics: PerClassMetric[];
}

const GRID_COLS = 7;
const SPACING = 1.35;
const MAX_HEIGHT = 5.5;

// Highlight any class with F1 below this — these are the "0.27% wrong" rows
const WEAK_F1_THRESHOLD = 0.99;

function f1ToColor(f1: number): string {
  // Sharp encoding: anything below 0.99 swings to rose so weak classes pop.
  if (f1 < 0.9) return "hsl(350, 70%, 50%)";
  if (f1 < 0.99) {
    // gradient amber→emerald between 0.9 and 0.99
    const t = (f1 - 0.9) / 0.09;
    const hue = 35 + (158 - 35) * t;
    return `hsl(${hue}, 65%, 50%)`;
  }
  return "hsl(158, 65%, 48%)";
}

interface BarProps {
  metric: PerClassMetric;
  x: number;
  z: number;
  isHovered: boolean;
  onHover: (index: number | null) => void;
}

function Bar({ metric, x, z, isHovered, onHover }: BarProps) {
  const height = Math.max(metric.recall * MAX_HEIGHT, 0.05);
  const color = f1ToColor(metric.f1);
  const isWeak = metric.f1 < WEAK_F1_THRESHOLD;

  return (
    <mesh
      position={[x, height / 2, z]}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(metric.index);
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        onHover(null);
        document.body.style.cursor = "auto";
      }}
      castShadow
      receiveShadow
    >
      <boxGeometry args={[0.9, height, 0.9]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isHovered ? 0.55 : isWeak ? 0.3 : 0}
        roughness={0.6}
      />
    </mesh>
  );
}

export function PerClassBars3D({ metrics }: PerClassBars3DProps) {
  const [hovered, setHovered] = React.useState<number | null>(null);

  // Sort by F1 ascending — weakest classes go to position 0 (front-left of grid)
  // so they're the most visually prominent.
  const ordered = React.useMemo(
    () => [...metrics].sort((a, b) => a.f1 - b.f1),
    [metrics],
  );

  const bottom5 = ordered.slice(0, 5);

  const hoveredMetric =
    hovered !== null ? ordered.find((m, i) => i === hovered) : null;
  const hoveredGridIdx =
    hovered !== null ? ordered.findIndex((m, i) => i === hovered) : -1;

  const gridXOffset = ((GRID_COLS - 1) / 2) * SPACING;
  const numRows = Math.ceil(ordered.length / GRID_COLS);
  const gridZOffset = ((numRows - 1) / 2) * SPACING;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <SceneCanvas
          cameraPosition={[9, 8, 9]}
          caption="Bars sorted by F1 — weakest classes front-left · rose tint highlights F1 < 0.99"
          height={520}
        >
          <mesh
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, -0.02, 0]}
            receiveShadow
          >
            <planeGeometry args={[24, 24]} />
            <shadowMaterial opacity={0.18} />
          </mesh>

          {[0.25, 0.5, 0.75, 1.0].map((tick) => (
            <Text
              key={tick}
              position={[-gridXOffset - 1.2, tick * MAX_HEIGHT, -gridZOffset]}
              fontSize={0.28}
              color="#94a3b8"
              anchorX="right"
              anchorY="middle"
            >
              {`${(tick * 100).toFixed(0)}%`}
            </Text>
          ))}

          {ordered.map((m, i) => {
            const col = i % GRID_COLS;
            const row = Math.floor(i / GRID_COLS);
            const x = col * SPACING - gridXOffset;
            const z = row * SPACING - gridZOffset;
            return (
              <Bar
                key={m.class_name}
                metric={m}
                x={x}
                z={z}
                isHovered={hovered === i}
                onHover={setHovered}
              />
            );
          })}

          {hoveredMetric && hoveredGridIdx >= 0 && (
            <Html
              position={[
                (hoveredGridIdx % GRID_COLS) * SPACING - gridXOffset,
                Math.max(hoveredMetric.recall * MAX_HEIGHT, 0.05) + 0.6,
                Math.floor(hoveredGridIdx / GRID_COLS) * SPACING - gridZOffset,
              ]}
              center
              distanceFactor={9}
            >
              <div className="pointer-events-none whitespace-nowrap rounded-md border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-lg">
                <div className="font-medium">{hoveredMetric.display_name}</div>
                <div className="tabular-nums text-muted-foreground">
                  F1 {(hoveredMetric.f1 * 100).toFixed(1)}% · recall{" "}
                  {(hoveredMetric.recall * 100).toFixed(1)}% · precision{" "}
                  {(hoveredMetric.precision * 100).toFixed(1)}% · n=
                  {hoveredMetric.support}
                </div>
              </div>
            </Html>
          )}
        </SceneCanvas>

        <div className="rounded-lg border bg-card p-4">
          <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
            Bottom {bottom5.length} by F1
          </p>
          <p className="mb-3 text-xs text-muted-foreground">
            Where the model's 0.27% errors live. Everything above 99% F1 is
            functionally perfect on the held-out test set.
          </p>
          {bottom5.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              All classes at 100% F1.
            </p>
          ) : (
            <ol className="space-y-2 text-xs">
              {bottom5.map((m, i) => (
                <li key={m.class_name} className="flex items-start gap-2">
                  <span className="mt-0.5 tabular-nums text-muted-foreground">
                    {i + 1}.
                  </span>
                  <div className="flex-1">
                    <div className="font-medium">{m.display_name}</div>
                    <div className="tabular-nums text-muted-foreground">
                      F1 {(m.f1 * 100).toFixed(1)}% · recall {(m.recall * 100).toFixed(1)}% · n={m.support}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
    </div>
  );
}
