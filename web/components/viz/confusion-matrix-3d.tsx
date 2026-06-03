"use client";

import * as React from "react";
import * as THREE from "three";

import { SceneCanvas } from "@/components/viz/scene-canvas";
import type { PerClassMetric } from "@/lib/types";

interface ConfusionMatrix3DProps {
  matrix: number[][];
  classNames: string[];
  perClassMetrics: PerClassMetric[];
}

const SPACING = 0.45;
const MAX_HEIGHT = 6;

function ConfusionGrid({ matrix }: { matrix: number[][] }) {
  const meshRef = React.useRef<THREE.InstancedMesh>(null);
  const n = matrix.length;
  const total = n * n;
  const offset = -((n - 1) / 2) * SPACING;

  React.useEffect(() => {
    if (!meshRef.current) return;

    let logMax = 0;
    for (const row of matrix) {
      for (const v of row) {
        if (v > 0) logMax = Math.max(logMax, Math.log1p(v));
      }
    }
    if (logMax === 0) logMax = 1;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    let i = 0;
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const count = matrix[row][col];
        const height = (Math.log1p(count) / logMax) * MAX_HEIGHT;
        const visible = count > 0;
        const drawHeight = visible ? Math.max(height, 0.02) : 0.001;

        dummy.position.set(
          offset + col * SPACING,
          drawHeight / 2,
          offset + row * SPACING,
        );
        dummy.scale.set(SPACING * 0.88, drawHeight, SPACING * 0.88);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);

        if (row === col) {
          // Correct predictions: emerald, brighter for tall bars
          color.setHSL(0.42, 0.55, 0.45 + (height / MAX_HEIGHT) * 0.15);
        } else if (count > 0) {
          // Confusions: rose, more saturated for bigger errors
          const intensity = Math.min(count / 5, 1);
          color.setHSL(0.97, 0.5 + intensity * 0.3, 0.45);
        } else {
          // Zero cells: very dim, just to suggest the floor
          color.setHSL(0, 0, 0.18);
        }
        meshRef.current.setColorAt(i, color);
        i++;
      }
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [matrix]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, total]}
      castShadow
      receiveShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.6} />
    </instancedMesh>
  );
}

function findTopConfusions(
  matrix: number[][],
  classNames: string[],
  perClassMetrics: PerClassMetric[],
  k: number = 5,
): { actual: string; predicted: string; count: number }[] {
  const lookup = new Map(perClassMetrics.map((m) => [m.class_name, m.display_name]));
  const pairs: { actual: string; predicted: string; count: number }[] = [];
  for (let row = 0; row < matrix.length; row++) {
    for (let col = 0; col < matrix[row].length; col++) {
      if (row !== col && matrix[row][col] > 0) {
        pairs.push({
          actual: lookup.get(classNames[row]) ?? classNames[row],
          predicted: lookup.get(classNames[col]) ?? classNames[col],
          count: matrix[row][col],
        });
      }
    }
  }
  pairs.sort((a, b) => b.count - a.count);
  return pairs.slice(0, k);
}

export function ConfusionMatrix3D({
  matrix,
  classNames,
  perClassMetrics,
}: ConfusionMatrix3DProps) {
  const topConfusions = React.useMemo(
    () => findTopConfusions(matrix, classNames, perClassMetrics, 5),
    [matrix, classNames, perClassMetrics],
  );

  const correctCount = matrix.reduce((sum, row, i) => sum + (row[i] ?? 0), 0);
  const totalCount = matrix.reduce(
    (sum, row) => sum + row.reduce((s, v) => s + v, 0),
    0,
  );
  const wrongCount = totalCount - correctCount;

  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <SceneCanvas
        cameraPosition={[14, 14, 14]}
        caption="Diagonal towers = correct · off-diagonal stubs = confusions"
      >
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, -0.02, 0]}
          receiveShadow
        >
          <planeGeometry args={[24, 24]} />
          <shadowMaterial opacity={0.2} />
        </mesh>
        <ConfusionGrid matrix={matrix} />
      </SceneCanvas>

      <div className="space-y-3">
        <div className="rounded-lg border bg-card p-4 space-y-1">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Test-set totals
          </p>
          <p className="text-2xl font-semibold tabular-nums">
            {correctCount.toLocaleString()} / {totalCount.toLocaleString()}{" "}
            <span className="text-sm font-normal text-muted-foreground">correct</span>
          </p>
          <p className="text-xs text-muted-foreground">
            {wrongCount} misclassifications · {((wrongCount / totalCount) * 100).toFixed(2)}% error rate
          </p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
            Top {topConfusions.length} confusions
          </p>
          {topConfusions.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Zero off-diagonal entries — perfect classification.
            </p>
          ) : (
            <ol className="space-y-2 text-xs">
              {topConfusions.map((c, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="mt-0.5 text-muted-foreground tabular-nums">
                    {i + 1}.
                  </span>
                  <div className="flex-1">
                    <div>
                      <span className="text-muted-foreground">Actual:</span>{" "}
                      {c.actual}
                    </div>
                    <div>
                      <span className="text-muted-foreground">Predicted:</span>{" "}
                      {c.predicted}
                    </div>
                  </div>
                  <span className="rounded-full bg-rose-500/15 px-2 py-0.5 text-[10px] font-medium text-rose-700 dark:text-rose-300 tabular-nums">
                    {c.count}×
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
    </div>
  );
}
