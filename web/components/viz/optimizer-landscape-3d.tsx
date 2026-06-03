"use client";

import * as React from "react";
import { useFrame } from "@react-three/fiber";
import { Line, Text } from "@react-three/drei";
import * as THREE from "three";
import { Play, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SceneCanvas } from "@/components/viz/scene-canvas";

// ──────────────────────────────────────────────────────────────────
// Loss function and optimizers — actual update equations, not mocks
// ──────────────────────────────────────────────────────────────────

const MIN_X = 1.5;
const MIN_Z = -1;
const START_X = -2.2;
const START_Z = 2.0;
const STEPS = 90;

function loss(x: number, z: number): number {
  const a = x - MIN_X;
  const b = z - MIN_Z;
  return 0.55 * a * a + 1.7 * b * b + 0.45 * Math.sin(2 * x) * Math.cos(2 * z);
}

function grad(x: number, z: number): [number, number] {
  const dx = 1.1 * (x - MIN_X) + 0.9 * Math.cos(2 * x) * Math.cos(2 * z);
  const dz = 3.4 * (z - MIN_Z) - 0.9 * Math.sin(2 * x) * Math.sin(2 * z);
  return [dx, dz];
}

type Trajectory = [number, number, number][]; // (x, y=loss, z)

function runSGD(lr = 0.045): Trajectory {
  let x = START_X;
  let z = START_Z;
  const path: Trajectory = [[x, loss(x, z), z]];
  for (let i = 0; i < STEPS; i++) {
    const [dx, dz] = grad(x, z);
    x -= lr * dx;
    z -= lr * dz;
    path.push([x, loss(x, z), z]);
  }
  return path;
}

function runAdam(lr = 0.12): Trajectory {
  let x = START_X;
  let z = START_Z;
  let mx = 0,
    mz = 0,
    vx = 0,
    vz = 0;
  const b1 = 0.9,
    b2 = 0.999,
    eps = 1e-8;
  const path: Trajectory = [[x, loss(x, z), z]];
  for (let i = 1; i <= STEPS; i++) {
    const [dx, dz] = grad(x, z);
    mx = b1 * mx + (1 - b1) * dx;
    mz = b1 * mz + (1 - b1) * dz;
    vx = b2 * vx + (1 - b2) * dx * dx;
    vz = b2 * vz + (1 - b2) * dz * dz;
    const mxh = mx / (1 - Math.pow(b1, i));
    const mzh = mz / (1 - Math.pow(b1, i));
    const vxh = vx / (1 - Math.pow(b2, i));
    const vzh = vz / (1 - Math.pow(b2, i));
    x -= (lr * mxh) / (Math.sqrt(vxh) + eps);
    z -= (lr * mzh) / (Math.sqrt(vzh) + eps);
    path.push([x, loss(x, z), z]);
  }
  return path;
}

function runAdamW(lr = 0.12, wd = 0.04): Trajectory {
  let x = START_X;
  let z = START_Z;
  let mx = 0,
    mz = 0,
    vx = 0,
    vz = 0;
  const b1 = 0.9,
    b2 = 0.999,
    eps = 1e-8;
  const path: Trajectory = [[x, loss(x, z), z]];
  for (let i = 1; i <= STEPS; i++) {
    // Decoupled weight decay — applied DIRECTLY to parameters,
    // not folded into the gradient. THIS is the AdamW innovation.
    x *= 1 - lr * wd;
    z *= 1 - lr * wd;
    const [dx, dz] = grad(x, z);
    mx = b1 * mx + (1 - b1) * dx;
    mz = b1 * mz + (1 - b1) * dz;
    vx = b2 * vx + (1 - b2) * dx * dx;
    vz = b2 * vz + (1 - b2) * dz * dz;
    const mxh = mx / (1 - Math.pow(b1, i));
    const mzh = mz / (1 - Math.pow(b1, i));
    const vxh = vx / (1 - Math.pow(b2, i));
    const vzh = vz / (1 - Math.pow(b2, i));
    x -= (lr * mxh) / (Math.sqrt(vxh) + eps);
    z -= (lr * mzh) / (Math.sqrt(vzh) + eps);
    path.push([x, loss(x, z), z]);
  }
  return path;
}

// ──────────────────────────────────────────────────────────────────
// Loss surface mesh
// ──────────────────────────────────────────────────────────────────

const RES = 70;
const XMIN = -3;
const XMAX = 4;
const ZMIN = -3;
const ZMAX = 3;

function buildLossSurface(): THREE.BufferGeometry {
  const positions = new Float32Array(RES * RES * 3);
  const colors = new Float32Array(RES * RES * 3);
  const losses: number[] = new Array(RES * RES);

  let minL = Infinity;
  let maxL = -Infinity;

  for (let i = 0; i < RES; i++) {
    for (let j = 0; j < RES; j++) {
      const x = XMIN + (i / (RES - 1)) * (XMAX - XMIN);
      const z = ZMIN + (j / (RES - 1)) * (ZMAX - ZMIN);
      const l = loss(x, z);
      const idx = i * RES + j;
      losses[idx] = l;
      positions[idx * 3] = x;
      positions[idx * 3 + 1] = l;
      positions[idx * 3 + 2] = z;
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
    }
  }

  const color = new THREE.Color();
  for (let k = 0; k < RES * RES; k++) {
    const t = (losses[k] - minL) / (maxL - minL || 1);
    // Low loss (cool emerald) → mid (amber) → high (rose)
    if (t < 0.5) {
      const u = t * 2;
      color.setHSL(0.4 - u * 0.06, 0.5 + u * 0.1, 0.5 - u * 0.08);
    } else {
      const u = (t - 0.5) * 2;
      color.setHSL(0.1 - u * 0.07, 0.6 + u * 0.05, 0.5 - u * 0.05);
    }
    colors[k * 3] = color.r;
    colors[k * 3 + 1] = color.g;
    colors[k * 3 + 2] = color.b;
  }

  const indices: number[] = [];
  for (let i = 0; i < RES - 1; i++) {
    for (let j = 0; j < RES - 1; j++) {
      const a = i * RES + j;
      const b = a + 1;
      const c = (i + 1) * RES + j;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

function LossSurface() {
  const geometry = React.useMemo(buildLossSurface, []);
  return (
    <mesh geometry={geometry} receiveShadow castShadow>
      <meshStandardMaterial
        vertexColors
        roughness={0.7}
        side={THREE.DoubleSide}
        flatShading={false}
      />
    </mesh>
  );
}

// ──────────────────────────────────────────────────────────────────
// Animated trajectory line + chasing particle
// ──────────────────────────────────────────────────────────────────

function liftedPath(path: Trajectory): [number, number, number][] {
  // Lift points slightly off the surface so the line doesn't z-fight
  return path.map(([x, y, z]) => [x, y + 0.12, z]);
}

interface AnimatedTrajectoryProps {
  path: Trajectory;
  color: string;
  progress: number;
  label: string;
  showParticle?: boolean;
}

function AnimatedTrajectory({
  path,
  color,
  progress,
  label,
  showParticle = false,
}: AnimatedTrajectoryProps) {
  const lifted = React.useMemo(() => liftedPath(path), [path]);
  const visibleCount = Math.max(2, Math.floor(progress * lifted.length));
  const visible = lifted.slice(0, visibleCount);
  const head = visible[visible.length - 1];
  const endPoint = lifted[lifted.length - 1];

  return (
    <group>
      <Line points={visible} color={color} lineWidth={3.5} />
      {showParticle && (
        <mesh position={head}>
          <icosahedronGeometry args={[0.12, 1]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={1.8}
          />
          <pointLight color={color} intensity={1.5} distance={2.5} />
        </mesh>
      )}
      {progress >= 1 && (
        <Text
          position={[endPoint[0] + 0.25, endPoint[1] + 0.45, endPoint[2]]}
          fontSize={0.3}
          color={color}
          anchorX="left"
          anchorY="middle"
        >
          {label}
        </Text>
      )}
    </group>
  );
}

// ──────────────────────────────────────────────────────────────────
// Markers
// ──────────────────────────────────────────────────────────────────

function StartMarker() {
  const y = loss(START_X, START_Z);
  return (
    <group position={[START_X, y + 0.15, START_Z]}>
      <mesh>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial color="#0f172a" emissive="#1e293b" />
      </mesh>
      <Text
        position={[0, 0.45, 0]}
        fontSize={0.26}
        color="#475569"
        anchorX="center"
        anchorY="middle"
      >
        start
      </Text>
    </group>
  );
}

function MinimumMarker() {
  const y = loss(MIN_X, MIN_Z);
  return (
    <group position={[MIN_X, y + 0.15, MIN_Z]}>
      <mesh>
        <torusGeometry args={[0.25, 0.04, 16, 32]} />
        <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={0.5} />
      </mesh>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.26}
        color="#059669"
        anchorX="center"
        anchorY="middle"
      >
        minimum
      </Text>
    </group>
  );
}

// ──────────────────────────────────────────────────────────────────
// Animation driver
// ──────────────────────────────────────────────────────────────────

const ANIM_DURATION = 5.5;

function AnimationDriver({
  playing,
  onComplete,
  onProgress,
}: {
  playing: boolean;
  onComplete: () => void;
  onProgress: (p: number) => void;
}) {
  const start = React.useRef<number | null>(null);
  useFrame((state) => {
    if (!playing) {
      start.current = null;
      return;
    }
    if (start.current === null) start.current = state.clock.elapsedTime;
    const elapsed = state.clock.elapsedTime - start.current;
    const p = Math.min(elapsed / ANIM_DURATION, 1);
    onProgress(p);
    if (p >= 1) {
      onComplete();
      start.current = null;
    }
  });
  return null;
}

// ──────────────────────────────────────────────────────────────────
// Public component
// ──────────────────────────────────────────────────────────────────

const SGD_COLOR = "#f43f5e"; // rose
const ADAM_COLOR = "#3b82f6"; // blue
const ADAMW_COLOR = "#10b981"; // emerald

export function OptimizerLandscape3D() {
  const [playing, setPlaying] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [visible, setVisible] = React.useState({
    sgd: true,
    adam: true,
    adamw: true,
  });

  const sgdPath = React.useMemo(() => runSGD(), []);
  const adamPath = React.useMemo(() => runAdam(), []);
  const adamwPath = React.useMemo(() => runAdamW(), []);

  function start() {
    setProgress(0);
    setPlaying(true);
  }

  function reset() {
    setPlaying(false);
    setProgress(0);
  }

  function onComplete() {
    setPlaying(false);
  }

  return (
    <div className="space-y-4">
      <SceneCanvas
        cameraPosition={[5, 7, 9]}
        height={520}
        caption="3D loss surface · three optimizers descend from the same start · AdamW is the one used to train GreenVision"
      >
        <LossSurface />
        <StartMarker />
        <MinimumMarker />

        {visible.sgd && (
          <AnimatedTrajectory
            path={sgdPath}
            color={SGD_COLOR}
            progress={progress}
            label="SGD"
          />
        )}
        {visible.adam && (
          <AnimatedTrajectory
            path={adamPath}
            color={ADAM_COLOR}
            progress={progress}
            label="Adam"
          />
        )}
        {visible.adamw && (
          <AnimatedTrajectory
            path={adamwPath}
            color={ADAMW_COLOR}
            progress={progress}
            label="AdamW"
            showParticle
          />
        )}

        <AnimationDriver
          playing={playing}
          onComplete={onComplete}
          onProgress={setProgress}
        />
      </SceneCanvas>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={start} disabled={playing}>
          <Play className="mr-2 h-4 w-4" />
          {playing ? "Descending…" : "Run optimizers"}
        </Button>
        <Button
          variant="outline"
          onClick={reset}
          disabled={!playing && progress === 0}
        >
          <RotateCcw className="mr-2 h-4 w-4" />
          Reset
        </Button>

        <div className="ml-auto flex flex-wrap gap-2 text-xs">
          {(["sgd", "adam", "adamw"] as const).map((key) => {
            const labels = { sgd: "SGD", adam: "Adam", adamw: "AdamW" };
            const colors = { sgd: SGD_COLOR, adam: ADAM_COLOR, adamw: ADAMW_COLOR };
            return (
              <button
                key={key}
                onClick={() =>
                  setVisible((v) => ({ ...v, [key]: !v[key] }))
                }
                className="inline-flex items-center gap-1.5 rounded-full border bg-card px-3 py-1 font-medium transition-opacity"
                style={{ opacity: visible[key] ? 1 : 0.35 }}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: colors[key] }}
                />
                {labels[key]}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <OptimizerExplain
          color={SGD_COLOR}
          name="SGD"
          formula="θ ← θ − η · ∇L(θ)"
          desc="Vanilla gradient descent. Sensitive to the gradient's per-axis magnitude — long thin valleys cause oscillation."
        />
        <OptimizerExplain
          color={ADAM_COLOR}
          name="Adam"
          formula="m, v ← exp-avg of g, g² · θ ← θ − η · m̂ / (√v̂ + ε)"
          desc="Per-axis adaptive learning rates via running variance. Faster convergence but L2 weight decay couples to the gradient and gets scaled by the same √v̂."
        />
        <OptimizerExplain
          color={ADAMW_COLOR}
          name="AdamW (used for training)"
          formula="θ ← θ(1 − ηλ) − η · m̂ / (√v̂ + ε)"
          desc="Decoupled weight decay — applied directly to parameters, NOT folded into the gradient. Restores the original L2 regularization behavior under adaptive LRs."
        />
      </div>
    </div>
  );
}

function OptimizerExplain({
  color,
  name,
  formula,
  desc,
}: {
  color: string;
  name: string;
  formula: string;
  desc: string;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <span
            className="h-3 w-3 rounded-full"
            style={{ background: color }}
          />
          <CardTitle className="text-sm">{name}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 pt-0">
        <p className="rounded bg-muted/60 px-2 py-1.5 font-mono text-[11px]">
          {formula}
        </p>
        <p className="text-xs text-muted-foreground">{desc}</p>
      </CardContent>
    </Card>
  );
}
