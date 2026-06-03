"use client";

import * as React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Pause, Play, RotateCcw } from "lucide-react";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SceneCanvasProps {
  children: React.ReactNode;
  cameraPosition?: [number, number, number];
  height?: number | string;
  className?: string;
  defaultAutoRotate?: boolean;
  /** Hide the floating controls (e.g. for tiny inset scenes). */
  hideControls?: boolean;
  /** Optional caption rendered top-left for accessibility. */
  caption?: string;
}

/**
 * Shared <Canvas> wrapper used by every 3D scene.
 *
 * Provides lighting, OrbitControls, a Reset Camera button, and an
 * Auto-rotate toggle. The reset works by remounting the Canvas (key bump),
 * which resets all R3F state — cheaper than reaching into the controls
 * ref and almost imperceptible to the user.
 */
export function SceneCanvas({
  children,
  cameraPosition = [10, 8, 10],
  height = 520,
  className,
  defaultAutoRotate = false,
  hideControls = false,
  caption,
}: SceneCanvasProps) {
  const { resolvedTheme } = useTheme();
  const [autoRotate, setAutoRotate] = React.useState(defaultAutoRotate);
  const [resetKey, setResetKey] = React.useState(0);
  const isDark = resolvedTheme === "dark";

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border bg-card",
        className,
      )}
      style={{ height }}
    >
      {caption && (
        <div className="pointer-events-none absolute left-3 top-3 z-10 text-xs text-muted-foreground">
          {caption}
        </div>
      )}

      {!hideControls && (
        <div className="absolute right-3 top-3 z-10 flex gap-1">
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8 bg-background/60 backdrop-blur"
            onClick={() => setAutoRotate((a) => !a)}
            aria-label={autoRotate ? "Pause rotation" : "Start rotation"}
          >
            {autoRotate ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8 bg-background/60 backdrop-blur"
            onClick={() => setResetKey((k) => k + 1)}
            aria-label="Reset camera"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      )}

      <Canvas
        key={resetKey}
        shadows
        camera={{ position: cameraPosition, fov: 50 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
      >
        <color attach="background" args={[isDark ? "#0a0a0a" : "#fafafa"]} />
        <fog
          attach="fog"
          args={[isDark ? "#0a0a0a" : "#fafafa", 25, 60]}
        />
        <ambientLight intensity={0.55} />
        <directionalLight
          position={[10, 14, 8]}
          intensity={0.85}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-15}
          shadow-camera-right={15}
          shadow-camera-top={15}
          shadow-camera-bottom={-15}
        />
        <directionalLight
          position={[-8, 6, -10]}
          intensity={0.25}
        />
        <OrbitControls
          autoRotate={autoRotate}
          autoRotateSpeed={1.4}
          enableDamping
          dampingFactor={0.06}
          minDistance={4}
          maxDistance={45}
        />
        <React.Suspense fallback={null}>{children}</React.Suspense>
      </Canvas>
    </div>
  );
}
